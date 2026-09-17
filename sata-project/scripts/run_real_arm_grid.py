"""Real-arm LLM runner for the mechanism x composition protocol grid.

Replaces NB02's inference loop. Two reasons this is a script and not a notebook:
the run is long and sharded (see SATA_RUN_SHARD / scripts/merge_sharded_results.py),
and a notebook kernel dying mid-run loses everything since the last manual save --
v1's git history carries a dozen consecutive "NB06 sync: periodic checkpoint of
in-progress results" commits, which is a notebook fighting its own lack of
resumability. Keep notebooks for NB00 diagnostics and NB08 figures.

Modes
-----
--mode corruption-gate
    The gate that should run BEFORE the grid (proposed Gate S0c in
    PROTOCOL_SPEC.md §5.1). Runs `random x balanced` with demonstration labels
    corrupted at 0% / 50% / 100% and nothing else varied. If accuracy is flat in
    corruption level, the model is not using the input-label mapping -- it is
    doing task recognition, every selection protocol has a zero ceiling on that
    dataset/model, and the grid is not worth renting a GPU for. This is the
    Min et al. 2022 vs Yoo et al. 2022 question asked directly of *your* models
    and *your* tasks, and it costs hours rather than days.

--mode grid
    The screen-reduced factorial: the mechanisms that cleared random-k on the
    CPU screen, crossed with all three compositions, plus zero-shot and a single
    counter_spurious negative control.

--dry-run
    Build every prompt, report token statistics and one rendered example, and
    exit without loading a model. Runs on a laptop. Use it to validate the
    pipeline end-to-end BEFORE paying for GPU time; a prompt-construction bug
    found on rented hardware costs money, the same bug found here costs nothing.

Resumability
------------
One parquet per (model, dataset, shard) unit under <out>/units/. A completed
unit is skipped on restart, so an interrupted rented instance resumes instead of
restarting. Merge with scripts/merge_sharded_results.py. Results are written
per unit rather than appended to one file because
results_schema.append_results rewrites the whole parquet on every call, which is
quadratic over a long run.

Prerequisite
------------
    PYTHONPATH=. python scripts/prep_shift_context.py --out _screen_cache
(CPU, ~1 min; fits the domain discriminator and target-prior estimate that the
shift-aware mechanisms and the `target_prior` composition read.)

Usage
-----
    # validate locally, no GPU
    PYTHONPATH=. python scripts/run_real_arm_grid.py --mode corruption-gate --dry-run

    # on the GPU box
    PYTHONPATH=. python scripts/run_real_arm_grid.py --mode corruption-gate \
        --model Llama-3.1-8B-Instruct --out results/v2/gate_corruption
    PYTHONPATH=. python scripts/run_real_arm_grid.py --mode grid \
        --model Llama-3.1-8B-Instruct --out results/v2/real_grid --shard 0 --n-shards 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.serialisation import serialise_row
from src.data.tableshift_loader import TASK_DESCRIPTIONS, load_codebook
from src.inference.calibration import calibrate, content_free_features
from src.inference.prompts import build_chat_messages
from src.selection.ordering import shuffle_order
from src.selection.protocols import ProtocolContext, select

warnings.filterwarnings("ignore")

DATASETS = ["brfss_diabetes", "acsincome", "acspubcov", "anes"]
LABEL_TOKENS = ("No", "Yes")  # overridden per-run by --label-tokens (see main())

# Screen-reduced grid (PROTOCOL_SPEC.md §4.1): mechanisms that cleared random-k,
# plus random as the baseline. counter_spurious is carried as ONE negative
# control cell, not a full row -- it scored below chance on all four datasets.
GRID_CELLS = [
    ("similarity", "free"), ("similarity", "balanced"), ("similarity", "target_prior"),
    ("feature_coverage", "free"), ("feature_coverage", "balanced"), ("feature_coverage", "target_prior"),
    ("importance_weighted", "free"), ("importance_weighted", "balanced"), ("importance_weighted", "target_prior"),
    ("random", "free"), ("random", "balanced"), ("random", "target_prior"),
    ("counter_spurious", "balanced"),
]


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #
# Implied-decimal rescaling, applied at serialisation time only (never to the
# values any selector or estimator sees, so no protocol behaviour changes).
#
# BRFSS stores several calculated fields with implied decimal places. BMI5 is
# the unambiguous case: it ranges 1275-9765 in the cache, and the codebook
# documents the scaling in BMI5CAT's own value labels ("Obese (3000 <= BMI <
# 9999)", "Normal Weight (1850 <= BMI < 2500)"). Left unscaled, the prompt tells
# the model a person's BMI is 3228 -- a quantity outside anything it has seen,
# which defeats the entire reason the real arm has exploitable priors.
#
# ONLY features whose scaling the codebook itself documents are listed here.
# Others flagged by the --dry-run range audit (e.g. brfss DRNK_PER_WEEK, max
# 58100, likely drinks-per-week x100) must be verified against the BRFSS
# codebook before being added -- do not guess a divisor.
FEATURE_SCALES: dict[str, dict[str, float]] = {
    "brfss_diabetes": {"BMI5": 0.01},
}

# Rescaling BMI5 creates a second problem: BMI5CAT's value labels quote the
# unscaled thresholds, so the prompt would read "BMI: 32.28" next to "BMI
# category: Obese (3000 <= BMI < 9999)" -- two mutually contradictory scales in
# the same row. Rewrite the labels onto the same scale as the rescaled feature.
VALUE_LABEL_REWRITES: dict[str, dict[str, dict[int, str]]] = {
    "brfss_diabetes": {
        "BMI5CAT": {
            0: "Obese (BMI >= 30)",
            1: "Normal Weight (18.5 <= BMI < 25)",
            2: "Overweight (25 <= BMI < 30)",
            3: "Underweight (BMI < 18.5)",
        }
    }
}


def apply_label_rewrites(codebook: dict, dataset: str) -> dict:
    """Return `codebook` with dataset-specific value-label corrections applied."""
    rewrites = VALUE_LABEL_REWRITES.get(dataset, {})
    if not rewrites:
        return codebook
    cb = {k: dict(v) for k, v in codebook.items()}
    for col, mapping in rewrites.items():
        if col in cb:
            cb[col]["values"] = dict(mapping)
    return cb


def _features(row: pd.Series, feature_cols: list[str], dataset: str | None = None) -> dict:
    scales = FEATURE_SCALES.get(dataset or "", {})
    out = {}
    for c in feature_cols:
        v = row[c]
        if c in scales:
            try:
                v = float(v) * scales[c]
            except (TypeError, ValueError):
                pass
        out[c] = v
    return out


def build_unit_prompts(
    ctx: ProtocolContext, pool: pd.DataFrame, queries: pd.DataFrame, codebook: dict,
    mechanism: str, composition: str, k: int, seed: int, corruption: float,
    task_desc: tuple[str, str, str, str], calibrated: bool, dataset: str,
    strip_label_meanings: bool = False,
) -> tuple[list[list[dict]], list[list[dict]], list[dict]]:
    """Return (chat message lists, content-free message lists, per-row metadata).

    `strip_label_meanings`: verbaliser diagnostic (Step 1, GATE_S0C_FINDINGS.md
    §7). When True, the system prompt drops "where {label_0} = ..., {label_1}
    = ..." and falls back to the bare "one word: {label_0} or {label_1}"
    instruction -- Wei et al. 2023's symbol-tuning setting, which relies
    purely on the in-context demonstrations rather than a semantic gloss.
    Combine with LABEL_TOKENS = ("A","B") or ("0","1") for the neutral-symbol
    arms; leave LABEL_TOKENS as ("No","Yes") with this False for baseline/swap.
    """
    rng = np.random.default_rng(seed)
    sentence, _noun, m0, m1 = task_desc
    label_meanings = None if strip_label_meanings else (m0, m1)
    msgs, cf_msgs, meta = [], [], []

    for qi in range(len(queries)):
        q = queries.iloc[qi]
        query_line = serialise_row(_features(q, ctx.feature_cols, dataset), label=None, codebook=codebook)

        if k == 0:                                     # zero-shot
            demo_lines, demo_ids, order_seed = [], [], -1
        else:
            demo_ids, _ = select(mechanism, composition, pool, q, k, seed, ctx)
            order_seed = int(rng.integers(0, 2**31 - 1))
            demo_ids = shuffle_order(list(demo_ids), order_seed)
            demo_lines = []
            for did in demo_ids:
                d = pool.loc[did]
                y = int(d[ctx.label_col])
                if corruption > 0 and rng.random() < corruption:
                    y = 1 - y                          # flip this demo's label
                demo_lines.append(
                    serialise_row(_features(d, ctx.feature_cols, dataset), label=LABEL_TOKENS[y], codebook=codebook)
                )

        msgs.append(build_chat_messages(sentence, LABEL_TOKENS, demo_lines, query_line,
                                        label_meanings=label_meanings))
        if calibrated:
            cf_line = serialise_row(content_free_features(_features(q, ctx.feature_cols, dataset)),
                                    label=None, codebook=codebook)
            cf_msgs.append(build_chat_messages(sentence, LABEL_TOKENS, demo_lines, cf_line,
                                               label_meanings=label_meanings))
        meta.append(dict(query_id=int(queries.index[qi]), label=int(q[ctx.label_col]),
                         demo_ids=json.dumps([int(i) for i in demo_ids]), order_seed=order_seed))
    return msgs, cf_msgs, meta


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
def enumerate_units(mode: str, seeds: list[int], datasets: list[str],
                     corruptions: tuple[float, ...] = (0.0, 0.5, 1.0)) -> list[dict]:
    units = []
    if mode == "corruption-gate":
        for ds in datasets:
            for c in corruptions:
                for s in seeds:
                    units.append(dict(dataset=ds, mechanism="random", composition="balanced",
                                      corruption=c, seed=s, k=8))
            for s in seeds[:1]:
                units.append(dict(dataset=ds, mechanism="zero_shot", composition="none",
                                  corruption=0.0, seed=s, k=0))
    else:
        for ds in datasets:
            for mech, comp in GRID_CELLS:
                for s in seeds:
                    units.append(dict(dataset=ds, mechanism=mech, composition=comp,
                                      corruption=0.0, seed=s, k=8))
            for s in seeds[:1]:
                units.append(dict(dataset=ds, mechanism="zero_shot", composition="none",
                                  corruption=0.0, seed=s, k=0))
    return units


def unit_path(out: Path, model: str, u: dict) -> Path:
    name = (f"{u['dataset']}__{u['mechanism']}__{u['composition']}"
            f"__corr{u['corruption']:.2f}__seed{u['seed']}.parquet")
    return out / "units" / model.replace("/", "_") / name


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["corruption-gate", "grid"], required=True)
    ap.add_argument("--cache-dir", default="_screen_cache")
    ap.add_argument("--raw-cache", default="data/tableshift_raw_cache")
    ap.add_argument("--out", default="results/v2/real_grid")
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct")
    ap.add_argument("--model-path", default=None, help="defaults to the HF id for --model in configs/v2.yaml")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=256)
    ap.add_argument("--n-queries", type=int, default=250)
    ap.add_argument("--query-split", default="ood", choices=["ood", "id", "both"])
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--label-tokens", nargs=2, default=["No", "Yes"],
                     metavar=("TOKEN_FOR_CLASS0", "TOKEN_FOR_CLASS1"),
                     help="verbaliser diagnostic: the literal tokens the model reads/emits for "
                          "class 0 / class 1. Swap the pair (e.g. 'Yes' 'No') to test token- vs "
                          "concept-anchoring -- semantic meanings stay attached to the same class "
                          "indices via TASK_DESCRIPTIONS, only which literal string denotes which "
                          "class changes.")
    ap.add_argument("--strip-label-meanings", action="store_true",
                     help="verbaliser diagnostic: drop the '{label_0} = <meaning>' gloss from the "
                          "system prompt (Wei et al. 2023 symbol-tuning setting). Pair with neutral "
                          "--label-tokens (A/B, 0/1) to remove semantic priors entirely.")
    ap.add_argument("--corruptions", type=float, nargs="+", default=None,
                     help="restrict corruption-gate mode to these levels (default: 0.0 0.5 1.0). "
                          "Pass '0.0' alone for a verbaliser-only diagnostic with no corruption sweep.")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global LABEL_TOKENS
    LABEL_TOKENS = tuple(args.label_tokens)  # ("No","Yes") default; swap/neutral for the verbaliser diagnostic

    out = Path(args.out)
    calibrated = not args.no_calibration
    splits = ["ood", "id"] if args.query_split == "both" else [args.query_split]
    corruptions = tuple(args.corruptions) if args.corruptions is not None else (0.0, 0.5, 1.0)

    # ---- load prepped contexts ---------------------------------------------
    ctxs = {}
    for ds in args.datasets:
        p = Path(args.cache_dir) / f"{ds}.pkl"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run: PYTHONPATH=. python scripts/prep_shift_context.py "
                f"--cache {args.raw_cache} --out {args.cache_dir}")
        with open(p, "rb") as f:
            blob = pickle.load(f)
        # load_codebook(), NOT json.load(): JSON object keys are always strings,
        # so a raw json.load leaves the `values` maps keyed "0"/"1" while the
        # parquet stores 0.0/1.0, every categorical lookup misses, and each
        # categorical feature silently renders as a bare number ("high blood
        # pressure: 0.0" instead of "high blood pressure: Yes"). That destroys
        # the feature semantics the real arm exists to exercise.
        blob["codebook"] = apply_label_rewrites(load_codebook(Path(args.raw_cache) / ds), ds)
        ctxs[ds] = blob

    units = enumerate_units(args.mode, args.seeds, args.datasets, corruptions=corruptions)
    units = [u for i, u in enumerate(units) if i % args.n_shards == args.shard]
    pending = [u for u in units if not unit_path(out, args.model, u).exists()]
    print(f"[grid] mode={args.mode} model={args.model} units={len(units)} "
          f"pending={len(pending)} splits={splits} calibration={calibrated}", flush=True)

    # ---- build prompts -----------------------------------------------------
    built = []
    for u in pending:
        blob = ctxs[u["dataset"]]
        ctx = ProtocolContext(feature_cols=blob["feats"], kinds=blob["kinds"], shift=blob["shift"],
                              train_ref=blob["train"], proxy_features=blob["proxy_features"])
        rng = np.random.default_rng(u["seed"])
        train = blob["train"]
        # unstratified pool: preserves the source label prior so `composition`
        # is a live factor (build_demo_pool's 50/50 pool would pre-empt it)
        pool = train.iloc[rng.choice(len(train), args.pool_size, replace=False)].copy()
        pool.index = pd.RangeIndex(len(pool))

        for split in splits:
            src = blob["test_ood"] if split == "ood" else blob["test_id"]
            q = src.iloc[rng.choice(len(src), min(args.n_queries, len(src)), replace=False)]
            mech = "random" if u["mechanism"] == "zero_shot" else u["mechanism"]
            comp = "balanced" if u["mechanism"] == "zero_shot" else u["composition"]
            msgs, cf, meta = build_unit_prompts(
                ctx, pool, q, blob["codebook"], mech, comp, u["k"], u["seed"],
                u["corruption"], TASK_DESCRIPTIONS[u["dataset"]], calibrated, u["dataset"],
                strip_label_meanings=args.strip_label_meanings)
            built.append(dict(unit=u, split=split, msgs=msgs, cf=cf, meta=meta))

    n_prompts = sum(len(b["msgs"]) + len(b["cf"]) for b in built)
    print(f"[grid] built {n_prompts} prompts across {len(built)} unit-splits", flush=True)

    if args.dry_run:
        ex = built[0]
        print("\n--- example prompt (system) ---\n" + ex["msgs"][0][0]["content"])
        print("\n--- example prompt (user, first 900 chars) ---\n" + ex["msgs"][0][1]["content"][:900])
        # Per-feature value audit. The point is that a human READS this before
        # renting a GPU. It is how the BRFSS BMI5 scaling was caught: the column
        # carries two implied decimals, so the prompt was showing the model
        # "Body Mass Index (BMI): 3228.00" for a BMI of 32.28 -- and the real arm
        # depends entirely on the model's priors over quantities like BMI being
        # meaningful. The codebook documents the scaling in BMI5CAT's own value
        # labels ("Obese (3000 <= BMI < 9999)"), but BMI5 is kind=numeric with no
        # value map, so nothing rescales it.
        print("\n[dry-run] numeric feature ranges (check these are physically plausible):")
        for ds in args.datasets:
            b = ctxs[ds]
            num = [f for f in b["feats"] if b["kinds"].get(f) != "categorical"]
            print(f"  {ds}:")
            for f in num:
                s = pd.to_numeric(b["train"][f], errors="coerce")
                print(f"    {f:22s} min={s.min():>10.2f} med={s.median():>10.2f} max={s.max():>10.2f}")
        chars = [len(m[0]["content"]) + len(m[1]["content"]) for b in built for m in b["msgs"]]
        print(f"\n[dry-run] prompt chars: mean={np.mean(chars):.0f} p95={np.percentile(chars,95):.0f} "
              f"max={max(chars)}  (~{max(chars)/3.6:.0f} tokens at max; limit {args.max_model_len})")
        print(f"[dry-run] total prompts if run: {n_prompts}")
        return

    # ---- model -------------------------------------------------------------
    from src.inference.llm_runner import VLLMWorkerRunner
    model_path = args.model_path or {
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
        "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    }.get(args.model, args.model)

    runner = VLLMWorkerRunner(model_path=model_path, tensor_parallel=args.tensor_parallel,
                              gpu_memory_utilisation=args.gpu_mem_util,
                              max_model_len=args.max_model_len, quantization=args.quantization)
    def _render(formatter, msgs: list[dict]) -> str:
        # ChatFormatter.render takes (system, user) strings, not a message
        # list -- build_chat_messages returns exactly the [system, user] pair
        # it expects. There is no .format()/__call__ on ChatFormatter.
        by_role = {m["role"]: m["content"] for m in msgs}
        return formatter.render(by_role["system"], by_role["user"])

    try:
        formatter = runner.chat_formatter()
        for b in built:
            u, split = b["unit"], b["split"]
            prompts = [_render(formatter, m) for m in b["msgs"]]
            preds = runner.batch_predict(prompts, LABEL_TOKENS)
            cf_preds = None
            if b["cf"]:
                cf_prompts = [_render(formatter, m) for m in b["cf"]]
                cf_preds = runner.batch_predict(cf_prompts, LABEL_TOKENS)

            rows = []
            for i, (p, m) in enumerate(zip(preds, b["meta"])):
                # PredictionResult.prediction is the label STRING ("No"/"Yes"),
                # not an int -- map through LABEL_TOKENS so it compares
                # correctly against `label` (int 0/1). Comparing the raw
                # string against an int label silently evaluates False for
                # every row (accuracy identically 0, no exception raised).
                pred_int = LABEL_TOKENS.index(p.prediction) if p.prediction in LABEL_TOKENS else -1
                r = dict(arm="real", dataset=u["dataset"], environment=split, model=args.model,
                         method=f"{u['mechanism']}x{u['composition']}", mechanism=u["mechanism"],
                         composition=u["composition"], corruption=u["corruption"],
                         seed=u["seed"], k=u["k"], pool_provenance="id_pool",
                         prompt_version="v2_chat", query_id=m["query_id"], label=m["label"],
                         demo_ids=m["demo_ids"], order_seed=m["order_seed"],
                         prediction_raw=pred_int, prediction=pred_int,
                         logprob_0=p.logprob_0, logprob_1=p.logprob_1,
                         logprob_0_cf=np.nan, logprob_1_cf=np.nan)
                if cf_preds is not None:
                    c = cf_preds[i]
                    r["logprob_0_cf"], r["logprob_1_cf"] = c.logprob_0, c.logprob_1
                    l0, l1 = calibrate(p.logprob_0, p.logprob_1, c.logprob_0, c.logprob_1)
                    r["prediction"] = int(l1 > l0)
                rows.append(r)

            path = unit_path(out, args.model, u)
            path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows)
            # written per unit, immediately -- an interrupted rented instance
            # keeps every completed unit
            df.to_parquet(path.with_name(path.stem + f"__{split}.parquet"), index=False)
            acc = float((df.prediction == df.label).mean())
            print(f"[grid] {u['dataset']} {u['mechanism']}x{u['composition']} "
                  f"corr={u['corruption']:.2f} seed={u['seed']} {split}: acc={acc:.3f}", flush=True)
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
