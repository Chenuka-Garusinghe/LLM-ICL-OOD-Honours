#!/usr/bin/env python3
"""P5 5b: SATA-MI and its SATA-names control (docs/research_plan.md, 5b implementation).

Subcommands:
  readers  Fit the per-naming reader models of the prior-augmented proxy on stage F's
           Qwen results for sata_llmval (src/models/prior_proxy.py) -> p5/readers.json.
  gate     Gate S3a-MI on sata_mi_val: top-k by the MI targets (lambda in 0, 0.5, 1) against
           label_diversity, counter_prior, counter_spurious and random, scored by the reader
           proxy; picks lambda -> p5/gate_s3a_mi.json.
  train    Train sata_mi or sata_names on sata_mi_train, early stopping on sata_mi_val.

The suites' pool priors and the queries' zero-shot margins are Qwen measurements
(stage G of scripts/run_5b_pod.sh): results/v3/synthetic/p5/mi/pool_prior_<suite>.parquet
and zero_shot_<suite>.parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first: sets thread env vars)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.naming import load_lexicons  # noqa: E402
from src.data.suites import suite_dir  # noqa: E402
from src.data.synthetic_bridge import FEATURE_NAMES, load_task_frame  # noqa: E402
from src.inference.priors import PoolPrior  # noqa: E402
from src.models.prior_proxy import Reader, demo_evidence, fit_reader, query_margins, reader_table  # noqa: E402
from src.models.sata import INPUT_WIDTH, build_sata  # noqa: E402
from src.models.sata_mi import ENVS, load_items, mi_targets, train_mi  # noqa: E402
from src.models.sata_train import sata_config  # noqa: E402
from src.evaluation.metrics import auroc  # noqa: E402
from src.selection.balanced_topk import balanced_top_k  # noqa: E402
from src.selection.protocols import SYNTHETIC_STRATEGIES, ProtocolContext, counter_prior_active, demo_seed, select  # noqa: E402

QI = "qwen2.5-7b-instruct"


def readers_path(R: Path) -> Path:
    return R / "p5" / "readers.json"


def load_readers(R: Path) -> dict[str, Reader]:
    raw = json.loads(readers_path(R).read_text())
    return {n: Reader(**v["reader"]) for n, v in raw.items()}


def mi_items(config, R: Path, suite: str):
    zs = pd.read_parquet(R / "p5" / "mi" / f"zero_shot_{suite}.parquet")
    pp = PoolPrior.load(R / "p5" / "mi" / f"pool_prior_{suite}.parquet")
    z = zs[zs["strategy"] == "zero_shot"].drop_duplicates(["task_id", "naming", "row_id"])
    qm = pd.Series((z["logprob_1"] - z["logprob_0"]).to_numpy(), index=pd.MultiIndex.from_frame(z[["task_id", "naming", "row_id"]]))
    return load_items(suite_dir(suite, config), pp, qm, load_lexicons("final"), config.selection.base_seed)


def cmd_readers(config, R):
    v = pd.read_parquet(R / "h200" / "p5" / "llmval.parquet")
    pp = PoolPrior.load(R / "h200" / "p5" / f"pool_prior_{QI}_llmval.parquet")
    frames = {t: load_task_frame(suite_dir("sata_llmval", config), t) for t in v["task_id"].unique()}
    tab = reader_table(v, frames, query_margins(v, pp))
    out = {}
    for n in ("abstract", "aligned", "flipped"):
        reader, info = fit_reader(tab[tab["naming"] == n])
        out[n] = {"reader": {"a1": reader.a1, "a2": reader.a2}, **info}
        print(n, out[n])
    readers_path(R).write_text(json.dumps(out, indent=2))


def _cell(item, env, sets, reader):
    Xq, yq, _, m0q = item.queries[env]
    ev = np.array([demo_evidence(item.X[s], item.y[s], Xq[i:i + 1])[0] for i, s in enumerate(sets)])
    return float(auroc(yq, reader.score(ev, m0q)))


def cmd_gate(config, R):
    readers = load_readers(R)
    items = mi_items(config, R, "sata_mi_val")
    k, base = config.selection.k, config.selection.base_seed
    rows = []
    for item in items:
        pool = pd.DataFrame(item.X, columns=FEATURE_NAMES)
        pool["label"] = item.y
        pool["y_clean"] = [m["y_clean"] for m in item.meta]
        pool["regime"] = [m["regime"] for m in item.meta]
        pool["agree_obs"] = [m["agree_obs"] for m in item.meta]
        pool["spurious_raw"] = item.shortcut
        ctx = ProtocolContext(feature_cols=FEATURE_NAMES, kinds={c: "numeric" for c in FEATURE_NAMES}, train_ref=pool,
                              prior_margin=pd.Series(item.m0, index=pool.index))
        active, _ = counter_prior_active(pool, ctx)
        for env, (Xq, yq, q_meta, m0q) in item.queries.items():
            methods = {}
            for lam in (0.0, 0.5, 1.0):
                methods[f"oracle_mi_{lam}"] = [balanced_top_k(mi_targets(qm, item.meta, item.shortcut, item.contra_clean,
                                                                         item.conflict, lam, k), item.y, k) for qm in q_meta]
            for strat in ("random", "label_diversity", "counter_spurious", "counter_prior"):
                s_ = strat if (strat != "counter_prior" or active) else "label_diversity"
                mech, comp = SYNTHETIC_STRATEGIES[s_]
                methods[strat] = [np.asarray(select(mech, comp, pool, None, k,
                                                    demo_seed(base, item.task_id, f"{strat}|{item.naming}|{env}", 0, "select", i), ctx)[0])
                                  for i in range(len(yq))]
            for m, sets in methods.items():
                rows.append({"method": m, "task_id": item.task_id, "naming": item.naming, "env": env,
                             "reader_auroc": _cell(item, env, sets, readers[item.naming])})
    df = pd.DataFrame(rows)
    out_dir = R / "p5"
    df.to_parquet(out_dir / "gate_s3a_mi.parquet", index=False)
    tab = df.groupby(["method", "env"])["reader_auroc"].mean().unstack()
    tab["worst"] = tab[list(ENVS)].min(axis=1)
    by_naming = df.groupby(["method", "naming"])["reader_auroc"].mean().unstack()
    print(tab.join(by_naming).sort_values("worst", ascending=False).round(3).to_string())
    oracles = tab.loc[[i for i in tab.index if i.startswith("oracle_mi")], "worst"]
    best_lam = float(oracles.idxmax().split("_")[-1])
    best_strategy = tab.loc[["random", "label_diversity", "counter_spurious", "counter_prior"], "worst"]
    verdict = {"lambda": best_lam, "oracle_worst": float(oracles.max()),
               "best_strategy": str(best_strategy.idxmax()), "best_strategy_worst": float(best_strategy.max()),
               "passed": bool(oracles.max() >= best_strategy.max()),
               "table": tab.join(by_naming).round(4).reset_index().to_dict(orient="records")}
    (out_dir / "gate_s3a_mi.json").write_text(json.dumps(verdict, indent=2))
    print(f"lambda {best_lam}: oracle {oracles.max():.4f} against {best_strategy.idxmax()} {best_strategy.max():.4f} "
          f"-> {'PASS' if verdict['passed'] else 'FAIL'}")


def cmd_train(config, R, variant: str, device: str | None, epochs: int | None):
    readers = load_readers(R)
    lam = json.loads((R / "p5" / "gate_s3a_mi.json").read_text())["lambda"]
    train, val = mi_items(config, R, "sata_mi_train"), mi_items(config, R, "sata_mi_val")
    overrides = {k: v for k, v in {"device": device, "epochs": epochs}.items() if v is not None}
    cfg = sata_config(config, train_seed=0, **overrides)
    ckpt = resolve_path(getattr(config.sata.checkpoints, variant))
    torch.manual_seed(0)
    model = build_sata(variant, INPUT_WIDTH[variant], cfg.d_model, cfg.n_heads, cfg.n_layers)
    print(f"{variant}: {len(train)} training and {len(val)} validation (task, naming) items, lambda {lam} -> {ckpt}")
    t0 = time.perf_counter()
    log = train_mi(model, variant, train, val, readers, cfg, checkpoint_path=ckpt,
                   resume_checkpoint_path=ckpt.with_suffix(".resume.pt"), lam=lam)
    best = max(log, key=lambda r: r["val_proxy"])
    (R / "p5" / f"train_{variant}.json").write_text(json.dumps(
        {"variant": variant, "lambda": lam, "minutes": (time.perf_counter() - t0) / 60, "best_epoch": best["epoch"],
         "best_val_proxy": best["val_proxy"], "best_topk": max(r["val_proxy_topk"] for r in log), "log": log}, indent=2))
    print(f"best reader-proxy {best['val_proxy']:.4f} at epoch {best['epoch']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("readers", "gate", "train"))
    ap.add_argument("--variant", choices=("sata_mi", "sata_names"), default="sata_mi")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    config = load_config()
    R = resolve_path(config.paths.results)
    if args.command == "readers":
        cmd_readers(config, R)
    elif args.command == "gate":
        cmd_gate(config, R)
    else:
        cmd_train(config, R, args.variant, args.device, args.epochs)


if __name__ == "__main__":
    main()
