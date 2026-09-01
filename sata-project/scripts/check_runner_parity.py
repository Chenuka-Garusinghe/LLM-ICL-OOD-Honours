"""Parity check: HFRunner (transformers) vs the previous vLLM implementation.

Run this ONCE on a box where vLLM actually works (e.g. Gadi's container) to
confirm that replacing vLLM with transformers did not change the numbers the
thesis depends on. `batch_predict`'s logprobs feed confidence -> R-AUC, so a
silent divergence here would quietly corrupt RQ1's headline metric.

Usage (from the project root, in an environment with BOTH transformers and a
working vllm):
    python scripts/check_runner_parity.py [dataset_name] [--n 20] [--model 0]

Exits non-zero if any comparison exceeds the tolerances below.

What "parity" can and cannot mean here
--------------------------------------
Exact equality is NOT expected, for two reasons that are both improvements
rather than regressions:

* vLLM requested `logprobs=20` and matched decoded token strings, so a label
  token outside the top 20 was floored to -100 or returned INVALID. HFRunner
  indexes the label token ids directly, so it reports an exact logprob where
  vLLM reported a floor. Those rows are counted and reported separately rather
  than failed, since disagreeing with a floor is the point.
* Batched fp16/bf16 matmuls are not bitwise deterministic across different
  kernel/batching strategies, so small numerical drift is normal.

The check therefore asserts on:
  - argmax prediction agreement (the thing accuracy is computed from), and
  - closeness of the constrained two-way probability p0, which is what
    confidence and R-AUC actually use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.serialisation import ordered_feature_names, serialise_row  # noqa: E402
from src.inference.prompts import build_classification_prompt  # noqa: E402
from src.selection.random_select import select as random_select  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

# fp16/bf16 batched matmuls drift slightly between kernel strategies; these
# bound "same answer, different arithmetic" without hiding a real regression.
P0_ATOL = 0.02
LOGPROB_ATOL = 0.15
VLLM_FLOOR = -100.0


def build_prompts(dataset_name: str, n_queries: int, config) -> tuple[list[str], tuple[str, str]]:
    data_dir = resolve_path(config.paths.data_real) / dataset_name
    pool = pd.read_parquet(data_dir / "train_pool.parquet")
    test = pd.read_parquet(data_dir / "test_id.parquet").head(n_queries)
    feature_cols = json.load(open(data_dir / "feature_list.json"))
    label_tokens = tuple(json.load(open(data_dir / "label_tokens.json")))
    task_description = f"the '{dataset_name}' outcome"

    prompts = []
    for _, query in test.iterrows():
        ordered = ordered_feature_names({f: query[f] for f in feature_cols})
        query_line = serialise_row({f: query[f] for f in ordered})
        demo_ids = random_select(pool, query, k=config.k_primary, seed=config.seed_accuracy[0])
        demo_lines = [
            serialise_row(
                {f: pool.loc[i, f] for f in ordered_feature_names({f: pool.loc[i, f] for f in feature_cols})},
                label=str(pool.loc[i, "label"]),
            )
            for i in demo_ids
        ]
        prompts.append(
            build_classification_prompt(task_description, label_tokens, demo_lines, query_line)
        )
    return prompts, label_tokens


class LegacyVLLMRunner:
    """The pre-transformers implementation, kept here (and only here) purely as
    the reference point for this comparison -- it is no longer used in the
    pipeline. See src/inference/llm_runner.py's docstring for why it was
    replaced.
    """

    def __init__(self, model_path: str, max_model_len: int = 4096,
                 batch_size: int = 16, tensor_parallel: int = 1,
                 gpu_memory_utilisation: float = 0.9):
        from vllm import LLM

        # batch_size is an HFRunner knob (config.inference carries it so both
        # runners can be constructed with the same **vars(...)); vLLM does its
        # own batching internally, so accept and ignore it here.
        del batch_size

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_memory_utilisation,
            max_model_len=max_model_len,
        )

    def batch_predict(self, prompts, label_tokens):
        from vllm import SamplingParams

        from src.inference.llm_runner import INVALID, PredictionResult, get_confidence

        outputs = self.llm.generate(prompts, SamplingParams(logprobs=20, max_tokens=1, temperature=0))
        results = []
        for out in outputs:
            token_logprobs = out.outputs[0].logprobs[0]
            logprobs_dict = {lp.decoded_token.strip(): lp.logprob for lp in token_logprobs.values()}
            if label_tokens[0] not in logprobs_dict and label_tokens[1] not in logprobs_dict:
                results.append(PredictionResult(INVALID, 0.0, 0.0, 0.0, VLLM_FLOOR, VLLM_FLOOR))
                continue
            results.append(get_confidence(logprobs_dict, label_tokens))
        return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default=None,
                        help="dataset name; defaults to the first SELECTED_DATASETS entry")
    parser.add_argument("--n", type=int, default=20, help="number of queries to compare")
    parser.add_argument("--model", type=int, default=0, help="index into config.base_llms")
    args = parser.parse_args()

    config = load_config()
    if args.dataset is None:
        from src.data.tableshift_loader import SELECTED_DATASETS

        args.dataset = SELECTED_DATASETS[0]

    model_cfg = config.base_llms[args.model]
    prompts, label_tokens = build_prompts(args.dataset, args.n, config)
    print(f"comparing {len(prompts)} prompts | dataset={args.dataset} | model={model_cfg.name} "
          f"| labels={label_tokens}\n")

    from src.inference.llm_runner import HFRunner

    # Sequential, not simultaneous: two 8B models resident at once will OOM most
    # single-GPU boxes. vLLM in particular pre-allocates a large KV cache.
    print("running vLLM (reference)...")
    vllm_results = LegacyVLLMRunner(model_cfg.path, **vars(config.inference)).batch_predict(prompts, label_tokens)
    print("running HFRunner (transformers)...")
    hf_results = HFRunner(model_cfg.path, **vars(config.inference)).batch_predict(prompts, label_tokens)

    pred_mismatches, p0_failures, floored = [], [], []
    for i, (v, h) in enumerate(zip(vllm_results, hf_results)):
        # vLLM floored this label (outside its top-20) -> HFRunner is expected to
        # differ, and to be the more correct of the two. Report, do not fail.
        if v.prediction == "INVALID" or VLLM_FLOOR in (v.logprob_0, v.logprob_1):
            floored.append((i, v.logprob_0, v.logprob_1, h.logprob_0, h.logprob_1))
            continue
        if v.prediction != h.prediction:
            pred_mismatches.append((i, v.prediction, h.prediction, v.p0, h.p0))
        if abs(v.p0 - h.p0) > P0_ATOL:
            p0_failures.append((i, v.p0, h.p0, abs(v.p0 - h.p0)))

    compared = len(prompts) - len(floored)
    print(f"\n=== results ===")
    print(f"compared            : {compared}/{len(prompts)} "
          f"({len(floored)} skipped: vLLM floored the label outside its top-20)")
    print(f"prediction mismatch : {len(pred_mismatches)}")
    print(f"p0 beyond ±{P0_ATOL}   : {len(p0_failures)}")

    if floored:
        print(f"\n-- rows where vLLM floored a label (HFRunner reports the exact value) --")
        for i, vlp0, vlp1, hlp0, hlp1 in floored[:10]:
            print(f"  [{i}] vllm=({vlp0:.2f}, {vlp1:.2f})  hf=({hlp0:.2f}, {hlp1:.2f})")
        if len(floored) > 10:
            print(f"  ... {len(floored) - 10} more")

    if pred_mismatches:
        print(f"\n-- prediction mismatches --")
        for i, vp, hp, vp0, hp0 in pred_mismatches[:10]:
            print(f"  [{i}] vllm={vp} (p0={vp0:.4f})  hf={hp} (p0={hp0:.4f})")

    if p0_failures:
        print(f"\n-- p0 divergences --")
        for i, vp0, hp0, delta in p0_failures[:10]:
            print(f"  [{i}] vllm={vp0:.4f}  hf={hp0:.4f}  delta={delta:.4f}")

    if compared:
        deltas = [abs(v.p0 - h.p0) for v, h in zip(vllm_results, hf_results)
                  if v.prediction != "INVALID" and VLLM_FLOOR not in (v.logprob_0, v.logprob_1)]
        print(f"\np0 delta: max={max(deltas):.5f} mean={np.mean(deltas):.5f}")

    ok = not pred_mismatches and not p0_failures
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{'transformers matches vLLM within tolerance' if ok else 'divergence beyond tolerance -- investigate before relying on results'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
