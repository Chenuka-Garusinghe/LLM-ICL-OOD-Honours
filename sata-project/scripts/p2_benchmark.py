#!/usr/bin/env python3
"""P2 benchmark: PyTorch throughput and cached-versus-full agreement (docs/research_plan.md, P2).

1. Throughput. On one pilot task's label-balanced demonstration set (k=8,
   about 1k tokens with Qwen), time a full forward pass, the prefix pass
   alone, and the scoring of each query suffix against the cached prefix.
   Timings are medians over repeats after a warm-up.
2. Agreement. For every pilot task, score its ID queries and the
   content-free query once through the prefix cache and once with full
   forward passes. The check passes when caching moves the mean per-task
   AUROC by at most 0.005 (src/evaluation/gates.py::cache_agreement).
   Prediction and calibrated-prediction agreement are reported too: margins
   differ slightly between the two paths (bf16 kernels), and a prediction
   flips only where |margin| is below that difference.

Writes results/v3/synthetic/p2/benchmark_<model>.json, and the per-suffix
scores to benchmark_<model>_agreement.parquet.

Usage: python scripts/p2_benchmark.py [--model qwen2.5-7b-instruct] [--k 8] [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first: sets thread/MPS env vars)
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.suites import load_suite, suite_dir  # noqa: E402
from src.evaluation.gates import cache_agreement  # noqa: E402
from src.experiments.synth_grid import GridRunner, Unit  # noqa: E402
from src.inference.hf_runner import runner_from_config  # noqa: E402


def _sync(device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device, repeats: int) -> list[float]:
    fn()                                   # warm-up
    out = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        out.append(time.perf_counter() - t0)
    return out


def _group(grid, model: str, task_id: str, k: int) -> dict:
    """The label-balanced gold demonstration set (seed 0) with the task's ID queries."""
    return grid.prompt_groups(Unit(model, "abstract", task_id, "label_diversity", "gold", k, 0))[0]


def compare_cached_full(runner, grid, model: str, task_ids: list[str], k: int) -> pd.DataFrame:
    """Score every task's suffixes through the prefix cache and with full passes.

    One row per suffix; the content-free suffix comes last with label -1.
    """
    rows = []
    for task_id in task_ids:
        g = _group(grid, model, task_id, k)
        cached = runner.score(g["prefix"], g["suffixes"])
        full = runner.batch_predict([g["prefix"] + s for s in g["suffixes"]])
        queries = [q for _, q in g["queries"].iterrows()]
        for i, (a, b) in enumerate(zip(cached, full)):
            q = queries[i] if i < len(queries) else None
            rows.append({
                "task_id": task_id,
                "query_id": -1 if q is None else int(q["query_id"]),
                "label": -1 if q is None else int(q["label"]),
                "logprob_0_cached": a.logprob_0, "logprob_1_cached": a.logprob_1, "prediction_cached": a.prediction,
                "logprob_0_full": b.logprob_0, "logprob_1_full": b.logprob_1, "prediction_full": b.prediction,
            })
    scores = pd.DataFrame(rows)
    scores["margin_cached"] = scores["logprob_1_cached"] - scores["logprob_0_cached"]
    scores["margin_full"] = scores["logprob_1_full"] - scores["logprob_0_full"]
    return scores


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-7b-instruct")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    models = {m.name: m for m in config.models}
    runner = runner_from_config(models[args.model], config.inference)
    manifest = load_suite("pilot", config)
    grid = GridRunner(runner, args.model, suite_dir("pilot", config), manifest,
                      base_seed=config.selection.base_seed, envs=["id"])
    task_ids = [t["task_id"] for t in manifest["tasks"]]

    # 1. Throughput on the first pilot task.
    g = _group(grid, args.model, task_ids[0], args.k)
    prefix, suffixes = g["prefix"], g["suffixes"]
    full = prefix + suffixes[0]
    n_full, n_prefix = len(runner.encode(full)), len(runner.encode(prefix))
    t_full = _time(lambda: runner.batch_predict([full]), runner.device, args.repeats)
    t_prefix = _time(lambda: runner._forward(runner.encode(prefix)), runner.device, args.repeats)
    t_score = _time(lambda: runner.score(prefix, suffixes), runner.device, args.repeats)
    per_suffix = (statistics.median(t_score) - statistics.median(t_prefix)) / len(suffixes)

    # 2. Cached versus full scoring on every pilot task.
    t0 = time.perf_counter()
    scores = compare_cached_full(runner, grid, args.model, task_ids, args.k)
    agreement_seconds = time.perf_counter() - t0
    agreement = cache_agreement(scores)

    report = {
        "model": args.model, "model_path": runner.model_path, "device": str(runner.device),
        "device_name": runner.device_name, "dtype": str(runner.dtype).replace("torch.", ""),
        "load_seconds": runner.load_seconds, "k": args.k, "repeats": args.repeats,
        "prompt_tokens": n_full, "prefix_tokens": n_prefix, "n_suffixes": len(suffixes),
        "full_forward_s": statistics.median(t_full), "full_forward_all_s": t_full,
        "prefill_tokens_per_s": n_full / statistics.median(t_full),
        "prefix_forward_s": statistics.median(t_prefix),
        "score_all_s": statistics.median(t_score), "cached_suffix_s": per_suffix,
        **agreement,
        "agreement_seconds": agreement_seconds, "prefix_fallbacks": runner.n_prefix_fallbacks,
        "passes_agreement": agreement.pop("passes"),     # the AUROC rule (gates.py)
    }
    out_dir = resolve_path(config.paths.results) / "p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"benchmark_{args.model}.json"
    out.write_text(json.dumps(report, indent=2))
    scores.to_parquet(out_dir / f"benchmark_{args.model}_agreement.parquet", index=False)
    print(json.dumps({k: v for k, v in report.items() if k != "full_forward_all_s"}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
