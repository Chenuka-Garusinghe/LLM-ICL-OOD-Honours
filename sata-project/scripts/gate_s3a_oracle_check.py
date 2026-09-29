#!/usr/bin/env python3
"""Gate S3a -- target-oracle sanity check, run BEFORE any SATA training.

Selects top-k directly via the (v2, structural) target function
compute_target_scores -- no trained model at all -- and evaluates it with
the same worst-environment/id-pool proxy evaluate_sata_proxy now uses.
Compares against a couple of hand-designed protocols computed the same way.

Pass criterion (REDESIGN_RATIONALE.md §5.3): oracle-target worst-env
accuracy >= best hand-designed protocol's worst-env accuracy. If the
supervision signal itself can't beat the best protocol, nothing trained on
it can -- this gate would have caught v1's label leak in minutes instead of
after a full training run + LLM evaluation.

Needs no torch at all (bypasses the SATA model entirely) -- only numpy,
pandas, and xgboost, so it runs on this Mac without a GPU.

Usage: python3 scripts/gate_s3a_oracle_check.py [--n-tasks 30] [--config configs/v2.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generator import generate_val_test_tasks  # noqa: E402
from src.models.sata_targets import compute_target_scores  # noqa: E402
from src.models.standardise import standardise  # noqa: E402
from src.models.xgb_proxy import fit_predict_one, fit_predict_one_soft  # noqa: E402
from src.selection import counter_spurious, label_diversity, random_select  # noqa: E402
from src.selection.balanced_topk import balanced_top_k  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _pool_and_query_frames(task, pool_env: str, env_type: str, n_demos: int, n_queries: int, seed: int):
    demo_X, demo_y, demo_meta = task.generate_environment(pool_env, n_samples=n_demos, seed=seed)
    query_X, query_y, query_meta = task.generate_environment(env_type, n_samples=n_queries, seed=seed + 1)
    demo_X, query_X = standardise(demo_X, query_X)

    feature_cols = [f"feature_{i}" for i in range(demo_X.shape[1])]
    pool_df = pd.DataFrame(demo_X, columns=feature_cols)
    pool_df["label"] = demo_y
    pool_df["regime"] = [m["regime"] for m in demo_meta]
    pool_df["is_counter_spurious"] = [m["is_counter_spurious"] for m in demo_meta]
    return pool_df, demo_X, demo_y, demo_meta, query_X, query_y, query_meta, feature_cols


def _oracle_top_k(query_meta_row: dict, demo_meta: list[dict], k: int) -> np.ndarray:
    target_scores = compute_target_scores(query_meta_row, demo_meta)
    labels = np.array([m["label"] for m in demo_meta])
    return balanced_top_k(target_scores, labels, k)


def evaluate_method(tasks: list, gen_config, method: str, k: int = 8) -> dict[str, float]:
    """Worst-case-over-environments proxy accuracy for one selection method,
    with pool_env="id" (mirrors evaluate_sata_proxy's deployment-realistic
    default) -- `method` in {"oracle_target", "random", "label_diversity",
    "counter_spurious"}.
    """
    env_accs: dict[str, list[int]] = {env: [] for env in gen_config.environments}
    for task in tasks:
        for env_type in gen_config.environments:
            # id-pool for every environment except 'mechanism' -- mechanism
            # changes the causal->label mapping itself (a concept shift), so
            # an id-drawn demo sharing the query's raw regime carries the
            # OPPOSITE label under mechanism's mapping for complement-paired
            # rule families (verified empirically; see sata_train.py's
            # train_sata loop for the full derivation). Always matched-pool
            # there instead.
            pool_env = "mechanism" if env_type == "mechanism" else "id"
            pool_df, demo_X, demo_y, demo_meta, query_X, query_y, query_meta, feature_cols = _pool_and_query_frames(
                task, pool_env, env_type, gen_config.demos_per_task, 32, seed=0
            )
            for qi in range(len(query_y)):
                if method == "oracle_target":
                    top_k_idx = _oracle_top_k(query_meta[qi], demo_meta, k)
                    fit_predict = fit_predict_one
                elif method == "random":
                    top_k_idx = np.array(random_select.select(pool_df, pool_df.iloc[0], k, seed=qi))
                    fit_predict = fit_predict_one_soft
                elif method == "label_diversity":
                    top_k_idx = np.array(label_diversity.select(pool_df, pool_df.iloc[0], k, seed=qi))
                    fit_predict = fit_predict_one_soft
                elif method == "counter_spurious":
                    top_k_idx = np.array(counter_spurious.select(
                        pool_df, pool_df.iloc[0], k, seed=qi, is_counter_spurious=pool_df["is_counter_spurious"]
                    ))
                    fit_predict = fit_predict_one_soft
                else:
                    raise ValueError(f"Unknown method: {method}")

                acc = fit_predict(demo_X[top_k_idx], demo_y[top_k_idx], query_X[qi], query_y[qi])
                env_accs[env_type].append(acc)

    return {env: (float(np.mean(v)) if v else 0.0) for env, v in env_accs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=30)
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    import os

    os.environ["SATA_CONFIG"] = args.config
    config = load_config()
    gen_config = config.generator

    _, test_tasks = generate_val_test_tasks(gen_config)
    tasks = test_tasks[: args.n_tasks]
    print(f"Gate S3a oracle check: {len(tasks)} tasks, config={args.config}, k={args.k}\n")

    protocol_methods = ["random", "label_diversity", "counter_spurious"]
    results = {}
    for method in ["oracle_target"] + protocol_methods:
        per_env = evaluate_method(tasks, gen_config, method, k=args.k)
        worst = min(per_env.values())
        results[method] = (worst, per_env)
        print(f"{method:20s} worst-env={worst:.4f}  per-env={ {e: round(v,3) for e,v in per_env.items()} }")

    oracle_worst = results["oracle_target"][0]
    best_protocol_worst = max(results[m][0] for m in protocol_methods)
    best_protocol_name = max(protocol_methods, key=lambda m: results[m][0])

    print(f"\nOracle-target worst-env accuracy: {oracle_worst:.4f}")
    print(f"Best protocol ({best_protocol_name}) worst-env accuracy: {best_protocol_worst:.4f}")
    passed = oracle_worst >= best_protocol_worst
    print(f"\nGate S3a: {'PASS' if passed else 'FAIL'} "
          f"(oracle {'>=' if passed else '<'} best protocol)")


if __name__ == "__main__":
    main()
