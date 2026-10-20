#!/usr/bin/env python3
"""Gate S3a -- target-oracle sanity check, run BEFORE any SATA training (P5).

Selects top-k directly by the target function (no trained model) and scores
it with the XGBoost proxy that SATA's early stopping uses, against the
grid's hand-designed strategies on the same pools and queries
(src/models/sata_gate.py).

Pass criterion: the oracle's worst evaluated environment (id, covariate,
spurious_reversal) is at least the best hand-designed strategy's. The oracle
is the main SATA's targets (`oracle_matched`, the neutral matched targets;
hiccups/21); the planned targets are reported alongside. If the
supervision signal itself cannot beat the strategies, nothing trained on it
can. The mechanism placebo is scored and reported, not used.

v3: the tasks are SATA's validation tasks, drawn from the evaluation-task
distribution (`sample_sata_tasks`), with a column-role permutation per task
and environment. The version ported from main used the older task sampler
and v1's selection modules.

Usage: python scripts/gate_s3a_oracle_check.py [--n-tasks 100] [--k 8] [--only oracle_matched]
(`--only` scores just those methods and merges them into the saved results.)
Writes results/v3/synthetic/p5/gate_s3a.parquet (per task and environment)
and gate_s3a.json (summary and verdict).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first: sets thread env vars)
from src.models.sata_gate import STRATEGIES, proxy_compare, summarise  # noqa: E402
from src.models.sata_train import sample_sata_tasks, sata_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tasks", type=int, default=None, help="first N validation tasks (default: all)")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--only", default=None, help="comma-separated methods to (re)score and merge into the saved results")
    args = parser.parse_args()

    config = load_config()
    g = config.generator
    cfg = sata_config(config)
    k = args.k or config.selection.k
    _, val_tasks = sample_sata_tasks(0, g.n_val_tasks, base_seed=g.task_seed,
                                     spurious_strength_range=tuple(g.spurious_strength_range),
                                     label_noise=g.label_noise, n_features=g.n_features)
    tasks = val_tasks[: args.n_tasks]
    print(f"Gate S3a: {len(tasks)} validation tasks, k={k}, environments {cfg.environments}")

    out_dir = Path(args.out_dir) if args.out_dir else resolve_path(config.paths.results) / "p5"
    if args.only:
        only = tuple(args.only.split(","))
        new = proxy_compare(tasks, cfg, methods=only, k=k, base_seed=config.selection.base_seed, n_jobs=args.n_jobs)
        old = pd.read_parquet(out_dir / "gate_s3a.parquet")
        per_task = pd.concat([old[~old.method.isin(only)], new], ignore_index=True)
    else:
        per_task = proxy_compare(tasks, cfg, k=k, base_seed=config.selection.base_seed, n_jobs=args.n_jobs)
    table = summarise(per_task, tuple(cfg.selection_environments))
    print(table.round(3).to_string())

    oracle = float(table.loc["oracle_matched", "worst"])
    best = table.loc[list(STRATEGIES), "worst"]
    verdict = {
        "n_tasks": len(tasks), "k": k, "selection_environments": cfg.selection_environments,
        "oracle": "oracle_matched", "oracle_worst": oracle,
        "planned_targets_worst": float(table.loc["oracle_target", "worst"]),
        "oracle_designed_worst": float(table.loc["oracle_target_designed", "worst"]),
        "best_strategy": str(best.idxmax()), "best_strategy_worst": float(best.max()),
        "passed": bool(oracle >= best.max()),
        "summary": table.round(4).reset_index().to_dict(orient="records"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    per_task.to_parquet(out_dir / "gate_s3a.parquet", index=False)
    (out_dir / "gate_s3a.json").write_text(json.dumps(verdict, indent=2))
    print(f"\nOracle worst {oracle:.4f} vs best strategy {verdict['best_strategy']} {best.max():.4f}: "
          f"{'PASS' if verdict['passed'] else 'FAIL'}  -> {out_dir / 'gate_s3a.json'}")


if __name__ == "__main__":
    main()
