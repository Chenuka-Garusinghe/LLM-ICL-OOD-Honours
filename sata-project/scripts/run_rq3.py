#!/usr/bin/env python3
"""RQ3 behavioural measurements (src/experiments/rq3.py; generator_spec.pdf,
faithfulness and reliance): hot-deck ablations (pi_behav), +/-0.5 nudges
(directional sensitivity, DFI) and the self-reported ranking (pi_self), on the
grid's label_diversity demonstration sets and the ID queries.

Units are (model, naming, task, seed), run seed-major and resumable like the
grid. Scores go to results/v3/synthetic/<run>.parquet and rankings to
<run>_rankings.parquet.

  python scripts/run_rq3.py --run p4/rq3 --models qwen2.5-7b-instruct --seeds 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first: sets thread/MPS env vars)
from src.data.naming import load_lexicons  # noqa: E402
from src.data.suites import load_suite, suite_dir  # noqa: E402
from src.experiments.rq3 import RQ3Runner, enumerate_rq3_units  # noqa: E402
from src.experiments.synth_grid import GridRunner  # noqa: E402
from src.inference.hf_runner import runner_from_config  # noqa: E402
from src.utils.shard import take_shard  # noqa: E402


def _csv(value: str) -> list[str]:
    return [v for v in value.split(",") if v]


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--suite", default="eval")
    parser.add_argument("--models", default="qwen2.5-7b-instruct")
    parser.add_argument("--namings", default="abstract,aligned,flipped")
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--queries", type=int, default=None, help="ID queries per task (half per class)")
    parser.add_argument("--seeds", type=int, default=config.selection.n_demo_seeds)
    parser.add_argument("--k", type=int, default=config.selection.k)
    parser.add_argument("--no-rank", action="store_true", help="skip the self-reported ranking")
    parser.add_argument("--shard", default=None, help="i/N: run every N-th unit starting at i (parallel workers)")
    args = parser.parse_args()

    manifest = load_suite(args.suite, config)
    task_ids = [t["task_id"] for t in manifest["tasks"]][: args.tasks]
    namings = _csv(args.namings)
    models = {m.name: m for m in config.models}
    units = take_shard(enumerate_rq3_units(_csv(args.models), namings, task_ids, args.seeds, args.k), args.shard)
    out = resolve_path(config.paths.results) / f"{args.run}.parquet"
    print(f"{len(units)} RQ3 units -> {out}")
    lexicons = load_lexicons("final") if any(n != "abstract" for n in namings) else {}
    for name in _csv(args.models):
        runner = runner_from_config(models[name], config.inference)
        print(f"loaded {name} on {runner.device_name} ({runner.dtype}) in {runner.load_seconds:.0f}s")
        grid = GridRunner(runner, name, suite_dir(args.suite, config), manifest,
                          base_seed=config.selection.base_seed, envs=["id"], run_name=args.run, lexicons=lexicons)
        RQ3Runner(grid, queries_per_env=args.queries, rank=not args.no_rank).run(units, out)
        runner.close()


if __name__ == "__main__":
    main()
