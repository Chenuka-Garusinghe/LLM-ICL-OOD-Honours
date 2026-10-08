#!/usr/bin/env python3
"""Measured zero-shot priors on the pool rows (src/inference/priors.py,
`measure_pool_prior`): for every (task, naming), the model scores each of the
256 pool rows with no demonstrations, under the same system message and names
as the grid. counter_prior and the prior-data conflict read these margins
(`PoolPrior`), centred on the pool median.

Resumable per (task, naming). Output: results/v3/synthetic/p3/pool_prior_<model>.parquet.

  python scripts/p3_pool_priors.py --models qwen2.5-7b-instruct --namings abstract,aligned,flipped

For notebook 03.1, --demo-mask shows each pool row without the hidden features
and --count-free-system drops the measurement count from the system message,
as the masked grid does.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first: sets thread/MPS env vars)

import pandas as pd  # noqa: E402

from src.data.demo_mask import DEMO_MASKS  # noqa: E402
from src.data.naming import load_lexicons  # noqa: E402
from src.data.suites import load_suite, suite_dir  # noqa: E402
from src.experiments.synth_grid import GridRunner  # noqa: E402
from src.inference.hf_runner import runner_from_config  # noqa: E402
from src.inference.priors import measure_pool_prior  # noqa: E402
from src.utils.results_schema import append_results  # noqa: E402
from src.utils.shard import take_shard  # noqa: E402


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="qwen2.5-7b-instruct")
    parser.add_argument("--namings", default="abstract,aligned,flipped")
    parser.add_argument("--suite", default="eval")
    parser.add_argument("--lexicon-stage", default="final")
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--shard", default=None, help="i/N: score every N-th (task, naming) pool starting at i")
    parser.add_argument("--demo-mask", default="none", choices=tuple(DEMO_MASKS),
                        help="features hidden from the pool rows, as in the masked grid (notebook 03.1)")
    parser.add_argument("--count-free-system", action="store_true",
                        help="system message without the measurement count")
    parser.add_argument("--out", default=None, help="output parquet (default p3/pool_prior_<model>.parquet)")
    args = parser.parse_args()

    manifest = load_suite(args.suite, config)
    task_ids = [t["task_id"] for t in manifest["tasks"]][: args.tasks]
    namings = [n for n in args.namings.split(",") if n]
    lexicons = load_lexicons(args.lexicon_stage) if any(n != "abstract" for n in namings) else {}
    models = {m.name: m for m in config.models}
    for name in args.models.split(","):
        out = Path(args.out) if args.out else resolve_path(config.paths.results) / "p3" / f"pool_prior_{name}.parquet"
        done = set()
        if out.exists():
            done = set(map(tuple, pd.read_parquet(out, columns=["task_id", "naming"]).drop_duplicates().to_numpy()))
        mine = take_shard([(t, n) for n in namings for t in task_ids], args.shard)   # fixed before resuming
        todo = [pair for pair in mine if pair not in done]
        print(f"{name}: {len(todo)} (task, naming) pools to score ({len(done)} saved) -> {out}")
        if not todo:
            continue
        runner = runner_from_config(models[name], config.inference)
        grid = GridRunner(runner, name, suite_dir(args.suite, config), manifest,
                          base_seed=config.selection.base_seed, envs=["id"], lexicons=lexicons,
                          demo_mask=args.demo_mask, system_n_features=None if args.count_free_system else 10)
        t_start = time.perf_counter()
        for i, (task_id, naming) in enumerate(todo, 1):
            t0 = time.perf_counter()
            rows = measure_pool_prior(grid, task_id, naming)
            rows.insert(0, "model", name)
            rows["lexicon_stage"] = args.lexicon_stage if naming != "abstract" else None
            append_results(rows, out)
            el = time.perf_counter() - t_start
            print(f"  [{i}/{len(todo)}] {task_id} {naming}: {len(rows)} rows, {time.perf_counter() - t0:.0f}s "
                  f"(eta {el / i * (len(todo) - i) / 60:.0f} min)")
        runner.close()


if __name__ == "__main__":
    main()
