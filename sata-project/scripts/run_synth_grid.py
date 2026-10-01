#!/usr/bin/env python3
"""Run the synthetic grid (generator_spec.pdf, orchestration).

Units are (model, naming, task, strategy, label mode, k, seed); each finished
unit's rows are appended to the results parquet, and units already in the
file are skipped, so an interrupted run resumes where it stopped. One model
is loaded at a time.

Examples:
  # P1 smoke run
  python scripts/run_synth_grid.py --run smoke --models qwen2.5-7b-instruct \
      --tasks 2 --queries 4 --strategies random,counter_spurious
  # P2 learnability gate (pilot suite, ID queries only)
  python scripts/run_synth_grid.py --run p2_learnability --suite pilot \
      --models qwen2.5-7b-instruct,llama-3.1-8b-instruct --envs id \
      --strategies zero_shot,label_diversity --label-modes gold,shuffled --k 8,16,32

  # P4 main grid, seed-major so complete grids for the early seeds finish first
  python scripts/run_synth_grid.py --run p4/grid --models qwen2.5-7b-instruct \
      --namings abstract,aligned,flipped --order seed \
      --strategies zero_shot,random,label_diversity,feature_range,rule_diversity,counter_spurious,counter_prior

Results go to results/v3/synthetic/<run>.parquet unless --out is given.
The aligned and flipped namings use the final names in configs/lexicons.yaml,
and counter_prior the model's measured zero-shot prior on the pool rows,
results/v3/synthetic/p3/pool_prior_<model>.parquet (scripts/p3_pool_priors.py).
The names every task was shown under are written next to the results
(<run>.names.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first: sets thread/MPS env vars)
import json  # noqa: E402

from src.data.naming import load_lexicons, naming_table  # noqa: E402
from src.data.suites import load_suite, suite_dir  # noqa: E402
from src.experiments.synth_grid import UNIT_ORDERS, GridRunner, completed_units, enumerate_units, order_units  # noqa: E402
from src.inference.hf_runner import runner_from_config  # noqa: E402
from src.inference.priors import PoolPrior  # noqa: E402
from src.utils.shard import take_shard  # noqa: E402


def pool_prior_path(config, model: str) -> Path:
    return resolve_path(config.paths.results) / "p3" / f"pool_prior_{model}.parquet"


def _csv(value: str) -> list[str]:
    return [v for v in value.split(",") if v]


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run name; also the default results file name")
    parser.add_argument("--suite", default="eval")
    parser.add_argument("--models", default=",".join(m.name for m in config.models))
    parser.add_argument("--namings", default="abstract")
    parser.add_argument("--tasks", type=int, default=None, help="use the first N tasks of the suite")
    parser.add_argument("--queries", type=int, default=None, help="queries per task and environment (half per class)")
    parser.add_argument("--envs", default=",".join(config.evaluation.environments))
    parser.add_argument("--strategies", default="zero_shot,random,label_diversity,feature_range,"
                                                "rule_diversity,counter_spurious,similarity")
    parser.add_argument("--label-modes", default="gold")
    parser.add_argument("--k", default=str(config.selection.k), help="comma-separated demonstration counts")
    parser.add_argument("--seeds", type=int, default=config.selection.n_demo_seeds)
    parser.add_argument("--lexicon-stage", default="final", help="block of configs/lexicons.yaml to name with")
    parser.add_argument("--order", default="grid", choices=UNIT_ORDERS,
                        help="grid: as enumerated; seed: all of seed 0 first, then seed 1, ...")
    parser.add_argument("--shard", default=None, help="i/N: run every N-th unit starting at i (parallel workers)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true", help="list the units and exit")
    args = parser.parse_args()

    manifest = load_suite(args.suite, config)
    task_ids = [t["task_id"] for t in manifest["tasks"]][: args.tasks]
    models = {m.name: m for m in config.models}
    model_names = _csv(args.models)
    unknown = [m for m in model_names if m not in models]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; configured: {list(models)}")
    namings = _csv(args.namings)
    units = order_units(enumerate_units(
        model_names, namings, task_ids, _csv(args.strategies),
        _csv(args.label_modes), [int(k) for k in _csv(args.k)], args.seeds,
    ), args.order)
    units = take_shard(units, args.shard)
    out = Path(args.out) if args.out else resolve_path(config.paths.results) / f"{args.run}.parquet"
    print(f"{len(units)} units -> {out}")
    lexicons = {}
    if any(n != "abstract" for n in namings):
        lexicons = load_lexicons(args.lexicon_stage)
        tasks = [t for t in manifest["tasks"] if t["task_id"] in task_ids]
        missing = sorted({t["domain"] for t in tasks} - set(lexicons))
        if missing:
            parser.error(f"no {args.lexicon_stage!r} names for domain(s) {missing} in configs/lexicons.yaml")
        out.parent.mkdir(parents=True, exist_ok=True)
        names_path = out.with_suffix(".names.json")
        table = json.loads(names_path.read_text()) if names_path.exists() else {}
        for task_id, by_naming in naming_table(tasks, lexicons, config.selection.base_seed, tuple(namings)).items():
            table.setdefault(task_id, {}).update(by_naming)
        names_path.write_text(json.dumps(table, indent=1))
    needs_prior = any(s.startswith("counter_prior") for s in _csv(args.strategies))
    priors = {}
    for name in model_names:
        if not needs_prior:
            continue
        path = pool_prior_path(config, name)
        if not path.exists():
            parser.error(f"counter_prior needs {path}; run scripts/p3_pool_priors.py first")
        priors[name] = PoolPrior.load(path)
        have = set(map(tuple, priors[name].table[["task_id", "naming"]].drop_duplicates().to_numpy()))
        missing = [(t, n) for n in namings for t in task_ids if (t, n) not in have]
        if missing:
            parser.error(f"{path} lacks the pool prior of {len(missing)} (task, naming) pairs, e.g. {missing[:3]}")
    if args.dry_run:
        for u in units[:20]:
            print("  ", u.key)
        return

    done = completed_units(out)
    for name in model_names:
        remaining = sum(u.model == name and u.key not in done for u in units)
        if not remaining:
            print(f"{name}: all units already saved")
            continue
        runner = runner_from_config(models[name], config.inference)
        print(f"loaded {name} on {runner.device_name} ({runner.dtype}) in {runner.load_seconds:.0f}s")
        grid = GridRunner(
            runner, name, suite_dir(args.suite, config), manifest,
            base_seed=config.selection.base_seed, envs=_csv(args.envs),
            queries_per_env=args.queries, run_name=args.run, lexicons=lexicons,
            pool_prior=priors.get(name),
        )
        grid.run(units, out)
        if runner.n_prefix_fallbacks:
            print(f"  note: {runner.n_prefix_fallbacks} suffixes needed a full forward pass")
        runner.close()


if __name__ == "__main__":
    main()
