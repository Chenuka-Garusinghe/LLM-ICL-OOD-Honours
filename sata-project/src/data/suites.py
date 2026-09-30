"""The standard task suites (`suites` in configs/default.yaml).

`eval` holds the evaluation tasks and `pilot` the P2 pilot tasks. They come
from the same distribution with disjoint task ids, so P2 picks k without
touching the evaluation tasks.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.data.generator import sample_eval_tasks
from src.data.synthetic_bridge import load_manifest, write_suite
from src.utils.config import resolve_path


def suite_dir(name: str, config: SimpleNamespace) -> Path:
    return resolve_path(config.paths.data_synthetic) / "v3" / name


def suite_tasks(name: str, config: SimpleNamespace):
    g = config.generator
    return sample_eval_tasks(
        n_tasks=getattr(config.suites, name),
        id_prefix=name,
        base_seed=g.task_seed,
        spurious_strength_range=tuple(g.spurious_strength_range),
        label_noise=g.label_noise,
        n_features=g.n_features,
    )


def make_suite(name: str, config: SimpleNamespace) -> dict:
    """Sample the suite's tasks and write their parquets and manifest."""
    g, ev = config.generator, config.evaluation
    return write_suite(
        suite_tasks(name, config),
        suite_dir(name, config),
        suite=name,
        n_pool=g.demos_per_task,
        n_test_id=g.n_test_id,
        n_test_ood=g.n_test_ood,
        ood_envs=tuple(e for e in ev.environments if e != "id"),
        queries_per_class=ev.queries_per_class,
    )


def load_suite(name: str, config: SimpleNamespace) -> dict:
    path = suite_dir(name, config)
    if not (path / "task_manifest.json").exists():
        raise FileNotFoundError(
            f"no {name!r} suite at {path}; run `python scripts/make_synth_data.py` (or NB01) first"
        )
    return load_manifest(path)
