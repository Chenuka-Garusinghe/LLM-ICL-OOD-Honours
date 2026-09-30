"""Per-task synthetic datasets (v3): generation, standardisation, column-role
permutation, query sampling and the task manifest.

Each task is stored as one long-format parquet holding its pool, its ID test
split and one OOD test split per evaluated environment
(generator_spec.pdf, dataset construction):

- ID rows are generated once; the first `n_pool` are the demonstration pool
  and the rest are ID test rows. Every environment is answered with
  demonstrations from this pool.
- Every split is z-scored with the pool's mean and standard deviation, so a
  covariate shift shows up as an offset in z-units instead of being
  standardised away, and f8 no longer stands out by its spread.
- Columns are shown in a per-task random order (the column-role permutation),
  so the spurious and noise features are not always f8 and f9. Downstream
  code reads column roles from the manifest, never from position.
- Queries are 10 per class per test split, drawn with a fixed seed.

v2 merged five tasks into one pool per shift type; that module is gone.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.generator import SyntheticTask

FEATURE_NAMES = [f"f{i}" for i in range(10)]
LABEL_TOKENS = ("0", "1")
EVAL_OOD_ENVS = ("covariate", "spurious_reversal")
META_COLUMNS = [
    "task_id", "split", "env", "row_id", "label", "y_clean", "regime",
    "agree_clean", "agree_obs", "spurious_raw", "is_query", "query_id",
]


def make_codebook(names: list[str] = FEATURE_NAMES) -> dict:
    """Abstract naming: every column is shown under its displayed name."""
    return {f: {"name_extended": f, "type": "continuous"} for f in names}


def _seed(data_seed: int, purpose: str) -> int:
    """Independent, order-free seed for one use of a task's data stream."""
    ss = np.random.SeedSequence([int(data_seed), zlib.crc32(purpose.encode())])
    return int(ss.generate_state(1)[0])


def _sample_queries(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Query ids (0..2*per_class-1, class-interleaved at random) for a test
    split, -1 for rows that are not queries."""
    query_id = np.full(len(labels), -1, dtype=int)
    chosen = []
    for cls in (0, 1):
        pos = np.flatnonzero(labels == cls)
        if len(pos) < per_class:
            raise ValueError(f"only {len(pos)} rows of class {cls}; need {per_class} queries")
        chosen.extend(rng.choice(pos, size=per_class, replace=False))
    chosen = np.array(chosen)
    rng.shuffle(chosen)
    query_id[chosen] = np.arange(len(chosen))
    return query_id


def build_task_frame(
    task: SyntheticTask,
    n_pool: int = 256,
    n_test_id: int = 100,
    n_test_ood: int = 100,
    ood_envs: tuple[str, ...] = EVAL_OOD_ENVS,
    queries_per_class: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one task's long-format frame and its manifest entry."""
    if task.data_seed is None:
        raise ValueError(f"task {task.task_id} has no data_seed; sample it with sample_eval_tasks")
    n_feat = task.n_features

    blocks: list[tuple[str, str, np.ndarray, np.ndarray, list[dict]]] = []
    X_id, y_id, meta_id = task.generate_environment("id", n_pool + n_test_id, seed=_seed(task.data_seed, "id"))
    blocks.append(("pool", "id", X_id[:n_pool], y_id[:n_pool], meta_id[:n_pool]))
    blocks.append(("test_id", "id", X_id[n_pool:], y_id[n_pool:], meta_id[n_pool:]))
    env_info: dict[str, dict] = {}
    for env in ood_envs:
        X_e, y_e, meta_e = task.generate_environment(env, n_test_ood, seed=_seed(task.data_seed, env))
        env_info[env] = dict(task.last_env_info)
        blocks.append(("test_ood", env, X_e, y_e, meta_e))

    # Standardisation statistics come from the pool only (generator columns).
    pool_X = blocks[0][2]
    mean = pool_X.mean(axis=0)
    std = pool_X.std(axis=0)
    std = np.where(std > 0, std, 1.0)

    # perm[g] = displayed position of generator column g.
    perm = np.random.default_rng(_seed(task.data_seed, "permutation")).permutation(n_feat)
    display = [f"f{perm[g]}" for g in range(n_feat)]
    q_rng = np.random.default_rng(_seed(task.data_seed, "queries"))

    frames, row_start = [], 0
    for split, env, X, y, meta in blocks:
        Z = (X - mean) / std
        df = pd.DataFrame({display[g]: Z[:, g] for g in range(n_feat)})
        df = df[[f"f{j}" for j in range(n_feat)]]            # displayed order
        df.insert(0, "row_id", np.arange(row_start, row_start + len(X)))
        df.insert(0, "env", env)
        df.insert(0, "split", split)
        df.insert(0, "task_id", task.task_id)
        df["label"] = y.astype(int)
        df["y_clean"] = [m["y_clean"] for m in meta]
        df["regime"] = [m["regime"] for m in meta]
        df["agree_clean"] = [m["agree_clean"] for m in meta]
        df["agree_obs"] = [m["agree_obs"] for m in meta]
        df["spurious_raw"] = X[:, task.spurious_idx]
        if split == "pool":
            df["query_id"] = -1
        else:
            df["query_id"] = _sample_queries(df["label"].to_numpy(), queries_per_class, q_rng)
        df["is_query"] = df["query_id"] >= 0
        frames.append(df)
        row_start += len(X)
    frame = pd.concat(frames, ignore_index=True)

    load_bearing = task.load_bearing_features()
    roles = {
        "load_bearing": [display[g] for g in load_bearing],
        "spurious": display[task.spurious_idx],
        "noise": display[task.noise_idx],
        "distractor": [display[g] for g in range(n_feat)
                       if g not in load_bearing and g not in (task.spurious_idx, task.noise_idx)],
    }
    pool = frame[frame["split"] == "pool"]
    entry = {
        "task_id": task.task_id,
        "family": task.rule_family,
        "domain": task.domain,
        "spurious_direction": int(task.spurious_sign),
        "generator": task.to_meta(),
        "load_bearing_generator_idx": [int(g) for g in load_bearing],
        "directions": dict(zip(roles["load_bearing"], task.directions())),
        "roles": roles,
        "permutation": [int(p) for p in perm],
        "standardisation": {
            "mean": {display[g]: float(mean[g]) for g in range(n_feat)},
            "std": {display[g]: float(std[g]) for g in range(n_feat)},
        },
        "seeds": {purpose: _seed(task.data_seed, purpose)
                  for purpose in ("id", *ood_envs, "permutation", "queries")},
        "covariate_shift": _covariate_entry(env_info.get("covariate"), display),
        "n_rows": {f"{s}/{e}": int(((frame["split"] == s) & (frame["env"] == e)).sum())
                   for s, e in frame[["split", "env"]].drop_duplicates().itertuples(index=False)},
        "pool_label_rate": float(pool["label"].mean()),
        "pool_agree_clean": float(pool["agree_clean"].mean()),
        "queries_per_class": queries_per_class,
    }
    return frame, entry


def _covariate_entry(info: dict | None, display: list[str]) -> dict | None:
    if not info:
        return None
    out = dict(info)
    out["shift_features_displayed"] = [display[g] for g in info["shift_features"]]
    return out


def write_suite(
    tasks: list[SyntheticTask],
    out_dir: str | Path,
    suite: str,
    **frame_kwargs: Any,
) -> dict[str, Any]:
    """Write every task's parquet plus `task_manifest.json` under `out_dir`."""
    out_dir = Path(out_dir)
    (out_dir / "tasks").mkdir(parents=True, exist_ok=True)
    entries = []
    for task in tasks:
        frame, entry = build_task_frame(task, **frame_kwargs)
        frame.to_parquet(out_dir / "tasks" / f"{task.task_id}.parquet", index=False)
        entries.append(entry)
    manifest = {
        "suite": suite,
        "version": "v3",
        "label_tokens": list(LABEL_TOKENS),
        "feature_columns": FEATURE_NAMES,
        "frame_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in frame_kwargs.items()},
        "tasks": entries,
    }
    with open(out_dir / "task_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_manifest(suite_dir: str | Path) -> dict[str, Any]:
    with open(Path(suite_dir) / "task_manifest.json") as f:
        return json.load(f)


def load_task_frame(suite_dir: str | Path, task_id: str) -> pd.DataFrame:
    return pd.read_parquet(Path(suite_dir) / "tasks" / f"{task_id}.parquet")


def pool_of(frame: pd.DataFrame) -> pd.DataFrame:
    """The task's demonstration pool, indexed by row_id."""
    return frame[frame["split"] == "pool"].set_index("row_id", drop=False)


def queries_of(frame: pd.DataFrame, env: str) -> pd.DataFrame:
    """The task's queries for one environment, ordered by query_id."""
    q = frame[(frame["split"] != "pool") & (frame["env"] == env) & frame["is_query"]]
    return q.sort_values("query_id").set_index("row_id", drop=False)
