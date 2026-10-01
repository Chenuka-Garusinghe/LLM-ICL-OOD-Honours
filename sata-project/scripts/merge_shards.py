#!/usr/bin/env python3
"""Merge per-worker result shards into one file (for runs split with --shard).

  python scripts/merge_shards.py OUT.parquet SHARD1.parquet SHARD2.parquet ...

Rows are concatenated; the script stops if two shards hold the same unit
(`unit_key`) or, for pool priors, the same (model, task_id, naming, row_id).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    out, shards = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    frames = [pd.read_parquet(p) for p in shards if p.exists()]
    missing = [str(p) for p in shards if not p.exists()]
    if missing:
        print("missing shards:", missing)
    df = pd.concat(frames, ignore_index=True)
    if "unit_key" in df.columns:
        by_shard = [set(f["unit_key"].unique()) for f in frames]
        assert not any(a & b for i, a in enumerate(by_shard) for b in by_shard[i + 1:]), "a unit appears in two shards"
        print(f"{out.name}: {df['unit_key'].nunique()} units, {len(df)} rows from {len(frames)} shards")
    else:
        key = ["model", "task_id", "naming", "row_id"]
        assert not df.duplicated(key).any(), "duplicate pool rows across shards"
        print(f"{out.name}: {df.groupby(['model', 'task_id', 'naming']).ngroups} pools, {len(df)} rows from {len(frames)} shards")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
