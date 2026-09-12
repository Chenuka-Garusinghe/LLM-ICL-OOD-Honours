"""Merge per-model sharded results (from a SATA_RUN_SHARD parallel run) back
into the canonical results/ path that the non-parallel notebooks expect.

Two concurrent processes -- one per model, launched with SATA_MODEL_FILTER +
SATA_RUN_SHARD set (see src/utils/config.py) -- write into results/<shard>/
instead of results/ directly, so they don't clobber each other's output.
Run this once both have finished to concatenate each shard pair back into
results/<filename>.parquet.

Usage:
    python scripts/merge_sharded_results.py [shard_name ...]
    # no args -> merges every shard dir found under results/
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def main(shards: list[str] | None = None) -> None:
    if shards is None:
        shards = [p.name for p in RESULTS_DIR.iterdir() if p.is_dir()]

    by_filename: dict[str, list[Path]] = {}
    for shard in shards:
        shard_dir = RESULTS_DIR / shard
        if not shard_dir.is_dir():
            continue
        for f in shard_dir.glob("*.parquet"):
            by_filename.setdefault(f.name, []).append(f)

    if not by_filename:
        print(f"No shard parquet files found under {RESULTS_DIR}/<shard>/")
        return

    for name, paths in by_filename.items():
        frames = [pd.read_parquet(p) for p in paths]
        merged = pd.concat(frames, ignore_index=True)
        out_path = RESULTS_DIR / name
        merged.to_parquet(out_path, index=False)
        print(f"merged {len(paths)} shard(s) -> {out_path} ({len(merged)} rows)")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
