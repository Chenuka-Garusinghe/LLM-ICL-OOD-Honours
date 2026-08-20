"""Parquet schema definition and I/O helpers for the results tables.

All predictions (real and synthetic arms) are stored as one row per prediction
in an append-only parquet file, per the schema in configs/default.yaml
(results_columns). Keeping the schema in one place prevents column drift
between notebooks 02, 03, 06, and 07.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

RESULTS_SCHEMA = pa.schema(
    [
        pa.field("arm", pa.string()),
        pa.field("dataset", pa.string()),
        pa.field("environment", pa.string()),
        pa.field("model", pa.string()),
        pa.field("method", pa.string()),
        pa.field("seed", pa.int64()),
        pa.field("query_id", pa.int64()),
        pa.field("prediction", pa.string()),
        pa.field("label", pa.string()),
        pa.field("logprob_0", pa.float64()),
        pa.field("logprob_1", pa.float64()),
        pa.field("demo_ids", pa.list_(pa.int64())),
        pa.field("k", pa.int64()),
    ]
)


def new_results_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the correct columns/dtypes for results rows."""
    return pd.DataFrame({field.name: pd.Series(dtype="object") for field in RESULTS_SCHEMA})


def append_results(rows: pd.DataFrame, path: str | Path) -> None:
    """Append rows to a results parquet, creating it if it doesn't exist.

    Re-running a notebook appends rather than overwrites, per the spec's
    reproducibility notes — callers that need a clean slate should delete
    the file explicitly first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, rows], ignore_index=True)
    else:
        combined = rows
    combined.to_parquet(path, index=False)


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
