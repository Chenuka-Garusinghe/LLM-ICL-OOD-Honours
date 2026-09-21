"""Bridge between SyntheticTask generator and the experiment pipeline.

Converts generator output (numpy arrays) into the DataFrame / parquet format
used by the demo-selection and prompting pipeline, so synthetic datasets
are interchangeable with the real-arm datasets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.data.generator import SyntheticTask

FEATURE_NAMES = [f"f{i}" for i in range(10)]
LABEL_TOKENS = ("0", "1")

SYNTHETIC_TASK_DESC = (
    "Classify the input into one of two classes based on numerical features.",
    "feature-based binary classification",
    "Class 0",
    "Class 1",
)


def task_to_dataframes(
    task: SyntheticTask,
    env_ood: str,
    n_pool: int = 256,
    n_test_id: int = 100,
    n_test_ood: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate train_pool, test_id, test_ood DataFrames from a SyntheticTask."""
    X_id, y_id, meta_id = task.generate_environment("id", n_pool + n_test_id, seed)
    X_ood, y_ood, meta_ood = task.generate_environment(env_ood, n_test_ood, seed + 1000)

    pool_df = pd.DataFrame(X_id[:n_pool], columns=FEATURE_NAMES)
    pool_df["label"] = y_id[:n_pool].astype(int)

    test_id_df = pd.DataFrame(X_id[n_pool:], columns=FEATURE_NAMES)
    test_id_df["label"] = y_id[n_pool:].astype(int)

    test_ood_df = pd.DataFrame(X_ood, columns=FEATURE_NAMES)
    test_ood_df["label"] = y_ood.astype(int)

    return pool_df, test_id_df, test_ood_df


def make_codebook() -> dict:
    """Simple codebook for synthetic features (all continuous, no encoding)."""
    return {f: {"name_extended": f, "type": "continuous"} for f in FEATURE_NAMES}


def save_dataset(
    pool_df: pd.DataFrame,
    test_id_df: pd.DataFrame,
    test_ood_df: pd.DataFrame,
    out_dir: str | Path,
) -> Path:
    """Save a synthetic dataset in the real-arm parquet format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_df.to_parquet(out_dir / "train_pool.parquet", index=False)
    test_id_df.to_parquet(out_dir / "test_id.parquet", index=False)
    test_ood_df.to_parquet(out_dir / "test_ood.parquet", index=False)

    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(FEATURE_NAMES, f)
    with open(out_dir / "label_tokens.json", "w") as f:
        json.dump(list(LABEL_TOKENS), f)
    with open(out_dir / "codebook.json", "w") as f:
        json.dump(make_codebook(), f, indent=2)

    return out_dir


def select_top_features(
    train_df: pd.DataFrame, n_features: int = 12, mi_sample_size: int = 5000
) -> list[str]:
    """Select top-N feature columns by mutual information with the label."""
    feature_cols = [c for c in train_df.columns if c != "label"]
    mi_df = (
        train_df.sample(n=mi_sample_size, random_state=0)
        if len(train_df) > mi_sample_size
        else train_df
    )
    X = mi_df[feature_cols].apply(
        lambda col: col.astype("category").cat.codes if col.dtype == "object" else col
    )
    mi = mutual_info_classif(X.fillna(X.median()), mi_df["label"], random_state=0)
    ranked = sorted(zip(feature_cols, mi), key=lambda t: -t[1])
    return [name for name, _ in ranked[:n_features]]
