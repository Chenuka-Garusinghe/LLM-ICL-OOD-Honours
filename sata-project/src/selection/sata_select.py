"""SATA-based demo selection — wraps models/sata.py for inference-time use.

Standardises pool features to zero mean / unit variance within the task's
demo pool before scoring (per the data-contamination guarantee in the spec:
SATA never sees raw TableShift feature scales during training).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.sata import SATA


def standardise(pool_features: np.ndarray, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = pool_features.mean(axis=0)
    std = pool_features.std(axis=0) + 1e-8
    return (pool_features - mean) / std, (query_features - mean) / std


def select(
    model: SATA,
    pool: pd.DataFrame,
    query: pd.Series,
    feature_cols: list[str],
    label_col: str,
    k: int,
    pre_filtered_idx: list[int] | None = None,
) -> list[int]:
    """Score demos with SATA and return the top-k indices.

    If `pre_filtered_idx` is given, SATA scores/ranks only that subset
    (Notebook 06's "best protocol + SATA selection" condition); otherwise it
    scores the full pool ("SATA alone").
    """
    candidates = pool.loc[pre_filtered_idx] if pre_filtered_idx is not None else pool

    pool_X = candidates[feature_cols].to_numpy(dtype=np.float32)
    query_X = query[feature_cols].to_numpy(dtype=np.float32)
    pool_X, query_X = standardise(pool_X, query_X)

    demo_features = torch.tensor(pool_X, dtype=torch.float32).unsqueeze(0)
    demo_labels = torch.tensor(candidates[label_col].to_numpy(dtype=np.int64)).unsqueeze(0)
    query_features = torch.tensor(query_X, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        scores = model(demo_features, demo_labels, query_features).squeeze(0).numpy()

    top_k_local = np.argsort(-scores)[:k]
    return list(candidates.index[top_k_local])
