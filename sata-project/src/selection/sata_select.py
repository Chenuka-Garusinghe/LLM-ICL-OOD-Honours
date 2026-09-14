"""SATA-based demo selection at inference time (Notebook 06).

Standardisation now lives in src/models/standardise.py, shared with
src/models/sata_train.py (see that module's docstring for why the mismatch
mattered). Top-k selection is now label-balanced (k//2 per class), matching
src/models/sata_train.py::evaluate_sata_proxy's proxy selection and
src/selection/label_diversity.py's stratify-then-fill pattern -- an
uncontrolled top-k let SATA's demo-label balance itself become the signal
the LLM reacted to, and let the proxy's now-removed single-class fallback
reward a degenerate label-copy policy (REDESIGN_RATIONALE.md §4.2 Fact C,
§4.3, §5.3).
"""

from __future__ import annotations

import numpy as np
import torch

from src.models.standardise import standardise  # noqa: F401 -- re-exported for backward compat
from src.selection.balanced_topk import balanced_top_k


def select(model, pool, query, feature_cols, label_col, k, pre_filtered_idx=None) -> list[int]:
    # Standardisation stats are fit on the FULL pool, before any
    # pre-filtering -- v1 fit them on `candidates` (the pre-filtered 32 rows
    # for best_protocol_sata but the full 64 for sata_alone), so the same
    # physical row arrived at SATA with different standardised values
    # depending on the condition (REDESIGN_RATIONALE.md §4.5).
    full_pool_X = pool[feature_cols].to_numpy(dtype=np.float32)
    candidates = pool.loc[pre_filtered_idx] if pre_filtered_idx is not None else pool
    candidate_X = candidates[feature_cols].to_numpy(dtype=np.float32)
    query_X = query[feature_cols].to_numpy(dtype=np.float32)

    mean = full_pool_X.mean(axis=0)
    std = full_pool_X.std(axis=0) + 1e-8
    candidate_X = (candidate_X - mean) / std
    query_X = (query_X - mean) / std

    demo_labels_np = candidates[label_col].to_numpy(dtype=np.int64)
    demo_features = torch.tensor(candidate_X, dtype=torch.float32).unsqueeze(0)
    demo_labels = torch.tensor(demo_labels_np).unsqueeze(0)
    query_features = torch.tensor(query_X, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        scores = model(demo_features, demo_labels, query_features).squeeze(0).numpy()
    top_k_local = balanced_top_k(scores, demo_labels_np, k)
    return list(candidates.index[top_k_local])
