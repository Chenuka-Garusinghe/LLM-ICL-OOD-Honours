"""Shared feature standardisation for SATA training and selection.

Moved out of src/selection/sata_select.py so src/models/sata_train.py can use
the exact same function -- v1 trained SATA on raw (un-standardised) generator
features but standardised at inference time, worst on feature 8 specifically
(mean~0.5, std~0.5 raw vs ~+-1 z-scored) -- see REDESIGN_RATIONALE.md §4.5.
"""

from __future__ import annotations

import numpy as np


def standardise(pool_features: np.ndarray, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both arrays using the pool's own mean/std.

    Callers that only want to select from a pre-filtered subset of the pool
    should still compute these statistics on the *full* pool before
    filtering (see src/selection/sata_select.py::select) -- fitting stats on
    a pre-filtered subset means the same physical row gets different
    standardised values depending on which condition filtered it first.
    """
    mean = pool_features.mean(axis=0)
    std = pool_features.std(axis=0) + 1e-8
    return (pool_features - mean) / std, (query_features - mean) / std
