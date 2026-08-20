"""Protocol A: label diversity — stratified sampling, k/2 demos per class."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select(pool: pd.DataFrame, query: pd.Series, k: int, seed: int, label_col: str = "label") -> list[int]:
    """Sample k/2 demos from each class (binary labels assumed)."""
    rng = np.random.default_rng(seed)
    classes = sorted(pool[label_col].unique())
    per_class = k // len(classes)

    selected: list[int] = []
    for cls in classes:
        cls_idx = pool.index[pool[label_col] == cls]
        n = min(per_class, len(cls_idx))
        selected.extend(rng.choice(cls_idx, size=n, replace=False))

    # Fill any remainder (e.g. odd k, or a class with too few rows) randomly.
    remaining = k - len(selected)
    if remaining > 0:
        leftover_idx = pool.index.difference(selected)
        selected.extend(rng.choice(leftover_idx, size=min(remaining, len(leftover_idx)), replace=False))

    return selected
