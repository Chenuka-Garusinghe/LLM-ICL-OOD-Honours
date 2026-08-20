"""Random-k baseline demo selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select(pool: pd.DataFrame, query: pd.Series, k: int, seed: int) -> list[int]:
    """Sample k demo indices uniformly at random from the pool."""
    rng = np.random.default_rng(seed)
    return list(rng.choice(pool.index, size=k, replace=False))
