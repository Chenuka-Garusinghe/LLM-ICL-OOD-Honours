"""Protocol B: feature-range diversity — quantile-bin coverage.

For each of the top 3 continuous features (by mutual information), bin into
3 quantiles and sample demos to cover all bins. Remaining slots filled
randomly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select(
    pool: pd.DataFrame, query: pd.Series, k: int, seed: int, top_features: list[str], n_bins: int = 3
) -> list[int]:
    """Sample demos to cover the quantile bins of `top_features` (up to 3 features)."""
    rng = np.random.default_rng(seed)
    selected: set[int] = set()

    bin_cells: dict[tuple, list[int]] = {}
    binned = pd.DataFrame(index=pool.index)
    for feat in top_features[:3]:
        binned[feat] = pd.qcut(pool[feat], q=n_bins, labels=False, duplicates="drop")

    for idx, row in binned.iterrows():
        cell = tuple(row.values)
        bin_cells.setdefault(cell, []).append(idx)

    cells = list(bin_cells.keys())
    rng.shuffle(cells)
    for cell in cells:
        if len(selected) >= k:
            break
        candidates = bin_cells[cell]
        selected.add(rng.choice(candidates))

    # Fill remaining slots randomly.
    remaining = k - len(selected)
    if remaining > 0:
        leftover_idx = pool.index.difference(list(selected))
        selected.update(rng.choice(leftover_idx, size=min(remaining, len(leftover_idx)), replace=False))

    return list(selected)[:k]
