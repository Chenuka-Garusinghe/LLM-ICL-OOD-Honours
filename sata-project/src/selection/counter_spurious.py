"""Protocol D: counter-spurious diversity.

Real arm: identify the 2-3 features most correlated with both the label and
the shift variable (the variable defining the TableShift domain split).
Over-sample demos from "minority cells" where the proxy-label correlation is
broken (e.g. high-income but label=0, if high-income normally predicts
label=1 and the shift is geographic).

Synthetic arm: skip the correlation search and use the generator's
`is_counter_spurious` ground-truth tag directly (Notebook 06) — pass
`is_counter_spurious` instead of `proxy_col`/`shift_col`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_spurious_proxy_features(
    train_df: pd.DataFrame, feature_cols: list[str], label_col: str, shift_col: str, top_n: int = 3
) -> list[str]:
    """Rank features by combined |corr with label| * |corr with shift variable|."""
    scores = {}
    for feat in feature_cols:
        try:
            corr_label = train_df[feat].corr(train_df[label_col])
            corr_shift = train_df[feat].corr(train_df[shift_col])
        except TypeError:
            continue
        if pd.isna(corr_label) or pd.isna(corr_shift):
            continue
        scores[feat] = abs(corr_label) * abs(corr_shift)
    ranked = sorted(scores.items(), key=lambda t: -t[1])
    return [name for name, _ in ranked[:top_n]]


def select(
    pool: pd.DataFrame,
    query: pd.Series,
    k: int,
    seed: int,
    label_col: str = "label",
    proxy_col: str | None = None,
    proxy_majority_label: int | None = None,
    is_counter_spurious: pd.Series | None = None,
) -> list[int]:
    """Over-sample minority cells (real arm: proxy_col disagrees with label;
    synthetic arm: pass `is_counter_spurious` directly).
    """
    rng = np.random.default_rng(seed)

    if is_counter_spurious is not None:
        counter_idx = pool.index[is_counter_spurious.loc[pool.index]]
    elif proxy_col is not None and proxy_majority_label is not None:
        proxy_high = pool[proxy_col] > pool[proxy_col].median()
        counter_idx = pool.index[
            (proxy_high & (pool[label_col] != proxy_majority_label))
            | (~proxy_high & (pool[label_col] == proxy_majority_label))
        ]
    else:
        raise ValueError("Provide either `is_counter_spurious` (synthetic) or `proxy_col`+`proxy_majority_label` (real).")

    n_counter = min(k // 2 + 1, len(counter_idx))
    selected = list(rng.choice(counter_idx, size=n_counter, replace=False)) if n_counter else []

    remaining = k - len(selected)
    if remaining > 0:
        leftover_idx = pool.index.difference(selected)
        selected.extend(rng.choice(leftover_idx, size=min(remaining, len(leftover_idx)), replace=False))

    return selected[:k]
