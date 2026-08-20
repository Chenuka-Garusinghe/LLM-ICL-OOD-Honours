"""Protocol C: rule diversity — decision-tree leaf sampling.

Real arm: fit a depth-3 decision tree on the training split, identify leaf
membership for each pool demo, sample demos to cover all populated leaves.

Synthetic arm: skip the tree fit and use the generator's ground-truth
`regime` metadata directly (Notebook 06) — pass `regimes` instead of fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def fit_leaf_tree(train_df: pd.DataFrame, feature_cols: list[str], label_col: str = "label") -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(train_df[feature_cols], train_df[label_col])
    return tree


def select(
    pool: pd.DataFrame,
    query: pd.Series,
    k: int,
    seed: int,
    feature_cols: list[str] | None = None,
    tree: DecisionTreeClassifier | None = None,
    regimes: pd.Series | None = None,
) -> list[int]:
    """Sample demos to cover populated tree leaves (real arm) or ground-truth
    regimes (synthetic arm — pass `regimes` and omit `tree`/`feature_cols`).
    """
    rng = np.random.default_rng(seed)

    if regimes is not None:
        leaf_ids = regimes.loc[pool.index]
    elif tree is not None and feature_cols is not None:
        leaf_ids = pd.Series(tree.apply(pool[feature_cols]), index=pool.index)
    else:
        raise ValueError("Provide either `regimes` (synthetic) or `tree`+`feature_cols` (real).")

    leaf_groups: dict[int, list[int]] = {}
    for idx, leaf in leaf_ids.items():
        leaf_groups.setdefault(leaf, []).append(idx)

    selected: list[int] = []
    leaves = list(leaf_groups.keys())
    rng.shuffle(leaves)
    for leaf in leaves:
        if len(selected) >= k:
            break
        selected.append(rng.choice(leaf_groups[leaf]))

    remaining = k - len(selected)
    if remaining > 0:
        leftover_idx = pool.index.difference(selected)
        selected.extend(rng.choice(leftover_idx, size=min(remaining, len(leftover_idx)), replace=False))

    return selected[:k]
