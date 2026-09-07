"""Correctness-of-reliance faithfulness (synthetic only): pi_true vs pi_behav.

pi_behav is computed exactly as in faithfulness.py (LOO ablation). pi_true
comes from the generator's known causal structure rather than an LLM
self-report, so this answers whether the model relies on the *correct*
features — not just whether it's internally consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def true_importance_scores(
    rule_family: str,
    n_features: int,
    causal_features: list[int],
    coefficients: np.ndarray,
    thresholds3=None,
    leaf_labels=None,
) -> np.ndarray:
    """One non-negative true-importance score per feature index.

    Family-aware, because `SyntheticTask._apply_rule` (src/data/generator.py)
    only reads `coefficients` for 'linear' -- 'threshold' uses `thresholds3`,
    'tree' uses `leaf_labels`, and 'sparse_interaction' only reads the first
    two `causal_features`. Scoring every family by |coefficients| (the
    previous behaviour) meant the within-causal ranking was noise from an
    unused random vector for three of the four families, and the whole
    heldout-family (sparse_interaction) RQ4 evaluation was affected.

    The spurious and noise features, and any `causal_features` entry a
    family doesn't actually use (e.g. 'threshold'/'tree' sample 3-5 causal
    features but only read the first `k`), all score 0 -- they are not
    load-bearing for the label regardless of how predictive the spurious
    feature happens to be in the data.
    """
    scores = np.zeros(n_features)

    if rule_family == "linear":
        for f, c in zip(causal_features, coefficients):
            scores[f] = abs(float(c))

    elif rule_family == "threshold":
        k = len(thresholds3) if thresholds3 is not None else len(causal_features)
        for f in causal_features[:k]:
            scores[f] = 1.0

    elif rule_family == "tree":
        # Boolean influence of each bit: the fraction of leaves whose label
        # flips when that bit (i.e. that causal feature's sign) is toggled --
        # 0 for a feature the leaf function ignores entirely, up to 1.0 for
        # a feature that alone determines the label (a "dictator", which
        # `_sample_tree_leaf_labels` now excludes -- see generator.py).
        leaf_labels_arr = np.asarray(leaf_labels)
        n_leaves = len(leaf_labels_arr)
        k = int(np.log2(n_leaves))
        leaves = np.arange(n_leaves)
        for i, f in enumerate(causal_features[:k]):
            flipped = leaves ^ (1 << i)
            scores[f] = float((leaf_labels_arr != leaf_labels_arr[flipped]).mean())

    elif rule_family == "sparse_interaction":
        for f in causal_features[:2]:
            scores[f] = 1.0

    else:
        raise ValueError(f"Unknown rule_family: {rule_family}")

    return scores


def rank_by_true_importance(
    rule_family: str,
    n_features: int,
    causal_features: list[int],
    coefficients: np.ndarray,
    spurious_idx: int,
    noise_idx: int,
    thresholds3=None,
    leaf_labels=None,
) -> list[int]:
    """Rank all feature indices by true importance: load-bearing causal
    features (by `true_importance_scores`, family-aware) descending, then
    the spurious feature, then the noise feature, then any remaining
    features (non-load-bearing causal-family entries and true decoys never
    sampled as causal at all -- none of these determine the label, so their
    relative order doesn't matter).
    """
    scores = true_importance_scores(
        rule_family, n_features, causal_features, coefficients, thresholds3, leaf_labels
    )
    causal_ranked = sorted((f for f in causal_features if scores[f] > 0), key=lambda f: -scores[f])
    remaining = [f for f in range(n_features) if f not in causal_ranked and f not in (spurious_idx, noise_idx)]
    return causal_ranked + [spurious_idx, noise_idx] + remaining


def correctness_rho(pi_true: list[int], pi_behav: list[int]) -> dict:
    true_rank = {f: r for r, f in enumerate(pi_true)}
    behav_rank = {f: r for r, f in enumerate(pi_behav)}
    features = list(true_rank.keys())
    rho, pval = spearmanr([true_rank[f] for f in features], [behav_rank[f] for f in features])
    return {"rho": float(rho), "pval": float(pval)}


def correctness_rho_from_scores(true_scores: np.ndarray, behav_scores: np.ndarray) -> dict:
    """Spearman rho directly on two per-feature score vectors (true
    importance vs. behavioural ablation deltas), rather than on two
    hand-built permutations -- `spearmanr` rank-transforms internally using
    average ranks for ties, which handles the many legitimate zero-score
    ties in `true_importance_scores` (every decoy/spurious/noise feature)
    correctly instead of needing an arbitrary tie-break order baked into a
    permutation ahead of time.
    """
    rho, pval = spearmanr(true_scores, behav_scores)
    return {"rho": float(rho), "pval": float(pval)}
