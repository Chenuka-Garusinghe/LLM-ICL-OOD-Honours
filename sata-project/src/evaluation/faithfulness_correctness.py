"""Correctness-of-reliance faithfulness (synthetic only): pi_true vs pi_behav.

pi_behav is computed exactly as in faithfulness.py (LOO ablation). pi_true
comes from the generator's known causal features rather than an LLM
self-report, so this answers whether the model relies on the *correct*
features — not just whether it's internally consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def rank_by_true_importance(
    n_features: int, causal_features: list[int], coefficients: np.ndarray, spurious_idx: int, noise_idx: int
) -> list[int]:
    """Rank all feature indices by true importance: causal features ranked by
    |coefficient| descending, then the spurious feature, then the noise
    feature, then any remaining (decoy) causal-family features.
    """
    causal_ranked = [
        f for f, _ in sorted(zip(causal_features, np.abs(coefficients)), key=lambda t: -t[1])
    ]
    remaining = [
        f for f in range(n_features) if f not in causal_ranked and f not in (spurious_idx, noise_idx)
    ]
    return causal_ranked + remaining + [spurious_idx, noise_idx]


def correctness_rho(pi_true: list[int], pi_behav: list[int]) -> dict:
    true_rank = {f: r for r, f in enumerate(pi_true)}
    behav_rank = {f: r for r, f in enumerate(pi_behav)}
    features = list(true_rank.keys())
    rho, pval = spearmanr([true_rank[f] for f in features], [behav_rank[f] for f in features])
    return {"rho": float(rho), "pval": float(pval)}
