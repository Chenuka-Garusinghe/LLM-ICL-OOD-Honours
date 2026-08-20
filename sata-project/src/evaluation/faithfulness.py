"""Internal-consistency faithfulness: pi_self, pi_behav (hot-deck LOO), Spearman rho.

Implements Notebook 03's three steps:
  1. Elicit pi_self via a feature-ranking prompt (parsing only — the prompt
     itself lives in src/inference/prompts.py::build_feature_ranking_prompt).
  2. Compute pi_behav by kNN hot-deck imputation ablation of each feature and
     measuring the accuracy drop.
  3. Spearman rho between pi_self and pi_behav, with bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def parse_feature_ranking(response_text: str, valid_features: list[str]) -> list[str]:
    """Parse a comma-separated feature-ranking LLM response into pi_self.

    Silently drops tokens that don't match a known feature name (typos,
    trailing punctuation) rather than raising, since this is free-text LLM
    output.
    """
    tokens = [t.strip() for t in response_text.split(",")]
    valid_set = set(valid_features)
    ranked = [t for t in tokens if t in valid_set]
    # Append any features the model omitted, in their original order, at the tail.
    missing = [f for f in valid_features if f not in ranked]
    return ranked + missing


def hot_deck_impute_feature(
    df: pd.DataFrame, feature: str, train_pool: pd.DataFrame, feature_cols: list[str], n_neighbours: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Replace `feature`'s value in every row of `df` via kNN hot-deck imputation:
    find the n_neighbours nearest rows in train_pool (Euclidean distance on all
    features except `feature`), then randomly sample one neighbour's value.
    """
    rng = np.random.default_rng(seed)
    other_cols = [c for c in feature_cols if c != feature]

    nbrs = NearestNeighbors(n_neighbors=n_neighbours).fit(train_pool[other_cols])
    _, neighbour_idx = nbrs.kneighbors(df[other_cols])

    out = df.copy()
    donor_choice = rng.integers(0, n_neighbours, size=len(df))
    chosen_neighbours = neighbour_idx[np.arange(len(df)), donor_choice]
    out[feature] = train_pool[feature].to_numpy()[chosen_neighbours]
    return out


def compute_accuracy_drop(original_correct: np.ndarray, modified_correct: np.ndarray) -> float:
    """Delta_j = accuracy(original) - accuracy(modified)."""
    return float(original_correct.mean() - modified_correct.mean())


def rank_from_deltas(deltas: dict[str, float]) -> list[str]:
    """pi_behav: features ranked by Delta_j descending (removal that hurts most is rank 1)."""
    return [f for f, _ in sorted(deltas.items(), key=lambda t: -t[1])]


def spearman_with_bootstrap(
    pi_self: list[str],
    pi_behav_deltas: dict[str, float],
    per_row_correct_by_feature: dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    """rho(pi_self, pi_behav) plus a bootstrap CI over the test rows.

    `per_row_correct_by_feature[feature]` is a boolean array (len = n_test_rows)
    of per-row correctness under that feature's hot-deck ablation, used to
    recompute Delta_j on each bootstrap resample.
    """
    features = list(pi_behav_deltas.keys())
    self_rank = {f: r for r, f in enumerate(pi_self)}
    behav_rank = {f: r for r, f in enumerate(rank_from_deltas(pi_behav_deltas))}

    self_ranks = [self_rank[f] for f in features]
    behav_ranks = [behav_rank[f] for f in features]
    rho, pval = spearmanr(self_ranks, behav_ranks)

    n_rows = len(next(iter(per_row_correct_by_feature.values())))
    rng = np.random.default_rng(seed)
    rho_samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_rows, size=n_rows)
        delta_j_boot = {f: per_row_correct_by_feature[f][idx].mean() for f in features}
        behav_rank_boot = {f: r for r, f in enumerate(rank_from_deltas(delta_j_boot))}
        rho_boot, _ = spearmanr(self_ranks, [behav_rank_boot[f] for f in features])
        rho_samples.append(rho_boot)

    return {
        "rho": float(rho),
        "pval": float(pval),
        "ci_low": float(np.percentile(rho_samples, 2.5)),
        "ci_high": float(np.percentile(rho_samples, 97.5)),
    }
