"""Target score computation from synthetic ground truth (Notebook 05).

These targets are the supervision signal for SATA's KL-divergence training
loss (see sata_train.py) — they encode which demos *should* be relevant for
a given query, based on generator ground truth that is unavailable at
real-data evaluation time.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_target_scores(
    query_metadata: dict[str, Any],
    demo_metadata: list[dict[str, Any]],
    temperature: float = 1.0,
) -> np.ndarray:
    """Assign target relevance weights based on generator ground truth.

    High weight:
        - Demos in the same decision regime as the query
        - Counter-spurious demos (break the shortcut)
    Low weight:
        - Demos only predictive via spurious feature
        - Demos from irrelevant regimes

    Returns: (n_demos,) array, normalised to sum to 1.
    """
    scores = np.zeros(len(demo_metadata))

    for i, demo in enumerate(demo_metadata):
        score = 0.0

        # Same regime bonus
        if demo["regime"] == query_metadata["regime"]:
            score += 2.0

        # Counter-spurious bonus
        if demo["is_counter_spurious"]:
            score += 1.5

        # Correct label bonus (mild)
        # Not too strong — we want structural alignment, not just label matching
        if demo["label"] == query_metadata["label"]:
            score += 0.5

        # Penalty for spurious-only demos
        if demo["spurious_consistent"] and demo["regime"] != query_metadata["regime"]:
            score += 0.1  # near-zero but not exactly zero for numerical stability

        scores[i] = score

    # Normalise to distribution via softmax with temperature
    scores = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))
    return scores
