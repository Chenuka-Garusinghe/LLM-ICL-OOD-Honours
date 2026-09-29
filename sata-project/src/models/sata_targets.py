"""Ground-truth target scores for training SATA (Notebook 05).

v1's target function included a +0.5 bonus when a demo's label matched the
*query's own true label* (query_metadata["label"]) -- information SATA
cannot have at inference time. Because the spurious feature (index 8) was a
near-photocopy of the label at v1's spurious_strength_range ([0.96, 0.995]),
that term made "read the query's label off feature 8, then select
label-matching demos" the cheapest path to reducing the KL loss -- SATA
learned exactly the shortcut it was meant to fight (Gate 2 proxy 0.961 ~=
the spurious strength). See REDESIGN_RATIONALE.md §4.2 for the full causal
chain to the 0.045-0.099 spurious-reversal accuracy this produced.

The v2 target function below is purely structural: every term is computable
from information SATA legitimately has access to (the query's features, and
each demo's own label/regime/spurious-consistency), and the counter-spurious
signal now has a real negative term rather than the +0.1 "penalty" that was
actually a small positive bonus.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_target_scores(
    query_metadata: dict[str, Any],
    demo_metadata: list[dict[str, Any]],
    temperature: float = 1.0,
) -> np.ndarray:
    scores = np.zeros(len(demo_metadata))

    for i, demo in enumerate(demo_metadata):
        score = 0.0

        # Same regime bonus -- structural alignment with the query's own
        # decision-rule region, computable from features alone.
        if demo["regime"] == query_metadata["regime"]:
            score += 2.0

        # Counter-spurious bonus -- a demo whose spurious feature does NOT
        # track its own label is informative about the causal rule rather
        # than the shortcut.
        if demo["is_counter_spurious"]:
            score += 1.5

        # Real penalty (not the +0.1 bonus v1 had) for a demo that is both
        # spurious-consistent AND outside the query's regime -- pure
        # shortcut bait with no structural payoff.
        if demo["spurious_consistent"] and demo["regime"] != query_metadata["regime"]:
            score -= 1.5

        scores[i] = score

    scores = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))
    return scores
