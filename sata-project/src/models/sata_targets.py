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

v3 (P5, hiccups/21): both bonuses go to clean rows only (label == y_clean).
A label-noise row's spurious feature follows its true label, so it looks
counter-spurious, and inside the query's regime it scored 3.5, the maximum.
On the evaluation distribution the targets then picked label-noise rows at
about four times the pool rate, and failed gate S3a. The penalty is unchanged.
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
        # A label-noise row shows the rule wrongly, so it earns no bonus.
        clean = demo["label"] == demo["y_clean"]

        # Same regime bonus -- structural alignment with the query's own
        # decision-rule region, computable from features alone.
        if clean and demo["regime"] == query_metadata["regime"]:
            score += 2.0

        # Counter-spurious bonus -- a demo whose spurious feature does NOT
        # track its own label is informative about the causal rule rather
        # than the shortcut.
        if clean and demo["is_counter_spurious"]:
            score += 1.5

        # Real penalty (not the +0.1 bonus v1 had) for a demo that is both
        # spurious-consistent AND outside the query's regime -- pure
        # shortcut bait with no structural payoff.
        if demo["spurious_consistent"] and demo["regime"] != query_metadata["regime"]:
            score -= 1.5

        scores[i] = score

    scores = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))
    return scores


def matched_target_scores(
    query_metadata: dict[str, Any],
    demo_metadata: list[dict[str, Any]],
    shortcut: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """Neutral targets (P5, the user's decision; hiccups/21): each row's chance
    of being in a matched set, normalised to sum to 1.

    The structural targets above reward rows that break the shortcut, so their
    top-k sets show it reversed (agreeing in about a quarter of rows, under a
    tenth on tree tasks). The matched set mirrors the magnitude-matched
    counter_spurious strategy, with the query's regime first: within each
    label, k/4 clean rows whose shortcut disagrees with the label (the query's
    regime first, at random among ties), each paired with the clean agreeing
    row of the same label nearest in |shortcut| (the query's regime first).
    Half the set agrees with the shortcut, so it carries no information about
    the label. A row's target is its probability of being chosen, computed
    exactly except that two disagreeing rows sharing a nearest partner are not
    resolved. A label short of disagreeing rows is filled evenly over its other
    clean rows. Label-noise rows are never chosen.

    `shortcut` holds every pool row's shortcut value (any scale centred on 0;
    only |shortcut| is compared). Like the structural targets, nothing here
    reads the query's label.
    """
    label = np.array([d["label"] for d in demo_metadata])
    clean = label == np.array([d["y_clean"] for d in demo_metadata])
    agree = np.array([d["agree_obs"] for d in demo_metadata], dtype=bool)
    same = np.array([d["regime"] == query_metadata["regime"] for d in demo_metadata])
    mag = np.abs(np.asarray(shortcut, dtype=float))
    per = k // 4
    p = np.zeros(len(label))
    for c in (0, 1):
        dis = np.flatnonzero((label == c) & clean & ~agree)
        agr = np.flatnonzero((label == c) & clean & agree)
        dis_same, dis_off = dis[same[dis]], dis[~same[dis]]
        p_dis = np.zeros(len(label))
        if len(dis_same) >= per:
            p_dis[dis_same] = per / len(dis_same)
        else:
            p_dis[dis_same] = 1.0
            if len(dis_off):
                p_dis[dis_off] = min(1.0, (per - len(dis_same)) / len(dis_off))
        p += p_dis
        if len(dis) and len(agr):
            cost = np.abs(mag[agr][None, :] - mag[dis][:, None]) - 10.0 * same[agr][None, :]
            np.add.at(p, agr[np.argmin(cost, axis=1)], p_dis[dis])
        n_dis = min(per, len(dis))                      # disagreeing rows the set takes
        n_partners = n_dis if len(agr) else 0
        short = k // 2 - n_dis - n_partners             # rows of this label still to fill
        if short > 0:
            rest = np.flatnonzero((label == c) & clean & (p == 0))
            if len(rest):
                p[rest] += short / len(rest)
    p = np.minimum(p, 1.0)
    return p / p.sum()
