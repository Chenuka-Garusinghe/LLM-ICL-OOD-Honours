"""Verifies label-balanced top-k selection (REDESIGN_RATIONALE.md §5.3 / Fact C):
selected sets are label-balanced, and a single-class top-k is unreachable
whenever both classes are present in reasonable numbers -- this is what
makes evaluate_sata_proxy's since-removed single-class fallback obsolete."""

from __future__ import annotations

import numpy as np

from src.selection.balanced_topk import balanced_top_k


def test_selection_is_label_balanced_for_several_pools():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 64
        labels = rng.integers(0, 2, size=n)
        # Ensure both classes present with reasonable counts (mirrors the
        # generator's [0.2, 0.8] label-rate guarantee).
        if labels.mean() < 0.2 or labels.mean() > 0.8:
            continue
        scores = rng.random(n)
        k = 8
        idx = balanced_top_k(scores, labels, k)
        assert len(idx) == k
        assert len(set(idx.tolist())) == k  # no duplicates
        selected_labels = labels[idx]
        n0 = (selected_labels == 0).sum()
        n1 = (selected_labels == 1).sum()
        assert abs(n0 - n1) <= 1  # off-by-one only when a class is short


def test_top_scoring_demo_per_class_is_included():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.3, 0.95, 0.05])
    idx = balanced_top_k(scores, labels, k=2)
    # Best class-0 demo is index 1 (score 0.9), best class-1 demo is index 4 (score 0.95).
    assert 1 in idx
    assert 4 in idx
