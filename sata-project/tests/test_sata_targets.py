"""Verifies the rewritten target function (REDESIGN_RATIONALE.md §5.3 / Fact A):
no label leak, and the expected structural ranking."""

from __future__ import annotations

import numpy as np

from src.models.sata_targets import compute_target_scores


def _demo(regime, is_counter_spurious, spurious_consistent, label):
    return {
        "regime": regime,
        "is_counter_spurious": is_counter_spurious,
        "spurious_consistent": spurious_consistent,
        "label": label,
    }


def test_target_function_never_reads_query_label():
    query_metadata = {"regime": 3, "label": 1}
    demo_metadata = [_demo(3, False, True, 0)]  # same regime, spurious-consistent

    # Deleting the query's label must not raise -- the function has no
    # legitimate reason to touch it (v1's +0.5 label-match term did).
    query_metadata_no_label = {"regime": 3}
    scores_with_label = compute_target_scores(query_metadata, demo_metadata)
    scores_without_label = compute_target_scores(query_metadata_no_label, demo_metadata)
    assert np.allclose(scores_with_label, scores_without_label)


def test_expected_structural_ranking():
    query_metadata = {"regime": 0}
    demos = [
        _demo(regime=0, is_counter_spurious=True, spurious_consistent=False, label=1),  # same-regime + counter-spurious: best
        _demo(regime=0, is_counter_spurious=False, spurious_consistent=False, label=0),  # same-regime only
        _demo(regime=1, is_counter_spurious=False, spurious_consistent=False, label=1),  # neither
        _demo(regime=1, is_counter_spurious=False, spurious_consistent=True, label=0),  # off-regime + spurious-consistent: worst (penalised)
    ]
    scores = compute_target_scores(query_metadata, demos, temperature=1.0)
    order = np.argsort(-scores)
    assert list(order) == [0, 1, 2, 3]


def test_scores_sum_to_one():
    query_metadata = {"regime": 2}
    demos = [_demo(r, r % 2 == 0, r % 3 == 0, r % 2) for r in range(10)]
    scores = compute_target_scores(query_metadata, demos)
    assert abs(scores.sum() - 1.0) < 1e-8
