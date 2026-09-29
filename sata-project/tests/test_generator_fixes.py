"""Verifies src/data/generator.py's Stage 2 fixes (REDESIGN_RATIONALE.md §5.2)
against a small locally-generated sample -- the full-scale regeneration is
Stage 4/HPC work, but the generator logic itself is fully checkable here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.data.generator import ENVIRONMENTS, _sample_tasks, generate_val_test_tasks
from src.evaluation.generator_checks import (
    check_extrapolation_flips_labels,
    check_label_rate_in_range,
    check_shift_preserves_base_rate,
    check_spurious_strength,
)

V2_GENERATOR_CONFIG = SimpleNamespace(
    n_features=10,
    n_causal_range=[3, 5],
    n_val_tasks=0,
    n_test_tasks=20,
    rule_families=["linear", "threshold", "tree", "sparse_interaction"],
    heldout_family="sparse_interaction",
    spurious_strength_range=[0.80, 0.90],
    label_noise=0.02,
    coefficient_scale=1.5,
)


@pytest.fixture(scope="module")
def tasks():
    _, test_tasks = generate_val_test_tasks(V2_GENERATOR_CONFIG)
    si_tasks = _sample_tasks(V2_GENERATOR_CONFIG, 15, "si", ["sparse_interaction"])
    return test_tasks + si_tasks


def test_no_environment_is_degenerate(tasks):
    result = check_label_rate_in_range(tasks, ENVIRONMENTS)
    assert result.passed, result.detail


def test_sparse_interaction_missing_feature_is_no_longer_constant(tasks):
    si_tasks = [t for t in tasks if t.rule_family == "sparse_interaction"]
    assert si_tasks, "expected at least one sparse_interaction task"
    rates = []
    for t in si_tasks:
        _, y, _ = t.generate_environment("missing_feature", n_samples=300, seed=7)
        rates.append(y.mean())
    rates = np.array(rates)
    assert rates.min() > 0.1, "missing_feature x sparse_interaction should not be near-constant-label"


def test_covariate_preserves_base_rate(tasks):
    # Only covariate shift is required to be a pure P(x) intervention that
    # preserves P(y) (REDESIGN_RATIONALE.md §4.4(b)/§5.2) -- extrapolation is
    # a support/coverage shift and is *expected* to move individual labels
    # (that's the point of check_extrapolation_flips_labels below); it has
    # no base-rate-preservation requirement of its own.
    result = check_shift_preserves_base_rate(tasks, "covariate", max_delta=0.10)
    assert result.passed, result.detail


def test_spurious_strength_matches_configured_range(tasks):
    result = check_spurious_strength(tasks, tuple(V2_GENERATOR_CONFIG.spurious_strength_range))
    assert result.passed, result.detail


def test_extrapolation_flips_labels_for_every_task(tasks):
    result = check_extrapolation_flips_labels(tasks, min_flip_frac=0.05)
    assert result.passed, result.detail


def test_v1_spurious_range_is_flagged_as_a_near_photocopy():
    """Sanity check on the check function itself: v1's default.yaml range
    ([0.96, 0.995]) should measure close to that (i.e. the check correctly
    reports it, it isn't the check that was wrong for v1)."""
    v1_config = SimpleNamespace(**{**vars(V2_GENERATOR_CONFIG), "spurious_strength_range": [0.96, 0.995]})
    _, v1_tasks = generate_val_test_tasks(v1_config)
    result = check_spurious_strength(v1_tasks[:10], (0.96, 0.995))
    assert result.passed
    assert result.value > 0.95
