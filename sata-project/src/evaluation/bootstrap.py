"""Generic bootstrap confidence intervals for any scalar metric."""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Resample `values` with replacement n_bootstrap times, apply metric_fn,
    and return the point estimate plus a percentile CI.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = np.array([metric_fn(values[rng.integers(0, n, size=n)]) for _ in range(n_bootstrap)])

    alpha = (1 - ci) / 2
    return {
        "estimate": float(metric_fn(values)),
        "ci_low": float(np.percentile(samples, 100 * alpha)),
        "ci_high": float(np.percentile(samples, 100 * (1 - alpha))),
    }


def paired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for metric_fn(a) - metric_fn(b), resampling paired indices jointly."""
    rng = np.random.default_rng(seed)
    n = len(values_a)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs.append(metric_fn(values_a[idx]) - metric_fn(values_b[idx]))
    diffs = np.array(diffs)

    alpha = (1 - ci) / 2
    return {
        "estimate": float(metric_fn(values_a) - metric_fn(values_b)),
        "ci_low": float(np.percentile(diffs, 100 * alpha)),
        "ci_high": float(np.percentile(diffs, 100 * (1 - alpha))),
    }
