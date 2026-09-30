"""Evaluation metrics for the synthetic grid (generator_spec.pdf, metrics).

AUROC is computed from p1 within a cell, so calibration (strictly increasing
in p1 within a demonstration set) never changes it. Balanced accuracy equals
accuracy on the 10/10 query sets.
"""

from __future__ import annotations

import numpy as np


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count 1/2)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def auroc_columns(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUROC for every column of (n_pos, c) and (n_neg, c) score matrices at once."""
    diff = pos[:, None, :] - neg[None, :, :]
    return (diff > 0).mean(axis=(0, 1)) + 0.5 * (diff == 0).mean(axis=(0, 1))


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Mean of the per-class accuracies."""
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    rates = [np.mean(predictions[labels == c] == c) for c in (0, 1) if np.any(labels == c)]
    return float(np.mean(rates)) if rates else float("nan")
