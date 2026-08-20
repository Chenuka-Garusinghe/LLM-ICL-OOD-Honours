"""Normalised logprob confidence, R-AUC, and F1@95% (Notebook 07, Pass 2).

Confidence for a prediction = max(exp(logprob_0), exp(logprob_1)), where
logprob_0/logprob_1 are already stored per-row in the results parquets.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def confidence_from_logprobs(logprob_0: np.ndarray, logprob_1: np.ndarray) -> np.ndarray:
    return np.maximum(np.exp(logprob_0), np.exp(logprob_1))


def compute_rauc(predictions: np.ndarray, labels: np.ndarray, confidences: np.ndarray) -> tuple[float, np.ndarray, list[float]]:
    """Error-retention curve: sort by ascending confidence, progressively
    replace least-confident predictions with ground truth, compute error at
    each retention level.
    """
    n = len(predictions)
    sorted_idx = np.argsort(confidences)  # least confident first

    errors = (predictions != labels).astype(float)

    retention_levels = np.linspace(0, 1, 101)  # 0% to 100% retention
    error_at_retention = []

    for r in retention_levels:
        n_retain = int(r * n)
        if n_retain == 0:
            error_at_retention.append(0.0)
            continue
        # Keep the n_retain most confident predictions
        retained_idx = sorted_idx[n - n_retain:]
        error_rate = errors[retained_idx].mean()
        error_at_retention.append(error_rate)

    rauc = np.trapz(error_at_retention, retention_levels)
    return rauc, retention_levels, error_at_retention


def compute_f1_at_retention(
    predictions: np.ndarray, labels: np.ndarray, confidences: np.ndarray, retention: float = 0.95
) -> float:
    """F1 score using only the top `retention` fraction most confident predictions."""
    n_retain = int(retention * len(predictions))
    sorted_idx = np.argsort(confidences)
    retained_idx = sorted_idx[len(predictions) - n_retain:]

    return f1_score(labels[retained_idx], predictions[retained_idx], average="macro")
