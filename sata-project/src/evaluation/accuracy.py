"""Accuracy, macro-F1, and shift gap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

INVALID = "INVALID"


def accuracy(predictions: pd.Series, labels: pd.Series) -> float:
    """Accuracy excluding INVALID predictions (still counted by the caller for reporting)."""
    valid = predictions != INVALID
    if valid.sum() == 0:
        return float("nan")
    return float((predictions[valid] == labels[valid]).mean())


def macro_f1(predictions: pd.Series, labels: pd.Series) -> float:
    valid = predictions != INVALID
    if valid.sum() == 0:
        return float("nan")
    return float(f1_score(labels[valid], predictions[valid], average="macro"))


def shift_gap(id_accuracy: float, ood_accuracy: float) -> float:
    """Positive value indicates OOD degradation relative to ID."""
    return id_accuracy - ood_accuracy


def invalid_rate(predictions: pd.Series) -> float:
    return float((predictions == INVALID).mean())


def summarise(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Accuracy/macro-F1 per group, averaged over seeds with std, per Notebook 02 output."""
    rows = []
    for keys, group in results.groupby(group_cols):
        rows.append(
            {
                **dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
                "accuracy": accuracy(group["prediction"], group["label"]),
                "macro_f1": macro_f1(group["prediction"], group["label"]),
                "invalid_rate": invalid_rate(group["prediction"]),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)
