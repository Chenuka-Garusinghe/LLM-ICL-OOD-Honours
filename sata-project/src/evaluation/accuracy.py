"""Accuracy, macro-F1, and shift gap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

INVALID = "INVALID"


def _canon(series: pd.Series) -> pd.Series:
    """Canonicalise a label/prediction column so scoring is format-insensitive.

    Predictions and stored labels reach here as strings, but not always the
    *same* string for the same class: a label that passed through
    DataFrame.iterrows() on an all-float feature row stringifies as "1.0",
    while the prediction (a single label token) is "1". Numeric-looking
    values are normalised to their integer form ("1.0" -> "1", " 0 " -> "0");
    "INVALID" and any genuinely non-numeric value pass through untouched.
    """
    def one(v: object) -> object:
        s = str(v).strip()
        try:
            return str(int(float(s)))
        except (TypeError, ValueError):
            return s

    return series.map(one)


def accuracy(predictions: pd.Series, labels: pd.Series) -> float:
    """Accuracy excluding INVALID predictions (still counted by the caller for reporting)."""
    valid = predictions != INVALID
    if valid.sum() == 0:
        return float("nan")
    return float((_canon(predictions[valid]) == _canon(labels[valid])).mean())


def macro_f1(predictions: pd.Series, labels: pd.Series) -> float:
    valid = predictions != INVALID
    if valid.sum() == 0:
        return float("nan")
    return float(f1_score(_canon(labels[valid]), _canon(predictions[valid]), average="macro"))


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
