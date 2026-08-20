"""Row -> text serialisation for LLM prompts.

Feature order is a nuisance variable: it must be frozen alphabetically per
dataset (see feature_list.json) and never randomised, so that demo/query
serialisation is deterministic across conditions and seeds.
"""

from __future__ import annotations

from typing import Any


def serialise_row(features: dict[str, Any], label: str | None = None) -> str:
    """Convert a tabular row to a text string for the LLM prompt.

    Format: "FeatureName1: value1; FeatureName2: value2; ... -> Label"
    Feature order is fixed per dataset (alphabetical) and recorded in config.
    If label is None (query row), end with "->"
    """
    parts = [f"{name}: {value}" for name, value in features.items()]
    line = "; ".join(parts)
    if label is None:
        return f"{line} ->"
    return f"{line} -> {label}"


def ordered_feature_names(features: dict[str, Any]) -> list[str]:
    """Alphabetical feature order — the single source of truth for column order.

    Callers should sort feature dicts through this function before serialising
    so that demo pool and query rows never drift out of sync on ordering.
    """
    return sorted(features.keys())
