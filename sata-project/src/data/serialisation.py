"""Row -> text serialisation for LLM prompts.

Feature order is a nuisance variable: it must be frozen per dataset (the
alphabetical order recorded in feature_list.json) and never randomised, so
that demo/query serialisation is deterministic across conditions and seeds.
Callers pass an already-ordered dict; serialise_row preserves that order.

When a `codebook` (from src/data/tableshift_loader.load_codebook) is given,
each feature is rendered with its human-readable name and, for categorical
features, its category label instead of the raw integer code -- e.g.
"Body Mass Index (BMI) category: Obese (3000 <= BMI < 9999)" rather than
"BMI5CAT: 3". Numeric features are rounded to `ndigits`. Without a codebook
the behaviour is the original "name: value" rendering.
"""

from __future__ import annotations

from typing import Any


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and v != v


def _format_number(v: Any, ndigits: int) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f.is_integer():
        return str(int(f))
    return f"{round(f, ndigits)}"


def _render_value(name: str, value: Any, codebook: dict | None, ndigits: int) -> str:
    entry = codebook.get(name) if codebook else None
    if _is_nan(value):
        return "unknown"
    if entry and entry.get("kind") == "categorical" and entry.get("values"):
        try:
            return str(entry["values"].get(int(value), value))
        except (TypeError, ValueError):
            return str(value)
    return _format_number(value, ndigits)


def serialise_row(
    features: dict[str, Any],
    label: str | None = None,
    codebook: dict | None = None,
    ndigits: int = 2,
) -> str:
    """Convert a tabular row to a text string for the LLM prompt.

    Format: "FeatureName1: value1; FeatureName2: value2; ... -> Label"
    Feature order is taken from `features` as given (do not re-sort here).
    If label is None (query row), end with "->".
    """
    parts = []
    for name, value in features.items():
        disp_name = name
        if codebook and name in codebook and codebook[name].get("name_extended"):
            disp_name = codebook[name]["name_extended"]
        parts.append(f"{disp_name}: {_render_value(name, value, codebook, ndigits)}")
    line = "; ".join(parts)
    if label is None:
        return f"{line} ->"
    return f"{line} -> {label}"


def ordered_feature_names(features: dict[str, Any]) -> list[str]:
    """Alphabetical feature order — the single source of truth for column order.

    Callers should sort feature dicts through this function before serialising
    so that demo pool and query rows never drift out of sync on ordering.
    Sorting is on the raw column name, not any codebook display name, so a
    codebook rename can never reorder the prompt.
    """
    return sorted(features.keys())
