"""Contextual calibration (Zhao et al. 2021, "Calibrate Before Use") -- Stage 1e.

v1 measured a 96-99% class-1 prediction rate with zero demonstrations (both
models, every synthetic environment -- see REDESIGN_RATIONALE.md §3/§4.1)
and did nothing to correct it, so every downstream method comparison was
partly a comparison of how well each demo set happened to counteract that
prior bias rather than a comparison of selection quality. Contextual
calibration estimates the model's output distribution on a content-free
("N/A") version of the same prompt and divides it out. Both raw and
calibrated predictions are reported (this is a correction for a *measured*
bias, not a silent change) -- see the new `logprob_0_cf`/`logprob_1_cf`/
`prediction_raw` columns in src/utils/results_schema.py::RESULTS_SCHEMA_V2.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CONTENT_FREE_PLACEHOLDER = "N/A"


def content_free_features(features: dict[str, Any]) -> dict[str, Any]:
    """Replace every feature value with a neutral placeholder, for building
    the paired content-free prompt (same demos, query values blanked out).
    """
    return {k: CONTENT_FREE_PLACEHOLDER for k in features}


def _normalise(logprob_0: float, logprob_1: float) -> tuple[float, float]:
    p0 = np.exp(logprob_0) / (np.exp(logprob_0) + np.exp(logprob_1))
    return float(p0), float(1 - p0)


def calibrate(logprob_0: float, logprob_1: float, cf_logprob_0: float, cf_logprob_1: float) -> tuple[float, float]:
    """Diagonal-W contextual calibration: divide the raw two-way probabilities
    by the content-free ones and renormalise.

    If raw == content-free exactly, the two predictions carry no evidence
    beyond the model's inherent output bias, and calibration correctly
    collapses to (0.5, 0.5) -- verified in tests/test_inference_fixes.py.
    """
    p0_raw, p1_raw = _normalise(logprob_0, logprob_1)
    p0_cf, p1_cf = _normalise(cf_logprob_0, cf_logprob_1)

    q0 = p0_raw / max(p0_cf, 1e-8)
    q1 = p1_raw / max(p1_cf, 1e-8)
    total = q0 + q1
    if total <= 0:
        return 0.5, 0.5
    return q0 / total, q1 / total
