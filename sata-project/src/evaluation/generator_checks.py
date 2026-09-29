"""Local, dependency-light checks for src/data/generator.py's Stage 2 fixes.

Not the full NB00 data-diagnostics audit suite from the redesign plan (that
also covers the real arm, prompt token budgets, and a machine-readable gate
report) -- these are the handful of check functions actually needed to
verify the generator fixes in REDESIGN_RATIONALE.md §5.2 against a small,
locally-generated sample of v2-config tasks, reusable later by the real
Gate S2 once the full suite is regenerated at production scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: str
    detail: str = ""


def check_label_rate_in_range(
    tasks: list, environments: list[str], n_samples: int = 200, seed: int = 1, low: float = 0.2, high: float = 0.8
) -> CheckResult:
    """No environment's label-1 rate should fall outside [low, high] for any
    task -- catches a degenerate/near-constant-label cell (Fact 4.4(a)).
    """
    offenders = []
    for env in environments:
        for t in tasks:
            _, y, _ = t.generate_environment(env, n_samples=n_samples, seed=seed)
            rate = float(y.mean())
            if rate < low or rate > high:
                offenders.append((t.task_id, env, rate))
    passed = len(offenders) == 0
    return CheckResult(
        name="label_rate_in_range",
        passed=passed,
        value=len(offenders),
        threshold=f"0 tasks outside [{low}, {high}]",
        detail=f"{len(offenders)} offending (task, env) cells" + (f": {offenders[:5]}" if offenders else ""),
    )


def check_shift_preserves_base_rate(
    tasks: list, env: str, n_samples: int = 500, seed: int = 9, max_delta: float = 0.10
) -> CheckResult:
    """|rate_env - rate_id| should stay within max_delta per task -- catches
    a shift that moves the label base rate instead of being a pure P(x)
    intervention (Fact 4.4(b), covariate; also used for extrapolation).
    """
    deltas = []
    for t in tasks:
        _, y_id, _ = t.generate_environment("id", n_samples=n_samples, seed=seed)
        _, y_env, _ = t.generate_environment(env, n_samples=n_samples, seed=seed)
        deltas.append(abs(float(y_id.mean()) - float(y_env.mean())))
    deltas = np.array(deltas)
    passed = bool((deltas <= max_delta).all())
    return CheckResult(
        name=f"shift_preserves_base_rate[{env}]",
        passed=passed,
        value=float(deltas.max()) if len(deltas) else 0.0,
        threshold=f"max |delta| <= {max_delta}",
        detail=f"mean={deltas.mean():.4f}, max={deltas.max():.4f}" if len(deltas) else "no tasks",
    )


def check_spurious_strength(
    tasks: list, spurious_strength_range: tuple[float, float], n_samples: int = 500, seed: int = 2, tolerance: float = 0.05
) -> CheckResult:
    """Empirical spurious sign-agreement should land within
    `spurious_strength_range` (+/- tolerance) -- catches a spurious feature
    that has drifted into being a near-photocopy of the label (Fact 4.2's
    enabling condition). Run against v1's default.yaml range, this is
    expected to still land inside that (too-strong) range -- the check
    itself doesn't know a range is "bad", it only reports what's configured
    vs what's empirically observed.
    """
    rates = []
    for t in tasks:
        _, _, meta = t.generate_environment("id", n_samples=n_samples, seed=seed)
        rates.append(np.mean([m["spurious_consistent"] for m in meta]))
    rates = np.array(rates)
    lo, hi = spurious_strength_range
    passed = bool(((rates >= lo - tolerance) & (rates <= hi + tolerance)).all())
    return CheckResult(
        name="spurious_strength_matches_config",
        passed=passed,
        value=float(rates.mean()),
        threshold=f"[{lo}, {hi}] +/- {tolerance}",
        detail=f"mean={rates.mean():.4f}, min={rates.min():.4f}, max={rates.max():.4f}",
    )


def check_extrapolation_flips_labels(
    tasks: list, n_samples: int = 300, seed: int = 3, min_flip_frac: float = 0.05
) -> CheckResult:
    """Extrapolation should actually change some labels relative to id, for
    every rule family -- catches a shift that's erased by inference-time
    standardisation or label-inert for some rule families (Fact 4.4(c)).
    """
    offenders = []
    for t in tasks:
        _, y_id, _ = t.generate_environment("id", n_samples=n_samples, seed=seed)
        _, y_ex, _ = t.generate_environment("extrapolation", n_samples=n_samples, seed=seed)
        flip_frac = float((y_id != y_ex).mean())
        if flip_frac < min_flip_frac:
            offenders.append((t.task_id, t.rule_family, flip_frac))
    passed = len(offenders) == 0
    return CheckResult(
        name="extrapolation_flips_labels",
        passed=passed,
        value=len(offenders),
        threshold=f"every task flips >= {min_flip_frac} of labels vs id",
        detail=f"{len(offenders)} offending tasks" + (f": {offenders[:5]}" if offenders else ""),
    )


def run_all_checks(tasks: list, environments: list[str], spurious_strength_range: tuple[float, float]) -> list[CheckResult]:
    results = [check_label_rate_in_range(tasks, environments)]
    for env in ("covariate", "extrapolation"):
        results.append(check_shift_preserves_base_rate(tasks, env))
    results.append(check_spurious_strength(tasks, spurious_strength_range))
    results.append(check_extrapolation_flips_labels(tasks))
    return results


def print_report(results: list[CheckResult]) -> bool:
    overall_pass = all(r.passed for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: value={r.value} threshold={r.threshold} -- {r.detail}")
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")
    return overall_pass
