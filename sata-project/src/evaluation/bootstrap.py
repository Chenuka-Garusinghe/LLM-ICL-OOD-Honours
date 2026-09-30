"""Generic bootstrap confidence intervals for any scalar metric, plus the
hierarchical (task, then query) bootstrap for paired AUROC contrasts."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.evaluation.metrics import auroc_columns


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


def _auroc_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, ip: np.ndarray, ineg: np.ndarray) -> float:
    """Mean over seed columns of AUROC(a) - AUROC(b) on the given query rows."""
    return float((auroc_columns(a[ip], a[ineg]) - auroc_columns(b[ip], b[ineg])).mean())


def hierarchical_auroc_diff(
    tasks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Hierarchical bootstrap for a paired AUROC contrast (generator_spec.pdf,
    design and inference).

    `tasks` holds one (labels, scores_a, scores_b) triple per task, where the
    score matrices are (n_queries, n_cells), for example one column per demo
    seed, paired row by row and column by column. The statistic is the mean
    over tasks of the mean over cells of AUROC(a) - AUROC(b). Each bootstrap
    draw resamples tasks with replacement, then, within each drawn task,
    positives and negatives separately with replacement (so AUROC stays
    defined), and recomputes the statistic.
    """
    rng = np.random.default_rng(seed)
    idx = [(np.flatnonzero(y == 1), np.flatnonzero(y == 0)) for y, _, _ in tasks]
    estimate = float(np.mean([_auroc_diff(y, a, b, p, n) for (y, a, b), (p, n) in zip(tasks, idx)]))
    stats = np.empty(n_bootstrap)
    for r in range(n_bootstrap):
        vals = []
        for t in rng.integers(0, len(tasks), size=len(tasks)):
            (y, a, b), (p, n) = tasks[t], idx[t]
            vals.append(_auroc_diff(y, a, b, rng.choice(p, size=len(p)), rng.choice(n, size=len(n))))
        stats[r] = np.mean(vals)
    alpha = (1 - ci) / 2
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(stats, 100 * alpha)),
        "ci_high": float(np.percentile(stats, 100 * (1 - alpha))),
        "n_tasks": len(tasks),
    }


def _ba_columns(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Balanced accuracy for every column of 0/1 prediction matrices (positives, negatives)."""
    return 0.5 * (pos.mean(axis=0) + (1 - neg).mean(axis=0))


def _mean_columns(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Plain mean over all rows (for per-row indicators such as the DFI)."""
    return np.concatenate([pos, neg]).mean(axis=0)


COLUMN_STATISTICS = {"auroc": auroc_columns, "ba": _ba_columns, "mean": _mean_columns}


def _block_value(block, statistic, ip, ineg) -> float:
    y, mats = block
    return float(sum(w * statistic(S[ip], S[ineg]).mean() for S, w in mats))


def hierarchical_contrast(
    tasks: list[list[tuple[np.ndarray, list[tuple[np.ndarray, float]]]]],
    statistic: str = "auroc",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Hierarchical bootstrap for a general contrast (generator_spec.pdf,
    design and inference).

    A task is a list of row blocks `(labels, [(S, w), ...])`. Every matrix S
    of a block is (n_rows, n_cells) and shares the block's rows (the same
    queries, so they are resampled together); cells are demo seeds, averaged
    after the statistic. The task's value is sum_blocks sum_S w * mean_cells
    stat(S). Examples: a paired strategy contrast is one block
    [(A, +1), (B, -1)]; an ID-minus-OOD gap is two blocks [(S_id, +1)] and
    [(S_ood, -1)]; an interaction is two blocks with opposite signs.

    Each bootstrap draw resamples tasks, then, within each block, positives
    and negatives separately. Returns the estimate (mean over tasks), the
    percentile interval, one-sided bootstrap p-values for "> 0" and "< 0",
    and the per-task values (for sign tests).
    """
    stat = COLUMN_STATISTICS[statistic]
    idx = [[(np.flatnonzero(y == 1), np.flatnonzero(y == 0)) for y, _ in task] for task in tasks]
    per_task = np.array([sum(_block_value(b, stat, p, n) for b, (p, n) in zip(task, ix))
                         for task, ix in zip(tasks, idx)])
    rng = np.random.default_rng(seed)
    stats = np.empty(n_bootstrap)
    for r in range(n_bootstrap):
        vals = []
        for t in rng.integers(0, len(tasks), size=len(tasks)):
            vals.append(sum(_block_value(b, stat, rng.choice(p, size=len(p)), rng.choice(n, size=len(n)))
                            for b, (p, n) in zip(tasks[t], idx[t])))
        stats[r] = np.mean(vals)
    alpha = (1 - ci) / 2
    return {
        "estimate": float(per_task.mean()),
        "ci_low": float(np.percentile(stats, 100 * alpha)),
        "ci_high": float(np.percentile(stats, 100 * (1 - alpha))),
        "p_greater": float((1 + np.sum(stats <= 0)) / (n_bootstrap + 1)),
        "p_less": float((1 + np.sum(stats >= 0)) / (n_bootstrap + 1)),
        "n_tasks": len(tasks),
        "per_task": per_task.tolist(),
    }


def holm(pvalues: list[float]) -> list[float]:
    """Holm-adjusted p-values (step-down, monotone), in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
