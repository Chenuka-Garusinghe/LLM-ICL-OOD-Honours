"""Go/no-go gates computed from grid results (docs/research_plan.md).

P2 learnability gate: with abstract names, do the demonstrations' labels
matter? For each model and k, compare AUROC with the gold labels against
AUROC with the same demonstrations and shuffled labels (Min et al. 2022),
paired by task, seed and query. The gate passes at a k when the mean
difference is at least 0.05 and its hierarchical-bootstrap 95% interval lies
above 0; the smallest passing k is used from then on.

A result counts only if the model answers with the label tokens: its median
label mass P("0") + P("1") must be at least 0.9. Llama-3.1-8B-Instruct put
about 2% of its next-token probability there (it began to explain its
reasoning), so its two-token scores carried almost no signal.

G3 naming gate (`naming_gate`): the prior surrogate must fit (out-of-fold
R^2 >= 0.5 in every domain) and the names must carry priors: zero-shot AUROC
under aligned names minus flipped names >= 0.15 on the ID queries, paired by
task and query. The spec's draft stated this in accuracy; zero-shot
predictions are uncalibrated, so raw accuracy mixes the model's label bias
into the contrast, while AUROC depends only on how the margins order the
queries (fixed on 30 September 2026, before the gate was run). Raw accuracy
is reported alongside.

P2 cache check (`cache_agreement`): scores from the prefix cache against
full forward passes, from scripts/p2_benchmark.py. It passes when caching
moves the mean per-task AUROC by at most 0.005, a tenth of the learnability
threshold. Prediction agreement is reported too, but not required: a
prediction can flip only on a near-tie, so its rate depends on the model's
margins rather than on the cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.bootstrap import hierarchical_auroc_diff
from src.evaluation.metrics import auroc, balanced_accuracy

LEARNABILITY_THRESHOLD = 0.05
LABEL_MASS_THRESHOLD = 0.9
CACHE_AUROC_TOLERANCE = 0.005
NAMING_R2_THRESHOLD = 0.5
NAMING_SEPARATION_THRESHOLD = 0.15


def _task_matrices(g: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per task: labels plus (query x seed) p1 matrices for gold and shuffled labels."""
    out = []
    for _, gt in g.groupby("task_id"):
        piv = {mode: gt[gt.label_mode == mode].pivot_table(index="row_id", columns="seed", values="p1")
               for mode in ("gold", "shuffled")}
        rows = piv["gold"].index.intersection(piv["shuffled"].index)
        seeds = piv["gold"].columns.intersection(piv["shuffled"].columns)
        if len(rows) == 0 or len(seeds) == 0:
            continue
        y = gt.drop_duplicates("row_id").set_index("row_id").loc[rows, "label"].astype(int).to_numpy()
        out.append((y, piv["gold"].loc[rows, seeds].to_numpy(), piv["shuffled"].loc[rows, seeds].to_numpy()))
    return out


def _mean_cell_auroc(g: pd.DataFrame) -> float:
    return float(g.groupby(["task_id", "seed"]).apply(
        lambda c: auroc(c["label"].astype(int), c["p1"]), include_groups=False).mean())


def label_mass(df: pd.DataFrame) -> pd.Series:
    """P("0") + P("1") per row, from the full-vocabulary label log-probabilities."""
    return np.exp(df["logprob_0"]) + np.exp(df["logprob_1"])


def _mean_cell_ba(g: pd.DataFrame, col: str) -> float:
    return float(g.groupby(["task_id", "seed"]).apply(
        lambda c: balanced_accuracy(c["label"].astype(int), c[col].astype(int)), include_groups=False).mean())


def learnability_gate(
    results: pd.DataFrame,
    strategy: str = "label_diversity",
    threshold: float = LEARNABILITY_THRESHOLD,
    n_bootstrap: int = 2000,
    seed: int = 0,
    min_label_mass: float = LABEL_MASS_THRESHOLD,
) -> pd.DataFrame:
    """One row per (model, k): the paired AUROC contrast, its interval, the
    median label mass, and the verdict (which requires a valid answer format)."""
    res = results[results["env"] == "id"]
    rows = []
    for (model, k), g in res[res["strategy"] == strategy].groupby(["model", "k"]):
        tasks = _task_matrices(g)
        diff = hierarchical_auroc_diff(tasks, n_bootstrap=n_bootstrap, seed=seed)
        gold, shuf = g[g.label_mode == "gold"], g[g.label_mode == "shuffled"]
        per_task = [np.mean([auroc(y, a[:, j]) - auroc(y, b[:, j]) for j in range(a.shape[1])])
                    for y, a, b in tasks]
        rows.append({
            "model": model, "k": int(k),
            "auroc_gold": _mean_cell_auroc(gold), "auroc_shuffled": _mean_cell_auroc(shuf),
            "delta_auroc": diff["estimate"], "ci_low": diff["ci_low"], "ci_high": diff["ci_high"],
            "tasks_positive": int(np.sum(np.array(per_task) > 0)), "n_tasks": diff["n_tasks"],
            "cal_ba_gold": _mean_cell_ba(gold, "prediction_cal"),
            "cal_ba_shuffled": _mean_cell_ba(shuf, "prediction_cal"),
            "label_mass": float(label_mass(g).median()),
        })
    gate = pd.DataFrame(rows)
    zs = res[res["strategy"] == "zero_shot"]
    if len(zs):
        zs_auroc = zs.groupby(["model", "task_id"]).apply(
            lambda c: auroc(c["label"].astype(int), c["p1"]), include_groups=False).groupby("model").mean()
        gate["auroc_zero_shot"] = gate["model"].map(zs_auroc)
        gate["label_mass_zero_shot"] = gate["model"].map(label_mass(zs).groupby(zs["model"]).median())
    gate["valid"] = gate["label_mass"] >= min_label_mass
    gate["passes"] = (gate["delta_auroc"] >= threshold) & (gate["ci_low"] > 0) & gate["valid"]
    return gate.sort_values(["model", "k"]).reset_index(drop=True)


def smallest_passing_k(gate: pd.DataFrame) -> dict[str, int | None]:
    out = {}
    for model, g in gate.groupby("model"):
        passing = g[g["passes"]]
        out[model] = int(passing["k"].min()) if len(passing) else None
    return out


def cache_agreement(scores: pd.DataFrame) -> dict[str, float | int | None]:
    """Compare scores from the prefix cache with full forward passes.

    `scores` has one row per scored suffix: `task_id`, `label` (-1 marks the
    task's content-free suffix), `margin_cached` and `margin_full`
    (logprob_1 - logprob_0), and `prediction_cached` and `prediction_full`.

    The check (`passes`): the mean per-task AUROC from the cache is within
    CACHE_AUROC_TOLERANCE of the full-pass value. Prediction agreement is
    reported but not required: a prediction can flip only where |margin| is
    smaller than the bf16 difference between the two passes, so that rate
    depends on how many margins sit near zero. Calibrated predictions (the
    margin minus the task's content-free margin) are compared as well.
    """
    mc, mf = scores["margin_cached"].to_numpy(float), scores["margin_full"].to_numpy(float)
    diff = np.sort(np.abs(mc - mf))
    flips = (scores["prediction_cached"] != scores["prediction_full"]).to_numpy()
    q = scores[scores["label"] >= 0]
    cf = scores[scores["label"] < 0].set_index("task_id")
    cal_cached = q["margin_cached"] - q["task_id"].map(cf["margin_cached"])
    cal_full = q["margin_full"] - q["task_id"].map(cf["margin_full"])
    per_task = pd.DataFrame([
        {"cached": auroc(g["label"], g["margin_cached"]), "full": auroc(g["label"], g["margin_full"]),
         "margin_sd": g["margin_full"].std()}
        for _, g in q.groupby("task_id")])
    d_auroc = (per_task["cached"] - per_task["full"]).abs()
    return {
        "agreement_rate": float(1 - flips.mean()), "n_compared": int(len(scores)), "n_flips": int(flips.sum()),
        "max_abs_margin_at_flip": float(np.abs(mf[flips]).max()) if flips.any() else None,
        "max_abs_margin_diff": float(diff[-1]), "median_abs_margin_diff": float(np.median(diff)),
        "p95_abs_margin_diff": float(diff[int(0.95 * (len(diff) - 1))]),
        "median_abs_margin": float(np.median(np.abs(mf))),
        "share_abs_margin_below_0.25": float(np.mean(np.abs(mf) < 0.25)),
        "median_task_margin_sd": float(per_task["margin_sd"].median()),
        "calibrated_agreement_rate": float(np.mean((cal_cached > 0) == (cal_full > 0))),
        "auroc_cached_mean": float(per_task["cached"].mean()), "auroc_full_mean": float(per_task["full"].mean()),
        "abs_mean_auroc_diff": abs(float(per_task["cached"].mean() - per_task["full"].mean())),
        "mean_abs_auroc_diff": float(d_auroc.mean()), "max_abs_auroc_diff": float(d_auroc.max()),
        "passes": abs(float(per_task["cached"].mean() - per_task["full"].mean())) <= CACHE_AUROC_TOLERANCE,
    }


def zero_shot_by_naming(results: pd.DataFrame, env: str = "id") -> pd.DataFrame:
    """Zero-shot mean per-task AUROC, raw accuracy, share predicted 1 and label mass per (model, naming)."""
    zs = results[(results["strategy"] == "zero_shot") & (results["env"] == env)]
    rows = []
    for (model, naming), g in zs.groupby(["model", "naming"]):
        per_task = g.groupby("task_id").apply(lambda c: pd.Series({
            "auroc": auroc(c["label"].astype(int), c["p1"]),
            "accuracy": float((c["prediction"] == c["label"]).mean()),
            "share_predicted_1": float((c["prediction"] == "1").mean()),
        }), include_groups=False)
        rows.append({"model": model, "naming": naming, "n_tasks": len(per_task), **per_task.mean().to_dict(),
                     "label_mass": float(label_mass(g).median())})
    return pd.DataFrame(rows)


def naming_gate(
    results: pd.DataFrame,
    r2_by_domain: dict[str, float],
    threshold: float = NAMING_SEPARATION_THRESHOLD,
    r2_threshold: float = NAMING_R2_THRESHOLD,
    env: str = "id",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """G3 for one model: surrogate fit and the zero-shot aligned-minus-flipped contrast."""
    zs = results[(results["strategy"] == "zero_shot") & (results["env"] == env)]
    tasks, acc_diff = [], []
    for _, g in zs.groupby("task_id"):
        piv = g.pivot_table(index="row_id", columns="naming", values="p1")
        pred = g.assign(correct=(g["prediction"] == g["label"]).astype(float)).pivot_table(
            index="row_id", columns="naming", values="correct")
        y = g.drop_duplicates("row_id").set_index("row_id").loc[piv.index, "label"].astype(int).to_numpy()
        tasks.append((y, piv[["aligned"]].to_numpy(), piv[["flipped"]].to_numpy()))
        acc_diff.append(float(pred["aligned"].mean() - pred["flipped"].mean()))
    diff = hierarchical_auroc_diff(tasks, n_bootstrap=n_bootstrap, seed=seed)
    rng = np.random.default_rng(seed)
    acc = np.array(acc_diff)
    boot = [acc[rng.integers(0, len(acc), len(acc))].mean() for _ in range(n_bootstrap)]
    per_task = [np.mean([auroc(y, a[:, 0]) - auroc(y, b[:, 0])]) for y, a, b in tasks]
    r2_min = min(r2_by_domain.values())
    return {
        "delta_auroc": diff["estimate"], "ci_low": diff["ci_low"], "ci_high": diff["ci_high"],
        "tasks_positive": int(np.sum(np.array(per_task) > 0)), "n_tasks": len(tasks),
        "delta_accuracy": float(acc.mean()), "delta_accuracy_ci": [float(np.percentile(boot, 2.5)),
                                                                   float(np.percentile(boot, 97.5))],
        "r2_by_domain": r2_by_domain, "r2_min": r2_min,
        "passes_r2": r2_min >= r2_threshold, "passes_separation": diff["estimate"] >= threshold,
        "passes": bool(r2_min >= r2_threshold and diff["estimate"] >= threshold),
        "rule": f"out-of-fold R^2 >= {r2_threshold} in every domain and zero-shot AUROC(aligned) - "
                f"AUROC(flipped) >= {threshold} ({env} queries)",
    }
