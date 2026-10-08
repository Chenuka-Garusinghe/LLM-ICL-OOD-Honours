"""P4 analysis (generator_spec.pdf, metrics and design and inference): cell
metrics, the pre-registered contrast family with Holm correction, and the
RQ3 reliance measures.

A cell is (model, naming, task, strategy, label mode, k, seed, environment):
one demonstration set (or one per query for similarity) scored on 20
queries. AUROC is computed within a cell from the calibrated p1 (`score`;
raw p1 for zero-shot, which is not calibrated). For a query-agnostic
strategy calibration shifts every query of a cell by the same amount, so
this equals the AUROC of the raw p1; for similarity, whose queries each have
their own prompt, it removes each prompt's bias. Balanced accuracy uses the
calibrated predictions.

The contrast family (`CONTRASTS`) was fixed on 30 September 2026 before any
P4 result existed. Every contrast is paired by task and query, and tested
one-sided with the hierarchical bootstrap (`hierarchical_contrast`); the
p-values are Holm-adjusted over the whole family. Statistic: AUROC for C1
and C2 (the user's decision the same day, after the first 11 tasks showed
that contextual calibration often leaves every query of a plain-random cell
on one side of the threshold, which pins balanced accuracy at 0.5;
hiccups/15), the demo-following index and the correctness correlation for
C3. Calibrated balanced accuracy is reported alongside and is not part of
the family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from src.evaluation.bootstrap import hierarchical_contrast, holm
from src.evaluation.metrics import auroc, balanced_accuracy

CELL_KEYS = ["model", "naming", "task_id", "family", "domain", "strategy", "label_mode", "k", "seed", "env"]

# Comparators: plain random (free composition) for the shift gaps of RQ1;
# label_diversity (random, balanced) for the balanced strategies of RQ2, so a
# contrast compares mechanisms at fixed label counts (Proposition "label
# counts are controlled").
CONTRASTS = [
    {"id": "C1a", "rq": "RQ1", "kind": "env_gap", "strategy": "random", "naming": "abstract",
     "env_a": "id", "env_b": "covariate", "direction": "greater",
     "label": "ID minus covariate, random demonstrations, abstract names"},
    {"id": "C1b", "rq": "RQ1", "kind": "env_gap", "strategy": "random", "naming": "abstract",
     "env_a": "id", "env_b": "spurious_reversal", "direction": "greater",
     "label": "ID minus spurious reversal, random demonstrations, abstract names"},
    {"id": "C1c", "rq": "RQ1", "kind": "naming_gap", "strategy": "random", "env": "id",
     "naming_a": "aligned", "naming_b": "flipped", "direction": "greater",
     "label": "aligned minus flipped names (concept shift), random demonstrations, ID"},
    {"id": "C2a", "rq": "RQ2", "kind": "strategy", "a": "counter_spurious", "b": "label_diversity",
     "naming": "abstract", "env": "spurious_reversal", "direction": "greater",
     "label": "counter_spurious minus label_diversity under spurious reversal"},
    {"id": "C2b", "rq": "RQ2", "kind": "strategy", "a": "similarity", "b": "label_diversity",
     "naming": "abstract", "env": "covariate", "direction": "greater",
     "label": "similarity (feature kNN) minus label_diversity under covariate shift"},
    {"id": "C2c", "rq": "RQ2", "kind": "strategy", "a": "feature_range", "b": "label_diversity",
     "naming": "abstract", "env": "covariate", "direction": "greater",
     "label": "feature_range minus label_diversity under covariate shift"},
    {"id": "C2d", "rq": "RQ2", "kind": "strategy", "a": "counter_prior", "b": "label_diversity",
     "naming": "flipped", "env": "id", "direction": "greater",
     "label": "counter_prior minus label_diversity under concept shift (flipped names, ID)"},
    {"id": "C2e", "rq": "RQ2", "kind": "interaction", "a": "counter_spurious", "b": "label_diversity",
     "naming": "abstract", "env_a": "spurious_reversal", "env_b": "covariate", "direction": "greater",
     "label": "(counter_spurious minus label_diversity) under spurious reversal minus the same under covariate shift"},
    {"id": "C3a", "rq": "RQ3", "kind": "rq3_dfi", "naming_a": "abstract", "naming_b": "flipped",
     "direction": "greater", "label": "demo-following index, abstract minus flipped names"},
    {"id": "C3b", "rq": "RQ3", "kind": "rq3_rho", "naming_a": "abstract", "naming_b": "flipped",
     "direction": "greater", "label": "rho(pi_true, pi_behav), abstract minus flipped names"},
]


# --------------------------------------------------------------------------- #
# grid metrics
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Integer labels and predictions; zero-shot's balanced accuracy uses its raw prediction."""
    df = df.copy()
    df["y"] = df["label"].astype(int)
    df["pred_raw"] = df["prediction"].astype(int)
    cal = pd.to_numeric(df["prediction_cal"], errors="coerce")
    df["pred"] = cal.fillna(df["pred_raw"]).astype(int)
    df["score"] = pd.to_numeric(df["calibrated_p1"], errors="coerce").fillna(df["p1"])
    df["margin"] = df["logprob_1"] - df["logprob_0"]
    return df


def cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: AUROC, calibrated and raw balanced accuracy, share predicted 1, label mass."""
    df = prepare(df)
    g = df.groupby(CELL_KEYS, dropna=False)
    out = g.apply(lambda c: pd.Series({
        "auroc": auroc(c["y"], c["score"]),
        "auroc_raw": auroc(c["y"], c["p1"]),
        "ba": balanced_accuracy(c["y"], c["pred"]),
        "ba_raw": balanced_accuracy(c["y"], c["pred_raw"]),
        "share_pred_1": float(c["pred"].mean()),
        "label_mass": float(np.median(np.exp(c["logprob_0"]) + np.exp(c["logprob_1"]))),
        "n": len(c),
    }), include_groups=False).reset_index()
    return out


def summary_table(cells: pd.DataFrame, value: str = "ba", index=("strategy",), columns=("naming", "env"),
                  label_mode: str = "gold") -> pd.DataFrame:
    """Mean over tasks of the per-task mean over seeds."""
    c = cells[cells["label_mode"] == label_mode]
    per_task = c.groupby([*index, *columns, "task_id"])[value].mean().reset_index()
    return per_task.pivot_table(index=list(index), columns=list(columns), values=value, aggfunc="mean")


def prior_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Share of predictions equal to the zero-shot prediction under the same naming, per cell."""
    df = prepare(df)
    zs = df[df["strategy"] == "zero_shot"].set_index(["model", "naming", "task_id", "env", "row_id"])["pred_raw"]
    rest = df[df["strategy"] != "zero_shot"].copy()
    key = pd.MultiIndex.from_frame(rest[["model", "naming", "task_id", "env", "row_id"]])
    rest["zs_pred"] = zs.reindex(key).to_numpy()
    rest["agree_cal"] = (rest["pred"] == rest["zs_pred"]).astype(float)
    rest["agree_raw"] = (rest["pred_raw"] == rest["zs_pred"]).astype(float)
    return rest.groupby(CELL_KEYS)[["agree_cal", "agree_raw"]].mean().reset_index()


# --------------------------------------------------------------------------- #
# contrasts
# --------------------------------------------------------------------------- #
def _pivot(g: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    piv = g.pivot_table(index="row_id", columns="seed", values=value)
    y = g.drop_duplicates("row_id").set_index("row_id").loc[piv.index, "y"].to_numpy()
    return y, piv.to_numpy(dtype=float), piv.index


def _paired(ga: pd.DataFrame, gb: pd.DataFrame, value: str):
    """(labels, A, B) on the queries both conditions scored, or None if either is missing
    (for example a task still running)."""
    if ga.empty or gb.empty:
        return None
    pa = ga.pivot_table(index="row_id", columns="seed", values=value)
    pb = gb.pivot_table(index="row_id", columns="seed", values=value)
    rows = pa.index.intersection(pb.index)
    if len(rows) == 0:
        return None
    y = ga.drop_duplicates("row_id").set_index("row_id").loc[rows, "y"].to_numpy()
    return y, pa.loc[rows].to_numpy(dtype=float), pb.loc[rows].to_numpy(dtype=float)


def _select(df, **kv):
    m = np.ones(len(df), dtype=bool)
    for k, v in kv.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def grid_blocks(df: pd.DataFrame, spec: dict, statistic: str) -> list:
    """Row blocks per task for a grid contrast (see `hierarchical_contrast`).
    Tasks missing either side of the contrast are left out."""
    value = "score" if statistic == "auroc" else "pred"
    d = _select(df, label_mode="gold")
    tasks = []
    for task_id, g in d.groupby("task_id"):
        kind = spec["kind"]
        blocks = []
        if kind == "env_gap":
            s = _select(g, strategy=spec["strategy"], naming=spec["naming"])
            a, b = _select(s, env=spec["env_a"]), _select(s, env=spec["env_b"])
            if a.empty or b.empty:
                continue
            ya, A, _ = _pivot(a, value)
            yb, B, _ = _pivot(b, value)
            blocks = [(ya, [(A, 1.0)]), (yb, [(B, -1.0)])]
        elif kind in ("naming_gap", "strategy"):
            if kind == "naming_gap":
                s = _select(g, strategy=spec["strategy"], env=spec["env"])
                pair = _paired(_select(s, naming=spec["naming_a"]), _select(s, naming=spec["naming_b"]), value)
            else:
                s = _select(g, naming=spec["naming"], env=spec["env"])
                pair = _paired(_select(s, strategy=spec["a"]), _select(s, strategy=spec["b"]), value)
            if pair is None:
                continue
            y, A, B = pair
            blocks = [(y, [(A, 1.0), (B, -1.0)])]
        elif kind == "interaction":
            s = _select(g, naming=spec["naming"])
            for env, w in ((spec["env_a"], 1.0), (spec["env_b"], -1.0)):
                e = _select(s, env=env)
                pair = _paired(_select(e, strategy=spec["a"]), _select(e, strategy=spec["b"]), value)
                if pair is None:
                    break
                y, A, B = pair
                blocks.append((y, [(A, w), (B, -w)]))
            if len(blocks) < 2:
                continue
        else:
            raise ValueError(f"not a grid contrast: {kind!r}")
        tasks.append(blocks)
    return tasks


def run_contrasts(grid: pd.DataFrame, rq3: pd.DataFrame | None = None, true_scores: dict | None = None,
                  n_bootstrap: int = 2000, seed: int = 0, contrasts: list[dict] = CONTRASTS) -> pd.DataFrame:
    """The contrast family: estimate, interval, one-sided p, Holm-adjusted p and sign test per contrast."""
    df = prepare(grid)
    rows = []
    for spec in contrasts:
        if spec["kind"].startswith("rq3"):
            if rq3 is None:
                continue
            res = rq3_contrast(rq3, spec, true_scores, n_bootstrap=n_bootstrap, seed=seed)
            secondary = None
        else:
            res = hierarchical_contrast(grid_blocks(df, spec, "auroc"), "auroc", n_bootstrap=n_bootstrap, seed=seed)
            res["statistic"] = "auroc"
            secondary = hierarchical_contrast(grid_blocks(df, spec, "ba"), "ba", n_bootstrap=n_bootstrap, seed=seed)
        per_task = np.array(res["per_task"], dtype=float)
        sign = per_task > 0 if spec["direction"] == "greater" else per_task < 0
        n_nonzero = int(np.sum(per_task != 0))
        rows.append({
            "id": spec["id"], "rq": spec["rq"], "label": spec["label"], "statistic": res["statistic"],
            "estimate": res["estimate"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "p_one_sided": res["p_greater"] if spec["direction"] == "greater" else res["p_less"],
            "tasks_in_direction": int(sign.sum()), "n_tasks": res["n_tasks"],
            "sign_test_p": float(binomtest(int(sign.sum()), n_nonzero, 0.5, alternative="greater").pvalue)
            if n_nonzero else float("nan"),
            "ba_estimate": None if secondary is None else secondary["estimate"],
            "ba_ci": None if secondary is None else [secondary["ci_low"], secondary["ci_high"]],
        })
    out = pd.DataFrame(rows)
    if len(out):
        ok = out["n_tasks"] > 0                          # a contrast without data is reported, not tested
        out["p_holm"] = float("nan")
        out.loc[ok, "p_holm"] = holm(out.loc[ok, "p_one_sided"].tolist())
        out["significant"] = out["p_holm"] < 0.05
    return out


# --------------------------------------------------------------------------- #
# RQ3
# --------------------------------------------------------------------------- #
def rq3_prepare(rq3: pd.DataFrame) -> pd.DataFrame:
    d = rq3.copy()
    d["sign"] = 2 * d["label"].astype(int) - 1
    d["correct_margin"] = d["sign"] * (d["margin"] - d["margin_cf"])        # calibration-invariant up to the cf shift
    d["correct"] = (d["prediction_cal"].astype(int) == d["label"].astype(int)).astype(float)
    return d


def behavioural_importance(rq3: pd.DataFrame) -> pd.DataFrame:
    """pi_behav per (model, naming, task, seed, feature): the mean drop in the
    correct-label margin (primary) and in calibrated accuracy when the feature
    is replaced by its hot-deck donor value."""
    d = rq3_prepare(rq3)
    key = ["model", "naming", "task_id", "seed", "row_id"]
    orig = d[d["variant"] == "orig"].set_index(key)[["correct_margin", "correct"]]
    hd = d[d["variant"] == "hotdeck"].set_index(key)
    hd = hd.join(orig, rsuffix="_orig")
    hd["margin_drop"] = hd["correct_margin_orig"] - hd["correct_margin"]
    hd["accuracy_drop"] = hd["correct_orig"] - hd["correct"]
    return hd.reset_index().groupby(["model", "naming", "task_id", "family", "seed", "feature", "feature_role"])[
        ["margin_drop", "accuracy_drop"]].mean().reset_index()


def directional_sensitivity(rq3: pd.DataFrame) -> pd.DataFrame:
    """g_j = m(x + delta e_j) - m(x - delta e_j) per (query, nudged feature), with
    whether it follows the data (sign g = s_j) and the name (sign g = name direction)."""
    d = rq3[rq3["variant"] == "nudge"]
    key = ["model", "naming", "task_id", "family", "seed", "row_id", "label", "feature", "feature_role",
           "data_direction", "name_direction"]
    piv = d.pivot_table(index=key, columns="delta", values="margin").reset_index()
    piv["g"] = piv[0.5] - piv[-0.5]
    piv["follows_data"] = (np.sign(piv["g"]) == piv["data_direction"]).astype(float)
    piv["follows_name"] = np.where(piv["name_direction"] != 0,
                                   (np.sign(piv["g"]) == piv["name_direction"]).astype(float), np.nan)
    return piv.drop(columns=[0.5, -0.5])


def dfi(rq3: pd.DataFrame) -> pd.DataFrame:
    """Demo-following index per (model, naming, task, seed): the share of
    (query, load-bearing feature) pairs whose sensitivity follows the data."""
    ds = directional_sensitivity(rq3)
    lb = ds[ds["feature_role"] == "load_bearing"]
    return lb.groupby(["model", "naming", "task_id", "family", "seed"])[["follows_data", "follows_name"]].mean().reset_index()


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if np.all(y == y[0]) or np.all(x == x[0]):
        return 0.0
    return float(spearmanr(x, y).statistic)


def correctness_rho(rq3: pd.DataFrame, true_scores: dict[str, dict[str, float]], value: str = "margin_drop") -> pd.DataFrame:
    """rho(pi_true, pi_behav) per (model, naming, task, seed); `true_scores[task][column]`."""
    bi = behavioural_importance(rq3)
    rows = []
    for (model, naming, task, seed), g in bi.groupby(["model", "naming", "task_id", "seed"]):
        t = np.array([true_scores[task][c] for c in g["feature"]])
        rows.append({"model": model, "naming": naming, "task_id": task, "seed": seed,
                     "rho": _rho(t, g[value].to_numpy())})
    return pd.DataFrame(rows)


def rq3_contrast(rq3: pd.DataFrame, spec: dict, true_scores: dict | None, n_bootstrap: int = 2000,
                 seed: int = 0) -> dict:
    """C3a (DFI) through `hierarchical_contrast` on per-query DFI; C3b (rho) with
    its own two-level bootstrap (tasks, then queries, shared by both namings)."""
    na, nb = spec["naming_a"], spec["naming_b"]
    if spec["kind"] == "rq3_dfi":
        ds = directional_sensitivity(rq3)
        lb = ds[ds["feature_role"] == "load_bearing"]
        per_q = lb.groupby(["naming", "task_id", "seed", "row_id", "label"])["follows_data"].mean().reset_index()
        tasks = []
        for _, g in per_q.groupby("task_id"):
            mats, y = [], None
            for naming, w in ((na, 1.0), (nb, -1.0)):
                piv = g[g["naming"] == naming].pivot_table(index=["row_id", "label"], columns="seed", values="follows_data")
                y = piv.index.get_level_values("label").to_numpy().astype(int)
                mats.append((piv.to_numpy(dtype=float), w))
            tasks.append([(y, mats)])
        res = hierarchical_contrast(tasks, "mean", n_bootstrap=n_bootstrap, seed=seed)
        res["statistic"] = "dfi"
        return res
    if true_scores is None:
        raise ValueError("C3b needs the true importance scores")
    d = rq3_prepare(rq3)
    key = ["naming", "task_id", "seed", "row_id"]
    orig = d[d["variant"] == "orig"].set_index(key)["correct_margin"]
    hd = d[d["variant"] == "hotdeck"].set_index(key)
    hd = hd.assign(drop=orig.reindex(hd.index).to_numpy() - hd["correct_margin"].to_numpy()).reset_index()
    cube = {}
    for (task, naming), g in hd.groupby(["task_id", "naming"]):
        piv = g.pivot_table(index="row_id", columns="feature", values="drop")   # queries x features (mean over seeds)
        cube[(task, naming)] = piv
    task_ids = sorted({t for t, _ in cube})

    def value(task, rows):
        out = 0.0
        for naming, w in ((na, 1.0), (nb, -1.0)):
            piv = cube[(task, naming)]
            t = np.array([true_scores[task][c] for c in piv.columns])
            out += w * _rho(t, piv.to_numpy()[rows].mean(axis=0))
        return out

    n_rows = {t: len(cube[(t, na)]) for t in task_ids}
    per_task = np.array([value(t, np.arange(n_rows[t])) for t in task_ids])
    rng = np.random.default_rng(seed)
    stats = np.empty(n_bootstrap)
    for r in range(n_bootstrap):
        draw = rng.integers(0, len(task_ids), size=len(task_ids))
        stats[r] = np.mean([value(task_ids[i], rng.integers(0, n_rows[task_ids[i]], n_rows[task_ids[i]])) for i in draw])
    return {"estimate": float(per_task.mean()), "ci_low": float(np.percentile(stats, 2.5)),
            "ci_high": float(np.percentile(stats, 97.5)),
            "p_greater": float((1 + np.sum(stats <= 0)) / (n_bootstrap + 1)),
            "p_less": float((1 + np.sum(stats >= 0)) / (n_bootstrap + 1)),
            "n_tasks": len(task_ids), "per_task": per_task.tolist(), "statistic": "rho"}


def true_importance_by_column(manifest: dict) -> dict[str, dict[str, float]]:
    """pi_true per displayed column: the generator's family-aware importance
    (true_importance_scores), zero for every column the rule does not read."""
    from src.evaluation.faithfulness_correctness import true_importance_scores

    out = {}
    for e in manifest["tasks"]:
        g = e["generator"]
        scores = true_importance_scores(g["rule_family"], g["n_features"], g["causal_features"],
                                        np.asarray(g["coefficients"]), g.get("thresholds3"), g.get("leaf_labels"))
        perm = e["permutation"]
        out[e["task_id"]] = {f"f{perm[i]}": float(scores[i]) for i in range(g["n_features"])}
    return out


def self_report_rho(rankings: pd.DataFrame, rq3: pd.DataFrame, value: str = "margin_drop") -> pd.DataFrame:
    """rho(pi_self, pi_behav) per (model, naming, task, seed): self-reported rank position vs behavioural drop."""
    bi = behavioural_importance(rq3)
    rows = []
    for _, r in rankings.iterrows():
        g = bi[(bi["model"] == r["model"]) & (bi["naming"] == r["naming"]) & (bi["task_id"] == r["task_id"])
               & (bi["seed"] == r["seed"])]
        if g.empty:
            continue
        position = {c: i for i, c in enumerate(r["ranking"])}
        self_score = np.array([-position[c] for c in g["feature"]], dtype=float)   # listed first = most important
        rows.append({"model": r["model"], "naming": r["naming"], "task_id": r["task_id"], "seed": r["seed"],
                     "n_listed": r["n_listed"], "rho": _rho(self_score, g[value].to_numpy())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# covariate shift beyond within-environment AUROC (hiccups/18; exploratory)
# --------------------------------------------------------------------------- #
def cross_environment_auroc(grid: pd.DataFrame, env_a: str = "id", env_b: str = "covariate") -> pd.DataFrame:
    """Per (strategy, naming, task, seed) cell pair: AUROC within each environment,
    and across them, ranking one environment's positives against the other's
    negatives. A shift that moves every query's score together leaves the
    within-environment AUROCs unchanged but shows up across environments.
    Also the mean score shift (calibrated margin, env_b minus env_a) and the
    shares predicted 1."""
    d = prepare(grid[grid["label_mode"] == "gold"])
    d = d.assign(cal_margin=np.log(d["score"].clip(1e-9, 1 - 1e-9)) - np.log1p(-d["score"].clip(1e-9, 1 - 1e-9)))
    rows = []
    for (s, n, t, seed), u in d.groupby(["strategy", "naming", "task_id", "seed"]):
        a, b = u[u["env"] == env_a], u[u["env"] == env_b]
        if a.empty or b.empty:
            continue
        ap, an, bp, bn = (a[a.y == 1].score, a[a.y == 0].score, b[b.y == 1].score, b[b.y == 0].score)
        cross = lambda pos, neg: auroc(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
        rows.append({"strategy": s, "naming": n, "task_id": t, "seed": seed,
                     f"auroc_{env_a}": auroc(a.y, a.score), f"auroc_{env_b}": auroc(b.y, b.score),
                     f"{env_a}_pos_vs_{env_b}_neg": cross(ap, bn), f"{env_b}_pos_vs_{env_a}_neg": cross(bp, an),
                     "pooled_auroc": auroc(np.r_[a.y, b.y], np.r_[a.score, b.score]),
                     "score_shift": float(b.cal_margin.mean() - a.cal_margin.mean()),
                     f"share_1_{env_a}": float(a.pred.mean()), f"share_1_{env_b}": float(b.pred.mean())})
    return pd.DataFrame(rows)


def task_level_interval(values: pd.Series, n_bootstrap: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over tasks of per-task means, with a percentile bootstrap interval over tasks."""
    per_task = values.groupby(level="task_id").mean().to_numpy()
    rng = np.random.default_rng(seed)
    boots = [per_task[rng.integers(0, len(per_task), len(per_task))].mean() for _ in range(n_bootstrap)]
    return float(per_task.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# Exploratory covariate probes (hiccups/18; not part of the frozen family). Gaps
# use the same hierarchical bootstrap as the family. f8-neutral copies share
# query ids with their source environment, so those pairs are matched query by
# query; other environment pairs are resampled separately.
EXPLORATORY_CONTRASTS = [
    {"id": "E1", "kind": "env_gap", "envs": ("id", "covariate"),
     "label": "ID minus covariate (the pre-registered C1a shift, all strategies)"},
    {"id": "E2", "kind": "env_gap", "envs": ("id", "covariate_scale"),
     "label": "ID minus covariate_scale (variance shift, shortcut intact)"},
    {"id": "E3a", "kind": "env_gap", "envs": ("id", "id_f8neutral"),
     "label": "what the shortcut adds on ID queries: ID minus ID with f8 neutralised"},
    {"id": "E3b", "kind": "protection", "envs": ("id", "covariate"),
     "label": "shortcut protection, uniform shift: (ID - covariate) with f8 neutralised minus with f8 intact"},
    {"id": "E3c", "kind": "protection", "envs": ("id", "covariate_scale"),
     "label": "shortcut protection, variance shift: (ID - covariate_scale) with f8 neutralised minus with f8 intact"},
]


def _by_query(g: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    piv = g.pivot_table(index="query_id", columns="seed", values=value)
    y = g.drop_duplicates("query_id").set_index("query_id").loc[piv.index, "y"].to_numpy()
    return y, piv.to_numpy(dtype=float), piv.index


def _env_block(s: pd.DataFrame, env_a: str, env_b: str, w: float, value: str):
    """Row blocks for w * (stat(env_a) - stat(env_b)); paired when env_b is env_a's f8-neutral copy."""
    a, b = s[s["env"] == env_a], s[s["env"] == env_b]
    if a.empty or b.empty:
        return None
    ya, A, ia = _by_query(a, value)
    yb, B, ib = _by_query(b, value)
    if env_b == f"{env_a}_f8neutral" or env_a == f"{env_b}_f8neutral":
        rows = ia.intersection(ib)
        A = pd.DataFrame(A, index=ia).loc[rows].to_numpy(); B = pd.DataFrame(B, index=ib).loc[rows].to_numpy()
        return [(ya[ia.get_indexer(rows)], [(A, w), (B, -w)])]
    return [(ya, [(A, w)]), (yb, [(B, -w)])]


def exploratory_contrasts(grid: pd.DataFrame, strategies=("random", "label_diversity", "counter_spurious"),
                          namings=("abstract", "aligned", "flipped"), n_bootstrap: int = 2000, seed: int = 0,
                          specs: list[dict] = EXPLORATORY_CONTRASTS) -> pd.DataFrame:
    """Every exploratory probe for every (strategy, naming): AUROC estimate,
    hierarchical-bootstrap interval and tasks in the positive direction."""
    df = prepare(grid[grid["label_mode"] == "gold"])
    rows = []
    for spec in specs:
        env_a, env_b = spec["envs"]
        for strategy in strategies:
            for naming in namings:
                s0 = _select(df, strategy=strategy, naming=naming)
                tasks = []
                for _, s in s0.groupby("task_id"):
                    if spec["kind"] == "env_gap":
                        blocks = _env_block(s, env_a, env_b, 1.0, "score")
                    else:                       # (a_n - b_n) - (a - b) = (a_n - a) - (b_n - b)
                        x = _env_block(s, f"{env_a}_f8neutral", env_a, 1.0, "score")
                        z = _env_block(s, f"{env_b}_f8neutral", env_b, -1.0, "score")
                        blocks = None if x is None or z is None else x + z
                    if blocks:
                        tasks.append(blocks)
                if not tasks:
                    continue
                res = hierarchical_contrast(tasks, "auroc", n_bootstrap=n_bootstrap, seed=seed)
                per_task = np.array(res["per_task"])
                rows.append({"id": spec["id"], "label": spec["label"], "strategy": strategy, "naming": naming,
                             "estimate": res["estimate"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                             "tasks_positive": int((per_task > 0).sum()), "n_tasks": res["n_tasks"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# demonstrations without the shortcut (exploratory, notebook 03.1; hiccups/20)
# --------------------------------------------------------------------------- #
# The same grid run three times: full demonstrations ("none"), demonstrations
# without the shortcut feature ("spurious"), and without the shortcut and the
# noise feature ("spurious_noise"). Queries keep all ten features. Contrasts
# are paired by task, seed and query; random, label_diversity and
# rule_diversity show the same rows in every arm.
DEMO_MASK_ARMS = ("none", "spurious", "spurious_noise")
DEMO_MASK_CONTRASTS = [
    {"id": "M1", "kind": "arm", "a": "spurious", "b": "none",
     "label": "demonstrations without the shortcut minus full demonstrations"},
    {"id": "M2", "kind": "arm", "a": "spurious_noise", "b": "none",
     "label": "demonstrations without the shortcut and the noise feature minus full demonstrations"},
    {"id": "M3", "kind": "arm", "a": "spurious_noise", "b": "spurious",
     "label": "what also removing the noise feature adds"},
    {"id": "M4", "kind": "gap_change", "a": "spurious", "b": "none", "envs": ("id", "spurious_reversal"),
     "label": "change in the ID minus spurious-reversal gap when the shortcut is removed"},
    {"id": "M5", "kind": "cross", "a": ("label_diversity", "spurious"), "b": ("counter_spurious", "none"),
     "label": "removing the shortcut (label_diversity, masked) minus neutralising it (counter_spurious, full)"},
]


def combine_arms(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One frame with an `arm` column, from each demo-mask arm's grid."""
    return pd.concat([df.assign(arm=arm) for arm, df in arms.items()], ignore_index=True)


def _arm_block(s: pd.DataFrame, a: dict, b: dict, w: float, value: str = "score"):
    """Row block for w * (stat(a) - stat(b)) on the queries both scored, paired by seed."""
    pair = _paired(_select(s, **a), _select(s, **b), value)
    if pair is None:
        return None
    y, A, B = pair
    return (y, [(A, w), (B, -w)])


def demo_mask_contrasts(arms: dict[str, pd.DataFrame],
                        strategies=("random", "label_diversity", "feature_range", "rule_diversity",
                                    "similarity", "counter_prior"),
                        namings=("abstract", "aligned", "flipped"),
                        envs=("id", "covariate", "spurious_reversal"),
                        n_bootstrap: int = 2000, seed: int = 0,
                        specs: list[dict] = DEMO_MASK_CONTRASTS) -> pd.DataFrame:
    """Every demo-mask contrast (M1-M5) for every strategy, naming and
    environment: AUROC estimate, hierarchical-bootstrap interval and tasks in
    the positive direction. M4 spans two environments; M5 compares two fixed
    strategies."""
    df = prepare(combine_arms(arms))
    df = df[df["label_mode"] == "gold"]
    rows = []
    for spec in specs:
        kind = spec["kind"]
        loop_strategies = [None] if kind == "cross" else strategies
        loop_envs = [None] if kind == "gap_change" else envs
        for strategy in loop_strategies:
            for naming in namings:
                for env in loop_envs:
                    tasks = []
                    for _, g in _select(df, naming=naming).groupby("task_id"):
                        if kind == "arm":
                            blk = _arm_block(g, dict(strategy=strategy, arm=spec["a"], env=env),
                                             dict(strategy=strategy, arm=spec["b"], env=env), 1.0)
                            blocks = None if blk is None else [blk]
                        elif kind == "gap_change":       # (a_id - a_rev) - (b_id - b_rev)
                            (ea, eb), parts = spec["envs"], []
                            for e, w in ((ea, 1.0), (eb, -1.0)):
                                parts.append(_arm_block(g, dict(strategy=strategy, arm=spec["a"], env=e),
                                                        dict(strategy=strategy, arm=spec["b"], env=e), w))
                            blocks = None if any(p is None for p in parts) else parts
                        elif kind == "cross":
                            (sa, aa), (sb, ab) = spec["a"], spec["b"]
                            blk = _arm_block(g, dict(strategy=sa, arm=aa, env=env), dict(strategy=sb, arm=ab, env=env), 1.0)
                            blocks = None if blk is None else [blk]
                        else:
                            raise ValueError(f"unknown demo-mask contrast kind {kind!r}")
                        if blocks:
                            tasks.append(blocks)
                    if not tasks:
                        continue
                    res = hierarchical_contrast(tasks, "auroc", n_bootstrap=n_bootstrap, seed=seed)
                    per_task = np.array(res["per_task"])
                    rows.append({
                        "id": spec["id"], "label": spec["label"],
                        "strategy": strategy if kind != "cross" else f"{spec['a'][0]} vs {spec['b'][0]}",
                        "naming": naming, "env": env if kind != "gap_change" else " - ".join(spec["envs"]),
                        "estimate": res["estimate"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                        "tasks_positive": int((per_task > 0).sum()), "n_tasks": res["n_tasks"],
                    })
    return pd.DataFrame(rows)
