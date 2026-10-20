"""P6 analysis: per-layer AUROC, the critical layer, linear probes and steering vectors.

Inputs are the probe records of `src/inference/probes.py` (one row per query
with the logit-lens margins m_0..m_L in `lens`) and, for random
demonstrations, the saved last-position residuals.

Predictions are centred before anyone is said to "follow" anything: Qwen
favours label 1, so a raw sign would call most predictions "1". A
prediction is 1 when its margin is above the median margin of its (task,
naming, condition, strategy) group, and Qwen's prior prediction for a query
is its zero-shot margin centred the same way (as SATA-MI centres m0,
src/models/sata_mi.py). The critical layer uses AUROC, which needs no
threshold (hiccups/15), instead of the spec's accuracy of 1[m_l > 0];
centred accuracy is reported alongside.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import auroc

GROUP = ["condition", "naming", "task_id", "strategy"]


def load_lens(paths) -> pd.DataFrame:
    """Probe records from one or more parquets; a unit written twice keeps its last copy."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return df.drop_duplicates(["condition", "unit_key", "row_id"], keep="last").reset_index(drop=True)


def lens_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["lens"].to_numpy()).astype(float)


def with_prior(df: pd.DataFrame) -> pd.DataFrame:
    """Add `pred` (centred prediction), `m0` (zero-shot margin) and `prior` (centred prior prediction)."""
    out = df.copy()
    out["pred"] = (out["margin"] > out.groupby(GROUP)["margin"].transform("median")).astype(int)
    zs = out[(out.strategy == "zero_shot") & (out.condition == "none")]
    zs = zs.assign(prior=(zs["margin"] > zs.groupby(["naming", "task_id"])["margin"].transform("median")).astype(int))
    key = ["naming", "task_id", "row_id"]
    out = out.merge(zs[key + ["margin", "prior"]].rename(columns={"margin": "m0"}), on=key, how="left")
    return out


def layer_auroc(df: pd.DataFrame, by=("condition", "naming", "strategy")) -> pd.DataFrame:
    """AUROC of m_l against the rule label (y_clean) per (task, environment) cell, averaged over
    environments per task: one row per (by..., task_id, layer)."""
    by = list(by)
    rows = []
    for key, g in df.groupby(by + ["task_id", "env"]):
        L = lens_matrix(g)
        y = g["y_clean"].to_numpy()
        for layer in range(L.shape[1]):
            rows.append((*key, layer, auroc(y, L[:, layer])))
    cells = pd.DataFrame(rows, columns=by + ["task_id", "env", "layer", "auroc"])
    return cells.groupby(by + ["task_id", "layer"], as_index=False)["auroc"].mean()


def layer_accuracy(df: pd.DataFrame, by=("condition", "naming", "strategy")) -> pd.DataFrame:
    """Accuracy of the centred prediction at every layer (1[m_l > its group's median]) against y_clean."""
    by = list(by)
    rows = []
    for key, g in df.groupby(by + ["task_id"]):
        L = lens_matrix(g)
        pred = L > np.median(L, axis=0, keepdims=True)
        acc = (pred == g["y_clean"].to_numpy()[:, None]).mean(axis=0)
        rows += [(*key, layer, a) for layer, a in enumerate(acc)]
    return pd.DataFrame(rows, columns=by + ["task_id", "layer", "accuracy"])


def critical_layer(per_task: pd.DataFrame, value: str = "auroc", a: str = "aligned", b: str = "flipped",
                   threshold: float = 0.1, n_boot: int = 2000, seed: int = 0) -> dict:
    """The first layer where the mean over tasks of value(a) - value(b) exceeds `threshold` with a
    task-bootstrap 95% interval above zero (generator_spec.pdf, critical layer; tasks resampled).
    `per_task` holds one strategy and condition: columns naming, task_id, layer, value."""
    wide = per_task.pivot_table(index=["task_id", "layer"], columns="naming", values=value)
    gap = (wide[a] - wide[b]).unstack("layer")              # tasks x layers
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gap), size=(n_boot, len(gap)))
    boots = np.stack([gap.to_numpy()[i].mean(axis=0) for i in idx])
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    mean = gap.mean(axis=0).to_numpy()
    table = pd.DataFrame({"layer": gap.columns, "gap": mean, "ci_low": lo, "ci_high": hi})
    hits = table[(table.gap > threshold) & (table.ci_low > 0)]
    return {"layer": int(hits.layer.iloc[0]) if len(hits) else None, "table": table}


def per_task_critical_layers(per_task: pd.DataFrame, value: str = "auroc", threshold: float = 0.1) -> pd.Series:
    """Each task's first layer with value(aligned) - value(flipped) > threshold (no interval; descriptive)."""
    wide = per_task.pivot_table(index=["task_id", "layer"], columns="naming", values=value)
    gap = (wide["aligned"] - wide["flipped"]).unstack("layer")
    return gap.apply(lambda r: next((int(l) for l, v in r.items() if v > threshold), np.nan), axis=1)


def load_residuals(df: pd.DataFrame, probe_dir: str | Path) -> np.ndarray:
    """Residuals (n_rows, L+1, d) for rows of `df` that carry a resid_file, in df's order."""
    probe_dir = Path(probe_dir)
    out, cache = [], {}
    pos = df.groupby("resid_file").cumcount().to_numpy()
    for f, p in zip(df["resid_file"], pos):
        if f not in cache:
            cache = {f: np.load(probe_dir / f)}
        out.append(cache[f][p])
    return np.stack(out)


def steering_vectors(df: pd.DataFrame, resid: np.ndarray, seed: int = 0) -> dict:
    """Difference of mean residuals between predictions that follow the demonstrations and
    predictions that follow the prior, on queries where the two disagree (y_clean != prior).
    Averaged over the two rule labels so the label direction cancels. Returns v (L+1, d), a random
    direction of the same norm per layer, and the counts."""
    conflict = (df["y_clean"] != df["prior"]).to_numpy() & df["prior"].notna().to_numpy()
    follows_demo = (df["pred"] == df["y_clean"]).to_numpy()
    v, counts = np.zeros(resid.shape[1:]), {}
    for c in (0, 1):
        lab = (df["y_clean"] == c).to_numpy() & conflict
        fd, fp = lab & follows_demo, lab & ~follows_demo
        counts[f"y{c}_follows_demos"], counts[f"y{c}_follows_prior"] = int(fd.sum()), int(fp.sum())
        if not fd.any() or not fp.any():
            raise ValueError(f"no conflict queries with rule label {c} on one side: {counts}")
        v += 0.5 * (resid[fd].astype(np.float64).mean(axis=0) - resid[fp].astype(np.float64).mean(axis=0))
    rng = np.random.default_rng(seed)
    r = rng.normal(size=v.shape)
    r *= np.linalg.norm(v, axis=1, keepdims=True) / np.linalg.norm(r, axis=1, keepdims=True)
    return {"v": v.astype(np.float32), "rand": r.astype(np.float32), "counts": counts,
            "norms": np.linalg.norm(v, axis=1)}


def probe_curves(resid: np.ndarray, targets: dict[str, np.ndarray], groups: np.ndarray,
                 layers=None, n_splits: int = 4, C: float = 0.1) -> pd.DataFrame:
    """Per layer, cross-validated AUROC of a logistic probe on h_l for each target (folds by task)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    layers = range(resid.shape[1]) if layers is None else layers
    rows = []
    for name, y in targets.items():
        y = np.asarray(y).astype(int)
        for layer in layers:
            X = resid[:, layer].astype(np.float32)
            score = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
                clf.fit(X[tr], y[tr])
                score[te] = clf.decision_function(X[te])
            per_task = [auroc(y[groups == g], score[groups == g]) for g in np.unique(groups)]
            rows.append((name, layer, float(np.nanmean(per_task))))
    return pd.DataFrame(rows, columns=["target", "layer", "auroc"])
