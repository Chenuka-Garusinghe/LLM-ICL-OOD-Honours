"""The prior-augmented proxy for SATA-MI (P5, 5b; generator_spec.pdf, SATA validation).

The XGBoost proxy cannot see names, so it cannot judge whether a selector
works with or against the model's prior. This proxy is a small reader model
of how Qwen combines the two:

    Qwen's score for query q  ~  a1 * w.x_q + a2 * m0_q   (within a prompt set)

- w.x_q is the demonstrations' evidence: w is a ridge logistic regression
  fitted on the k chosen demonstrations alone (their z-scored features and
  labels), evaluated at the query.
- m0_q is Qwen's measured zero-shot margin for the query, centred on the
  task's pool median (`PoolPrior`), i.e. its prior.
- a1, a2 are fitted once, by least squares on Qwen's own calibrated scores
  (logit of calibrated p1) from stage F, every strategy on the 8 `sata_llmval`
  tasks and all namings, with every variable centred within its prompt set
  and environment (`fit_reader`). Only their ratio matters for a ranking.

A first version fitted Qwen's thresholded predictions by logistic
regression. Contextual calibration leaves about two thirds of them at 1
(hiccups/15), so it learned "always 1" (agreement 0.657, the base rate).
The analysis rests on AUROC, so the reader models Qwen's *ranking* of
queries, and a selection's proxy score is the AUROC of the reader's score
against the true labels, within each (set, environment) cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = [f"f{i}" for i in range(10)]


def demo_evidence(demo_X: np.ndarray, demo_y: np.ndarray, query_X: np.ndarray, C: float = 1.0) -> np.ndarray:
    """w.x for each query, with w fitted on the demonstrations alone (ridge logistic, no intercept
    beyond sklearn's default). A single-class set gives no evidence (0)."""
    query_X = np.atleast_2d(query_X)
    if len(np.unique(demo_y)) < 2:
        return np.zeros(len(query_X))
    clf = LogisticRegression(C=C, max_iter=200).fit(demo_X, demo_y)
    return clf.decision_function(query_X)


@dataclass
class Reader:
    a1: float
    a2: float

    def score(self, evidence: np.ndarray, prior: np.ndarray) -> np.ndarray:
        return self.a1 * np.asarray(evidence, dtype=float) + self.a2 * np.asarray(prior, dtype=float)

    def auroc(self, evidence: np.ndarray, prior: np.ndarray, labels: np.ndarray) -> float:
        from src.evaluation.metrics import auroc
        return float(auroc(np.asarray(labels).astype(int), self.score(evidence, prior)))


def query_margins(zero_shot: pd.DataFrame, pool_prior) -> pd.Series:
    """Zero-shot margin of every query, centred on its (task, naming) pool median,
    indexed by (task_id, naming, row_id)."""
    z = zero_shot[zero_shot["strategy"] == "zero_shot"]
    med = (pool_prior.table.assign(m=lambda t: t["logprob_1"] - t["logprob_0"])
           .groupby(["task_id", "naming"])["m"].median())
    m = z["logprob_1"] - z["logprob_0"] - med.reindex(pd.MultiIndex.from_frame(z[["task_id", "naming"]])).to_numpy()
    return pd.Series(m.to_numpy(), index=pd.MultiIndex.from_frame(z[["task_id", "naming", "row_id"]]))


def reader_table(grid: pd.DataFrame, frames: dict[str, pd.DataFrame], qmargin: pd.Series) -> pd.DataFrame:
    """One row per scored query of every non-zero-shot set: demonstration evidence, prior,
    Qwen's calibrated prediction and the true label."""
    rows = []
    g = grid[(grid["strategy"] != "zero_shot") & (grid["label_mode"] == "gold")]
    for (task, naming, ids_key), part in g.groupby(["task_id", "naming", g["demo_ids"].map(tuple)], sort=False):
        cal = part["calibrated_p1"].clip(1e-6, 1 - 1e-6).to_numpy()
        frame = frames[task].set_index("row_id")
        ids = list(ids_key)
        dX = frame.loc[ids, FEATURES].to_numpy(float)
        dy = np.asarray(part["demo_labels"].iloc[0], dtype=int)
        qX = frame.loc[part["row_id"].to_numpy(), FEATURES].to_numpy(float)
        ev = demo_evidence(dX, dy, qX)
        pr = qmargin.reindex(pd.MultiIndex.from_arrays([[task] * len(part), [naming] * len(part), part["row_id"].to_numpy()])).to_numpy()
        rows.append(pd.DataFrame({"task_id": task, "naming": naming, "strategy": part["strategy"].to_numpy(),
                                  "unit_key": part["unit_key"].to_numpy(),
                                  "env": part["env"].to_numpy(), "evidence": ev, "prior": pr,
                                  "qwen_score": np.log(cal / (1 - cal)),
                                  "label": part["label"].astype(int).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def fit_reader(table: pd.DataFrame) -> tuple[Reader, dict]:
    """Least squares of Qwen's calibrated score on (evidence, prior), every variable
    centred within its (unit, environment) cell; R² is the within-cell fit."""
    t = table.dropna(subset=["evidence", "prior", "qwen_score"]).copy()
    cols = ["evidence", "prior", "qwen_score"]
    t[cols] = t[cols] - t.groupby(["unit_key", "env"])[cols].transform("mean")
    A = t[["evidence", "prior"]].to_numpy()
    coef, *_ = np.linalg.lstsq(A, t["qwen_score"].to_numpy(), rcond=None)
    resid = t["qwen_score"].to_numpy() - A @ coef
    r2 = 1 - resid.var() / t["qwen_score"].var()
    return Reader(float(coef[0]), float(coef[1])), {"n": len(t), "within_cell_r2": float(r2)}
