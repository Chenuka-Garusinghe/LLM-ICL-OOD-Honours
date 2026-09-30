"""The P2 learnability gate and the metrics it rests on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.bootstrap import hierarchical_auroc_diff
from src.evaluation.gates import cache_agreement, learnability_gate, smallest_passing_k
from src.evaluation.metrics import auroc, auroc_columns, balanced_accuracy


def test_auroc_matches_sklearn_including_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 10)
    s = np.round(rng.random(20), 1)          # coarse scores, so there are ties
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    cols = auroc_columns(s[y == 1][:, None], s[y == 0][:, None])
    assert cols[0] == pytest.approx(roc_auc_score(y, s))


def test_balanced_accuracy():
    assert balanced_accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.75)


def _fake_results(signal_by_k: dict[int, float], n_tasks=12, n_seeds=3, seed=0):
    """Gold p1 carries `signal` about the label; shuffled p1 carries none."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, signal in signal_by_k.items():
        for t in range(n_tasks):
            y = np.array([0, 1] * 10)
            for s in range(n_seeds):
                noise = rng.normal(size=(2, 20))
                for mode, p in (("gold", signal * y + noise[0]), ("shuffled", noise[1])):
                    for q in range(20):
                        rows.append({"model": "m", "k": k, "task_id": f"t{t}", "seed": s, "label_mode": mode,
                                     "strategy": "label_diversity", "env": "id", "row_id": q,
                                     "label": str(y[q]), "p1": 1 / (1 + np.exp(-p[q])),
                                     "logprob_0": -np.logaddexp(0, p[q]), "logprob_1": -np.logaddexp(0, -p[q]),
                                     "prediction_cal": str(int(p[q] > 0))})
    return pd.DataFrame(rows)


def test_learnability_gate_passes_only_with_signal():
    df = _fake_results({8: 0.0, 16: 1.5, 32: 3.0})
    gate = learnability_gate(df, n_bootstrap=300)
    by_k = gate.set_index("k")
    assert not by_k.loc[8, "passes"]
    assert by_k.loc[16, "passes"] and by_k.loc[32, "passes"]
    assert by_k.loc[32, "delta_auroc"] > by_k.loc[16, "delta_auroc"]
    assert smallest_passing_k(gate) == {"m": 16}
    assert gate["label_mass"].tolist() == pytest.approx([1.0, 1.0, 1.0]) and gate["valid"].all()


def test_learnability_gate_needs_the_label_tokens():
    """A model that puts little probability on "0"/"1" cannot pass, whatever its contrast."""
    df = _fake_results({16: 3.0})
    df["logprob_0"] -= np.log(50)
    df["logprob_1"] -= np.log(50)                 # label mass 0.02, as for Llama in P2
    gate = learnability_gate(df, n_bootstrap=300)
    assert gate.loc[0, "delta_auroc"] > 0.05 and gate.loc[0, "ci_low"] > 0
    assert gate.loc[0, "label_mass"] == pytest.approx(0.02) and not gate.loc[0, "valid"] and not gate.loc[0, "passes"]
    assert smallest_passing_k(gate) == {"m": None}


def test_hierarchical_interval_covers_zero_without_signal():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 10)
    tasks = [(y, rng.random((20, 3)), rng.random((20, 3))) for _ in range(12)]
    res = hierarchical_auroc_diff(tasks, n_bootstrap=500)
    assert res["ci_low"] < 0 < res["ci_high"]
    assert res["n_tasks"] == 12


def test_cache_agreement():
    """Prediction flips, calibrated agreement and per-task AUROC, cached against full margins."""
    def rows(task, labels, full, cached):
        return [{"task_id": task, "label": y, "margin_full": f, "margin_cached": c,
                 "prediction_full": "1" if f > 0 else "0", "prediction_cached": "1" if c > 0 else "0"}
                for y, f, c in zip(labels, full, cached)]
    # Task a: one near-tie query flips sign, the ranking is unchanged. The last row is the content-free suffix.
    # Task b: no prediction flips, but its two queries swap order, so AUROC goes from 1 to 0.
    scores = pd.DataFrame(rows("a", [0, 0, 1, 1, -1], [-1.0, 0.02, 0.5, 2.0, 0.1], [-1.0, -0.01, 0.5, 2.0, 0.1])
                          + rows("b", [0, 1, -1], [0.3, 0.4, 0.35], [0.45, 0.4, 0.35]))
    rep = cache_agreement(scores)
    assert rep["n_compared"] == 8 and rep["n_flips"] == 1
    assert rep["agreement_rate"] == pytest.approx(7 / 8)
    assert rep["max_abs_margin_at_flip"] == pytest.approx(0.02)
    assert rep["max_abs_margin_diff"] == pytest.approx(0.15)
    assert rep["calibrated_agreement_rate"] == pytest.approx(5 / 6)
    assert rep["auroc_full_mean"] == pytest.approx(1.0) and rep["auroc_cached_mean"] == pytest.approx(0.5)
    assert rep["max_abs_auroc_diff"] == pytest.approx(1.0) and rep["mean_abs_auroc_diff"] == pytest.approx(0.5)
    assert rep["abs_mean_auroc_diff"] == pytest.approx(0.5) and not rep["passes"]
    same = scores.assign(margin_cached=scores.margin_full, prediction_cached=scores.prediction_full)
    assert cache_agreement(same)["passes"]


def test_naming_gate_needs_fit_and_separation():
    from src.evaluation.gates import naming_gate, zero_shot_by_naming

    rng = np.random.default_rng(0)
    rows = []
    for t in range(12):
        y = np.array([0, 1] * 10)
        for naming, signal in (("abstract", 0.0), ("aligned", 2.0), ("flipped", -2.0)):
            m = signal * (2 * y - 1) + rng.normal(size=20) + 3.0      # a strong label bias towards 1
            for q in range(20):
                rows.append({"model": "m", "naming": naming, "strategy": "zero_shot", "env": "id",
                             "task_id": f"t{t}", "row_id": q, "label": str(y[q]), "p1": 1 / (1 + np.exp(-m[q])),
                             "prediction": str(int(m[q] > 0)),
                             "logprob_0": -np.logaddexp(0, m[q]), "logprob_1": -np.logaddexp(0, -m[q])})
    df = pd.DataFrame(rows)
    gate = naming_gate(df, {"loan": 0.8, "medical": 0.6}, n_bootstrap=300)
    assert gate["passes"] and gate["delta_auroc"] > 0.5 and gate["ci_low"] > 0
    assert gate["delta_accuracy"] < gate["delta_auroc"]       # the label bias hides part of it in raw accuracy
    assert not naming_gate(df, {"loan": 0.8, "medical": 0.4}, n_bootstrap=100)["passes"]
    by = zero_shot_by_naming(df).set_index("naming")
    assert by.loc["aligned", "auroc"] > 0.8 > 0.2 > by.loc["flipped", "auroc"]


def test_hierarchical_contrast_matches_the_auroc_version_and_holm():
    from src.evaluation.bootstrap import hierarchical_contrast, holm

    rng = np.random.default_rng(3)
    tasks, blocks = [], []
    for _ in range(10):
        y = np.array([0, 1] * 10)
        a = (1.0 * y)[:, None] + rng.normal(size=(20, 3))
        b = rng.normal(size=(20, 3))
        tasks.append((y, a, b))
        blocks.append([(y, [(a, 1.0), (b, -1.0)])])
    old = hierarchical_auroc_diff(tasks, n_bootstrap=200, seed=1)
    new = hierarchical_contrast(blocks, "auroc", n_bootstrap=200, seed=1)
    assert new["estimate"] == pytest.approx(old["estimate"])
    assert new["ci_low"] == pytest.approx(old["ci_low"]) and new["ci_high"] == pytest.approx(old["ci_high"])
    assert new["p_greater"] < 0.05 < new["p_less"]
    # BA on predictions, with an unpaired two-block gap
    pa, pb = (a > 0.5).astype(float), (b > 0.5).astype(float)
    gap = hierarchical_contrast([[(y, [(pa, 1.0)]), (y, [(pb, -1.0)])] for y, a, b in
                                 [(t[0], (t[1] > 0.5).astype(float), (t[2] > 0.5).astype(float)) for t in tasks]],
                                "ba", n_bootstrap=200)
    assert gap["estimate"] > 0
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
