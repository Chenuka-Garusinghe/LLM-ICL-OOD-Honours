"""P5 SATA: training data on the evaluation distribution, the proxy gate and the grid strategies."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.demo_mask import check_strategies
from src.data.generator import sample_eval_tasks
from src.data.synthetic_bridge import load_manifest, write_suite
from src.experiments.synth_grid import GridRunner, enumerate_units
from src.models.sata import build_sata
from src.models.sata_gate import proxy_compare, summarise
from src.models.sata_train import _task_batch, sample_sata_tasks, sata_config, train_sata
from src.utils.config import load_config
from tests.test_synth_pipeline import StubRunner


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_sata_tasks_come_from_the_evaluation_distribution():
    train, val = sample_sata_tasks(6, 4)
    assert [t.task_id for t in train][:2] == ["sata_train_0000", "sata_train_0001"]
    assert [t.rule_family for t in train] == ["linear"] * 3 + ["tree"] * 3
    eval_seeds = {t.data_seed for t in sample_eval_tasks(24)}
    assert not eval_seeds & {t.data_seed for t in train + val}


def test_column_permutation_moves_the_shortcut_and_keeps_the_metadata():
    task = sample_eval_tasks(1)[0]
    perm = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 9])
    (X, y, meta), (Xq, yq, mq) = _task_batch(task, "id", 64, 8, seed=3, pool_env="id")
    (Xp, yp, metap), (Xqp, _, mqp) = _task_batch(task, "id", 64, 8, seed=3, pool_env="id", permutation=perm)
    assert np.allclose(Xp, X[:, perm]) and np.allclose(Xqp, Xq[:, perm])
    assert (yp == y).all() and metap == meta and mqp == mq
    corr = [np.corrcoef(Xp[:, j], yp)[0, 1] for j in range(10)]
    assert int(np.argmax(np.abs(corr))) == 0                                   # the shortcut is now column 0
    assert np.sign(corr[0]) == task.spurious_sign


def test_oracle_targets_pick_no_label_noise_rows(config):
    cfg = sata_config(config, environments=["id", "spurious_reversal"], queries_per_env=4)
    _, val = sample_sata_tasks(0, 2)
    methods = ("oracle_matched", "oracle_target", "oracle_target_designed", "label_diversity")
    per_task = proxy_compare(val, cfg, methods=methods, n_jobs=2)
    table = summarise(per_task, ("id", "spurious_reversal"))
    assert table.loc["oracle_target", "label_noise"] == 0 and table.loc["oracle_matched", "label_noise"] == 0
    assert set(table.index) == set(methods)


@pytest.mark.parametrize("targets", ["designed", "matched"])
def test_training_runs_and_scores_on_the_proxy(config, tmp_path, targets):
    cfg = sata_config(config, epochs=1, max_demos=32, queries_per_env=4, environments=["id", "covariate"],
                      selection_environments=["id", "covariate"], device="cpu", targets=targets, train_seed=0)
    train, val = sample_sata_tasks(2, 1)
    model = build_sata("sata", 10, d_model=16, n_heads=2, n_layers=1)
    log = train_sata(model, train, val, cfg, checkpoint_path=tmp_path / "m.pt")
    assert len(log) == 1 and 0 <= log[0]["val_proxy"] <= 1 and (tmp_path / "m.pt").exists()


def test_balanced_sample_is_balanced_seeded_and_follows_the_weights():
    from src.selection.balanced_topk import balanced_sample
    labels = np.array([0] * 50 + [1] * 50)
    w = np.zeros(100); w[:10] = 1; w[50:60] = 1                      # only rows 0-9 and 50-59 have weight
    a = balanced_sample(w, labels, 8, np.random.default_rng(1))
    assert labels[a].sum() == 4 and len(set(a)) == 8
    assert set(a) <= set(range(10)) | set(range(50, 60))
    assert (a == balanced_sample(w, labels, 8, np.random.default_rng(1))).all()


def test_training_logs_both_selection_rules(config, tmp_path):
    cfg = sata_config(config, epochs=1, max_demos=32, queries_per_env=4, environments=["id"],
                      selection_environments=["id"], device="cpu", targets="matched", train_seed=0, selection="sample")
    train, val = sample_sata_tasks(2, 1)
    log = train_sata(build_sata("sata", 10, d_model=16, n_heads=2, n_layers=1), train, val, cfg,
                     checkpoint_path=tmp_path / "m.pt")
    assert {"val_proxy", "val_proxy_sample", "val_proxy_topk", "val_sample_id", "val_topk_id"} <= set(log[0])
    assert (tmp_path / "m.pt").exists() and (tmp_path / "m_topk.pt").exists()


def test_sata_is_refused_under_a_demo_mask():
    check_strategies(["sata", "sata_agnostic"], "none")
    with pytest.raises(ValueError, match="all ten features"):
        check_strategies(["sata"], "spurious")


@pytest.mark.parametrize("selection", ["sample", "topk"])
def test_grid_runs_sata_per_query_and_agnostic_per_task(tmp_path, selection):
    suite = tmp_path / "suite"
    write_suite(sample_eval_tasks(1, id_prefix="t"), suite, suite="t")
    manifest = load_manifest(suite)
    torch.manual_seed(0)
    models = {v: build_sata(v, 10, d_model=16, n_heads=2, n_layers=1).eval()
              for v in ("sata", "sata_agnostic", "sata_designed")}
    grid = GridRunner(StubRunner(), "stub", suite, manifest, base_seed=42, envs=["id", "spurious_reversal"],
                      queries_per_env=4, run_name="test", sata_models=models, sata_hashes={"sata": "abc"},
                      sata_selection=selection)
    units = enumerate_units(["stub"], ["abstract"], [manifest["tasks"][0]["task_id"]],
                            ["sata", "sata_agnostic", "sata_designed"], ["gold"], [8], 1)
    out = tmp_path / "grid.parquet"
    grid.run(units, out, log=lambda *_: None)
    df = pd.read_parquet(out)
    sets = df.groupby("strategy")["demo_ids"].apply(lambda s: {tuple(x) for x in s})
    assert len(sets["sata_agnostic"]) == 1                          # one set for every query
    assert len(sets["sata"]) > 1                                    # a set per query
    assert sets["sata"] != sets["sata_designed"]                    # its own model, through the same mechanism
    for ids in df["demo_ids"]:
        assert len(ids) == 8
    for labels in df["demo_labels"]:
        assert sum(labels) == 4                                     # balanced, as in training
    meta = json.loads(df.loc[df.strategy == "sata", "selection_meta"].iloc[0])
    assert meta["sata_checkpoint"] == "abc" and meta["sata_selection"] == selection
    assert meta["kind"] == ("weight" if selection == "sample" else "score")
