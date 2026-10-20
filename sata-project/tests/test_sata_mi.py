"""5b SATA-MI: items from fixed frames with a (stand-in) measured prior, targets, training and the grid."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.generator import sample_eval_tasks
from src.data.naming import load_lexicons
from src.data.synthetic_bridge import FEATURE_NAMES, load_manifest, load_task_frame, write_suite
from src.experiments.synth_grid import GridRunner, enumerate_units
from src.inference.priors import PoolPrior
from src.models.prior_proxy import Reader
from src.models.sata import INPUT_WIDTH, build_sata
from src.models.sata_mi import load_items, mi_inputs, mi_targets, train_mi
from src.models.sata_train import sata_config
from src.utils.config import load_config
from tests.test_synth_pipeline import StubRunner

NAMINGS = ("abstract", "aligned", "flipped")


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    """A 2-task suite with a stand-in prior: 'bigger numbers mean 1' (the sum of the features)."""
    out = tmp_path_factory.mktemp("mi")
    write_suite(sample_eval_tasks(2, id_prefix="mi"), out, suite="mi", ood_envs=("covariate", "spurious_reversal"),
                queries_per_class=4)
    rows = []
    for t in load_manifest(out)["tasks"]:
        f = load_task_frame(out, t["task_id"])
        for n in NAMINGS:
            m = f[FEATURE_NAMES].sum(axis=1).to_numpy() * 0.3 + (0.5 if n == "flipped" else 0.0)
            rows.append(pd.DataFrame({"task_id": t["task_id"], "naming": n, "row_id": f["row_id"],
                                      "split": f["split"], "logprob_1": m / 2, "logprob_0": -m / 2,
                                      "strategy": "zero_shot"}))
    allrows = pd.concat(rows, ignore_index=True)
    pp = PoolPrior(allrows[allrows["split"] == "pool"][["task_id", "naming", "row_id", "logprob_0", "logprob_1"]])
    qm = pd.Series((allrows["logprob_1"] - allrows["logprob_0"]).to_numpy(),
                   index=pd.MultiIndex.from_frame(allrows[["task_id", "naming", "row_id"]]))
    return out, pp, qm, allrows


def test_items_inputs_and_targets(suite):
    out, pp, qm, _ = suite
    items = load_items(out, pp, qm, load_lexicons("final"), 42)
    assert len(items) == 6 and all(set(it.queries) == {"id", "covariate", "spurious_reversal"} for it in items)
    it = next(i for i in items if i.naming == "aligned")
    assert abs(np.median(it.m0)) < 1e-9                                  # centred on the pool median
    assert set(np.unique(it.directions)) <= {-1.0, 0.0, 1.0} and np.abs(it.directions).sum() > 0
    assert np.abs(next(i for i in items if i.naming == "abstract").directions).sum() == 0
    Xq, yq, q_meta, m0q = it.queries["id"]
    d_in, q_in = mi_inputs("sata_mi", it.X, Xq, it.m0, m0q, it.directions)
    assert d_in.shape == (256, INPUT_WIDTH["sata_mi"]) and q_in.shape == (len(yq), 11)
    d_in, q_in = mi_inputs("sata_names", it.X, Xq, it.m0, m0q, it.directions)
    assert d_in.shape[1] == INPUT_WIDTH["sata_names"]
    q = mi_targets(q_meta[0], it.meta, it.shortcut, it.contra_clean, 0.9, 1.0)
    assert abs(q.sum() - 1) < 1e-9
    assert all(q[i] == 0 for i, m in enumerate(it.meta) if m["label"] != m["y_clean"])
    assert q[it.contra_clean].sum() > mi_targets(q_meta[0], it.meta, it.shortcut, it.contra_clean, 0.9, 0.0)[it.contra_clean].sum()


def test_train_one_epoch(suite, tmp_path):
    out, pp, qm, _ = suite
    items = load_items(out, pp, qm, load_lexicons("final"), 42)
    readers = {n: Reader(0.5, 0.3) for n in NAMINGS}
    cfg = sata_config(load_config(), epochs=1, device="cpu", train_seed=0)
    model = build_sata("sata_mi", 11, d_model=16, n_heads=2, n_layers=1)
    log = train_mi(model, "sata_mi", items[:4], items[4:], readers, cfg, checkpoint_path=tmp_path / "mi.pt")
    assert len(log) == 1 and {"val_proxy_sample", "val_proxy_topk"} <= set(log[0])
    assert (tmp_path / "mi.pt").exists() and (tmp_path / "mi_topk.pt").exists()


def test_grid_runs_sata_mi_and_names(suite, tmp_path):
    out, pp, qm, _ = suite
    manifest = load_manifest(out)
    torch.manual_seed(0)
    models = {v: build_sata(v, INPUT_WIDTH[v], d_model=16, n_heads=2, n_layers=1).eval() for v in ("sata_mi", "sata_names")}
    grid = GridRunner(StubRunner(), "stub", out, manifest, base_seed=42, envs=["id"], queries_per_env=4, run_name="t",
                      lexicons=load_lexicons("final"), pool_prior=pp, query_prior=qm, sata_models=models)
    units = enumerate_units(["stub"], ["flipped"], [manifest["tasks"][0]["task_id"]], ["sata_mi", "sata_names"],
                            ["gold"], [8], 1)
    path = tmp_path / "g.parquet"
    grid.run(units, path, log=lambda *_: None)
    df = pd.read_parquet(path)
    assert set(df["strategy"]) == {"sata_mi", "sata_names"}
    assert all(sum(l) == 4 for l in df["demo_labels"])
