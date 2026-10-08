"""Demonstrations without the shortcut (and noise) feature: notebook 03.1,
src/data/demo_mask.py, hiccups/20.

Hidden features come from the ground truth (generator columns through the
recorded permutation); demonstrations and selection never show or read them;
queries keep all ten features; the rows of naming- and feature-blind
strategies are the same in every arm.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import pytest

from src.data.demo_mask import DEMO_MASKS, GENERATOR_COLUMNS, check_strategies, hidden_features, visible_features
from src.data.generator import sample_eval_tasks
from src.data.synthetic_bridge import FEATURE_NAMES, load_manifest, pool_of, write_suite
from src.evaluation.analysis import DEMO_MASK_CONTRASTS, demo_mask_contrasts
from src.experiments.synth_grid import GridRunner, Unit, enumerate_units
from src.inference.priors import measure_pool_prior
from src.inference.prompts import system_message
from src.selection.protocols import ProtocolContext, select
from tests.test_synth_pipeline import StubRunner

ENVS = ["id", "covariate", "spurious_reversal"]


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    out = tmp_path_factory.mktemp("mask_suite")
    write_suite(sample_eval_tasks(4, id_prefix="m"), out, suite="m")
    return out


@pytest.fixture(scope="module")
def manifest(suite):
    return load_manifest(suite)


def _grid(suite, manifest, mask, runner=None, n_features=None):
    return GridRunner(runner or StubRunner(), "stub", suite, manifest, base_seed=42, envs=ENVS,
                      queries_per_env=4, run_name=f"mask_{mask}", demo_mask=mask, system_n_features=n_features)


def _names_in(line: str) -> list[str]:
    """Feature names of one serialised row ("f0: 0.12; f3: -1.00 -> 1")."""
    return re.findall(r"(f\d): ", line)


# ---------------------------------------------------------------- ground truth
def test_hidden_features_come_from_the_permutation(manifest):
    assert GENERATOR_COLUMNS == {"spurious": 8, "noise": 9}
    for entry in manifest["tasks"]:
        perm = entry["permutation"]
        assert hidden_features(entry, "none") == ()
        assert hidden_features(entry, "spurious") == (f"f{perm[8]}",) == (entry["roles"]["spurious"],)
        assert hidden_features(entry, "spurious_noise") == (entry["roles"]["spurious"], entry["roles"]["noise"])
        assert len(visible_features(entry, "spurious_noise")) == 8
        assert entry["roles"]["spurious"] not in visible_features(entry, "spurious")


def test_a_manifest_that_disagrees_with_its_permutation_is_refused(manifest):
    entry = copy.deepcopy(manifest["tasks"][0])
    entry["roles"]["spurious"] = entry["roles"]["noise"]
    with pytest.raises(ValueError, match="generator column 8"):
        hidden_features(entry, "spurious")
    with pytest.raises(ValueError, match="unknown demo mask"):
        hidden_features(entry, "rule")


# ---------------------------------------------------------------- prompts
@pytest.mark.parametrize("mask", list(DEMO_MASKS))
def test_demonstrations_hide_only_the_masked_features(suite, manifest, mask):
    grid = _grid(suite, manifest, mask)
    for entry in manifest["tasks"]:
        unit = Unit("stub", "abstract", entry["task_id"], "label_diversity", "gold", 8, 0)
        (group,) = grid.prompt_groups(unit)
        visible = visible_features(entry, mask)
        assert len(group["demo"]["lines"]) == 8
        for line in group["demo"]["lines"]:
            assert _names_in(line) == visible
        for suffix in group["suffixes"]:                 # every query and the content-free query
            assert _names_in(suffix) == FEATURE_NAMES
        assert "N/A" in group["suffixes"][-1]
        if mask != "none":
            assert group["demo"]["meta"]["hidden_features"] == list(hidden_features(entry, mask))


def test_count_free_system_message(suite, manifest):
    assert "10" not in system_message(None, None) and "10" not in system_message("loan", None)
    assert "with 10 measurements" in system_message("loan")          # the P4 wording is unchanged
    grid = _grid(suite, manifest, "spurious", n_features=None)
    unit = Unit("stub", "abstract", manifest["tasks"][0]["task_id"], "random", "gold", 8, 0)
    (group,) = grid.prompt_groups(unit)
    assert group["prefix"].startswith(system_message(None, None))


@pytest.mark.parametrize("strategy", ["random", "label_diversity", "rule_diversity"])
def test_feature_blind_strategies_show_the_same_rows_in_every_arm(suite, manifest, strategy):
    for entry in manifest["tasks"]:
        unit = Unit("stub", "abstract", entry["task_id"], strategy, "gold", 8, 1)
        ids = [_grid(suite, manifest, m).prompt_groups(unit)[0]["demo"]["ids"] for m in DEMO_MASKS]
        assert ids[0] == ids[1] == ids[2]


# ---------------------------------------------------------------- selection
def test_feature_range_never_covers_a_hidden_feature(suite, manifest):
    for mask in ("spurious", "spurious_noise"):
        grid = _grid(suite, manifest, mask)
        for entry in manifest["tasks"]:
            unit = Unit("stub", "abstract", entry["task_id"], "feature_range", "gold", 8, 0)
            meta = grid.prompt_groups(unit)[0]["demo"]["meta"]
            assert set(meta["features"]) <= set(visible_features(entry, mask))


def test_similarity_ignores_hidden_values(suite, manifest):
    grid = _grid(suite, manifest, "spurious_noise")
    entry = manifest["tasks"][0]
    pool = pool_of(grid.frame(entry["task_id"]))
    unit = Unit("stub", "abstract", entry["task_id"], "similarity", "gold", 8, 0)
    ctx = grid.context(unit, pool)
    assert ctx.feature_cols == visible_features(entry, "spurious_noise")
    query = pool.iloc[0]
    scrambled = pool.copy()
    rng = np.random.default_rng(0)
    for f in hidden_features(entry, "spurious_noise"):
        scrambled[f] = rng.permutation(scrambled[f].to_numpy())
    ctx2 = ProtocolContext(feature_cols=ctx.feature_cols, kinds=ctx.kinds, train_ref=scrambled)
    assert select("feature_knn", "balanced", pool, query, 8, 7, ctx)[0] == \
        select("feature_knn", "balanced", scrambled, query, 8, 7, ctx2)[0]


def test_strategies_that_select_on_the_shortcut_are_refused(suite, manifest):
    check_strategies(["counter_spurious", "counter_prior_matched"], "none")
    for mask in ("spurious", "spurious_noise"):
        with pytest.raises(ValueError, match="select on the shortcut"):
            check_strategies(["random", "counter_spurious"], mask)
        unit = Unit("stub", "abstract", manifest["tasks"][0]["task_id"], "counter_spurious", "gold", 8, 0)
        with pytest.raises(ValueError, match="select on the shortcut"):
            _grid(suite, manifest, mask).prompt_groups(unit)


def test_pool_prior_scores_the_masked_rows(suite, manifest):
    runner = StubRunner()
    grid = _grid(suite, manifest, "spurious", runner=runner, n_features=None)
    entry = manifest["tasks"][0]
    out = measure_pool_prior(grid, entry["task_id"], "abstract")
    assert len(out) == len(pool_of(grid.frame(entry["task_id"])))
    (prefix, suffixes), = runner.calls
    assert prefix.startswith(system_message(None, None))
    assert all(_names_in(s) == visible_features(entry, "spurious") for s in suffixes)


# ---------------------------------------------------------------- analysis
def test_demo_mask_contrasts_pair_arms(suite, manifest, tmp_path):
    task_ids = [t["task_id"] for t in manifest["tasks"]]
    units = enumerate_units(["stub"], ["abstract"], task_ids, ["label_diversity"], ["gold"], [8], 2)
    arms = {}
    for mask in ("none", "spurious"):
        out = tmp_path / f"{mask}.parquet"
        _grid(suite, manifest, mask, n_features=None).run(units, out, log=lambda *_: None)
        arms[mask] = pd.read_parquet(out)
    # The same frame as both arms: every paired difference is exactly zero.
    same = demo_mask_contrasts({"none": arms["none"], "spurious": arms["none"]}, strategies=("label_diversity",),
                               namings=("abstract",), n_bootstrap=50,
                               specs=[s for s in DEMO_MASK_CONTRASTS if s["id"] in ("M1", "M4")])
    assert len(same) == 4 and (same["estimate"] == 0).all() and (same["n_tasks"] == len(task_ids)).all()
    # Real arms: same rows, different prompts, so the scores differ.
    res = demo_mask_contrasts(arms, strategies=("label_diversity",), namings=("abstract",), n_bootstrap=50)
    m1 = res[res["id"] == "M1"]
    assert len(m1) == 3 and m1["n_tasks"].eq(len(task_ids)).all() and np.isfinite(m1["estimate"]).all()


def test_the_hidden_shortcut_is_the_shortcut_in_the_data(suite, manifest):
    """Independent of the lookup: the hidden feature is the one displayed feature holding the
    generator's own shortcut values (`spurious_raw`, saved before the shuffle), and the one whose
    agreement with the label collapses under spurious reversal. The hidden noise feature is
    unrelated to the label."""
    from src.data.synthetic_bridge import load_task_frame
    for entry in manifest["tasks"]:
        df = load_task_frame(suite, entry["task_id"])
        shortcut, noise = hidden_features(entry, "spurious_noise")
        corr = {f: abs(np.corrcoef(df[f], df["spurious_raw"])[0, 1]) for f in FEATURE_NAMES}
        assert corr[shortcut] > 0.999999 and max(v for f, v in corr.items() if f != shortcut) < 0.9
        sign = entry["spurious_direction"]
        agree = {}
        for env in ("id", "spurious_reversal"):
            d = df[(df["env"] == env) & (df["split"] != "pool")]
            agree[env] = {f: float((np.sign(sign * (d[f] - df.loc[df["split"] == "pool", f].mean()))
                                    == np.sign(2 * d["y_clean"] - 1)).mean()) for f in FEATURE_NAMES}
        drop = {f: agree["id"][f] - agree["spurious_reversal"][f] for f in FEATURE_NAMES}
        assert max(drop, key=drop.get) == shortcut and drop[shortcut] > 0.4
        pool = df[df["split"] == "pool"]
        assert abs(np.corrcoef(pool[noise], pool["label"])[0, 1]) < 0.25
