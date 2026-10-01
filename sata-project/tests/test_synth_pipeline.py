"""Gate G1 for the synthetic pipeline (docs/research_plan.md, P1), without a model.

Covers the generator (evaluation sampler, directions, manifest round trip),
the per-task bridge (single-task pools, standardisation, column roles,
queries), the selection engine (balanced sets, decorrelated counter-spurious
sets, seeding) and the grid engine run against a stub runner (resume, pairing,
calibration). Model-side checks (single BOS, prefix caching, prompt
snapshots) are in tests/test_hf_runner.py.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data.generator import SyntheticTask, _is_degenerate_leaf_labels, sample_eval_tasks
from src.data.synthetic_bridge import FEATURE_NAMES, build_task_frame, load_manifest, pool_of, queries_of, write_suite
from src.experiments.synth_grid import GridRunner, completed_units, enumerate_units
from src.inference.llm_runner import get_confidence
from src.inference.prompts import completion_prompt
from src.selection.protocols import SYNTHETIC_STRATEGIES, ProtocolContext, demo_seed, select

N_TASKS = 24


@pytest.fixture(scope="module")
def tasks():
    return sample_eval_tasks(N_TASKS)


@pytest.fixture(scope="module")
def frames(tasks):
    return [build_task_frame(t) for t in tasks]


def _ctx(pool, prior_slopes=None):
    """Selection context; the default prior is "bigger values mean 1" on every column (for counter_prior)."""
    return ProtocolContext(feature_cols=FEATURE_NAMES, kinds={f: "numeric" for f in FEATURE_NAMES}, train_ref=pool,
                           prior_slopes=prior_slopes or {f: 1.0 for f in FEATURE_NAMES})


# ---------------------------------------------------------------- generator
def test_eval_sampler_design(tasks):
    assert [t.rule_family for t in tasks] == ["linear"] * 12 + ["tree"] * 12
    for fam in ("linear", "tree"):
        doms = [t.domain for t in tasks if t.rule_family == fam]
        assert doms.count("loan") == doms.count("medical") == 6
    for t in tasks:
        assert len(t.load_bearing_features()) == 3
        assert set(t.load_bearing_features()) <= set(range(8))
        assert 0.80 <= t.spurious_strength <= 0.90
        if t.rule_family == "linear":
            assert np.all((np.abs(t.coefficients) >= 1) & (np.abs(t.coefficients) <= 2))
        else:
            assert not _is_degenerate_leaf_labels(np.asarray(t.leaf_labels), 3)


def test_sampler_is_deterministic(tasks):
    again = sample_eval_tasks(N_TASKS)
    assert [t.to_meta() for t in again] == [t.to_meta() for t in tasks]
    # A task's parameters come from its own stream: with the same family, task i
    # is the same in a smaller suite.
    assert sample_eval_tasks(4)[1].to_meta()["coefficients"] == tasks[1].to_meta()["coefficients"]


def test_suite_matches_the_config(tasks):
    from src.data.suites import suite_tasks
    from src.utils.config import load_config

    assert [t.to_meta() for t in suite_tasks("eval", load_config())] == [t.to_meta() for t in tasks]


def test_manifest_round_trip(tasks):
    for t in tasks[:3] + tasks[12:15]:
        rebuilt = SyntheticTask.from_meta(json.loads(json.dumps(t.to_meta())))
        assert rebuilt.to_meta() == t.to_meta()
        X, y, _ = t.generate_environment("id", 200, seed=5)
        X2, y2, _ = rebuilt.generate_environment("id", 200, seed=5)
        assert np.array_equal(X, X2) and np.array_equal(y, y2)


def test_directions_match_the_rule(tasks):
    """Nudging a load-bearing feature in its direction never lowers P(y=1)."""
    rng = np.random.default_rng(0)
    for t in tasks:
        X = rng.normal(size=(4000, 10))
        base = t._apply_rule(X).mean()
        for g, s in zip(t.load_bearing_features(), t.directions()):
            Xp = X.copy()
            Xp[:, g] += 1.0 * s
            assert t._apply_rule(Xp).mean() > base


def test_tree_regime_is_the_leaf(tasks):
    t = tasks[12]
    _, _, meta = t.generate_environment("id", 2000, seed=3)
    regimes = np.array([m["regime"] for m in meta])
    y_clean = np.array([m["y_clean"] for m in meta])
    assert np.array_equal(np.asarray(t.leaf_labels)[regimes], y_clean)


def test_f8_agreement_matches_strength(tasks):
    """G1: clean sign agreement within +-0.03 of s (large sample, generator scale).
    Agreement reads f8 in the task's direction; the raw column agrees with the
    label with probability s when the direction is +1 and 1 - s when it is -1."""
    for t in tasks:
        X, _, meta = t.generate_environment("id", 20000, seed=11)
        agree = np.mean([m["agree_clean"] for m in meta])
        assert abs(agree - t.spurious_strength) <= 0.03, (t.task_id, agree, t.spurious_strength)
        y_clean = np.array([m["y_clean"] for m in meta])
        raw = np.mean(np.sign(X[:, t.spurious_idx]) == np.sign(2 * y_clean - 1))
        expected = t.spurious_strength if t.spurious_sign > 0 else 1 - t.spurious_strength
        assert abs(raw - expected) <= 0.03, (t.task_id, raw, expected)


def test_spurious_direction_is_balanced(tasks):
    """Half the tasks of each (family, domain) cell get each direction; in the
    pilot suite (odd cells) each family is still balanced."""
    for suite in (tasks, sample_eval_tasks(12, id_prefix="pilot")):
        cells = {}
        for t in suite:
            cells.setdefault((t.rule_family, t.domain), []).append(t.spurious_sign)
        for signs in cells.values():
            assert set(signs) <= {-1, 1} and abs(sum(signs)) <= 1
        for fam in ("linear", "tree"):
            assert sum(t.spurious_sign for t in suite if t.rule_family == fam) == 0


def test_spurious_direction_reflects_only_f8(tasks):
    """With direction -1 the other columns, the labels and the metadata equal
    those of direction +1, and f8 is negated."""
    import copy

    pos, neg = copy.deepcopy(tasks[0]), copy.deepcopy(tasks[0])
    pos.spurious_sign, neg.spurious_sign = 1, -1
    Xp, yp, mp = pos.generate_environment("spurious_reversal", 300, seed=4)
    Xn, yn, mn = neg.generate_environment("spurious_reversal", 300, seed=4)
    assert np.array_equal(np.delete(Xp, 8, axis=1), np.delete(Xn, 8, axis=1))
    assert np.array_equal(Xp[:, 8], -Xn[:, 8])
    assert np.array_equal(yp, yn) and mp == mn


def test_covariate_shift_moves_two_rule_features(tasks):
    """The covariate shift moves two load-bearing features, one towards each
    label by the same amount in label terms, plus one distractor, each by 1-2
    standard deviations. (v2's rejection sampling left 21 of 24 tasks
    shifting distractors only.)"""
    for t in tasks:
        t.generate_environment("covariate", 50, seed=1)
        info = t.last_env_info
        f, d = info["shift_features"], np.array(info["shift_delta"])
        lb, s = t.load_bearing_features(), t.directions()
        assert info["method"] == "label_neutral" and info["shift_roles"] == ["towards_1", "towards_0", "distractor"]
        assert f[0] in lb and f[1] in lb and f[0] != f[1]
        assert f[2] < 8 and f[2] not in t.causal_features
        assert np.all((np.abs(d) >= 1 - 1e-9) & (np.abs(d) <= 2 + 1e-9)), (t.task_id, d)
        assert s[lb.index(f[0])] * d[0] > 0 and s[lb.index(f[1])] * d[1] < 0
        if t.rule_family == "linear":
            c = dict(zip(t.causal_features, t.coefficients))
            assert c[f[0]] * d[0] + c[f[1]] * d[1] == pytest.approx(0, abs=1e-9)
        else:
            assert abs(d[0]) == pytest.approx(abs(d[1]))


def test_covariate_shift_keeps_the_label_rate(tasks):
    """P(y) is unchanged: exactly for linear tasks (c . x keeps its distribution),
    and for tree tasks because the two shifted votes' probabilities sum to 1."""
    for t in (tasks[0], tasks[12]):
        _, _, meta_id = t.generate_environment("id", 100000, seed=5)
        _, _, meta_cov = t.generate_environment("covariate", 100000, seed=5)
        rate_id = np.mean([m["y_clean"] for m in meta_id])
        rate_cov = np.mean([m["y_clean"] for m in meta_cov])
        assert abs(rate_cov - rate_id) < 0.01, (t.task_id, rate_id, rate_cov)
        assert abs(t.last_env_info["probe_diff"]) < 0.02


# ---------------------------------------------------------------- bridge
def test_frames_hold_one_task_each(tasks, frames):
    for t, (frame, entry) in zip(tasks, frames):
        assert set(frame["task_id"]) == {t.task_id}
        assert entry["task_id"] == t.task_id
        counts = frame.groupby(["split", "env"]).size().to_dict()
        assert counts == {("pool", "id"): 256, ("test_id", "id"): 100,
                          ("test_ood", "covariate"): 100, ("test_ood", "spurious_reversal"): 100}
        assert frame["row_id"].is_unique


def test_label_rate_in_range(frames):
    """G1: per-task pool label rate in [0.35, 0.65]."""
    for frame, entry in frames:
        assert 0.35 <= entry["pool_label_rate"] <= 0.65, entry["task_id"]


def test_standardised_with_pool_statistics(frames):
    for frame, entry in frames:
        pool = pool_of(frame)[FEATURE_NAMES]
        assert np.allclose(pool.mean(), 0, atol=1e-9)
        assert np.allclose(pool.std(ddof=0), 1, atol=1e-9)
        # The covariate split is not re-standardised: its shifted columns keep an offset.
        cov = frame[frame["env"] == "covariate"]
        shifted = entry["covariate_shift"]["shift_features_displayed"]
        assert (cov[shifted].mean().abs() > 0.5).all()


def test_column_roles_come_from_the_permutation(tasks, frames):
    positions = []
    for t, (frame, entry) in zip(tasks, frames):
        perm = entry["permutation"]
        assert sorted(perm) == list(range(10))
        spur = entry["roles"]["spurious"]
        assert spur == f"f{perm[t.spurious_idx]}"
        assert np.corrcoef(frame[spur], frame["spurious_raw"])[0, 1] == pytest.approx(1.0)
        roles = entry["roles"]
        assert sorted(roles["load_bearing"] + [roles["spurious"], roles["noise"]] + roles["distractor"]) == sorted(FEATURE_NAMES)
        positions.append(perm[t.spurious_idx])
    assert len(set(positions)) > 3, "the spurious column should move between tasks"


def test_queries_are_balanced(frames):
    for frame, _ in frames:
        for env in ("id", "covariate", "spurious_reversal"):
            q = queries_of(frame, env)
            assert len(q) == 20
            assert q["label"].value_counts().to_dict() == {0: 10, 1: 10}
            assert (q["split"] != "pool").all()


# ---------------------------------------------------------------- selection
@pytest.mark.parametrize("strategy", list(SYNTHETIC_STRATEGIES))
@pytest.mark.parametrize("k", [8, 16, 32])
def test_selected_demos_come_from_the_pool_and_are_balanced(frames, strategy, k):
    mech, comp = SYNTHETIC_STRATEGIES[strategy]
    for frame, entry in frames[::4]:
        pool = pool_of(frame)
        query = queries_of(frame, "id").iloc[0] if mech == "feature_knn" else None
        seed = demo_seed(42, entry["task_id"], strategy, 0)
        ids, _ = select(mech, comp, pool, query, k, seed, _ctx(pool))
        assert len(set(ids)) == k
        assert set(ids) <= set(pool.index)
        assert ids == select(mech, comp, pool, query, k, seed, _ctx(pool))[0]
        if comp == "balanced":
            assert int(pool.loc[ids, "label"].sum()) == k // 2


def test_counter_spurious_sets_are_decorrelated(frames):
    """G1: pooled |corr(f8, y)| <= 0.1 over counter-spurious sets (naive cell balancing gives ~0.33)."""
    f8, y = [], []
    for frame, entry in frames:
        pool = pool_of(frame)
        for seed in range(3):
            ids, meta = select("gt_counter_spurious", "balanced", pool, None, 8,
                               demo_seed(42, entry["task_id"], "counter_spurious", seed), _ctx(pool))
            f8 += list(entry["spurious_direction"] * pool.loc[ids, "spurious_raw"])   # in the task's direction
            y += list(pool.loc[ids, "label"])
    assert abs(np.corrcoef(f8, y)[0, 1]) <= 0.1
    agree = np.mean(np.sign(f8) == np.sign(2 * np.array(y) - 1))
    assert agree == pytest.approx(0.5, abs=0.02)


def test_rule_diversity_covers_every_tree_leaf(tasks, frames):
    for t, (frame, entry) in list(zip(tasks, frames))[12:]:
        pool = pool_of(frame)
        ids, _ = select("gt_regime", "balanced", pool, None, 8, demo_seed(42, t.task_id, "rule_diversity", 0), _ctx(pool))
        assert sorted(pool.loc[ids, "regime"]) == list(range(8))
        assert (pool.loc[ids, "label"].to_numpy() == np.asarray(t.leaf_labels)[pool.loc[ids, "regime"]]).all()


def test_demo_seed_separates_streams():
    a = demo_seed(42, "eval_0000", "random", 0)
    assert a == demo_seed(42, "eval_0000", "random", 0)
    assert len({a, demo_seed(42, "eval_0001", "random", 0), demo_seed(42, "eval_0000", "rule_diversity", 0),
                demo_seed(42, "eval_0000", "random", 1), demo_seed(42, "eval_0000", "random", 0, "order")}) == 5


# ---------------------------------------------------------------- grid engine
class StubRunner:
    """Deterministic fake runner: p1 is a hash of the full prompt."""

    model_path, device, device_name, dtype = "stub", "cpu", "stub", "float32"

    def __init__(self):
        self.calls = []

    def render(self, system, demo_lines, query_line):
        return completion_prompt(system, demo_lines, query_line)

    def score(self, prefix, suffixes, label_tokens=("0", "1")):
        self.calls.append((prefix, list(suffixes)))
        self.last_token_counts = [len(prefix) + len(s) for s in suffixes]
        out = []
        for s in suffixes:
            h = int(hashlib.md5((prefix + s).encode()).hexdigest()[:8], 16) / 16 ** 8
            p1 = 0.01 + 0.98 * h
            out.append(get_confidence({"0": np.log(1 - p1), "1": np.log(p1)}, label_tokens))
        return out


@pytest.fixture(scope="module")
def small_suite(tmp_path_factory):
    out = tmp_path_factory.mktemp("suite")
    write_suite(sample_eval_tasks(2, id_prefix="t"), out, suite="t")
    return out


def test_grid_runs_resumes_and_pairs(small_suite, tmp_path):
    manifest = load_manifest(small_suite)
    task_ids = [t["task_id"] for t in manifest["tasks"]]
    units = enumerate_units(["stub"], ["abstract"], task_ids,
                            ["zero_shot", "random", "counter_spurious", "similarity"],
                            ["gold", "shuffled"], [8], 2)
    assert len(units) == 2 * (1 + 3 * 2 * 2)
    runner = StubRunner()
    grid = GridRunner(runner, "stub", small_suite, manifest, base_seed=42,
                      envs=["id", "covariate", "spurious_reversal"], queries_per_env=4, run_name="test")
    out = tmp_path / "grid.parquet"
    assert grid.run(units, out, log=lambda *_: None) == len(units)
    df = pd.read_parquet(out)
    assert len(df) == len(units) * 12                     # 4 queries x 3 environments per unit
    assert completed_units(out) == {u.key for u in units}
    assert grid.run(units, out, log=lambda *_: None) == 0  # resume: nothing left to do
    assert len(pd.read_parquet(out)) == len(df)

    # Queries per environment are balanced.
    assert (df.groupby(["unit_key", "env"])["label"].apply(lambda s: (s == "1").sum()) == 2).all()
    # Zero-shot is uncalibrated; everything else carries a calibrated probability.
    assert df.loc[df.strategy == "zero_shot", "calibrated_p1"].isna().all()
    assert df.loc[df.strategy != "zero_shot", "calibrated_p1"].notna().all()
    # A query-agnostic unit shows one demonstration set to every query and environment.
    rnd = df[(df.strategy == "random") & (df.label_mode == "gold") & (df.seed == 0)]
    assert rnd.groupby("task_id")["demo_ids"].apply(lambda s: len({tuple(x) for x in s})).eq(1).all()
    # Similarity selects per query.
    sim = df[(df.strategy == "similarity") & (df.label_mode == "gold") & (df.seed == 0)]
    assert sim.groupby("task_id")["demo_ids"].apply(lambda s: len({tuple(x) for x in s})).gt(1).all()
    # Gold and shuffled units show the same rows with permuted labels.
    key = ["task_id", "strategy", "seed", "row_id"]
    gold = df[df.label_mode == "gold"].set_index(key)
    shuf = df[df.label_mode == "shuffled"].set_index(key)
    joined = gold.join(shuf, rsuffix="_s", how="inner")
    assert len(joined) > 0
    assert all(list(a) == list(b) for a, b in zip(joined["demo_ids"], joined["demo_ids_s"]))
    assert all(sorted(a) == sorted(b) for a, b in zip(joined["demo_labels"], joined["demo_labels_s"]))
    # Every calibrated prefix was also scored on the content-free query.
    assert all(any("N/A" in s for s in suffixes) for prefix, suffixes in runner.calls if "->" in prefix)


# ---------------------------------------------------------------- probe splits (hiccups/18)
PROBE_KW = dict(ood_envs=("covariate", "spurious_reversal", "covariate_scale"),
                f8_neutral_from=("id", "covariate", "covariate_scale"))


def test_probe_splits_leave_existing_rows_unchanged(tasks):
    for t in tasks[::6]:
        base, _ = build_task_frame(t)
        probed, entry = build_task_frame(t, **PROBE_KW)
        assert probed.iloc[:len(base)].reset_index(drop=True).equals(base.reset_index(drop=True))
        assert set(probed.env) - set(base.env) == {"covariate_scale", "id_f8neutral", "covariate_f8neutral",
                                                   "covariate_scale_f8neutral"}
        assert len(entry["covariate_scale"]["scale"]) == 9


def test_covariate_scale_stretches_everything_but_the_shortcut(tasks):
    for t in tasks[::4]:
        frame, entry = build_task_frame(t, **PROBE_KW)
        cs, ident = frame[frame.env == "covariate_scale"], frame[frame.split == "test_id"]
        others = [c for c in FEATURE_NAMES if c != entry["roles"]["spurious"]]
        assert (cs[others].std() > 1.3 * ident[others].std()).mean() > 0.8
        q = cs[cs.is_query]
        assert q.label.value_counts().to_dict() == {0: 10, 1: 10}
        assert 0.6 < cs.agree_obs.mean()                     # the shortcut is intact


def test_f8_neutral_copies_differ_only_in_the_shortcut(tasks):
    for t in tasks[::4]:
        frame, entry = build_task_frame(t, **PROBE_KW)
        f8 = entry["roles"]["spurious"]
        others = [c for c in FEATURE_NAMES if c != f8]
        for src in ("id", "covariate", "covariate_scale"):
            a = queries_of(frame, src).sort_values("query_id")
            b = queries_of(frame, f"{src}_f8neutral").sort_values("query_id")
            assert np.allclose(a[others].to_numpy(), b[others].to_numpy())
            assert (a.label.to_numpy() == b.label.to_numpy()).all() and (a.query_id.to_numpy() == b.query_id.to_numpy()).all()
            assert not np.allclose(a[f8].to_numpy(), b[f8].to_numpy())
