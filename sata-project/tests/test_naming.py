"""P3: namings, the prior surrogate, counter_prior and the named grid (no model needed)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data.generator import sample_eval_tasks
from src.data.naming import NAMINGS, Lexicon, assign_names, codebook, load_lexicons, name_directions
from src.data.synthetic_bridge import FEATURE_NAMES, build_task_frame, load_manifest, pool_of, write_suite
from src.experiments.synth_grid import GridRunner, enumerate_units, order_units
from src.inference.llm_runner import get_confidence
from src.inference.priors import PriorSurrogate, fit_surrogate, measure_margins, random_profiles, screen_lexicon
from src.inference.prompts import completion_prompt, system_message
from src.selection.protocols import ProtocolContext, counter_prior_active, demo_seed, prior_scores, select

LEX = {
    "loan": Lexicon("loan", (("good a", "bad a"), ("good b", "bad b"), ("good c", "bad c"), ("good d", "bad d")),
                    tuple(f"weak loan {i}" for i in range(7))),
    "medical": Lexicon("medical", (("risk a", "safe a"), ("risk b", "safe b"), ("risk c", "safe c")),
                       tuple(f"weak medical {i}" for i in range(8))),
}
RULES = {"pair_min_strength_ratio": 0.5, "weak_max_relative_slope": 0.25, "max_pairs": 5, "n_weak": 7}


@pytest.fixture(scope="module")
def frames():
    return [build_task_frame(t) for t in sample_eval_tasks(24)]


def _true_surrogate(lexicons=LEX, abstract_slope=0.5):
    """Slopes that follow the lexicon's directions exactly; weak names and f0-f9 get `abstract_slope` / 0."""
    fits = {"abstract": {"slopes": {f: abstract_slope for f in FEATURE_NAMES}}}
    for d, lex in lexicons.items():
        fits[d] = {"slopes": {n: 2.0 * lex.direction(n) for n in lex.names}}
    return PriorSurrogate("stub", fits)


# ---------------------------------------------------------------- namings
def test_candidate_and_screening_config_load():
    cand = load_lexicons("candidates")
    assert set(cand) == {"loan", "medical"}
    for lex in cand.values():
        assert len(lex.pairs) >= 3 and len(lex.weak) >= 7
        names = lex.names
        assert len(names) == len(set(names))
        assert not any(ch in n for n in names for ch in ":;,")


def test_aligned_and_flipped_differ_only_in_the_rule_features(frames):
    for _, entry in frames:
        lex = LEX[entry["domain"]]
        a, f = (assign_names(entry, n, lex, 42) for n in ("aligned", "flipped"))
        ab = assign_names(entry, "abstract", lex, 42)
        assert ab == {c: c for c in FEATURE_NAMES}
        lb = set(entry["roles"]["load_bearing"])
        assert {c for c in FEATURE_NAMES if a[c] != f[c]} == lb
        assert len(set(a.values())) == len(set(f.values())) == 10
        da, df = name_directions(a, lex), name_directions(f, lex)
        for c in FEATURE_NAMES:
            if c in lb:
                assert da[c] == entry["directions"][c] and df[c] == -entry["directions"][c]
            else:
                assert da[c] == df[c] == 0
        assert a == assign_names(entry, "aligned", lex, 42)            # deterministic


def test_named_system_messages_depend_only_on_the_domain():
    assert system_message("loan") != system_message("medical") != system_message(None)
    assert "approved" in system_message("loan") and "high risk" in system_message("medical")


def test_codebook_renames_without_reordering():
    from src.data.serialisation import serialise_row

    row = {f: float(i) for i, f in enumerate(FEATURE_NAMES)}
    names = {f: f"name {9 - i}" for i, f in enumerate(FEATURE_NAMES)}
    line = serialise_row(row, codebook=codebook(names))
    assert line.startswith("name 9: 0.00; name 8: 1.00;") and line.endswith("name 0: 9.00 ->")


# ---------------------------------------------------------------- surrogate
def test_surrogate_recovers_known_slopes():
    rng = np.random.default_rng(0)
    lex = LEX["loan"]
    truth = {n: 1.5 * lex.direction(n) for n in lex.names}
    truth["good d"], truth["bad d"] = 2.0, -0.4        # an unbalanced pair
    truth["weak loan 0"] = 1.0                          # a weak name with a strong prior
    prof = random_profiles(800, lex, rng)
    prof["margin"] = [0.7 + sum(truth[n] * v for n, v in zip(ns, vs)) + rng.normal(0, 0.5)
                      for ns, vs in zip(prof["names"], prof["values"])]
    prof["logprob_0"] = -np.logaddexp(0, prof["margin"])
    prof["logprob_1"] = -np.logaddexp(0, -prof["margin"])
    fit = fit_surrogate(prof, lex.names, n_bootstrap=100)
    assert fit["r2_cv"] > 0.9
    assert fit["intercept"] == pytest.approx(0.7, abs=0.1)
    for n in lex.names:
        assert fit["slopes"][n] == pytest.approx(truth[n], abs=0.15)
    sc = screen_lexicon(fit, lex, RULES)
    verdict = {tuple(p["pair"]): p["admitted"] for p in sc["pairs"]}
    assert verdict == {("good a", "bad a"): True, ("good b", "bad b"): True, ("good c", "bad c"): True,
                       ("good d", "bad d"): False}
    assert "weak loan 0" not in sc["final"]["weak"] and len(sc["final"]["weak"]) == 6
    assert not sc["enough"]                              # only 6 usable weak names


def test_profiles_mimic_a_named_prompt():
    prof = random_profiles(50, LEX["loan"], np.random.default_rng(1))
    for names in prof["names"]:
        dirs = [LEX["loan"].direction(n) for n in names]
        assert len(set(names)) == 10 and sum(d != 0 for d in dirs) == 3
        members = {n for n in names if LEX["loan"].direction(n) != 0}
        assert all(not ({p, q} <= members) for p, q in LEX["loan"].pairs)   # one member per pair


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


def test_measure_margins_shares_one_prefix():
    runner = StubRunner()
    prof = random_profiles(7, LEX["loan"], np.random.default_rng(2))
    out = measure_margins(runner, system_message("loan"), prof, chunk=3)
    assert len(out) == 7 and len({p for p, _ in runner.calls}) == 1
    assert out["margin"].to_numpy() == pytest.approx((out["logprob_1"] - out["logprob_0"]).to_numpy())


# ---------------------------------------------------------------- counter_prior
def _ctx(pool, slopes):
    return ProtocolContext(feature_cols=FEATURE_NAMES, kinds={f: "numeric" for f in FEATURE_NAMES},
                           train_ref=pool, prior_slopes=slopes)


def test_prior_data_conflict_separates_aligned_from_flipped(frames):
    sur = _true_surrogate()
    for frame, entry in frames:
        pool = pool_of(frame)
        conflicts = {}
        for naming in ("aligned", "flipped"):
            names = assign_names(entry, naming, LEX[entry["domain"]], 42)
            conflicts[naming] = counter_prior_active(pool, _ctx(pool, sur.column_slopes(naming, entry["domain"], names)))
        assert not conflicts["aligned"][0] and conflicts["aligned"][1] < 0.3
        assert conflicts["flipped"][0] and conflicts["flipped"][1] > 0.7


def test_counter_prior_shows_rows_that_contradict_the_prior(frames):
    sur = _true_surrogate()
    for frame, entry in frames[::3]:
        pool = pool_of(frame)
        names = assign_names(entry, "flipped", LEX[entry["domain"]], 42)
        ctx = _ctx(pool, sur.column_slopes("flipped", entry["domain"], names))
        ids, meta = select("counter_prior", "balanced", pool, None, 8,
                           demo_seed(42, entry["task_id"], "counter_prior", 0), ctx)
        rows = pool.loc[ids]
        assert int(rows["label"].sum()) == 4
        assert ((prior_scores(rows, ctx) > 0) != (rows["label"] == 1)).all()
        assert (rows["label"] == rows["y_clean"]).all()
        assert meta["conflict"] > 0.5


# ---------------------------------------------------------------- named grid
@pytest.fixture(scope="module")
def small_suite(tmp_path_factory):
    out = tmp_path_factory.mktemp("suite")
    write_suite(sample_eval_tasks(2, id_prefix="t"), out, suite="t")
    return out


def test_named_grid_pairs_the_namings(small_suite, tmp_path):
    manifest = load_manifest(small_suite)
    task_ids = [t["task_id"] for t in manifest["tasks"]]
    units = enumerate_units(["stub"], list(NAMINGS), task_ids,
                            ["zero_shot", "label_diversity", "counter_prior"], ["gold"], [8], 2)
    assert [u.seed for u in order_units(units, "seed")] == sorted(u.seed for u in units)
    runner = StubRunner()
    grid = GridRunner(runner, "stub", small_suite, manifest, base_seed=42, envs=["id", "covariate"],
                      queries_per_env=4, run_name="test", lexicons=LEX, surrogate=_true_surrogate())
    out = tmp_path / "grid.parquet"
    grid.run(units, out, log=lambda *_: None)
    df = pd.read_parquet(out)
    assert set(df["naming"]) == set(NAMINGS)

    # Naming-blind strategies show the same rows under every naming.
    ld = df[df.strategy == "label_diversity"]
    assert ld.groupby(["task_id", "seed", "row_id"])["demo_ids"].apply(lambda s: len({tuple(x) for x in s})).eq(1).all()
    # Aligned and flipped prompts differ only in the load-bearing names.
    entry = manifest["tasks"][0]
    unit = next(u for u in units if u.task_id == entry["task_id"] and u.strategy == "label_diversity" and u.seed == 0)
    texts = {}
    for naming in ("aligned", "flipped"):
        g = grid.prompt_groups(unit.__class__(**{**unit.__dict__, "naming": naming}))[0]
        texts[naming] = g["fulls"][0]
        names = grid.names(entry["task_id"], naming)
        for c in entry["roles"]["load_bearing"]:
            texts[naming] = texts[naming].replace(names[c] + ":", f"<{c}>:")
    assert texts["aligned"] == texts["flipped"]

    # counter_prior: inactive under aligned names (it is then label_diversity's set), active under flipped.
    cp = df[df.strategy == "counter_prior"]
    meta = cp.drop_duplicates(["naming", "task_id", "seed"]).set_index(["naming", "task_id", "seed"])["selection_meta"]
    active = meta.map(lambda m: json.loads(m)["counter_prior_active"])
    assert not active.loc["aligned"].any() and active.loc["flipped"].all()
    key = ["naming", "task_id", "seed", "row_id"]
    joined = cp.set_index(key).join(ld.set_index(key), rsuffix="_ld")
    same = joined.apply(lambda r: list(r["demo_ids"]) == list(r["demo_ids_ld"]), axis=1)
    assert same.loc["aligned"].all() and not same.loc["flipped"].any()
    assert (joined.loc["aligned", "p1"] == joined.loc["aligned", "p1_ld"]).all()


# ---------------------------------------------------------------- RQ3
def test_parse_ranking_matches_display_names():
    from src.experiments.rq3 import parse_ranking

    names = {"f0": "credit score", "f1": "branch number", "f2": "LDL cholesterol"}
    ranking, n = parse_ranking("1. LDL Cholesterol, credit score., unknown thing, credit score", names)
    assert ranking == ["f2", "f0", "f1"] and n == 2
    assert parse_ranking("f2, f0", {c: c for c in ("f0", "f1", "f2")})[0] == ["f2", "f0", "f1"]


def test_rq3_unit_reuses_the_grid_demo_set(small_suite, tmp_path):
    from src.experiments.rq3 import RQ3Runner, enumerate_rq3_units

    class RankingStub(StubRunner):
        def generate_text(self, prompts, max_new_tokens=128):
            return ["f3, f1, f0" for _ in prompts]

    manifest = load_manifest(small_suite)
    task_ids = [t["task_id"] for t in manifest["tasks"]]
    grid = GridRunner(RankingStub(), "stub", small_suite, manifest, base_seed=42, envs=["id"],
                      queries_per_env=4, lexicons=LEX)
    units = enumerate_rq3_units(["stub"], ["abstract", "flipped"], task_ids, 1)
    out = tmp_path / "rq3.parquet"
    assert RQ3Runner(grid, queries_per_env=4).run(units, out, log=lambda *_: None) == len(units)
    df = pd.read_parquet(out)
    # 4 queries x (1 original + 10 hot-deck + 8 nudges) per unit
    assert len(df) == len(units) * 4 * 19
    assert (df.groupby(["unit_key", "row_id", "variant"]).size().unstack()[["orig", "hotdeck", "nudge"]]
            .eq([1, 10, 8]).all().all())
    # The same donors under every naming, and the grid's label_diversity rows.
    hd = df[df.variant == "hotdeck"].pivot_table(index=["task_id", "row_id", "feature"], columns="naming", values="delta")
    assert np.allclose(hd["abstract"], hd["flipped"])
    grid_set = grid._demo_set(units[0].grid_unit(), pool_of(grid.frame(units[0].task_id)), None)
    assert list(df[df.unit_key == units[0].key]["demo_ids"].iloc[0]) == grid_set["ids"]
    nd = df[df.variant == "nudge"]
    assert set(nd["feature_role"]) == {"load_bearing", "spurious"} and set(nd["delta"]) == {0.5, -0.5}
    flipped_lb = nd[(nd.naming == "flipped") & (nd.feature_role == "load_bearing")]
    assert (flipped_lb["name_direction"] == -flipped_lb["data_direction"]).all()
    ranks = pd.read_parquet(tmp_path / "rq3_rankings.parquet")
    assert len(ranks) == len(units)
    assert list(ranks[ranks.naming == "abstract"]["ranking"].iloc[0][:3]) == ["f3", "f1", "f0"]
    assert RQ3Runner(grid, queries_per_env=4).run(units, out, log=lambda *_: None) == 0


# ---------------------------------------------------------------- P4 analysis
def test_contrast_family_runs_on_a_stub_grid(small_suite, tmp_path):
    from src.evaluation.analysis import (CONTRASTS, cell_metrics, correctness_rho, dfi, prior_agreement,
                                         run_contrasts, self_report_rho, summary_table, true_importance_by_column)
    from src.experiments.rq3 import RQ3Runner, enumerate_rq3_units

    class RankingStub(StubRunner):
        def generate_text(self, prompts, max_new_tokens=128):
            return ["f3, f1, f0" for _ in prompts]

    manifest = load_manifest(small_suite)
    task_ids = [t["task_id"] for t in manifest["tasks"]]
    strategies = ["zero_shot", "random", "label_diversity", "feature_range", "counter_spurious",
                  "counter_prior", "similarity"]
    units = enumerate_units(["stub"], list(NAMINGS), task_ids, strategies, ["gold"], [8], 2)
    grid = GridRunner(RankingStub(), "stub", small_suite, manifest, base_seed=42,
                      envs=["id", "covariate", "spurious_reversal"], queries_per_env=4, lexicons=LEX,
                      surrogate=_true_surrogate())
    grid.run(units, tmp_path / "grid.parquet", log=lambda *_: None)
    RQ3Runner(grid, queries_per_env=4).run(enumerate_rq3_units(["stub"], ["abstract", "flipped"], task_ids, 2),
                                           tmp_path / "rq3.parquet", log=lambda *_: None)
    df = pd.read_parquet(tmp_path / "grid.parquet")
    rq3 = pd.read_parquet(tmp_path / "rq3.parquet")
    cells = cell_metrics(df)
    assert len(cells) == len(units) * 3
    assert summary_table(cells).shape[0] == len(strategies)
    pa = prior_agreement(df)
    assert pa["agree_raw"].between(0, 1).all()
    truth = true_importance_by_column(manifest)
    assert all(sum(v > 0 for v in t.values()) == 3 for t in truth.values())
    res = run_contrasts(df, rq3, truth, n_bootstrap=50)
    assert res["id"].tolist() == [c["id"] for c in CONTRASTS]
    assert res["p_holm"].between(0, 1).all() and (res["p_holm"] >= res["p_one_sided"] - 1e-12).all()
    assert dfi(rq3)["follows_data"].between(0, 1).all()
    assert len(correctness_rho(rq3, truth)) == 2 * 2 * 2
    ranks = pd.read_parquet(tmp_path / "rq3_rankings.parquet")
    assert len(self_report_rho(ranks, rq3)) == len(ranks)


def test_counter_prior_matched_keeps_the_shortcut_fixed(small_suite, tmp_path):
    manifest = load_manifest(small_suite)
    task_ids = [t["task_id"] for t in manifest["tasks"]]
    units = enumerate_units(["stub"], ["abstract", "flipped"], task_ids,
                            ["label_diversity", "counter_prior", "counter_prior_matched"], ["gold"], [8], 3)
    grid = GridRunner(StubRunner(), "stub", small_suite, manifest, base_seed=42, envs=["id"],
                      queries_per_env=4, lexicons=LEX, surrogate=_true_surrogate())
    grid.run(units, tmp_path / "g.parquet", log=lambda *_: None)
    df = pd.read_parquet(tmp_path / "g.parquet").drop_duplicates("unit_key")
    for (naming, task, seed), g in df.groupby(["naming", "task_id", "seed"]):
        pool = pool_of(grid.frame(task))
        sets = {r.strategy: list(r.demo_ids) for r in g.itertuples()}
        cell = lambda ids: sorted(zip(pool.loc[ids, "label"], pool.loc[ids, "agree_obs"]))
        assert cell(sets["counter_prior_matched"]) == cell(sets["label_diversity"])   # same label x f8-agreement counts
        meta = json.loads(g.set_index("strategy").loc["counter_prior_matched", "selection_meta"])
        if meta["counter_prior_active"]:
            plain = json.loads(g.set_index("strategy").loc["counter_prior", "selection_meta"])
            assert plain["counter_prior_active"] and meta["matched"]
        else:
            assert sets["counter_prior_matched"] == sets["label_diversity"]


def test_probe_environments_run_and_pair(tmp_path):
    from src.evaluation.analysis import cross_environment_auroc, exploratory_contrasts

    out = tmp_path / "suite"
    write_suite(sample_eval_tasks(2, id_prefix="p"), out, suite="p",
                ood_envs=("covariate", "spurious_reversal", "covariate_scale"),
                f8_neutral_from=("id", "covariate", "covariate_scale"))
    manifest = load_manifest(out)
    task_ids = [t["task_id"] for t in manifest["tasks"]]
    envs = ["id", "covariate", "covariate_scale", "id_f8neutral", "covariate_f8neutral", "covariate_scale_f8neutral"]
    units = enumerate_units(["stub"], ["abstract"], task_ids, ["random"], ["gold"], [8], 2)
    grid = GridRunner(StubRunner(), "stub", out, manifest, base_seed=42, envs=envs, queries_per_env=4)
    grid.run(units, tmp_path / "g.parquet", log=lambda *_: None)
    df = pd.read_parquet(tmp_path / "g.parquet")
    assert set(df.env) == set(envs) and len(df) == len(units) * 4 * len(envs)
    # f8-neutral queries are the source queries with the same ids
    a = df[df.env == "covariate"].sort_values(["unit_key", "query_id"])
    b = df[df.env == "covariate_f8neutral"].sort_values(["unit_key", "query_id"])
    assert (a.query_id.to_numpy() == b.query_id.to_numpy()).all() and (a.label.to_numpy() == b.label.to_numpy()).all()
    r = exploratory_contrasts(df, strategies=("random",), namings=("abstract",), n_bootstrap=20)
    assert set(r.id) == {"E1", "E2", "E3a", "E3b", "E3c"} and (r.n_tasks == 2).all()
    ce = cross_environment_auroc(df)
    assert {"id_pos_vs_covariate_neg", "score_shift"} <= set(ce.columns)
