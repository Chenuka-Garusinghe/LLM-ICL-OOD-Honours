"""RQ3 behavioural measurements (generator_spec.pdf, faithfulness and reliance).

A unit is (model, naming, task, seed). It uses the grid's label_diversity
demonstration set for that task and seed (the same rows and order as in the
grid) and the task's ID queries, and scores against the one cached prefix:

- `orig`: every query as it is;
- `hotdeck`: each of the 10 columns of each query replaced by a donor value
  (kNN hot-deck from the pool on the other 9 columns, `hot_deck_impute_feature`).
  Donors depend only on the task and the column, so every naming and seed sees
  the same perturbed queries. The drop in accuracy (or in the correct-label
  margin) gives pi_behav;
- `nudge`: each load-bearing column and the spurious column moved by +/- delta
  (0.5 z-units). g_j = m(x + delta e_j) - m(x - delta e_j) gives the
  directional sensitivity: it follows the data if sign g_j = s_j and the name
  if sign g_j equals the name's direction (demo-following index, DFI);
- the content-free query, for calibrated predictions.

After scoring, the model is asked for its own ranking of the columns
(pi_self): the same demonstrations, then a ranking request instead of a query,
answered greedily and parsed against the display names.
"""

from __future__ import annotations

import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.naming import name_directions
from src.data.serialisation import serialise_row
from src.data.synthetic_bridge import FEATURE_NAMES, LABEL_TOKENS, pool_of, queries_of
from src.evaluation.faithfulness import hot_deck_impute_feature
from src.experiments.synth_grid import GridRunner, Unit, _features, _split, take_queries
from src.inference.calibration import calibrate, content_free_features
from src.inference.prompts import ranking_request, ranking_system_message, system_message
from src.utils.results_schema import append_results

NUDGE_DELTA = 0.5
RQ3_STRATEGY = "label_diversity"


@dataclass(frozen=True)
class RQ3Unit:
    model: str
    naming: str
    task_id: str
    seed: int
    k: int = 8

    @property
    def key(self) -> str:
        return f"rq3|{self.model}|{self.naming}|{self.task_id}|k{self.k}|s{self.seed}"

    def grid_unit(self) -> Unit:
        return Unit(self.model, self.naming, self.task_id, RQ3_STRATEGY, "gold", self.k, self.seed)


def enumerate_rq3_units(models, namings, task_ids, n_seeds, k=8) -> list[RQ3Unit]:
    """Seed-major: every (model, naming, task) of seed 0 first."""
    return [RQ3Unit(m, n, t, s, k) for s in range(n_seeds) for m in models for n in namings for t in task_ids]


def hotdeck_seed(base_seed: int, task_id: str, column: str) -> int:
    ss = np.random.SeedSequence([base_seed, zlib.crc32(task_id.encode()), zlib.crc32(b"hotdeck"),
                                 zlib.crc32(column.encode())])
    return int(ss.generate_state(1)[0])


def _normalise(text: str) -> str:
    """Lower case, list markers and quotes removed, underscores and hyphens read as spaces."""
    text = text.lower().strip()
    text = re.sub(r"^[\s\-\*\d\.\)]+", "", text)      # list markers such as "1." or "- "
    text = re.sub(r"[_\-]+", " ", text)                    # "waist_circumference", "on-time"
    return re.sub(r"\s+", " ", text).strip(" .\"'`*")


def parse_ranking(response: str, names: dict[str, str]) -> tuple[list[str], int]:
    """Columns in the order the model listed their display names, then the
    unlisted columns in displayed order; also returns how many it listed.

    A token matches a name after normalisation, or failing that the closest
    name with a similarity of at least 0.85 (difflib), which absorbs small
    variants such as "hours of submission" for "hour of submission"."""
    import difflib

    lookup = {_normalise(n): c for c, n in names.items()}
    ranked = []
    for token in re.split(r"[,\n;]", response):
        key = _normalise(token)
        col = lookup.get(key)
        if col is None and key:
            close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=0.85)
            col = lookup[close[0]] if close else None
        if col is not None and col not in ranked:
            ranked.append(col)
    n_listed = len(ranked)
    return ranked + [c for c in names if c not in ranked], n_listed


class RQ3Runner:
    def __init__(self, grid: GridRunner, delta: float = NUDGE_DELTA, n_neighbours: int = 5,
                 queries_per_env: int | None = None, rank: bool = True, max_new_tokens: int = 160):
        self.grid = grid
        self.delta = delta
        self.n_neighbours = n_neighbours
        self.queries_per_env = queries_per_env
        self.rank = rank
        self.max_new_tokens = max_new_tokens

    def _variants(self, unit: RQ3Unit, pool: pd.DataFrame, queries: pd.DataFrame, entry: dict) -> list[dict]:
        roles = entry["roles"]
        role = {c: "distractor" for c in FEATURE_NAMES}
        role.update({c: "load_bearing" for c in roles["load_bearing"]})
        role[roles["spurious"]], role[roles["noise"]] = "spurious", "noise"
        data_dir = {c: 0 for c in FEATURE_NAMES}
        data_dir.update({c: int(v) for c, v in entry["directions"].items()})
        data_dir[roles["spurious"]] = int(entry["spurious_direction"])
        donors = {c: hot_deck_impute_feature(queries, c, pool, FEATURE_NAMES, self.n_neighbours,
                                             seed=hotdeck_seed(self.grid.base_seed, unit.task_id, c))
                  for c in FEATURE_NAMES}
        nudged = list(roles["load_bearing"]) + [roles["spurious"]]
        out = []
        for rid, q in queries.iterrows():
            base = _features(q)
            common = {"query_id": int(q["query_id"]), "row_id": int(rid), "label": int(q["label"]),
                      "y_clean": int(q["y_clean"])}
            out.append({**common, "variant": "orig", "feature": None, "delta": 0.0, "features": base})
            for c in FEATURE_NAMES:
                out.append({**common, "variant": "hotdeck", "feature": c, "delta": float(donors[c].loc[rid, c] - q[c]),
                            "features": {**base, c: float(donors[c].loc[rid, c])}})
            for c in nudged:
                for sign in (1, -1):
                    out.append({**common, "variant": "nudge", "feature": c, "delta": sign * self.delta,
                                "features": {**base, c: float(q[c]) + sign * self.delta}})
        for v in out:
            if v["feature"] is not None:
                v["feature_role"], v["data_direction"] = role[v["feature"]], data_dir[v["feature"]]
            else:
                v["feature_role"], v["data_direction"] = None, 0
        return out

    def run_unit(self, unit: RQ3Unit) -> tuple[pd.DataFrame, dict | None]:
        t0 = time.perf_counter()
        grid, entry = self.grid, self.grid.entries[unit.task_id]
        frame = grid.frame(unit.task_id)
        pool = pool_of(frame)
        queries = take_queries(queries_of(frame, "id"), self.queries_per_env)
        names = grid.names(unit.task_id, unit.naming)
        cb = {c: {"name_extended": n} for c, n in names.items()}
        lex = grid.lexicons.get(entry["domain"]) if unit.naming != "abstract" else None
        name_dir = name_directions(names, lex)
        domain = None if unit.naming == "abstract" else entry["domain"]
        demo = grid._demo_set(unit.grid_unit(), pool, None, cb)
        system = system_message(domain)
        variants = self._variants(unit, pool, queries, entry)
        lines = [serialise_row(v["features"], codebook=cb) for v in variants]
        cf_line = serialise_row(content_free_features(_features(queries.iloc[0])), codebook=cb)
        prefix, _ = _split(grid.runner.render(system, demo["lines"], lines[0]), lines[0])
        suffixes = []
        for line in lines + [cf_line]:
            p, s = _split(grid.runner.render(system, demo["lines"], line), line)
            assert p == prefix
            suffixes.append(s)
        results = grid.runner.score(prefix, suffixes, LABEL_TOKENS)
        cf = results[-1]
        rows = []
        for v, r in zip(variants, results[:-1]):
            cal_p1 = calibrate(r.logprob_0, r.logprob_1, cf.logprob_0, cf.logprob_1)[1]
            rows.append({
                "unit_key": unit.key, "model": unit.model, "naming": unit.naming, "domain": entry["domain"],
                "task_id": unit.task_id, "family": entry["family"], "strategy": RQ3_STRATEGY, "k": unit.k,
                "seed": unit.seed, "query_id": v["query_id"], "row_id": v["row_id"], "label": v["label"],
                "y_clean": v["y_clean"], "variant": v["variant"], "feature": v["feature"],
                "feature_role": v["feature_role"], "data_direction": v["data_direction"],
                "name_direction": None if v["feature"] is None else name_dir[v["feature"]],
                "display_name": None if v["feature"] is None else names[v["feature"]],
                "delta": v["delta"], "logprob_0": r.logprob_0, "logprob_1": r.logprob_1,
                "margin": r.logprob_1 - r.logprob_0, "margin_cf": cf.logprob_1 - cf.logprob_0,
                "calibrated_p1": cal_p1, "prediction_cal": int(cal_p1 > 0.5),
                "demo_ids": demo["ids"], "unit_seconds": None,
            })
        rank_row = None
        if self.rank:
            prompt = grid.runner.render(ranking_system_message(domain), demo["lines"], ranking_request())
            response = grid.runner.generate_text([prompt], max_new_tokens=self.max_new_tokens)[0]
            ranking, n_listed = parse_ranking(response, names)
            rank_row = {"unit_key": unit.key, "model": unit.model, "naming": unit.naming, "domain": entry["domain"],
                        "task_id": unit.task_id, "family": entry["family"], "seed": unit.seed,
                        "response": response, "ranking": ranking, "n_listed": n_listed,
                        "display_names": [names[c] for c in FEATURE_NAMES]}
        out = pd.DataFrame(rows)
        out["unit_seconds"] = time.perf_counter() - t0
        return out, rank_row

    def run(self, units: list[RQ3Unit], out_path: str | Path, log=print) -> int:
        out_path = Path(out_path)
        rank_path = out_path.with_name(out_path.stem + "_rankings.parquet")
        done, ranked = set(), set()
        if out_path.exists():
            done = set(pd.read_parquet(out_path, columns=["unit_key"])["unit_key"].unique())
        if rank_path.exists():
            ranked = set(pd.read_parquet(rank_path, columns=["unit_key"])["unit_key"].unique())
        todo = [u for u in units if u.key not in done and u.model == self.grid.model_name]
        log(f"{self.grid.model_name}: {len(todo)} RQ3 units to run ({len(done)} already saved)")
        t_start = time.perf_counter()
        for i, unit in enumerate(todo, 1):
            rows, rank_row = self.run_unit(unit)
            if rank_row is not None and unit.key not in ranked:   # a unit counts as done once its scores exist
                append_results(pd.DataFrame([rank_row]), rank_path)
            append_results(rows, out_path)
            elapsed = time.perf_counter() - t_start
            log(f"  [{i}/{len(todo)}] {unit.key}: {len(rows)} rows, {rows['unit_seconds'].iloc[0]:.1f}s "
                f"(eta {elapsed / i * (len(todo) - i) / 60:.1f} min)")
        return len(todo)
