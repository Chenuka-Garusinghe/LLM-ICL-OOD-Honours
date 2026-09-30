"""The synthetic grid: resumable units scored by one runner.

A unit is (model, naming, task, strategy, label mode, k, seed). For a
query-agnostic strategy it holds one demonstration set, drawn once from the
task's pool and reused for every query of the task in every environment;
for a query-conditional strategy (feature kNN) each query gets its own set.
Demonstration sets depend on (task, strategy, seed, k) only, never on the
model, naming or label mode, so those comparisons are paired
(generator_spec.pdf, seeding, reuse and pairing).

Each set becomes one prompt prefix, and every query of the unit, plus the
content-free query used for calibration, is scored against it through the
runner's prefix cache. Zero-shot units score the queries alone and are not
calibrated. A unit's rows are appended to the results parquet once the unit
finishes, so an interrupted run resumes by skipping finished units.

`label_mode="shuffled"` permutes the shown labels within the demonstration
set (same rows, same label counts, input-label mapping broken): the P2
learnability control and Min et al.'s random-label condition.

Namings (src/data/naming.py) change only the display names: abstract keeps
f0 to f9, aligned and flipped use the P3 lexicons. counter_prior reads the
loaded model's prior surrogate (src/inference/priors.py), so its sets depend
on the model and the naming; where a task's prior-data conflict is at most
1/2 it shows label_diversity's set instead.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.naming import NAMINGS, Lexicon, assign_names, codebook
from src.data.serialisation import serialise_row
from src.data.synthetic_bridge import FEATURE_NAMES, LABEL_TOKENS, load_task_frame, pool_of, queries_of
from src.inference.calibration import calibrate, content_free_features
from src.inference.prompts import system_message
from src.selection.ordering import shuffle_order
from src.selection.protocols import (
    COUNTER_PRIOR_FALLBACK,
    QUERY_CONDITIONAL,
    SYNTHETIC_STRATEGIES,
    ProtocolContext,
    counter_prior_active,
    demo_seed,
    select,
)
from src.utils.results_schema import RESULTS_COLUMNS_V3, append_results

STRATEGIES = ("zero_shot", *SYNTHETIC_STRATEGIES)
LABEL_MODES = ("gold", "shuffled")
UNIT_ORDERS = ("grid", "seed")


@dataclass(frozen=True)
class Unit:
    model: str
    naming: str
    task_id: str
    strategy: str
    label_mode: str
    k: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model}|{self.naming}|{self.task_id}|{self.strategy}|{self.label_mode}|k{self.k}|s{self.seed}"


def enumerate_units(
    models: list[str],
    namings: list[str],
    task_ids: list[str],
    strategies: list[str],
    label_modes: list[str],
    ks: list[int],
    n_seeds: int,
) -> list[Unit]:
    """Every unit of the grid; zero-shot has one unit per (model, naming, task)."""
    for name in namings:
        if name not in NAMINGS:
            raise ValueError(f"unknown naming {name!r}; expected one of {NAMINGS}")
    for s in strategies:
        if s not in STRATEGIES:
            raise ValueError(f"unknown strategy {s!r}; expected one of {STRATEGIES}")
    units = []
    for model in models:
        for naming in namings:
            for task_id in task_ids:
                for strategy in strategies:
                    if strategy == "zero_shot":
                        units.append(Unit(model, naming, task_id, "zero_shot", "gold", 0, 0))
                        continue
                    for label_mode in label_modes:
                        for k in ks:
                            for seed in range(n_seeds):
                                units.append(Unit(model, naming, task_id, strategy, label_mode, k, seed))
    return units


def order_units(units: list[Unit], order: str = "grid") -> list[Unit]:
    """`grid`: as enumerated; `seed`: every unit of seed 0 first, then seed 1, and so
    on, so an interrupted run already holds complete grids for the early seeds."""
    if order == "grid":
        return list(units)
    if order == "seed":
        return sorted(units, key=lambda u: u.seed)          # stable: the grid order within a seed
    raise ValueError(f"unknown order {order!r}; expected one of {UNIT_ORDERS}")


def completed_units(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["unit_key"])["unit_key"].unique())


def take_queries(queries: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """The first n queries of an environment, n/2 per class (all if n is None)."""
    if n is None:
        return queries
    per_class = n // 2
    parts = [queries[queries["label"] == c].sort_values("query_id").head(per_class) for c in (0, 1)]
    return pd.concat(parts).sort_values("query_id")


def _features(row: pd.Series) -> dict[str, Any]:
    return {f: row[f] for f in FEATURE_NAMES}


def _split(full: str, query_line: str) -> tuple[str, str]:
    """Split a rendered prompt at the start of its query line."""
    idx = full.rfind(query_line)
    if idx <= 0:
        raise ValueError("query line not found in the rendered prompt")
    return full[:idx], full[idx:]


class GridRunner:
    """Runs units of one model against one task suite."""

    def __init__(
        self,
        runner,
        model_name: str,
        suite_dir: str | Path,
        manifest: dict,
        base_seed: int,
        envs: list[str],
        queries_per_env: int | None = None,
        run_name: str = "",
        lexicons: dict[str, Lexicon] | None = None,
        surrogate=None,
        pool_prior=None,
    ):
        self.runner = runner
        self.model_name = model_name
        self.suite_dir = Path(suite_dir)
        self.entries = {e["task_id"]: e for e in manifest["tasks"]}
        self.base_seed = base_seed
        self.envs = list(envs)
        self.queries_per_env = queries_per_env
        self.run_name = run_name
        self.lexicons = lexicons or {}
        self.surrogate = surrogate            # PriorSurrogate of this model (counter_prior fallback)
        self.pool_prior = pool_prior          # PoolPrior of this model: measured zero-shot margins (preferred)
        self._frames: dict[str, pd.DataFrame] = {}
        self._names: dict[tuple[str, str], dict[str, str]] = {}

    def frame(self, task_id: str) -> pd.DataFrame:
        if task_id not in self._frames:
            self._frames[task_id] = load_task_frame(self.suite_dir, task_id)
        return self._frames[task_id]

    def names(self, task_id: str, naming: str) -> dict[str, str]:
        """Display name of every column of a task under a naming."""
        key = (task_id, naming)
        if key not in self._names:
            entry = self.entries[task_id]
            self._names[key] = assign_names(entry, naming, self.lexicons.get(entry["domain"]), self.base_seed)
        return self._names[key]

    def context(self, unit: Unit, pool: pd.DataFrame) -> ProtocolContext:
        """The selection context; counter_prior adds the model's prior slopes for the unit's naming."""
        ctx = ProtocolContext(feature_cols=FEATURE_NAMES, kinds={f: "numeric" for f in FEATURE_NAMES}, train_ref=pool)
        if unit.strategy == "counter_prior":
            if self.pool_prior is not None:
                ctx.prior_margin = self.pool_prior.centred(unit.task_id, unit.naming)
            elif self.surrogate is not None:
                entry = self.entries[unit.task_id]
                ctx.prior_slopes = self.surrogate.column_slopes(unit.naming, entry["domain"],
                                                                self.names(unit.task_id, unit.naming))
            else:
                raise ValueError("counter_prior needs the model's measured pool prior or its prior surrogate")
        return ctx

    # -------------------------------------------------------------- #
    def _demo_set(self, unit: Unit, pool: pd.DataFrame, query: pd.Series | None, cb: dict | None = None) -> dict:
        """Select, order and label one demonstration set."""
        extra = () if query is None else (int(query["row_id"]),)
        ctx = self.context(unit, pool)
        strategy, note = unit.strategy, {}
        if strategy == "counter_prior":
            active, conflict = counter_prior_active(pool, ctx)
            note = {"counter_prior_active": active, "prior_data_conflict": conflict}
            if not active:
                strategy = COUNTER_PRIOR_FALLBACK     # the same rows and order as that strategy
                note["fallback"] = strategy
        mechanism, composition = SYNTHETIC_STRATEGIES[strategy]
        ids, meta = select(
            mechanism, composition, pool, query, unit.k,
            demo_seed(self.base_seed, unit.task_id, strategy, unit.seed, "select", *extra), ctx,
        )
        meta.update(note)
        order_seed = demo_seed(self.base_seed, unit.task_id, strategy, unit.seed, "order", *extra)
        ordered = [int(i) for i in shuffle_order(ids, seed=order_seed)]
        labels = pool.loc[ordered, "label"].astype(int).to_numpy()
        if unit.label_mode == "shuffled":
            rng = np.random.default_rng(demo_seed(self.base_seed, unit.task_id, strategy, unit.seed, "shuffle", *extra))
            labels = rng.permutation(labels)
        elif unit.label_mode != "gold":
            raise ValueError(f"unknown label mode {unit.label_mode!r}")
        lines = [serialise_row(_features(pool.loc[i]), label=str(int(lab)), codebook=cb)
                 for i, lab in zip(ordered, labels)]
        return {"ids": ordered, "labels": [int(x) for x in labels], "lines": lines,
                "order_seed": order_seed, "meta": meta}

    def prompt_groups(self, unit: Unit) -> list[dict]:
        """The unit's prompts, grouped by shared prefix.

        Each group holds one demonstration set, its queries, the shared prompt
        prefix and one suffix per query, followed by the content-free suffix
        when the group is calibrated (every strategy except zero-shot).
        """
        entry = self.entries[unit.task_id]
        frame = self.frame(unit.task_id)
        pool = pool_of(frame)
        queries = pd.concat([take_queries(queries_of(frame, env), self.queries_per_env) for env in self.envs])
        system = system_message(None if unit.naming == "abstract" else entry["domain"])
        cb = codebook(self.names(unit.task_id, unit.naming))
        cf_line = serialise_row(content_free_features(_features(queries.iloc[0])), codebook=cb)

        if unit.strategy == "zero_shot":
            sets = [({"ids": [], "labels": [], "lines": [], "order_seed": None, "meta": {}}, queries, False)]
        elif SYNTHETIC_STRATEGIES[unit.strategy][0] in QUERY_CONDITIONAL:
            sets = [(self._demo_set(unit, pool, q, cb), queries.loc[[rid]], True) for rid, q in queries.iterrows()]
        else:
            sets = [(self._demo_set(unit, pool, None, cb), queries, True)]

        groups = []
        for demo, qs, calibrated in sets:
            lines = [serialise_row(_features(q), codebook=cb) for _, q in qs.iterrows()]
            fulls = [self.runner.render(system, demo["lines"], line) for line in lines]
            prefix, _ = _split(fulls[0], lines[0])
            suffixes = []
            for full, line in zip(fulls, lines):
                p, s = _split(full, line)
                if p != prefix:
                    raise AssertionError("queries sharing a demonstration set must share the prompt prefix")
                suffixes.append(s)
            if calibrated:
                p, s = _split(self.runner.render(system, demo["lines"], cf_line), cf_line)
                assert p == prefix
                suffixes.append(s)
            groups.append({"demo": demo, "queries": qs, "fulls": fulls, "prefix": prefix,
                           "suffixes": suffixes, "calibrated": calibrated})
        return groups

    def run_unit(self, unit: Unit) -> pd.DataFrame:
        t0 = time.perf_counter()
        entry = self.entries[unit.task_id]
        rows = []
        for group in self.prompt_groups(unit):
            demo, qs, fulls, calibrated = group["demo"], group["queries"], group["fulls"], group["calibrated"]
            results = self.runner.score(group["prefix"], group["suffixes"], LABEL_TOKENS)
            n_tokens = list(getattr(self.runner, "last_token_counts", [None] * len(group["suffixes"])))
            cf = results[-1] if calibrated else None
            for i, ((_, q), full) in enumerate(zip(qs.iterrows(), fulls)):
                r = results[i]
                cal_p1 = calibrate(r.logprob_0, r.logprob_1, cf.logprob_0, cf.logprob_1)[1] if cf else float("nan")
                rows.append({
                    "unit_key": unit.key, "run_name": self.run_name, "model": self.model_name,
                    "model_path": getattr(self.runner, "model_path", None),
                    "device": str(getattr(self.runner, "device", "")), "device_name": getattr(self.runner, "device_name", None),
                    "dtype": str(getattr(self.runner, "dtype", "")).replace("torch.", ""),
                    "naming": unit.naming, "domain": entry["domain"], "task_id": unit.task_id,
                    "family": entry["family"], "env": q["env"], "strategy": unit.strategy,
                    "mechanism": None if unit.strategy == "zero_shot" else SYNTHETIC_STRATEGIES[unit.strategy][0],
                    "composition": None if unit.strategy == "zero_shot" else SYNTHETIC_STRATEGIES[unit.strategy][1],
                    "label_mode": unit.label_mode, "k": unit.k, "seed": unit.seed,
                    "query_id": int(q["query_id"]), "row_id": int(q["row_id"]),
                    "demo_ids": demo["ids"], "demo_labels": demo["labels"], "order_seed": demo["order_seed"],
                    "prompt_hash": hashlib.sha1(full.encode()).hexdigest()[:16], "n_prompt_tokens": n_tokens[i],
                    "logprob_0": r.logprob_0, "logprob_1": r.logprob_1, "p1": r.p1,
                    "logprob_0_cf": cf.logprob_0 if cf else float("nan"),
                    "logprob_1_cf": cf.logprob_1 if cf else float("nan"),
                    "calibrated_p1": cal_p1, "prediction": r.prediction,
                    "prediction_cal": None if cf is None else ("1" if cal_p1 > 0.5 else "0"),
                    "label": str(int(q["label"])), "y_clean": int(q["y_clean"]),
                    "selection_meta": json.dumps(demo["meta"], default=str), "unit_seconds": None,
                })
        seconds = time.perf_counter() - t0
        out = pd.DataFrame(rows, columns=RESULTS_COLUMNS_V3)
        out["unit_seconds"] = seconds
        return out

    def run(self, units: list[Unit], out_path: str | Path, log=print) -> int:
        """Run the units not already in `out_path`; returns how many ran."""
        done = completed_units(out_path)
        todo = [u for u in units if u.key not in done and u.model == self.model_name]
        log(f"{self.model_name}: {len(todo)} units to run ({len(done)} already saved)")
        t_start = time.perf_counter()
        for i, unit in enumerate(todo, 1):
            rows = self.run_unit(unit)
            append_results(rows, out_path)
            elapsed = time.perf_counter() - t_start
            log(f"  [{i}/{len(todo)}] {unit.key}: {len(rows)} rows, {rows['unit_seconds'].iloc[0]:.1f}s "
                f"(eta {elapsed / i * (len(todo) - i) / 60:.1f} min)")
        return len(todo)
