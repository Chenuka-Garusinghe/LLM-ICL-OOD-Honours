"""Demonstration selection as a mechanism x composition factorial.

The composition fixes how many demos of each label are shown, so label counts
are never confounded with the mechanism (`label_diversity` is random x balanced).
Every mechanism returns pool index labels, never positions.

Synthetic (v3) strategies map onto mechanism x composition through
`SYNTHETIC_STRATEGIES` (generator_spec.pdf, selection engine). The `gt_*`
mechanisms read the generator's ground-truth row tags (`regime`, `agree_obs`,
`spurious_raw`, `y_clean`), which SATA does not get; the spec discloses this
asymmetry. `counter_prior` reads the model's prior surrogate
(src/inference/priors.py) through `ProtocolContext.prior_slopes`. Seeds come
from `demo_seed`, one stream per (task, strategy, seed).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

COMPOSITIONS = ("free", "balanced")
MECHANISMS = (
    "random",
    "similarity",
    "feature_coverage",
    "rule_coverage",
    "counter_spurious",
    "gt_regime",
    "gt_counter_spurious",
    "feature_knn",
    "counter_prior",
)


@dataclass
class ProtocolContext:
    """Everything a mechanism may condition on. Target labels are never included."""

    feature_cols: list[str]
    kinds: dict[str, str]
    train_ref: pd.DataFrame                     # labelled ID reference (medians, tree fit)
    label_col: str = "label"
    proxy_features: list[str] = field(default_factory=list)
    prior_slopes: dict[str, float] = field(default_factory=dict)  # column -> prior surrogate slope
    prior_margin: pd.Series | None = None      # pool row id -> measured zero-shot margin (centred); preferred
    _cache: dict = field(default_factory=dict, repr=False)

    def ranges(self) -> pd.Series:
        if "ranges" not in self._cache:
            r = self.train_ref[self.feature_cols].max() - self.train_ref[self.feature_cols].min()
            self._cache["ranges"] = r.replace(0, 1.0)
        return self._cache["ranges"]

    def leaf_tree(self) -> DecisionTreeClassifier:
        if "tree" not in self._cache:
            t = DecisionTreeClassifier(max_depth=3, random_state=0)
            t.fit(self.train_ref[self.feature_cols], self.train_ref[self.label_col])
            self._cache["tree"] = t
        return self._cache["tree"]

    def medians(self) -> pd.Series:
        if "medians" not in self._cache:
            self._cache["medians"] = self.train_ref[self.feature_cols].median()
        return self._cache["medians"]


# --------------------------------------------------------------------------- #
# distance
# --------------------------------------------------------------------------- #
def gower_distance(pool: pd.DataFrame, query: pd.Series, ctx: ProtocolContext) -> np.ndarray:
    """Mixed-type row distance: scaled |delta| for numeric, 0/1 mismatch for categorical.

    v1's `similarity_select` embedded *serialised text* with all-MiniLM-L6-v2 and
    took cosine similarity. For numeric tabular rows that similarity is driven by
    surface token overlap of the formatted numbers ("age: 67.00" vs "age: 67.40"
    share tokens; 67 vs 82 do not, but neither do 67 vs 6.7 in a predictable way),
    and it is worst behaved exactly where OOD rows carry values outside the
    demonstrated range. A metric defined on the feature values themselves is the
    defensible baseline for a "retrieve similar rows" protocol; the text-embedding
    variant is worth keeping only as a deliberate ablation.
    """
    num = [f for f in ctx.feature_cols if ctx.kinds.get(f) != "categorical"]
    cat = [f for f in ctx.feature_cols if ctx.kinds.get(f) == "categorical"]
    parts, n_used = np.zeros(len(pool)), 0
    if num:
        rng_ = ctx.ranges()[num].to_numpy(dtype=float)
        d = np.abs(pool[num].to_numpy(dtype=float) - query[num].to_numpy(dtype=float)) / rng_
        parts += np.clip(d, 0, 1).sum(axis=1)
        n_used += len(num)
    if cat:
        parts += (pool[cat].to_numpy() != query[cat].to_numpy()).sum(axis=1)
        n_used += len(cat)
    return parts / max(n_used, 1)


# --------------------------------------------------------------------------- #
# mechanisms: each returns (kind, values, meta)
#   kind == "score"  -> higher is better, chosen top-down
#   kind == "weight" -> non-negative sampling weights
#   kind == "strata" -> group ids, covered round-robin
# --------------------------------------------------------------------------- #
def _m_random(pool, query, ctx, rng):
    return "weight", np.ones(len(pool)), {}


def _m_similarity(pool, query, ctx, rng):
    if query is None:
        raise ValueError("mechanism 'similarity' is query-conditional; `query` is required")
    return "score", -gower_distance(pool, query, ctx), {}


def _m_feature_coverage(pool, query, ctx, rng, n_bins: int = 3, n_feats: int = 3):
    """Coverage of the joint quantile cells of the pool's top features by mutual
    information with the label: each of the `n_feats` features is cut into
    `n_bins` quantile bins, a row's cell is its bin combination, and cells are
    covered round-robin (generator_spec.pdf, mechanisms).

    The version ported from main assigned rows feature by feature, so the
    top feature's bins absorbed every row and the other features never
    formed a stratum.
    """
    mi = mutual_info_classif(pool[ctx.feature_cols], pool[ctx.label_col], random_state=0)
    order = [ctx.feature_cols[i] for i in np.argsort(-mi)[:n_feats]]
    cells = np.zeros(len(pool), dtype=int)
    for f in order:
        b = pd.qcut(pool[f], q=n_bins, labels=False, duplicates="drop").fillna(0).astype(int).to_numpy()
        cells = cells * n_bins + b
    return "strata", cells, {"n_strata": int(len(np.unique(cells))), "features": order}


def _m_rule_coverage(pool, query, ctx, rng):
    leaves = ctx.leaf_tree().apply(pool[ctx.feature_cols])
    return "strata", leaves.astype(int), {"n_strata": int(len(np.unique(leaves)))}


def _m_counter_spurious(pool, query, ctx, rng):
    """Rank by how many shortcut features the row *contradicts*.

    A shortcut feature is one that both drifts across domains and predicts the
    label in-domain (`shift_estimation.shift_proxy_features`) -- v1 needed a
    `shift_col` naming the domain variable, which the parquet cache does not
    carry. Thresholds come from the labelled ID reference split, not from the
    256-row pool: a pool-internal median is a noisy threshold that moves with
    the sampling seed, so the same physical row could count as counter-shortcut
    in one seed and not the next.
    """
    proxies = [f for f in (ctx.proxy_features or []) if f in pool.columns]
    if not proxies:
        return "weight", np.ones(len(pool)), {"degraded": True}
    med, ref, y = ctx.medians(), ctx.train_ref, ctx.train_ref[ctx.label_col]
    score = np.zeros(len(pool), dtype=float)
    for f in proxies:
        hi_ref = ref[f] > med[f]
        # label that the shortcut "votes for" when the feature is high
        vote_high = int(y[hi_ref].mean() >= y[~hi_ref].mean())
        hi = (pool[f] > med[f]).to_numpy()
        implied = np.where(hi, vote_high, 1 - vote_high)
        score += (implied != pool[ctx.label_col].to_numpy()).astype(float)
    return "score", score, {"n_proxies": len(proxies)}


def _m_gt_regime(pool, query, ctx, rng):
    """Ground-truth regime strata (sign pattern of the load-bearing features).

    A row is eligible for its class when it is not a label-noise row
    (label == y_clean) and its class is the pool majority in its regime, so
    each class covers the regimes where it actually lives: for a tree task
    the 4 leaves carrying that label. Without the filter, the few noise rows
    sitting in the other class's leaves would each form a stratum and be
    over-sampled. Ineligible rows get stratum -1, used only as a fallback.
    """
    labels = pool[ctx.label_col].to_numpy()
    regimes = pool["regime"].to_numpy().astype(int)
    clean = labels == pool["y_clean"].to_numpy()
    rate = pd.Series(labels).groupby(regimes).mean()
    majority_ok = np.where(labels == 1, rate.loc[regimes].to_numpy() >= 0.5, rate.loc[regimes].to_numpy() <= 0.5)
    groups = np.where(clean & majority_ok, regimes, -1)
    return "strata", groups, {"n_strata": int(len(np.unique(groups[groups >= 0])))}


def _m_feature_knn(pool, query, ctx, rng):
    """Euclidean distance to the query on the z-scored features. It reads
    only numbers, so it picks the same rows under every naming (v2's MiniLM
    text similarity changed with the display names)."""
    if query is None:
        raise ValueError("mechanism 'feature_knn' is query-conditional; `query` is required")
    diff = pool[ctx.feature_cols].to_numpy(dtype=float) - query[ctx.feature_cols].to_numpy(dtype=float)
    return "score", -np.sqrt((diff ** 2).sum(axis=1)), {}


def prior_scores(frame: pd.DataFrame, ctx: ProtocolContext) -> np.ndarray:
    """The model's prior for every row: its measured zero-shot margin, centred
    on the pool median (`prior_margin`, when given), else the intercept-free
    surrogate sum_j b_j x_j (z-scored columns)."""
    if ctx.prior_margin is not None:
        return ctx.prior_margin.reindex(frame.index).to_numpy(dtype=float)
    if not ctx.prior_slopes:
        raise ValueError("counter_prior needs the measured prior (prior_margin) or the surrogate (prior_slopes)")
    b = np.array([ctx.prior_slopes[f] for f in ctx.feature_cols], dtype=float)
    return frame[ctx.feature_cols].to_numpy(dtype=float) @ b


def prior_data_conflict(pool: pd.DataFrame, ctx: ProtocolContext) -> float:
    """C_t: the share of pool rows whose label the surrogate's prior gets wrong."""
    return float(np.mean((prior_scores(pool, ctx) > 0) != (pool[ctx.label_col].to_numpy() == 1)))


def _m_counter_prior(pool, query, ctx, rng):
    """Rows whose label contradicts the model's prior and follows the rule
    (generator_spec.pdf, mechanisms): score 1 when the intercept-free
    surrogate predicts the other label and the row is not a label-noise row,
    else 0; top-scoring rows are taken in random order. The grid applies it
    only when the task's prior-data conflict exceeds 1/2 (`counter_prior_active`).
    """
    z = prior_scores(pool, ctx)
    labels = pool[ctx.label_col].to_numpy()
    contra = (z > 0) != (labels == 1)
    clean = labels == pool["y_clean"].to_numpy()
    score = (contra & clean).astype(float)
    return "score", score, {"conflict": float(contra.mean()), "n_eligible": int(score.sum()),
                            "n_eligible_by_class": {int(c): int(score[labels == c].sum()) for c in (0, 1)}}


def counter_prior_active(pool: pd.DataFrame, ctx: ProtocolContext) -> tuple[bool, float]:
    """Whether counter_prior applies to a task: its prior-data conflict exceeds 1/2.

    Otherwise the strategy falls back to label_diversity's demonstration set
    (the same rows and order), so the two strategies coincide exactly.
    """
    c = prior_data_conflict(pool, ctx)
    return c > 0.5, c


def select_counter_prior_matched(pool: pd.DataFrame, ctx: ProtocolContext, reference_ids: list,
                                 seed: int) -> tuple[list, dict]:
    """counter_prior with the shortcut held fixed (exploratory; hiccups/16).

    Takes the same number of rows in each (label, f8 agrees with label) cell as
    the reference set (label_diversity's set for the same seed), so label
    counts and the shortcut's apparent reliability match label_diversity's.
    Within each cell it prefers clean rows whose label contradicts the model's
    measured prior, then other clean rows, then label-noise rows, in random
    order. Plain counter_prior over-selects shortcut-agreeing rows where the
    model's numeric prior opposes the shortcut's direction.
    """
    rng = np.random.default_rng(seed)
    labels = pool[ctx.label_col].to_numpy()
    agree = pool["agree_obs"].to_numpy().astype(bool)
    clean = labels == pool["y_clean"].to_numpy()
    contra = (prior_scores(pool, ctx) > 0) != (labels == 1)
    score = 2.0 * (contra & clean) + 1.0 * (clean & ~contra)
    ref = pool.loc[reference_ids]
    picked: list[int] = []
    for cls in (0, 1):
        for ag in (True, False):
            n = int(((ref[ctx.label_col] == cls).to_numpy() & (ref["agree_obs"].to_numpy().astype(bool) == ag)).sum())
            cell = np.flatnonzero((labels == cls) & (agree == ag))
            got = _choose(cell, n, "score", score, rng)
            picked.extend(got)
            if len(got) < n:                      # a cell ran short: fill from the class
                rest = np.setdiff1d(np.flatnonzero(labels == cls), np.array(picked, dtype=int))
                picked.extend(_choose(rest, n - len(got), "score", score, rng))
    picked = [int(p) for p in picked]
    meta = {"mechanism": "counter_prior", "matched": True, "composition": "balanced", "k": len(reference_ids),
            "seed": seed, "kind": "matched", "n_selected": len(picked),
            "n_contradicting_clean": int(np.sum((contra & clean)[picked])),
            "f8_agreeing": int(np.sum(agree[picked]))}
    return list(pool.index[picked]), meta


_MECH: dict[str, Callable] = {
    "random": _m_random,
    "similarity": _m_similarity,
    "feature_coverage": _m_feature_coverage,
    "rule_coverage": _m_rule_coverage,
    "counter_spurious": _m_counter_spurious,
    "gt_regime": _m_gt_regime,
    "feature_knn": _m_feature_knn,
    "counter_prior": _m_counter_prior,
}

QUERY_CONDITIONAL = {"similarity", "feature_knn"}


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def label_quota(composition: str, k: int, ctx: ProtocolContext, pool: pd.DataFrame) -> dict[int, int] | None:
    """Demos per class, or None for 'free' (mechanism decides)."""
    if composition == "free":
        return None
    if composition == "balanced":
        n1 = k // 2
    else:
        raise ValueError(f"unknown composition {composition!r}")
    return {1: n1, 0: k - n1}


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def _choose(positions: np.ndarray, n: int, kind: str, values: np.ndarray, rng) -> list[int]:
    """Take n positions from `positions` using the mechanism's output."""
    if n <= 0 or len(positions) == 0:
        return []
    n = min(n, len(positions))
    v = values[positions]
    if kind == "score":
        # random tie-break so equal scores don't inherit pool row order
        return list(positions[np.lexsort((rng.random(len(positions)), -v))][:n])
    if kind == "weight":
        w = np.clip(v.astype(float), 0, None)
        p = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
        return list(rng.choice(positions, size=n, replace=False, p=p))
    if kind == "strata":
        # Negative strata are fallback rows, drawn only once every real
        # stratum is exhausted.
        groups = {}
        for pos, g in zip(positions, v):
            groups.setdefault(int(g), []).append(pos)
        keys = [g for g in groups if g >= 0]
        rng.shuffle(keys)
        for g in groups:
            rng.shuffle(groups[g])
        out: list[int] = []
        while len(out) < n:
            progressed = False
            for g in keys:
                if groups[g]:
                    out.append(groups[g].pop())
                    progressed = True
                    if len(out) == n:
                        break
            if not progressed:
                break
        fallback = [p for g in groups if g < 0 for p in groups[g]]
        rng.shuffle(fallback)
        out.extend(fallback[: n - len(out)])
        return out
    raise ValueError(f"unknown mechanism kind {kind!r}")


def select(
    mechanism: str,
    composition: str,
    pool: pd.DataFrame,
    query: pd.Series | None,
    k: int,
    seed: int,
    ctx: ProtocolContext,
) -> tuple[list, dict]:
    """Select k demonstrations. Returns (pool index labels, metadata).

    Deterministic given `seed`. Ordering of the returned list is NOT the prompt
    order -- pass it through src/selection/ordering.py::shuffle_order, as every
    condition must, so demo position stays a uniformly controlled nuisance.
    """
    if mechanism == "gt_counter_spurious":
        return _select_counter_spurious_matched(pool, k, seed, ctx, composition)
    if mechanism not in _MECH:
        raise ValueError(f"unknown mechanism {mechanism!r}; expected one of {MECHANISMS}")
    rng = np.random.default_rng(seed)
    kind, values, meta = _MECH[mechanism](pool, query, ctx, rng)
    values = np.asarray(values)

    quota = label_quota(composition, k, ctx, pool)
    labels = pool[ctx.label_col].to_numpy()

    if quota is None:
        picked = _choose(np.arange(len(pool)), k, kind, values, rng)
    else:
        picked = []
        for cls, n in sorted(quota.items()):
            cls_pos = np.where(labels == cls)[0]
            picked.extend(_choose(cls_pos, n, kind, values, rng))
        if len(picked) < k:                       # a class ran short
            leftover = np.setdiff1d(np.arange(len(pool)), np.array(picked, dtype=int))
            picked.extend(_choose(leftover, k - len(picked), kind, values, rng))

    picked = [int(p) for p in picked][:k]
    meta = dict(meta)
    meta.update(mechanism=mechanism, composition=composition, k=k, seed=seed,
                kind=kind, n_selected=len(picked),
                quota=None if quota is None else {int(c): int(n) for c, n in quota.items()})
    return list(pool.index[picked]), meta


def _select_counter_spurious_matched(
    pool: pd.DataFrame, k: int, seed: int, ctx: ProtocolContext, composition: str = "balanced"
) -> tuple[list, dict]:
    """Magnitude-matched counter-spurious set (generator_spec.pdf, counter-spurious
    selection must match on |f8|).

    Within each class, draw k/4 rows whose spurious feature disagrees with the
    observed label and pair each with the agreeing row of the same class with
    the nearest |f8| (generator scale), without replacement. Sign agreement is
    then 1/2 and f8 carries no information about the label: simply balancing
    the four (label x agreement) cells leaves corr(f8, y) near 0.33 at
    s = 0.85, because agreeing rows lie further from zero.

    Label-noise rows (label != y_clean) are excluded on both sides. They are
    mostly disagreeing rows, so leaving them in would give counter-spurious
    sets several times the pool's label-noise rate.
    """
    if composition != "balanced":
        raise ValueError("gt_counter_spurious is defined for the balanced composition only")
    if k % 4:
        raise ValueError(f"magnitude-matched counter-spurious needs k divisible by 4, got {k}")
    rng = np.random.default_rng(seed)
    labels = pool[ctx.label_col].to_numpy()
    clean = labels == pool["y_clean"].to_numpy()
    agree = pool["agree_obs"].to_numpy().astype(bool)
    mag = np.abs(pool["spurious_raw"].to_numpy(dtype=float))
    picked: list[int] = []
    n_unmatched, gaps = 0, []
    for cls in (0, 1):
        in_cls = (labels == cls) & clean
        disagree = np.flatnonzero(in_cls & ~agree)
        agreeing = list(np.flatnonzero(in_cls & agree))
        n_pairs = min(k // 4, len(disagree), len(agreeing))
        for d in rng.choice(disagree, size=n_pairs, replace=False):
            dist = np.abs(mag[agreeing] - mag[d])
            j = int(rng.choice(np.flatnonzero(dist == dist.min())))
            gaps.append(float(dist[j]))
            picked.extend([int(d), int(agreeing.pop(j))])
        short = k // 2 - 2 * n_pairs
        if short:
            # Too few disagreeing rows in this class: fill with random rows of
            # the class so the label counts stay k/2 (recorded in the metadata).
            rest = np.setdiff1d(np.flatnonzero(labels == cls), np.array(picked, dtype=int))
            picked.extend(int(p) for p in rng.choice(rest, size=min(short, len(rest)), replace=False))
            n_unmatched += short
    meta = {
        "mechanism": "gt_counter_spurious", "composition": composition, "k": k, "seed": seed,
        "kind": "matched", "n_selected": len(picked), "quota": {0: k // 2, 1: k - k // 2},
        "n_unmatched": n_unmatched, "mean_abs_f8_gap": float(np.mean(gaps)) if gaps else float("nan"),
    }
    return list(pool.index[picked]), meta


def protocol_grid(include_degenerate: bool = False) -> list[tuple[str, str]]:
    """All (mechanism, composition) cells.

    `random x free` is the uniform baseline. `random x balanced` reproduces v1's
    `label_diversity`; `similarity x free` reproduces v1's `similarity`.
    """
    cells = [(m, c) for m in MECHANISMS for c in COMPOSITIONS]
    if not include_degenerate:
        # counter_spurious under 'free' is a pure score ranking that tends to
        # one label class by construction -- kept only when explicitly asked for.
        cells = [(m, c) for m, c in cells if not (m == "counter_spurious" and c == "free")]
        cells = [(m, c) for m, c in cells if not (m == "gt_counter_spurious" and c == "free")]
    return cells


# The synthetic arm's strategies (generator_spec.pdf, strategy table).
# Structured strategies are balanced (k/2 per class); plain random is free.
# zero_shot (no demonstrations) is handled by the runner, and SATA arrives in P5.
SYNTHETIC_STRATEGIES = {
    "random": ("random", "free"),
    "label_diversity": ("random", "balanced"),
    "feature_range": ("feature_coverage", "balanced"),
    "rule_diversity": ("gt_regime", "balanced"),
    "counter_spurious": ("gt_counter_spurious", "balanced"),
    "similarity": ("feature_knn", "balanced"),
    "counter_prior": ("counter_prior", "balanced"),
    "counter_prior_matched": ("counter_prior", "balanced"),   # exploratory (hiccups/16)
}
# Where counter_prior is inactive (prior-data conflict <= 1/2) it shows this
# strategy's demonstration set.
COUNTER_PRIOR_FALLBACK = "label_diversity"

# The v1 strategy names, kept as the names of the synthetic strategies above.
V1_EQUIVALENTS = {k: v for k, v in SYNTHETIC_STRATEGIES.items() if not k.startswith("counter_prior")}


def demo_seed(base_seed: int, task_id: str, strategy: str, seed: int, purpose: str = "select", *extra: int) -> int:
    """Seed for one random choice: SeedSequence([base, task, strategy, seed, purpose, ...]).

    Every (task, strategy, seed) gets its own stream, so a demonstration set
    does not change when other tasks or strategies are added. `purpose`
    separates the selection draw from the order shuffle and label shuffle;
    `extra` carries a query row id for query-conditional strategies.
    """
    parts = [base_seed, zlib.crc32(task_id.encode()), zlib.crc32(strategy.encode()), seed,
             zlib.crc32(purpose.encode()), *extra]
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])
