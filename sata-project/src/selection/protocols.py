"""Demonstration protocols for tabular OOD tasks, as a mechanism x composition factorial.

Why a factorial rather than a list of named protocols
-----------------------------------------------------
The v1 suite is a list of six mutually exclusive arms (random, label_diversity,
feature_range, rule_diversity, counter_spurious, similarity). Two problems:

1. They confound *what gets selected* with *the label composition of what gets
   selected*. `label_diversity` is "random, forced to k/2 per class";
   `counter_spurious` over-samples minority cells, which are label-skewed by
   construction, so it silently ships a different label composition too. Since
   the demonstration set's label statistics are a first-order driver of an
   LLM's output distribution at this scale (Min et al. 2022; Zhao et al. 2021's
   majority-label bias), an arm-vs-arm comparison cannot tell you whether an
   effect came from the selection mechanism or from the label counts.

2. Pinning everything to k/2 per class (src/selection/balanced_topk.py) fixes
   problem 1 by removing the channel entirely -- but on these four TableShift
   datasets the ID->OOD failure is dominated by label-prior shift, not by loss
   of discriminative signal (an ID-trained HistGradientBoosting model holds its
   AUROC almost exactly across the shift on all four, while accuracy collapses
   where the prior moves: acspubcov 0.795 -> 0.627 as P(y=1) goes 0.23 -> 0.64).
   Label composition is therefore the single highest-leverage knob available to
   a demonstration protocol on this benchmark, and a 50/50 pin is the one
   setting guaranteed not to use it.

So composition is promoted to an explicit, crossed factor. The v1 arms become
cells of the grid (`label_diversity` == random x balanced), the confound becomes
an estimable interaction, and prior matching becomes testable rather than
assumed away.

    mechanism  x  composition  ->  protocol
    ---------     -----------
    random        free            mechanism decides the label counts
    similarity    balanced        k/2 per class (v1's pinned behaviour)
    feature_coverage              target_prior  counts track the estimated
    rule_coverage                               target prior pi_T from
    importance_weighted                         src/selection/shift_estimation.py
    shift_axis_coverage
    counter_spurious

Every mechanism returns *pool index labels* (not positional offsets). v1's
`similarity_select.select` returned positions into `pool_texts` while every
other selector returned index labels; that asymmetry is a bug magnet at the
call site and is not reproduced here.

Shift-aware mechanisms degrade gracefully: when the domain discriminator
cannot separate the domains (`DomainShift.detectable` is False -- observed on
anes, AUC 0.564), the density ratio is noise, so `importance_weighted` and
`shift_axis_coverage` fall back to uniform behaviour and set
`degraded=True` in the returned metadata rather than sampling on noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

COMPOSITIONS = ("free", "balanced", "target_prior")
MECHANISMS = (
    "random",
    "similarity",
    "feature_coverage",
    "rule_coverage",
    "counter_spurious",
    "importance_weighted",
    "shift_axis_coverage",
)


@dataclass
class ProtocolContext:
    """Everything a mechanism may condition on. Target labels are never included."""

    feature_cols: list[str]
    kinds: dict[str, str]
    shift: Any                                  # shift_estimation.DomainShift
    train_ref: pd.DataFrame                     # labelled ID reference (medians, tree fit)
    label_col: str = "label"
    proxy_features: list[str] = field(default_factory=list)
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
    """Marginal quantile-bin coverage over the highest-drift features.

    v1's `feature_range.select` binned the top 3 features jointly and sampled
    distinct *joint* cells. With 3 features x 3 bins there are up to 27 joint
    cells and k=8 slots, so it covered 8 arbitrary cells out of 27 and gave no
    guarantee that any individual feature's three bins were all represented --
    which is what "feature-range diversity" is supposed to deliver, and is the
    property that matters when the target domain sits in a range the demos
    never show. Stratifying on (feature, bin) pairs instead makes marginal
    coverage the thing being guaranteed.

    Features are ordered by source->target drift rather than by mutual
    information: the ranges worth spanning are the ones that actually move.
    """
    order = [f for f in ctx.shift.drift["feature"].tolist() if f in pool.columns][:n_feats]
    if not order:
        return "weight", np.ones(len(pool)), {"degraded": True}
    groups = np.full(len(pool), -1, dtype=int)
    gid, assigned = 0, np.zeros(len(pool), dtype=bool)
    for f in order:
        b = pd.qcut(pool[f], q=n_bins, labels=False, duplicates="drop")
        for lvl in sorted(pd.unique(b.dropna())):
            mask = (b == lvl).to_numpy() & ~assigned
            if mask.any():
                groups[mask] = gid
                assigned |= mask
                gid += 1
    groups[groups < 0] = gid
    return "strata", groups, {"n_strata": int(len(np.unique(groups)))}


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


def _m_importance_weighted(pool, query, ctx, rng):
    """Sample ID demos in proportion to the density ratio w(x) = p_T(x)/p_S(x).

    The demonstration set then looks like the target domain while still carrying
    trustworthy source labels -- the demonstration-design analogue of importance
    weighting. Targets covariate shift and extrapolation.
    """
    if not ctx.shift.detectable:
        return "weight", np.ones(len(pool)), {"degraded": True, "reason": "domain AUC < 0.60"}
    return "weight", ctx.shift.weight(pool), {"degraded": False}


def _m_shift_axis_coverage(pool, query, ctx, rng, n_bins: int = 4):
    """Span the shift axis: strata over quantiles of the discriminator score s(x).

    Importance weighting concentrates the demo set at the target-like end, which
    at k=8 can collapse it onto one region. This instead guarantees the prompt
    contains demos from source-like *and* target-like ends, so the context spans
    the direction the distribution moved rather than sitting at one point on it.
    """
    if not ctx.shift.detectable:
        return "weight", np.ones(len(pool)), {"degraded": True, "reason": "domain AUC < 0.60"}
    s = ctx.shift.score(pool)
    b = pd.qcut(pd.Series(s), q=n_bins, labels=False, duplicates="drop").fillna(0).to_numpy()
    return "strata", b.astype(int), {"n_strata": int(len(np.unique(b)))}


_MECH: dict[str, Callable] = {
    "random": _m_random,
    "similarity": _m_similarity,
    "feature_coverage": _m_feature_coverage,
    "rule_coverage": _m_rule_coverage,
    "counter_spurious": _m_counter_spurious,
    "importance_weighted": _m_importance_weighted,
    "shift_axis_coverage": _m_shift_axis_coverage,
}

QUERY_CONDITIONAL = {"similarity"}
SHIFT_CONDITIONAL = {"importance_weighted", "shift_axis_coverage", "feature_coverage", "counter_spurious"}


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def label_quota(composition: str, k: int, ctx: ProtocolContext, pool: pd.DataFrame) -> dict[int, int] | None:
    """Demos per class, or None for 'free' (mechanism decides)."""
    if composition == "free":
        return None
    if composition == "balanced":
        n1 = k // 2
    elif composition == "target_prior":
        # Round to the nearest achievable count, but never let a class vanish:
        # a single-class prompt makes the LLM's answer the majority label almost
        # regardless of the query (Zhao et al. 2021), which would make the
        # composition factor a degenerate constant-prediction arm.
        n1 = int(np.clip(round(k * float(ctx.shift.prior_target)), 1, k - 1))
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
        groups = {}
        for pos in positions:
            groups.setdefault(int(v[np.where(positions == pos)[0][0]]), []).append(pos)
        keys = list(groups)
        rng.shuffle(keys)
        for g in keys:
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
    return cells


V1_EQUIVALENTS = {
    "random": ("random", "free"),
    "label_diversity": ("random", "balanced"),
    "feature_range": ("feature_coverage", "free"),
    "rule_diversity": ("rule_coverage", "free"),
    "similarity": ("similarity", "free"),
    "counter_spurious": ("counter_spurious", "free"),
}
