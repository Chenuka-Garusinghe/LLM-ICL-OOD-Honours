"""CPU-only pre-GPU screen of the demonstration-protocol grid.

What this measures, and what it deliberately does not
-----------------------------------------------------
An LLM prompt at k=8 can only convey what is in the eight rows. This screen asks
the question that precedes "does the LLM do better": *is the information even
present in the demonstration set?* For every (mechanism x composition) cell it
builds the demo set the protocol would put in the prompt and scores it with
three order-invariant surrogate readers, each standing in for a different
documented ICL behaviour:

  prior_only  predict the demo set's majority label for every query. The
              label-copy / majority-label behaviour (Zhao et al. 2021's
              majority-label bias; the "label space, not input-label mapping"
              reading of Min et al. 2022). Measures the composition channel
              alone -- no query is read at all.

  nn1         predict the label of the nearest demo (Gower). Retrieval-style
              behaviour: uses the query, uses the input-label pairing, but
              learns no rule.

  logreg      L2 logistic regression fit on the eight demo rows. Task
              *learning*: extracting a mapping from the demonstrations. The
              optimistic end of what k=8 rows can support.

The spread between `prior_only` and `logreg` is the headroom that demonstration
*content* offers over demonstration *label counts*. A cell where the two are
equal is one where no LLM, however good, can distinguish the protocol from its
label composition -- which is the pre-registered scale-dependent prediction the
ICL literature licenses (Wei et al. 2023; Krishna Kumar 2025, "Semantic Anchors
in In-Context Learning: Why Small LLMs Cannot Flip Their Labels", arXiv
2511.21038, reports semantic override rates of exactly zero across eight
open-source LLMs of 1-12B parameters, so 7-8B models are expected to sit near
`prior_only`, with `logreg`-like behaviour only reachable at larger scale).

Limits, stated plainly. These surrogates are order-invariant, so the screen
cannot speak to demo ordering, serialisation, or prompt format -- the axes that
need the actual LLM. It is a necessary-condition filter, not a substitute for
the GPU runs: a protocol whose demo sets carry no more OOD information than
random-k is not worth GPU hours, but passing the screen does not imply an LLM
will exploit it.

Pool provenance. `id_pool` draws demos from the labelled source split and
queries from the target split -- the deployment-realistic setting. The pool is
sampled WITHOUT label stratification, unlike
tableshift_loader.build_demo_pool's 50/50 stratified pool: with composition now
an explicit factor, pre-balancing the pool would pre-empt it.

Usage
-----
    cd sata-project
    PYTHONPATH=. python scripts/screen_protocols.py --cache-dir /tmp/screen_cache \
        --out ../_screen_out --n-queries 250 --seeds 42 123 456 789 1024
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.selection.protocols import ProtocolContext, gower_distance, protocol_grid, select

warnings.filterwarnings("ignore")

DATASETS = ["brfss_diabetes", "acsincome", "acspubcov", "anes"]
SURROGATES = ("prior_only", "nn1", "logreg")


# --------------------------------------------------------------------------- #
# surrogate readers
# --------------------------------------------------------------------------- #
def _fit_predict(demos: pd.DataFrame, queries: pd.DataFrame, ctx: ProtocolContext,
                 surrogate: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (hard predictions, scores-or-None) for `queries`."""
    y = demos[ctx.label_col].to_numpy()
    f = ctx.feature_cols

    if surrogate == "prior_only":
        # Bernoulli(demo positive fraction) rather than a hard argmax. Under
        # `balanced` composition the demo majority is exactly tied at even k,
        # and breaking that tie on an arbitrary demo's label would hand a third
        # of the grid a coin-flip result that looks like a measurement. Sampling
        # in proportion to the demo label fraction makes this reader a
        # continuous, monotone function of the composition factor -- which is
        # the channel it exists to isolate -- and gives `balanced` its correct
        # expected accuracy of 0.5 instead of an artefact.
        p = float(y.mean())
        rng = np.random.default_rng(seed)
        return (rng.random(len(queries)) < p).astype(int), np.full(len(queries), p)

    if surrogate == "nn1":
        D = np.vstack([gower_distance(demos, queries.iloc[i], ctx) for i in range(len(queries))])
        nn = D.argmin(axis=1)
        pred = y[nn]
        # score = inverse-distance-weighted positive share, for AUROC
        with np.errstate(divide="ignore"):
            W = 1.0 / np.maximum(D, 1e-6)
        score = (W * (y == 1)).sum(axis=1) / W.sum(axis=1)
        return pred, score

    if surrogate == "logreg":
        if len(np.unique(y)) < 2:
            maj = int(y[0])
            return np.full(len(queries), maj), np.full(len(queries), float(maj))
        sc = StandardScaler().fit(demos[f])
        # Strong regularisation: 8 rows, 12 features. C=0.5 keeps the fit from
        # memorising the demos and is held fixed across every cell so the
        # comparison is between demo sets, not between hyperparameters.
        m = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(demos[f]), y)
        p = m.predict_proba(sc.transform(queries[f]))[:, 1]
        return (p > 0.5).astype(int), p

    raise ValueError(surrogate)


def _score(y_true: np.ndarray, pred: np.ndarray, score: np.ndarray | None) -> dict:
    """Accuracy plus the metrics that survive class imbalance.

    Raw accuracy is not a safe headline here: on brfss_diabetes (12.5% positive)
    a demo set that simply carries the source label prior makes the surrogate
    predict the majority class, which scores ~0.75 without using the query at
    all. That would credit the `free` composition for class imbalance rather
    than for conveying anything. Balanced accuracy (mean per-class recall) and
    AUROC are reported alongside so the composition factor can be read off a
    metric it cannot game.
    """
    out = {"accuracy": float(accuracy_score(y_true, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
           "pred_pos_rate": float(np.mean(pred == 1)),
           "query_pos_rate": float(np.mean(y_true == 1))}
    try:
        out["auroc"] = float(roc_auc_score(y_true, score)) if score is not None and len(np.unique(y_true)) == 2 else np.nan
    except ValueError:
        out["auroc"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# demo-set descriptors
# --------------------------------------------------------------------------- #
def descriptors(demos: pd.DataFrame, ctx: ProtocolContext) -> dict:
    s = ctx.shift.score(demos) if ctx.shift is not None else np.full(len(demos), np.nan)
    leaves = ctx.leaf_tree().apply(demos[ctx.feature_cols])
    return dict(
        demo_pos_frac=float(demos[ctx.label_col].mean()),
        demo_target_likeness=float(np.mean(s)),
        demo_leaf_coverage=int(len(np.unique(leaves))),
        demo_feature_spread=float(np.mean(demos[ctx.feature_cols].std(ddof=0) /
                                          ctx.train_ref[ctx.feature_cols].std(ddof=0).replace(0, 1))),
    )


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def run_dataset(ctx_blob: dict, k: int, seeds: list[int], n_queries: int,
                pool_size: int) -> pd.DataFrame:
    ds = ctx_blob["dataset"]
    ctx = ProtocolContext(
        feature_cols=ctx_blob["feats"], kinds=ctx_blob["kinds"], shift=ctx_blob["shift"],
        train_ref=ctx_blob["train"], proxy_features=ctx_blob["proxy_features"],
    )
    train, test_id, test_ood = ctx_blob["train"], ctx_blob["test_id"], ctx_blob["test_ood"]
    grid = protocol_grid()
    rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # natural (unstratified) pool: preserves the source label prior
        pool = train.iloc[rng.choice(len(train), pool_size, replace=False)].copy()
        pool.index = pd.RangeIndex(len(pool))

        qsets = {
            "ood": test_ood.iloc[rng.choice(len(test_ood), min(n_queries, len(test_ood)), replace=False)].reset_index(drop=True),
            "id": test_id.iloc[rng.choice(len(test_id), min(n_queries, len(test_id)), replace=False)].reset_index(drop=True),
        }

        for mech, comp in grid:
            query_conditional = mech == "similarity"
            for qsplit, Q in qsets.items():
                y_true = Q[ctx.label_col].to_numpy()

                if not query_conditional:
                    demo_idx, meta = select(mech, comp, pool, Q.iloc[0], k, seed, ctx)
                    demos = pool.loc[demo_idx]
                    desc = descriptors(demos, ctx)
                    desc["demo_query_dist"] = float(np.mean(
                        [gower_distance(demos, Q.iloc[i], ctx).mean() for i in range(min(len(Q), 100))]))
                    for sur in SURROGATES:
                        pred, sco = _fit_predict(demos, Q, ctx, sur, seed=seed)
                        rows.append(dict(dataset=ds, mechanism=mech, composition=comp, seed=seed,
                                         query_split=qsplit, surrogate=sur, k=k,
                                         query_conditional=False,
                                         degraded=bool(meta.get("degraded", False)),
                                         **_score(y_true, pred, sco), **desc))
                else:
                    # one demo set per query: accumulate per-query predictions
                    per_sur = {s: (np.empty(len(Q), dtype=int), np.empty(len(Q))) for s in SURROGATES}
                    acc_desc, dists = [], []
                    for i in range(len(Q)):
                        demo_idx, meta = select(mech, comp, pool, Q.iloc[i], k, seed + i, ctx)
                        demos = pool.loc[demo_idx]
                        if i < 100:
                            acc_desc.append(descriptors(demos, ctx))
                            dists.append(float(gower_distance(demos, Q.iloc[i], ctx).mean()))
                        for sur in SURROGATES:
                            p, sc_ = _fit_predict(demos, Q.iloc[[i]], ctx, sur, seed=seed + i)
                            per_sur[sur][0][i], per_sur[sur][1][i] = p[0], sc_[0]
                    desc = {kk: float(np.mean([d[kk] for d in acc_desc])) for kk in acc_desc[0]}
                    desc["demo_query_dist"] = float(np.mean(dists))
                    for sur in SURROGATES:
                        pred, sco = per_sur[sur]
                        rows.append(dict(dataset=ds, mechanism=mech, composition=comp, seed=seed,
                                         query_split=qsplit, surrogate=sur, k=k,
                                         query_conditional=True, degraded=False,
                                         **_score(y_true, pred, sco), **desc))
        print(f"[screen] {ds} seed={seed} done ({len(rows)} rows)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/tmp/screen_cache")
    ap.add_argument("--out", default="_screen_out")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=256)
    ap.add_argument("--n-queries", type=int, default=250)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frames = []
    for ds in DATASETS:
        with open(Path(args.cache_dir) / f"{ds}.pkl", "rb") as f:
            blob = pickle.load(f)
        frames.append(run_dataset(blob, args.k, args.seeds, args.n_queries, args.pool_size))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out / "protocol_screen.parquet", index=False)
    print(f"[screen] wrote {out / 'protocol_screen.parquet'} rows={len(df)}")


if __name__ == "__main__":
    main()
