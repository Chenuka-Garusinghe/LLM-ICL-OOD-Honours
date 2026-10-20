"""The XGBoost proxy comparison behind gate S3a and notebook 04 (P5).

Every method selects k demonstrations for each validation query, an XGBoost
classifier is fitted on them and predicts the query, and accuracy is averaged
per environment. The pools, queries and column-role permutations are those
of `evaluate_sata_proxy` (`_task_batch` with `_val_seed`), so a trained SATA
scored there and the methods scored here see the same data.

Methods:
- `oracle_matched`: balanced top-k by the main SATA's targets themselves
  (`matched_target_scores`, the user's decision; hiccups/21), no model.
  Gate S3a needs it to beat every hand-designed strategy, or nothing trained
  on the targets can.
- `oracle_target`: the same for the planned structural targets with
  label-noise rows excluded (`compute_target_scores`; sata_designed's).
- `oracle_target_designed`: the structural targets as first designed,
  before label-noise rows lost their bonus; kept for the record.
- the grid's strategies through `protocols.select` (`counter_prior` needs the
  model's measured prior, so it is not scored here). Query-agnostic
  strategies draw a fresh set for every query, which has the expectation of
  the grid's one set per task with less variance.
- trained SATA models passed in `models`, each under both selection rules:
  `<name>` draws k/2 rows per label in proportion to its scores (SATA's rule
  from P5; the draws are those of `evaluate_sata_proxy`), `<name> (top-k)`
  takes the top k/2.

Alongside accuracy, each method's sets are described by the share of rows
whose shortcut agrees with the label, the share of label-noise rows and the
share in the query's regime.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models.sata_targets import compute_target_scores, matched_target_scores
from src.models.sata_train import _n_columns, _task_batch, _val_seed, select_demos
from src.models.xgb_proxy import fit_predict_one_soft
from src.selection.balanced_topk import balanced_top_k
from src.selection.protocols import SYNTHETIC_STRATEGIES, ProtocolContext, demo_seed, select

STRATEGIES = ("random", "label_diversity", "feature_range", "rule_diversity", "counter_spurious", "similarity")
TOPK_TAG = " (top-k)"


def designed_target_scores(query_meta: dict, demo_meta: list[dict], temperature: float = 1.0) -> np.ndarray:
    """The targets as first designed (generator_spec.pdf before P5): both
    bonuses also go to label-noise rows."""
    s = np.array([2.0 * (d["regime"] == query_meta["regime"]) + 1.5 * d["is_counter_spurious"]
                  - 1.5 * (d["spurious_consistent"] and d["regime"] != query_meta["regime"]) for d in demo_meta])
    e = np.exp(s / temperature)
    return e / e.sum()


def _pool_frame(demo_X: np.ndarray, demo_y: np.ndarray, demo_meta: list[dict], spurious_col: int):
    cols = [f"f{i}" for i in range(demo_X.shape[1])]
    df = pd.DataFrame(demo_X, columns=cols)
    df["label"] = demo_y
    df["y_clean"] = [m["y_clean"] for m in demo_meta]
    df["regime"] = [m["regime"] for m in demo_meta]
    df["agree_obs"] = [m["agree_obs"] for m in demo_meta]
    # Magnitude matching reads |shortcut|; the pool's z-scored values serve,
    # since the shortcut is centred on 0 by construction.
    df["spurious_raw"] = demo_X[:, spurious_col]
    return df, cols


def proxy_compare(
    tasks: list,
    config: Any,
    methods: tuple[str, ...] = ("oracle_matched", "oracle_target", "oracle_target_designed", *STRATEGIES),
    models: dict[str, torch.nn.Module] | None = None,
    k: int = 8,
    base_seed: int = 42,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """One row per (method, task, environment): proxy accuracy and set statistics."""
    models = models or {}
    for m in models.values():
        m.eval()
    rows = []
    with ThreadPoolExecutor(max_workers=n_jobs) as pool, torch.no_grad():
        for task in tasks:
            for env in config.environments:
                pool_env = "mechanism" if env == "mechanism" else "id"
                seed = _val_seed(task, env)
                perm = (np.random.default_rng(seed).permutation(_n_columns(task))
                        if getattr(config, "permute_columns", True) else np.arange(_n_columns(task)))
                (demo_X, demo_y, demo_meta), (query_X, query_y, query_meta) = _task_batch(
                    task, env, n_demos=config.max_demos, n_queries=getattr(config, "queries_per_env", 32),
                    seed=seed, pool_env=pool_env, permutation=perm,
                )
                spurious_col = int(np.flatnonzero(perm == task.spurious_idx)[0])
                df, cols = _pool_frame(demo_X, demo_y, demo_meta, spurious_col)
                ctx = ProtocolContext(feature_cols=cols, kinds={c: "numeric" for c in cols}, train_ref=df)
                noisy = (df["label"] != df["y_clean"]).to_numpy()
                agree = df["agree_obs"].to_numpy().astype(bool)
                regimes = df["regime"].to_numpy()
                scored = {}
                for name, model in models.items():
                    dev = next(model.parameters()).device
                    n = len(query_y)
                    scored[name] = model(
                        torch.tensor(demo_X, dtype=torch.float32, device=dev).unsqueeze(0).expand(n, -1, -1),
                        torch.tensor(demo_y, dtype=torch.long, device=dev).unsqueeze(0).expand(n, -1),
                        torch.tensor(query_X, dtype=torch.float32, device=dev),
                    ).cpu().numpy()
                model_methods = [f"{name}{tag}" for name in models for tag in ("", TOPK_TAG)]
                for method in (*methods, *model_methods):
                    sets = []
                    for qi in range(len(query_y)):
                        if method == "oracle_matched":
                            idx = balanced_top_k(matched_target_scores(query_meta[qi], demo_meta, demo_X[:, spurious_col], k),
                                                 demo_y, k)
                        elif method == "oracle_target":
                            idx = balanced_top_k(compute_target_scores(query_meta[qi], demo_meta), demo_y, k)
                        elif method == "oracle_target_designed":
                            idx = balanced_top_k(designed_target_scores(query_meta[qi], demo_meta), demo_y, k)
                        elif method in model_methods:
                            name, rule = (method[: -len(TOPK_TAG)], "topk") if method.endswith(TOPK_TAG) else (method, "sample")
                            idx = select_demos(scored[name][qi], demo_y, k, rule, np.random.default_rng([seed, qi]))
                        else:
                            mech, comp = SYNTHETIC_STRATEGIES[method]
                            ids, _ = select(mech, comp, df, pd.Series(query_X[qi], index=cols), k,
                                            demo_seed(base_seed, task.task_id, f"{method}|{env}", 0, "select", qi), ctx)
                            idx = np.asarray(ids)
                        sets.append(np.asarray(idx))
                    futures = [pool.submit(fit_predict_one_soft, demo_X[s], demo_y[s], query_X[qi], query_y[qi])
                               for qi, s in enumerate(sets)]
                    acc = [f.result() for f in futures]
                    rows.append({
                        "method": method, "task_id": task.task_id, "family": task.rule_family, "env": env,
                        "accuracy": float(np.mean(acc)),
                        "shortcut_agrees": float(np.mean([agree[s].mean() for s in sets])),
                        "label_noise": float(np.mean([noisy[s].mean() for s in sets])),
                        "same_regime": float(np.mean([(regimes[s] == query_meta[qi]["regime"]).mean()
                                                      for qi, s in enumerate(sets)])),
                    })
    return pd.DataFrame(rows)


def summarise(per_task: pd.DataFrame, selection_envs: tuple[str, ...] = ("id", "covariate", "spurious_reversal")) -> pd.DataFrame:
    """Mean accuracy per (method, environment), the worst evaluated environment,
    and the set statistics, one row per method."""
    acc = per_task.pivot_table(index="method", columns="env", values="accuracy", aggfunc="mean")
    out = acc.copy()
    out["worst"] = acc[list(selection_envs)].min(axis=1)
    stats = per_task.groupby("method")[["shortcut_agrees", "label_noise", "same_regime"]].mean()
    return out.join(stats).sort_values("worst", ascending=False)
