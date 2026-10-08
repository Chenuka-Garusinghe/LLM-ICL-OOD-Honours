"""Measuring a model's zero-shot priors over feature names
(generator_spec.pdf, measuring priors).

The model scores random profiles with no demonstrations: values
x ~ N(0, 1) in z-units, rounded to 2 decimals as in every prompt, with names
drawn from a lexicon and placed in random columns, so that name and position
are uncorrelated. A ridge regression of the zero-shot margin
m0 = log P("1") - log P("0") on the named values,

    m0(x; N) ~ b0 + sum_j b_{N(j)} x_j,

gives every name a prior slope: its sign is the direction the model expects
and |b| its strength. This linear model is the prior surrogate; its fit is
the out-of-fold R^2 (5 folds). In the abstract condition the names are the
column names f0 to f9, so the slopes describe a purely numeric prior, such as
the "bigger values mean 1" tendency found in P2.

The screening admits a pair (p, q) when b_p > 0 > b_q, both bootstrap
intervals exclude 0, and the weaker slope is at least half the stronger, so
flipping a name changes the prior's direction but not its strength. A weak
name is admitted when |b| is at most a quarter of the median |b| of the
admitted pair members. Thresholds are in configs/lexicons.yaml.

`PriorSurrogate` serves the fitted slopes to the grid: the counter-prior
strategy and the prior-data conflict use the intercept-free surrogate
sum_j b_{N(j)} x_j. The intercept is the model's label bias, which
contextual calibration removes and AUROC ignores; with it included the
surrogate would predict one label for almost every row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.naming import Lexicon
from src.data.serialisation import serialise_row

N_FEATURES = 10
N_PAIRS_PER_PROFILE = 3


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #
def random_profiles(n: int, lexicon: Lexicon | None, rng: np.random.Generator,
                    n_features: int = N_FEATURES) -> pd.DataFrame:
    """`n` profiles: display names per column and values (z-units, 2 decimals).

    With a lexicon, each profile names 3 columns from 3 distinct pairs (a
    random member of each) and the other columns with distinct weak names, in
    random positions, as in a named task prompt. Without one (abstract), the
    names are f0 to f{n-1} in order.
    """
    rows = []
    for i in range(n):
        values = np.round(rng.normal(size=n_features), 2)
        if lexicon is None:
            names = [f"f{j}" for j in range(n_features)]
        else:
            pairs = rng.choice(len(lexicon.pairs), size=N_PAIRS_PER_PROFILE, replace=False)
            members = rng.integers(0, 2, size=N_PAIRS_PER_PROFILE)
            weak = rng.choice(len(lexicon.weak), size=n_features - N_PAIRS_PER_PROFILE, replace=False)
            chosen = [lexicon.pairs[p][m] for p, m in zip(pairs, members)] + [lexicon.weak[w] for w in weak]
            names = [chosen[o] for o in rng.permutation(n_features)]
        rows.append({"profile": i, "names": names, "values": values.tolist()})
    return pd.DataFrame(rows)


def profile_line(names: list[str], values: list[float]) -> str:
    return serialise_row(dict(zip(names, values)))


def measure_margins(runner, system: str, profiles: pd.DataFrame, chunk: int = 100) -> pd.DataFrame:
    """Zero-shot label log-probabilities for every profile (one shared cached prefix)."""
    lines = [profile_line(n, v) for n, v in zip(profiles["names"], profiles["values"])]
    full0 = runner.render(system, [], lines[0])
    prefix = full0[: full0.rfind(lines[0])]
    suffixes = []
    for line in lines:
        full = runner.render(system, [], line)
        if not full.startswith(prefix):
            raise AssertionError("zero-shot prompts must share the prefix up to the query line")
        suffixes.append(full[len(prefix):])
    lp0, lp1 = [], []
    for start in range(0, len(suffixes), chunk):
        for r in runner.score(prefix, suffixes[start:start + chunk]):
            lp0.append(r.logprob_0)
            lp1.append(r.logprob_1)
    out = profiles.copy()
    out["logprob_0"], out["logprob_1"] = lp0, lp1
    out["margin"] = out["logprob_1"] - out["logprob_0"]
    return out


# --------------------------------------------------------------------------- #
# the surrogate
# --------------------------------------------------------------------------- #
def design_matrix(profiles: pd.DataFrame, vocab: list[str]) -> np.ndarray:
    """Z[i, name] = the value shown under that name in profile i (0 if absent)."""
    col = {n: j for j, n in enumerate(vocab)}
    Z = np.zeros((len(profiles), len(vocab)))
    for i, (names, values) in enumerate(zip(profiles["names"], profiles["values"])):
        for n, v in zip(names, values):
            Z[i, col[n]] = v
    return Z


def fit_ridge(Z: np.ndarray, m: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """Ridge with an unpenalised intercept (closed form on centred data)."""
    zm, mm = Z.mean(axis=0), m.mean()
    Zc = Z - zm
    beta = np.linalg.solve(Zc.T @ Zc + alpha * np.eye(Z.shape[1]), Zc.T @ (m - mm))
    return float(mm - zm @ beta), beta


def cv_r2(Z: np.ndarray, m: np.ndarray, alpha: float, n_folds: int = 5, seed: int = 0) -> float:
    """Out-of-fold R^2 (pooled over folds)."""
    folds = np.random.default_rng(seed).permutation(len(m)) % n_folds
    pred = np.empty(len(m))
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        b0, b = fit_ridge(Z[tr], m[tr], alpha)
        pred[te] = b0 + Z[te] @ b
    return float(1 - np.sum((m - pred) ** 2) / np.sum((m - m.mean()) ** 2))


def fit_surrogate(profiles: pd.DataFrame, vocab: list[str], alpha: float = 1.0,
                  n_bootstrap: int = 500, seed: int = 0) -> dict:
    """Slopes per name, with bootstrap intervals and the out-of-fold R^2."""
    Z = design_matrix(profiles, vocab)
    m = profiles["margin"].to_numpy(dtype=float)
    b0, beta = fit_ridge(Z, m, alpha)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_bootstrap, len(vocab)))
    for r in range(n_bootstrap):
        idx = rng.integers(0, len(m), size=len(m))
        boot[r] = fit_ridge(Z[idx], m[idx], alpha)[1]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    resid = m - (b0 + Z @ beta)
    return {
        "intercept": b0,
        "slopes": dict(zip(vocab, beta.tolist())),
        "slope_se": dict(zip(vocab, boot.std(axis=0).tolist())),
        "slope_ci": {n: [float(a), float(b)] for n, a, b in zip(vocab, lo, hi)},
        "n_shown": {n: int(c) for n, c in zip(vocab, (Z != 0).sum(axis=0))},
        "r2_cv": cv_r2(Z, m, alpha, seed=seed),
        "r2_train": float(1 - np.sum(resid ** 2) / np.sum((m - m.mean()) ** 2)),
        "residual_sd": float(resid.std()),
        "margin_mean": float(m.mean()), "margin_sd": float(m.std()),
        "label_mass_median": float(np.median(np.exp(profiles["logprob_0"]) + np.exp(profiles["logprob_1"]))),
        "share_predicted_1": float(np.mean(m > 0)),
        "n_profiles": int(len(m)), "ridge_alpha": alpha,
    }


def screen_lexicon(fit: dict, lexicon: Lexicon, rules: dict) -> dict:
    """Apply the pre-registered screening rules to one domain's fit."""
    s, ci = fit["slopes"], fit["slope_ci"]
    pairs = []
    for pos, neg in lexicon.pairs:
        bp, bq = s[pos], s[neg]
        strength = min(abs(bp), abs(bq)) / max(abs(bp), abs(bq), 1e-12)
        signs = bp > 0 > bq
        excl = ci[pos][0] > 0 and ci[neg][1] < 0
        pairs.append({"pair": [pos, neg], "slope_pos": bp, "slope_neg": bq, "strength_ratio": strength,
                      "min_abs_slope": min(abs(bp), abs(bq)), "signs_ok": bool(signs),
                      "intervals_exclude_zero": bool(excl),
                      "admitted": bool(signs and excl and strength >= rules["pair_min_strength_ratio"])})
    admitted = [p for p in pairs if p["admitted"]]
    ref = float(np.median([abs(v) for p in admitted for v in (p["slope_pos"], p["slope_neg"])])) if admitted else float("nan")
    weak = [{"name": n, "slope": s[n], "abs_slope_relative": abs(s[n]) / ref if admitted else float("nan"),
             "admitted": bool(admitted and abs(s[n]) <= rules["weak_max_relative_slope"] * ref)}
            for n in lexicon.weak]
    final_pairs = sorted(admitted, key=lambda p: -p["min_abs_slope"])[: rules["max_pairs"]]
    final_weak = sorted([w for w in weak if w["admitted"]], key=lambda w: abs(w["slope"]))[: rules["n_weak"]]
    return {
        "pairs": pairs, "weak": weak, "reference_abs_slope": ref,
        "final": {"pairs": [p["pair"] for p in final_pairs], "weak": [w["name"] for w in final_weak]},
        "enough": len(final_pairs) >= N_PAIRS_PER_PROFILE and len(final_weak) >= rules["n_weak"],
    }


@dataclass
class PriorSurrogate:
    """A model's fitted prior slopes, by condition ("abstract" or a domain)."""

    model: str
    fits: dict

    @classmethod
    def load(cls, path: str | Path) -> "PriorSurrogate":
        with open(path) as f:
            raw = json.load(f)
        return cls(raw["model"], raw["fits"])

    def column_slopes(self, naming: str, domain: str, names: dict[str, str]) -> dict[str, float]:
        """Slope of every displayed column under a naming (names from `assign_names`)."""
        slopes = self.fits["abstract" if naming == "abstract" else domain]["slopes"]
        return {c: float(slopes[n]) for c, n in names.items()}


def task_conflicts(entries: list[dict], frames: dict, surrogate: PriorSurrogate, lexicons: dict,
                   base_seed: int, namings: tuple[str, ...] = ("abstract", "aligned", "flipped")) -> pd.DataFrame:
    """Prior-data conflict C_t of every task's pool under every naming (intercept-free surrogate)."""
    from src.data.naming import assign_names
    from src.data.synthetic_bridge import FEATURE_NAMES, pool_of
    from src.selection.protocols import ProtocolContext, prior_data_conflict

    rows = []
    for e in entries:
        pool = pool_of(frames[e["task_id"]])
        for naming in namings:
            names = assign_names(e, naming, lexicons.get(e["domain"]), base_seed)
            ctx = ProtocolContext(feature_cols=FEATURE_NAMES, kinds={f: "numeric" for f in FEATURE_NAMES},
                                  train_ref=pool, prior_slopes=surrogate.column_slopes(naming, e["domain"], names))
            rows.append({"task_id": e["task_id"], "family": e["family"], "domain": e["domain"],
                         "spurious_direction": e["spurious_direction"], "naming": naming,
                         "conflict": prior_data_conflict(pool, ctx)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# measured priors on the pool rows
# --------------------------------------------------------------------------- #
def measure_pool_prior(grid, task_id: str, naming: str) -> pd.DataFrame:
    """Zero-shot label log-probabilities of every pool row of a task under a
    naming (the same system message and names as the grid's prompts). Each
    row is shown as it would appear as a demonstration, so under a demo mask
    (notebook 03.1) without the hidden features."""
    from src.data.naming import codebook
    from src.data.synthetic_bridge import pool_of

    pool = pool_of(grid.frame(task_id))
    cb = codebook(grid.names(task_id, naming))
    system = grid.system_text(task_id, naming)
    shown = grid.visible(task_id)
    lines = [serialise_row({f: r[f] for f in shown}, codebook=cb) for _, r in pool.iterrows()]
    full0 = grid.runner.render(system, [], lines[0])
    prefix = full0[: full0.rfind(lines[0])]
    suffixes = []
    for line in lines:
        full = grid.runner.render(system, [], line)
        if not full.startswith(prefix):
            raise AssertionError("zero-shot prompts must share the prefix up to the query line")
        suffixes.append(full[len(prefix):])
    res = grid.runner.score(prefix, suffixes)
    return pd.DataFrame({"task_id": task_id, "naming": naming, "row_id": pool.index.to_numpy(),
                         "label": pool["label"].to_numpy(), "y_clean": pool["y_clean"].to_numpy(),
                         "logprob_0": [r.logprob_0 for r in res], "logprob_1": [r.logprob_1 for r in res]})


@dataclass
class PoolPrior:
    """A model's measured zero-shot margins on every pool row, by (task, naming).

    `centred` subtracts the median over the task's pool, so the prior
    predicts label 1 for half the rows: the model's label bias is removed,
    as contextual calibration and AUROC remove it elsewhere.
    """

    table: pd.DataFrame

    @classmethod
    def load(cls, path: str | Path) -> "PoolPrior":
        return cls(pd.read_parquet(path))

    def centred(self, task_id: str, naming: str) -> pd.Series:
        t = self.table[(self.table["task_id"] == task_id) & (self.table["naming"] == naming)]
        if t.empty:
            raise KeyError(f"no measured pool prior for {task_id} under {naming!r}")
        m = (t["logprob_1"] - t["logprob_0"]).to_numpy()
        return pd.Series(m - np.median(m), index=t["row_id"].to_numpy())
