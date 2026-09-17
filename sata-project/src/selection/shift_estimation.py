"""Deployment-legal shift estimation for shift-aware demonstration selection.

Motivation
----------
The v1 protocol suite (src/selection/{label_diversity,feature_range,
rule_diversity,counter_spurious}.py) is entirely a function of the *demo pool*.
None of its members look at the target domain, so none of them can adapt to a
shift. `counter_spurious` is the only one that tries, and it needs a
`shift_col` naming the domain variable -- which the extracted TableShift
parquet cache does not contain (the domain splitter column is consumed by
TableShift when it forms train/test_id/test_ood), so on the real arm that
protocol has no shift signal to use as written.

This module supplies the shift signal that protocols are allowed to condition
on at deployment time. The hard constraint is *what a deployed system can
actually see*: unlabelled target-domain inputs. Everything here is estimated
from (labelled ID pool, UNLABELLED target inputs). No target label is ever
read. That makes the resulting protocols shift-type-agnostic -- they do not
need to be told whether the shift is covariate, prior, or mechanism, which is
the setting the thesis actually cares about.

Three estimands, each mapping to a different failure mode:

  s(x), w(x)   domain-discriminator score and density ratio
               p_T(x)/p_S(x). Targets covariate shift and extrapolation:
               tells you which pool rows resemble the target region.
               Trustworthy only when `DomainShift.auc` is meaningfully
               above 0.5 -- below that the shift is not in p(x) at all and
               every w(x) is noise.

  pi_T         target label prior, via BBSE (Lipton et al. 2018, "Detecting
               and Correcting for Label Shift with Black Box Predictors").
               Targets prior/label shift. BBSE assumes p(x|y) is invariant;
               where that fails the estimate is badly biased (observed on
               acspubcov: 0.98 estimated vs 0.64 true), hence the shrinkage
               in `estimate_target_prior`.

  per-feature  standardised mean difference / total-variation distance.
  drift        Diagnostic + feature ranking for `counter_spurious`, and the
               replacement for its missing `shift_col`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DomainShift:
    """Fitted, deployment-legal description of the source -> target shift."""

    feature_cols: list[str]
    scaler: StandardScaler
    discriminator: HistGradientBoostingClassifier
    auc: float                       # held-out ID-vs-target AUC; 0.5 => no detectable p(x) shift
    prior_source: float              # P(y=1) on the labelled source pool
    prior_target: float              # shrunk BBSE estimate of P(y=1) on target
    prior_target_raw: float          # ungated BBSE estimate, for reporting
    prior_reliable: bool             # False => prior_target fell back to prior_source
    prior_reason: str                # why the gate passed or fired
    drift: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def detectable(self) -> bool:
        """Whether p(x) shift is strong enough for w(x) to carry signal."""
        return self.auc >= 0.60

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """P(target | x) in [0, 1]."""
        return self.discriminator.predict_proba(self.scaler.transform(X[self.feature_cols]))[:, 1]

    def weight(self, X: pd.DataFrame, clip: float = 20.0) -> np.ndarray:
        """Density ratio p_T(x)/p_S(x) = s/(1-s) for a balanced discriminator.

        Clipped because an unclipped ratio lets a handful of pool rows absorb
        all of the sampling mass, which at k=8 collapses the demo set onto a
        near-duplicate cluster.
        """
        s = np.clip(self.score(X), 1e-6, 1 - 1e-6)
        return np.clip(s / (1.0 - s), 1.0 / clip, clip)


def fit_domain_discriminator(
    source_X: pd.DataFrame,
    target_X: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 0,
    max_n: int = 40_000,
) -> tuple[HistGradientBoostingClassifier, StandardScaler, float]:
    """Discriminate source inputs from target inputs. Uses no labels at all.

    Classes are balanced by subsampling to the smaller domain so that
    s/(1-s) is the density ratio without a base-rate correction term.
    """
    rng = np.random.default_rng(seed)
    n = min(len(source_X), len(target_X), max_n)
    src = source_X[feature_cols].iloc[rng.choice(len(source_X), n, replace=False)]
    tgt = target_X[feature_cols].iloc[rng.choice(len(target_X), n, replace=False)]

    X = pd.concat([src, tgt], ignore_index=True)
    y = np.r_[np.zeros(n), np.ones(n)]
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state=seed, stratify=y)
    disc = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=seed)
    disc.fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, disc.predict_proba(X_te)[:, 1]))
    return disc, scaler, auc


def estimate_target_prior(
    label_clf,
    scaler: StandardScaler,
    source_holdout: pd.DataFrame,
    target_X: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    boundary_margin: float = 0.05,
    max_cond: float = 20.0,
) -> tuple[float, float, bool, str]:
    """BBSE estimate of P_T(y=1), with an identifiability gate.

    Returns (prior_used, prior_raw, reliable, reason).

    BBSE (Lipton et al. 2018) solves q = A p for p, where A[j, i] =
    P(pred = j | y = i) measured on a labelled *source* holdout and q is the
    predicted-label distribution on unlabelled target inputs. It is consistent
    under pure label shift (p(x|y) invariant) and biased when that fails.

    Rather than blend a possibly-biased estimate toward the source prior --
    which taxes the datasets where BBSE is *right* in order to partially
    protect the one where it is wrong -- this gates on two detectable failure
    signatures and falls back to the source prior when either fires:

      * the solution sits on the simplex boundary. An interior p is what a
        genuine prior shift produces; a solve pinned at ~1.0 means the linear
        system had no valid interior solution, i.e. the observed target
        prediction distribution is not reachable by *any* reweighting of the
        source class-conditionals. That is precisely the assumption violation.
        (Fires on acspubcov: raw 0.999 against a true 0.637.)

      * A is ill-conditioned -- a classifier too weak or too one-sided on the
        source holdout for the inversion to be stable.

    The fallback is not a failure of the protocol: "the target prior is not
    identifiable from unlabelled inputs here" is a reportable finding, and it
    leaves `target_prior` composition degrading to the source prior rather
    than to a confidently wrong number.
    """
    pred_src = label_clf.predict(scaler.transform(source_holdout[feature_cols]))
    Cm = confusion_matrix(source_holdout[label_col], pred_src, labels=[0, 1])
    A = (Cm / np.maximum(Cm.sum(axis=1, keepdims=True), 1)).T      # A[pred, true]

    q = np.bincount(label_clf.predict(scaler.transform(target_X[feature_cols])), minlength=2) / len(target_X)
    p_raw = np.linalg.lstsq(A, q, rcond=None)[0]
    p_norm = np.clip(p_raw, 1e-6, None)
    p_norm = p_norm / p_norm.sum()
    raw = float(p_norm[1])

    cond = float(np.linalg.cond(A)) if np.linalg.matrix_rank(A) == 2 else np.inf
    on_boundary = bool(p_raw.min() < boundary_margin or p_raw.max() > 1 - boundary_margin)
    ill_conditioned = bool(not np.isfinite(cond) or cond > max_cond)

    prior_src = float(source_holdout[label_col].mean())
    if on_boundary:
        return prior_src, raw, False, f"BBSE solution on simplex boundary (raw={raw:.3f}); prior not identifiable"
    if ill_conditioned:
        return prior_src, raw, False, f"confusion matrix ill-conditioned (cond={cond:.1f})"
    return float(np.clip(raw, 0.02, 0.98)), raw, True, f"interior BBSE solution (cond={cond:.1f})"


def feature_drift(
    source_X: pd.DataFrame,
    target_X: pd.DataFrame,
    feature_cols: list[str],
    kinds: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-feature source->target drift: SMD for numeric, TV distance for categorical.

    `kinds` comes from the codebook (`kind` field). Anything not marked
    categorical is treated as numeric.
    """
    kinds = kinds or {}
    rows = []
    for f in feature_cols:
        a, b = source_X[f].dropna(), target_X[f].dropna()
        if kinds.get(f) == "categorical" or a.nunique() <= 10:
            cats = sorted(set(a.unique()) | set(b.unique()))
            pa = a.value_counts(normalize=True).reindex(cats).fillna(0.0)
            pb = b.value_counts(normalize=True).reindex(cats).fillna(0.0)
            stat, kind = 0.5 * float(np.abs(pa - pb).sum()), "tv"
        else:
            pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) + 1e-12
            stat, kind = float(abs(a.mean() - b.mean()) / pooled), "smd"
        rows.append(dict(feature=f, drift=stat, statistic=kind))
    return pd.DataFrame(rows).sort_values("drift", ascending=False).reset_index(drop=True)


def shift_proxy_features(
    drift: pd.DataFrame,
    pool: pd.DataFrame,
    label_col: str = "label",
    top_n: int = 3,
) -> list[str]:
    """Features that both drift and predict the label -- the shortcut candidates.

    This is the replacement for `counter_spurious.find_spurious_proxy_features`'s
    `shift_col` argument: instead of correlating each feature with a domain
    column that the parquet cache does not have, rank by
    |corr(feature, label)| * drift(feature). A feature that moves across
    domains *and* predicts the label in-domain is exactly the shortcut a
    demonstration set should be designed to break.
    """
    d = drift.set_index("feature")["drift"]
    scores = {}
    for f in d.index:
        if f not in pool.columns:
            continue
        c = pool[f].corr(pool[label_col])
        if pd.isna(c):
            continue
        scores[f] = abs(float(c)) * float(d[f])
    return [f for f, _ in sorted(scores.items(), key=lambda t: -t[1])[:top_n]]
