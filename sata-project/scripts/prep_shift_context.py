"""Fit the deployment-legal shift context for each real-arm TableShift dataset.

Reads the extracted parquet cache (data/tableshift_raw_cache/<dataset>/), selects
features by mutual information exactly as Notebook 01 does, and fits the
estimands in src/selection/shift_estimation.py: the ID-vs-target domain
discriminator, the shrunk BBSE target-prior estimate, and the per-feature drift
table. Caches the fitted context so the protocol screen does not refit it.

Uses NO target-domain labels. `test_ood`'s label column is loaded only so the
screen can score predictions afterwards; nothing in the fitted context touches it.

Usage
-----
    cd sata-project
    PYTHONPATH=. python scripts/prep_shift_context.py --out ../_screen_cache
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.tableshift_loader import impute_missing, select_top_features
from src.selection.shift_estimation import (
    DomainShift,
    estimate_target_prior,
    feature_drift,
    fit_domain_discriminator,
    shift_proxy_features,
)

warnings.filterwarnings("ignore")

DATASETS = ["brfss_diabetes", "acsincome", "acspubcov", "anes"]


def build(cache_root: Path, dataset: str, n_features: int, subsample: int, seed: int) -> dict:
    d = cache_root / dataset
    splits = {s: pd.read_parquet(d / f"{s}.parquet") for s in ("train", "test_id", "test_ood")}
    for s, df in splits.items():
        df["label"] = pd.to_numeric(df["label"], errors="coerce").round().astype(int)

    codebook = json.load(open(d / "codebook.json"))
    kinds = {c: v.get("kind", "numeric") for c, v in codebook.items()}

    feats = select_top_features(splits["train"], n_features=n_features, mi_sample_size=5000)

    rng = np.random.default_rng(seed)
    def prep(df):
        idx = rng.choice(len(df), size=min(subsample, len(df)), replace=False)
        out = df.iloc[idx].reset_index(drop=True)
        return impute_missing(out, feats)[feats + ["label"]]

    train, test_id, test_ood = prep(splits["train"]), prep(splits["test_id"]), prep(splits["test_ood"])

    # --- shift context (no target labels) -------------------------------------
    disc, disc_scaler, dom_auc = fit_domain_discriminator(train, test_ood, feats, seed=seed)

    label_scaler = StandardScaler().fit(train[feats])
    label_clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=seed)
    label_clf.fit(label_scaler.transform(train[feats]), train["label"])

    prior_t, prior_raw, prior_ok, prior_why = estimate_target_prior(
        label_clf, label_scaler, test_id, test_ood, feats
    )
    drift = feature_drift(train, test_ood, feats, kinds)

    shift = DomainShift(
        feature_cols=feats, scaler=disc_scaler, discriminator=disc, auc=dom_auc,
        prior_source=float(train["label"].mean()), prior_target=prior_t,
        prior_target_raw=prior_raw, prior_reliable=prior_ok, prior_reason=prior_why,
        drift=drift,
    )

    # --- reference points: full-data supervised learner, ID vs OOD ------------
    p_id = label_clf.predict_proba(label_scaler.transform(test_id[feats]))[:, 1]
    p_ood = label_clf.predict_proba(label_scaler.transform(test_ood[feats]))[:, 1]
    reference = dict(
        acc_id=accuracy_score(test_id["label"], p_id > 0.5),
        acc_ood=accuracy_score(test_ood["label"], p_ood > 0.5),
        auc_id=roc_auc_score(test_id["label"], p_id),
        auc_ood=roc_auc_score(test_ood["label"], p_ood),
        majority_id=max(test_id["label"].mean(), 1 - test_id["label"].mean()),
        majority_ood=max(test_ood["label"].mean(), 1 - test_ood["label"].mean()),
    )

    return dict(
        dataset=dataset, feats=feats, kinds={f: kinds.get(f, "numeric") for f in feats},
        train=train, test_id=test_id, test_ood=test_ood, shift=shift,
        proxy_features=shift_proxy_features(drift, train), reference=reference,
        true_prior_ood=float(test_ood["label"].mean()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/tableshift_raw_cache")
    ap.add_argument("--out", default="_screen_cache")
    ap.add_argument("--n-features", type=int, default=12)
    ap.add_argument("--subsample", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for ds in DATASETS:
        ctx = build(Path(args.cache), ds, args.n_features, args.subsample, args.seed)
        with open(out / f"{ds}.pkl", "wb") as f:
            pickle.dump(ctx, f)
        s = ctx["shift"]
        summary.append(dict(
            dataset=ds, n_features=len(ctx["feats"]), domain_auc=s.auc,
            detectable=s.detectable, prior_source=s.prior_source,
            prior_target_est=s.prior_target, prior_target_bbse_raw=s.prior_target_raw,
            prior_reliable=s.prior_reliable, prior_reason=s.prior_reason,
            prior_target_true=ctx["true_prior_ood"],
            top_drift=ctx["shift"].drift.iloc[0]["feature"],
            proxy_features="|".join(ctx["proxy_features"]), **ctx["reference"],
        ))
        print(f"[prep] {ds}: domain_auc={s.auc:.3f} detectable={s.detectable} | "
              f"prior_est={s.prior_target:.3f} raw={s.prior_target_raw:.3f} "
              f"true={ctx['true_prior_ood']:.3f} reliable={s.prior_reliable} | {s.prior_reason}", flush=True)

    pd.DataFrame(summary).to_csv(out / "shift_context_summary.csv", index=False)


if __name__ == "__main__":
    main()
