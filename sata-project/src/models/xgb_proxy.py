"""XGBoost proxy fit/predict, shared by src/models/sata_train.py's proxy
validation and scripts/gate_s3a_oracle_check.py's target-oracle sanity
check. Kept torch-free (only numpy + xgboost) so Gate S3a can run without
installing torch at all -- it bypasses the SATA model entirely.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier


def fit_predict_one(demo_X_topk: np.ndarray, demo_y_topk: np.ndarray, query_row: np.ndarray, query_label) -> int:
    """Fit XGBoost on one query's top-k demos and check the prediction.

    `n_jobs=1` is deliberate: this is called from inside a ThreadPoolExecutor
    (see evaluate_sata_proxy), so each individual fit must stay
    single-threaded -- letting XGBoost also spawn its own OMP thread pool
    per call would oversubscribe the allocated cores and run slower than
    either alone.
    """
    if len(np.unique(demo_y_topk)) < 2:
        # After balanced top-k (src/selection/balanced_topk.py) this should
        # be unreachable -- v1's fallback here (silently predict the lone
        # class) is exactly the reward path that let a degenerate
        # label-copy policy pass Gate 2 at 0.961 proxy accuracy
        # (REDESIGN_RATIONALE.md §4.2 Fact C). Raise loudly instead so a
        # regression in the balancing logic is caught immediately.
        raise ValueError(
            f"fit_predict_one: top-k demo selection was single-class (labels={demo_y_topk}) "
            "-- balanced top-k should make this unreachable; check this task/environment's label balance."
        )
    clf = XGBClassifier(max_depth=4, n_estimators=100, verbosity=0, n_jobs=1)
    clf.fit(demo_X_topk, demo_y_topk)
    pred = clf.predict(query_row[None, :])[0]
    return int(pred == query_label)


def fit_predict_one_soft(demo_X_topk: np.ndarray, demo_y_topk: np.ndarray, query_row: np.ndarray, query_label) -> int:
    """Same as `fit_predict_one` but with the old majority-class fallback for
    a single-class top-k -- for hand-designed baseline protocols that don't
    guarantee class balance by construction (e.g. plain random-k), where an
    occasional single-class draw is an expected property of the protocol,
    not a bug to raise loudly about.
    """
    if len(np.unique(demo_y_topk)) < 2:
        return int(demo_y_topk[0] == query_label)
    clf = XGBClassifier(max_depth=4, n_estimators=100, verbosity=0, n_jobs=1)
    clf.fit(demo_X_topk, demo_y_topk)
    pred = clf.predict(query_row[None, :])[0]
    return int(pred == query_label)
