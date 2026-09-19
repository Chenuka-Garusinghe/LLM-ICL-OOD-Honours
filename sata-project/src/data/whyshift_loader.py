"""WHYSHIFT (ACS PUMS via folktables) data loading + DISDE shift screening
(Notebook 01).

Two tasks, each with one source state and several candidate target states
(see scripts/extract_whyshift_cache.py for the exact lists and extraction
details). Unlike TableShift -- whose "concept shift" metric turns out to be
covariate shift under another name (../TABLESHIFT_AUDIT.md) -- WHYSHIFT
(Liu et al., NeurIPS 2023) gives each (task, target_state) pair a DISDE
decomposition of the source->target performance degradation into a
covariate (P(X)) component and a concept (P(Y|X)) component. So here, the
shift type of a given real-world pair is a measured quantity, not an
assumption or a benchmark's own unverified label.

This module never downloads anything -- cached parquet files are produced
once by scripts/extract_whyshift_cache.py. `screen_disde_shifts` additionally
needs the `whyshift` and `xgboost` packages (both plain-Python-installable,
no environment conflict like TableShift's `tableshift` package).

Reuses from tableshift_loader.py rather than duplicating (all fully generic
over any DataFrame + codebook, not TableShift-specific): load_codebook,
select_top_features, impute_missing, build_demo_pool, save_dataset_artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WHYSHIFT_DATASETS = ["whyshift_income", "whyshift_pubcov"]

# Same 4-tuple convention as tableshift_loader.TASK_DESCRIPTIONS: (classification
# sentence, short noun phrase, meaning of label 0, meaning of label 1).
WHYSHIFT_TASK_DESCRIPTIONS: dict[str, tuple[str, str, str, str]] = {
    "whyshift_income": (
        "whether this person's total annual income is above $50,000",
        "annual income above $50,000",
        "income is at or below $50,000",
        "income is above $50,000",
    ),
    "whyshift_pubcov": (
        "whether this person is covered by public health insurance",
        "public health insurance coverage",
        "not covered by public health insurance",
        "covered by public health insurance",
    ),
}


def default_whyshift_cache_dir() -> str:
    """Parquet cache produced by scripts/extract_whyshift_cache.py."""
    from src.utils.config import PROJECT_ROOT

    return str(PROJECT_ROOT / "data" / "whyshift_raw_cache")


def list_available_targets(task_name: str, cache_dir: str | None = None) -> list[str]:
    """Candidate target states already cached for `task_name`, read off the
    `test_ood_{STATE}.parquet` filenames scripts/extract_whyshift_cache.py wrote.
    """
    cache_root = Path(cache_dir or default_whyshift_cache_dir()) / task_name
    prefix = "test_ood_"
    return sorted(p.stem[len(prefix):] for p in cache_root.glob(f"{prefix}*.parquet"))


def load_whyshift_splits(
    task_name: str, target_state: str, cache_dir: str | None = None
) -> dict[str, pd.DataFrame]:
    """Load train / ID-test / OOD-test splits for one (task, target_state) pair.

    Same shape as tableshift_loader.load_tableshift_splits(): a dict with
    keys "train", "test_id", "test_ood", each with a "label" column plus raw
    feature columns (small integer codes for categorical features, natural
    units for numeric ones) -- pair with load_codebook() to render as text.
    """
    cache_root = Path(cache_dir or default_whyshift_cache_dir()) / task_name

    splits: dict[str, pd.DataFrame] = {}
    for split_name, filename in (
        ("train", "train.parquet"),
        ("test_id", "test_id.parquet"),
        ("test_ood", f"test_ood_{target_state}.parquet"),
    ):
        path = cache_root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/extract_whyshift_cache.py {task_name}` "
                "first -- see that script's docstring for the candidate target states."
            )
        df = pd.read_parquet(path)
        df["label"] = df["label"].astype(int)
        splits[split_name] = df
    return splits


def screen_disde_shifts(
    task_name: str,
    target_states: list[str] | None = None,
    cache_dir: str | None = None,
    data_sum: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    """DISDE decomposition (Liu et al., NeurIPS 2023) of the source->target
    degradation for each candidate target state, on the full raw feature set
    (before this project's own MI-based top-10 reduction -- the decomposition
    should reflect the real population shift, not an artefact of a later
    feature-selection step).

    Trains one quick XGBoost baseline on the source state's training split,
    then calls whyshift.degradation_decomp() against each target state. This
    baseline model is a diagnostic only, never used in the ICL experiments.

    Returns a DataFrame with columns: target_state, total_degradation,
    proportion_yx_shift (concept), proportion_x_shift (covariate) -- sorted
    by proportion_yx_shift descending, so the first row is the most
    concept-shift-dominant pair and the last is the most covariate-dominant
    (proportion_yx_shift < 0.5).
    """
    from whyshift import degradation_decomp
    from xgboost import XGBClassifier

    cache_root = Path(cache_dir or default_whyshift_cache_dir()) / task_name
    if target_states is None:
        target_states = list_available_targets(task_name, cache_dir)

    train_df = pd.read_parquet(cache_root / "train.parquet")
    feature_cols = [c for c in train_df.columns if c != "label"]
    train_df = train_df.dropna(subset=feature_cols).reset_index(drop=True)

    source_X = train_df[feature_cols].to_numpy(dtype=float)
    source_y = train_df["label"].to_numpy(dtype=int)

    model = XGBClassifier(eval_metric="logloss", random_state=seed)
    model.fit(source_X, source_y)

    rows = []
    for target_state in target_states:
        target_df = pd.read_parquet(cache_root / f"test_ood_{target_state}.parquet")
        target_df = target_df.dropna(subset=feature_cols).reset_index(drop=True)
        target_X = target_df[feature_cols].to_numpy(dtype=float)
        target_y = target_df["label"].to_numpy(dtype=int)

        p2p, q2q, p2s, s2q = degradation_decomp(source_X, source_y, target_X, target_y, model, data_sum=data_sum)
        total_degradation = p2p - q2q
        proportion_yx = (p2s - s2q) / total_degradation if total_degradation != 0 else float("nan")
        rows.append(
            {
                "target_state": target_state,
                "total_degradation": total_degradation,
                "proportion_yx_shift": proportion_yx,
                "proportion_x_shift": 1 - proportion_yx,
            }
        )

    return pd.DataFrame(rows).sort_values("proportion_yx_shift", ascending=False).reset_index(drop=True)
