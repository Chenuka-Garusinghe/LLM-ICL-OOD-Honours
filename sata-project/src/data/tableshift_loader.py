"""TableShift data loading + preprocessing (Notebook 01).

Candidate datasets (ranked by published shift gap and public accessibility).
Names are the exact keys registered in tableshift.configs.benchmark_configs
(verified against the installed tableshift source — do not rename these):
  - acsincome            geographic shift, predict income >= $50k across US states
  - acspubcov            demographic shift (ACS Public Coverage)
  - brfss_diabetes       temporal/geographic shift
  - anes                 temporal shift (ANES Voting)

Selection criteria: public access (no credentialed/MIMIC-derived data),
binary classification, <=15 usable features after reduction, nontrivial
published shift gap.

Final selection (all 4 verified to load end-to-end via load_tableshift_splits):
  brfss_diabetes, acsincome, acspubcov, anes.

Unlike the other three, `anes` is an OfflineDataSource that TableShift
cannot auto-download: it requires manually registering at
electionstudies.org, downloading the Time Series Cumulative Data File, and
placing it as anes_timeseries_cdf_csv_20220916.csv under the cache dir
(tableshift hardcodes this filename/date regardless of the actual release
downloaded — renaming a newer release's CSV to match works fine as long as
the VCF* columns tableshift.datasets.anes.ANES_FEATURES expects are present).

IMPORTANT: this module never imports `tableshift` at runtime. TableShift
hard-pins numpy==1.23.5 / ray==2.2 and its `xport` dependency breaks on
pandas>=3 — all incompatible with this project's modern stack (torch 2.x,
vllm, current pandas/numpy/sklearn). Raw datasets are extracted ONCE, in a
separate isolated environment, via scripts/extract_tableshift_cache.py, and
cached here as plain parquet files; this module only ever reads those. See
that script's docstring for the extraction instructions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

CANDIDATE_DATASETS = ["acsincome", "acspubcov", "brfss_diabetes", "anes"]
SELECTED_DATASETS = ["brfss_diabetes", "acsincome", "acspubcov", "anes"]

# Per dataset: (classification sentence, short noun phrase, meaning of label
# 0, meaning of label 1). The sentence goes in the classifier prompt ("predict
# {sentence}"); the noun phrase goes in the feature-ranking prompt (Notebook
# 03, "analyzing a {noun} prediction task"). Wording is from the target
# Feature's description in tableshift/datasets/{brfss,acs,anes}.py; label
# polarity was cross-checked against the observed P(label==1) base rate on
# the extracted data. Used by src/inference/prompts.py so the model is
# actually told what it is predicting instead of "the 'brfss_diabetes' outcome".
TASK_DESCRIPTIONS: dict[str, tuple[str, str, str, str]] = {
    "brfss_diabetes": (
        "whether this person has ever been told by a doctor that they have diabetes",
        "a diabetes diagnosis",
        "has not been told they have diabetes",
        "has been told they have diabetes",
    ),
    "acsincome": (
        "whether this person's total annual income is above $50,000",
        "annual income above $50,000",
        "income is at or below $50,000",
        "income is above $50,000",
    ),
    "acspubcov": (
        "whether this person is covered by public health insurance",
        "public health insurance coverage",
        "not covered by public health insurance",
        "covered by public health insurance",
    ),
    "anes": (
        "whether this person voted in the national election",
        "turnout in the national election",
        "did not vote",
        "voted",
    ),
}


def default_raw_cache_dir() -> str:
    """Parquet cache produced by scripts/extract_tableshift_cache.py."""
    from src.utils.config import PROJECT_ROOT

    return str(PROJECT_ROOT / "data" / "tableshift_raw_cache")


def load_tableshift_splits(dataset_name: str, cache_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Load train / ID-test / OOD-test splits from the cached parquet files.

    Returns a dict with keys "train", "test_id", "test_ood", each holding a
    DataFrame with a "label" column plus raw feature columns.

    Feature columns are all numeric (continuous features in natural units,
    categorical features as small integer codes); pair with load_codebook()
    to render them as text. The "label" column is coerced to plain `int`
    here regardless of how the parquet stored it — a float label stringifies
    to the 3-token "0.0", which the single-token classifier in
    src/inference/llm_runner.py can never match against the model's one-token
    "0"/"1" output (100% INVALID / NaN accuracy in Notebook 02).

    Does NOT import `tableshift` — the cache must already exist (run
    `python scripts/extract_tableshift_cache.py` from a separate environment
    with `tableshift` installed first; see that script's docstring for why).
    """
    cache_root = Path(cache_dir or default_raw_cache_dir()) / dataset_name

    splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "test_id", "test_ood"):
        path = cache_root / f"{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/extract_tableshift_cache.py {dataset_name}` "
                "from a separate, isolated environment with `tableshift` installed first — this "
                "project's own environment never installs tableshift. See that script's docstring."
            )
        df = pd.read_parquet(path)
        label = pd.to_numeric(df["label"], errors="coerce")
        if label.isna().any():
            raise ValueError(
                f"{path}: {int(label.isna().sum())} of {len(label)} rows have a "
                "missing or non-numeric 'label' after parsing."
            )
        df["label"] = label.round().astype(int)
        splits[split_name] = df
    return splits


def load_codebook(artifacts_dir: str | Path) -> dict:
    """Read a codebook.json produced by scripts/extract_tableshift_cache.py
    (or the per-dataset subset written by save_dataset_artifacts).

    Shape: {column: {"name_extended": str, "description": str,
                     "kind": "numeric" | "categorical",
                     "values": {int_code: text} | None}}.

    JSON object keys are always strings, so the `values` maps arrive as
    {"0": ...}; this restores integer keys so src/data/serialisation.py can
    look them up by the integer actually stored in the parquet.
    """
    with open(Path(artifacts_dir) / "codebook.json") as f:
        codebook = json.load(f)
    for entry in codebook.values():
        if entry.get("values"):
            entry["values"] = {int(k): v for k, v in entry["values"].items()}
    return codebook


def select_top_features(train_df: pd.DataFrame, n_features: int = 12, mi_sample_size: int = 5000) -> list[str]:
    """Select top-N feature columns by mutual information with the label
    on the training split. Document which features were kept in the caller
    (feature_list.json) so the choice is auditable.

    `mutual_info_classif` is single-threaded with no `n_jobs`, and its
    continuous-feature k-NN estimator is expensive per row — on TableShift's
    ACS-derived datasets (hundreds of thousands of rows) it dominates this
    notebook's wall time. The MI ranking is only used to pick which features
    survive, not the eventual (256-row) demo pool, so it's computed on a
    fixed-seed subsample rather than the full training split.
    """
    feature_cols = [c for c in train_df.columns if c != "label"]
    mi_df = (
        train_df.sample(n=mi_sample_size, random_state=0)
        if len(train_df) > mi_sample_size
        else train_df
    )
    X = mi_df[feature_cols].apply(lambda col: col.astype("category").cat.codes if col.dtype == "object" else col)
    mi = mutual_info_classif(X.fillna(X.median()), mi_df["label"], random_state=0)
    ranked = sorted(zip(feature_cols, mi), key=lambda t: -t[1])
    return [name for name, _ in ranked[:n_features]]


def impute_missing(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Mode imputation for categorical columns, median for continuous."""
    out = df.copy()
    for col in feature_cols:
        if out[col].dtype == "object" or out[col].dtype.name == "category":
            mode = out[col].mode(dropna=True)
            out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else "missing")
        else:
            out[col] = out[col].fillna(out[col].median())
    return out


def build_demo_pool(train_df: pd.DataFrame, pool_size: int, seed: int) -> pd.DataFrame:
    """Stratified sample of `pool_size` rows from the training split.

    This pool is fixed across all conditions and seeds for a given
    (dataset, seed) — only the *selection from* the pool varies per method.
    """
    rng = np.random.default_rng(seed)
    per_class = pool_size // train_df["label"].nunique()
    parts = []
    for _, group in train_df.groupby("label"):
        idx = rng.choice(group.index, size=min(per_class, len(group)), replace=False)
        parts.append(group.loc[idx])
    pool = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return pool


def save_dataset_artifacts(
    dataset_name: str,
    train_pool: pd.DataFrame,
    test_id: pd.DataFrame,
    test_ood: pd.DataFrame,
    feature_list: list[str],
    label_tokens: list[str],
    out_root: str | Path,
    codebook: dict | None = None,
) -> None:
    """Write train_pool/test_id/test_ood parquets + feature_list/label_tokens JSON.

    If `codebook` (the full raw-cache codebook from load_codebook()) is given,
    the subset covering `feature_list` is written next to the parquets as
    codebook.json, so Notebooks 02/03 never need to touch the raw cache at
    prompt time.
    """
    out_dir = Path(out_root) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pool.to_parquet(out_dir / "train_pool.parquet", index=False)
    test_id.to_parquet(out_dir / "test_id.parquet", index=False)
    test_ood.to_parquet(out_dir / "test_ood.parquet", index=False)

    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(sorted(feature_list), f, indent=2)
    with open(out_dir / "label_tokens.json", "w") as f:
        json.dump(label_tokens, f, indent=2)
    if codebook is not None:
        subset = {c: codebook[c] for c in feature_list if c in codebook}
        with open(out_dir / "codebook.json", "w") as f:
            json.dump(subset, f, indent=2, default=str)
