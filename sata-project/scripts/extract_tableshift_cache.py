"""One-time TableShift raw-data extraction.

Run this with a SEPARATE Python environment that has `tableshift` installed
(clone github.com/mlfoundations/tableshift and `pip install -e . --no-deps`,
then install its runtime deps manually) -- NOT this project's main .venv.
On Gadi that environment already exists at
/scratch/pp49/cg3543/tmp/tableshift_extract_env (py3.9, numpy 1.22 / pandas
1.3, tableshift editable-installed from /scratch/pp49/cg3543/tmp/tableshift_src).

Why isolated: TableShift hard-pins numpy==1.23.5 / ray==2.2 and its `xport`
dependency breaks on pandas>=3, all of which conflict with this project's
modern stack (torch 2.x, vllm, current pandas/numpy/sklearn). Fighting that
version conflict once, in a throwaway environment, is far cheaper than
carrying tableshift's constraints into the main project's environment
permanently.

Usage (from that isolated environment, at the project root):
    python scripts/extract_tableshift_cache.py [dataset_name ...]
    # no args -> extracts all of CANDIDATE_DATASETS

Writes, per dataset, under data/tableshift_raw_cache/{dataset_name}/:
  - {train,test_id,test_ood}.parquet -- ALL columns numeric: continuous
    features in their natural units (NOT z-scored), categorical features as
    small integer codes (0..K), plus an integer `label` column.
  - codebook.json -- {column: {name_extended, description, kind, values}}
    where `values` maps each integer code to human-readable text for
    categorical columns (null for numeric). src/data/serialisation.py uses
    this to render prompts like
    "Body Mass Index (BMI) category: Obese; ... -> 1".
  - feature_list.jsonl -- raw dump of tableshift's own FeatureList for
    provenance (via FeatureList.to_jsonl).

WHY numeric + codebook rather than raw strings: every demo-selection
protocol in src/selection/ (counter_spurious's `> median`, rule_diversity's
DecisionTree.fit, feature_range, sata_select's `.to_numpy(float32)`) and
select_top_features require numeric feature columns. Keeping the parquet
numeric and doing the string rendering only at serialisation time means the
selection code is untouched by this change.

WHY not z-scored: an 8B LLM cannot reason over "BMI5: -0.55"; it can over
"Body Mass Index (BMI) category: Obese". A previous version of this script
called get_dataset() with no preprocessor_config, so TableShift's default
Preprocessor (StandardScaler on numerics + one-hot on categoricals) ran
before caching -- the cached parquets held z-scored values and opaque
one-hot column names (BMI5CAT_20, VCF0718_00). This version disables both
transforms via PreprocessorConfig(numeric_features="passthrough",
categorical_features="passthrough") and factorises the categoricals here.

`anes` is TableShift's one OfflineDataSource and won't auto-download; see
Notebook 01's Step 2 markdown for the manual-download instructions. On Gadi
the CSV lives at data/anes_timeseries_cdf_csv_20260205/; tableshift hardcodes
the filename anes_timeseries_cdf_csv_20220916.csv, so
data/tableshift_cache/anes_timeseries_cdf_csv_20220916/ must be a symlink to
it (see the deploy notes).
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "tableshift_raw_cache"
TABLESHIFT_DOWNLOAD_CACHE = PROJECT_ROOT / "data" / "tableshift_cache"

CANDIDATE_DATASETS = ["acsincome", "acspubcov", "brfss_diabetes", "anes"]

# BRFSS survey convention: for these day-count questions the code 88 means
# "None" (i.e. zero days), not a real 88. 77 ("Don't know") / 99 ("Refused")
# are already declared as na_values on the tableshift Feature and become NaN
# (imputed downstream by impute_missing). Only 88 needs the -> 0 remap.
_ZERO_CODED_NUMERIC = {"PHYSHLTH": 88.0, "MENTHLTH": 88.0}

_MISSING_TEXT = "(missing / not asked)"
_NA_STRINGS = {"", "nan", "none", "na", "n/a", "null"}


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip().lower() in _NA_STRINGS


def _patch_xport_underscore_fields() -> None:
    """xport.v56 builds a namedtuple from raw SAS column names without
    rename=True; BRFSS/ANES have underscore-prefixed names (e.g. "_STATE")
    that namedtuple rejects outright without this patch.

    Only applies to older xport releases (verified against the version
    pinned by tableshift's original, ~2023 requirements.txt) that build
    records via a bare `namedtuple` call in this module. xport 3.6.1
    rewrote v56's internals around proper classes (Namestr, MemberHeader,
    Observations, ...) and no longer references `namedtuple` there at all --
    on that version there's nothing to patch, and the underscore-field
    problem may not even apply, so just skip rather than crash.
    """
    import collections

    try:
        import xport.v56 as _xport_v56
    except ImportError:
        return

    if not hasattr(_xport_v56, "namedtuple"):
        return

    if getattr(_xport_v56.namedtuple, "_extract_script_patch", False):
        return

    _original_namedtuple = collections.namedtuple

    def _namedtuple_with_rename(typename, field_names, **kwargs):
        kwargs.setdefault("rename", True)
        return _original_namedtuple(typename, field_names, **kwargs)

    _namedtuple_with_rename._extract_script_patch = True
    _xport_v56.namedtuple = _namedtuple_with_rename


def _patch_domain_label_passthrough() -> None:
    """Preprocessor.fit_transform (tableshift/core/features.py, as of
    upstream commit fca9429) calls self.get_passthrough_columns(...)
    without forwarding its own domain_label_colname argument, so
    get_passthrough_columns never adds the domain label (e.g. "DIVISION"
    for ACS's geographic shift) to the passthrough list -- it gets dropped
    by the ColumnTransformer, and fit_transform's later unconditional
    `transformed.loc[:, domain_label_colname]` raises KeyError. This is a
    real upstream bug affecting every domain-split dataset, not just ACS.

    Reimplements fit_transform with that one argument forwarded, rather
    than patching the installed source file -- keeps the fix inside this
    script so it applies to any fresh `git clone` of upstream tableshift.
    """
    try:
        from tableshift.core.features import Preprocessor
    except ImportError:
        return

    if getattr(Preprocessor.fit_transform, "_extract_script_patch", False):
        return

    def fit_transform(self, data, train_idxs, domain_label_colname=None,
                      target_colname=None, passthrough_columns=None):
        """Fit a feature_transformer and apply it to the input features."""
        logging.info("transforming columns")
        if self.config.passthrough_columns == "all":
            logging.info("passthrough is 'all'; data will not be preprocessed "
                         "by tableshift.")
            if self.config.use_extended_names:
                logging.warning(
                    "passthrough is 'all' but "
                    "config.use_extended_names is True; extended "
                    "names are not applied when passthrough is 'all'. Try "
                    "setting numeric_columns='passthough', "
                    "categorical_columns='passthrough' instead.")
            return data

        passthrough_columns = self.get_passthrough_columns(
            data,
            passthrough_columns,
            domain_label_colname=domain_label_colname,
            target_colname=target_colname)

        dtypes_in = data.dtypes.to_dict()

        post_transform_cast_dtypes = (
            {c: dtypes_in[c] for c in passthrough_columns if
             c != domain_label_colname}
            if passthrough_columns else None)

        self._check_inputs(data)

        self.fit_feature_transformer(data, train_idxs, passthrough_columns)
        transformed = self.transform_features(data)

        transformed = self._post_transform(
            transformed, cast_dtypes=post_transform_cast_dtypes)

        if domain_label_colname:
            transformed.loc[:, domain_label_colname] = \
                self.fit_transform_domain_labels(
                    transformed.loc[:, domain_label_colname])
        self._post_transform_summary(transformed)
        logging.info("transforming columns complete.")
        return transformed

    fit_transform._extract_script_patch = True
    Preprocessor.fit_transform = fit_transform


def _patch_acs_incremental_year_loading() -> None:
    """ACSDataSource._get_acs_data() (tableshift/core/data_source.py) loads
    every year in self.years fully into memory -- all 51 states x
    person+household joined, full raw PUMS schema (hundreds of columns) --
    appending each year's complete frame to a list before concatenating,
    and only AFTER that does _load_data() reduce down to the task's actual
    ~15-20 predictor/target columns via ACSProblem.df_to_numpy(). For a
    single-year task (acsincome) that's fine; for a multi-year task like
    acspubcov (ACS_YEARS = 5 years), peak memory holds 5 full-schema years
    simultaneously and reliably OOM-kills even on machines with tens of
    GB of RAM (confirmed: reliably kills processes on a 36GB machine).

    Reimplements ACSDataSource._load_data() to reduce each year down to
    just its predictor/target columns immediately after loading it, before
    fetching the next year. Safe to do per-year rather than after
    concatenating: the task's preprocess filters (folktables.acs.adult_filter,
    public_coverage_filter, etc.) are simple row-wise filters with no
    cross-year/global statistics, so this is functionally identical to the
    original -- just with peak memory bounded by ~1 year's raw data instead
    of len(self.years) years' worth.

    Must still set year_data["ACS_YEAR"] = year per-year before reducing --
    it's a real predictor column (part of the feature set some tasks use),
    not vestigial; dropping it raises KeyError inside df_to_numpy().
    """
    try:
        from tableshift.core.data_source import (
            ACSDataSource, ACS_TASK_CONFIGS, acs_data_to_df,
            get_acs_data_source,
        )
        import folktables
        import pandas as pd
        from functools import partial
    except ImportError:
        return

    if getattr(ACSDataSource._load_data, "_extract_script_patch", False):
        return

    def _load_data(self):
        task_config = ACS_TASK_CONFIGS[self.acs_task]
        target_transform = partial(task_config.target_transform,
                                   threshold=task_config.threshold)
        acs_problem = folktables.BasicProblem(
            features=task_config.features_to_use.predictors,
            target=task_config.target,
            target_transform=target_transform,
            preprocess=task_config.preprocess,
            postprocess=task_config.postprocess,
        )

        year_dfs = []
        for year in self.years:
            logging.info(f"fetching ACS data for year {year}...")
            data_source = get_acs_data_source(year, self.cache_dir)
            year_data = data_source.get_data(states=self.states,
                                             join_household=True,
                                             download=True)
            year_data["ACS_YEAR"] = year
            X, y, _ = acs_problem.df_to_numpy(year_data)
            year_dfs.append(acs_data_to_df(
                X, y, task_config.features_to_use,
                feature_mapping=self.feature_mapping))
            del year_data
        logging.info("fetching ACS data complete.")
        return pd.concat(year_dfs, axis=0)

    _load_data._extract_script_patch = True
    ACSDataSource._load_data = _load_data


# --------------------------------------------------------------------------
# codebook construction
# --------------------------------------------------------------------------
def _cat_dtype():
    from pandas.api.types import CategoricalDtype
    return CategoricalDtype


def _clean_scalar(v) -> str:
    """Human-ish string for a raw category code with no value_mapping:
    2.0 / "2.0" -> "2", "-9.0" -> "-9", "02" -> "2", NaN -> missing text."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return _MISSING_TEXT
    s = str(v).strip()
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    return str(int(f)) if f.is_integer() else s


def _map_value(raw, value_mapping: dict | None) -> str:
    """Resolve one raw category code to display text via a tableshift
    Feature.value_mapping, tolerating key-type drift (mappings use str keys
    like "2.0", int keys like 2, and occasionally np.nan)."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        if value_mapping:
            for k in (np.nan, "nan", "NaN"):
                if k in value_mapping:
                    return str(value_mapping[k]).strip()
        return _MISSING_TEXT
    if not value_mapping:
        return _clean_scalar(raw)

    candidates = [raw, str(raw), _clean_scalar(raw)]
    try:
        f = float(raw)
        candidates += [f, str(f), int(f), str(int(f))]
    except (TypeError, ValueError):
        pass
    for key in candidates:
        if key in value_mapping:
            return str(value_mapping[key]).strip()
    return _clean_scalar(raw)


def _feature_list(dset):
    for attr in ("preprocessor", "_preprocessor"):
        pp = getattr(dset, attr, None)
        if pp is not None and getattr(pp, "feature_list", None) is not None:
            return pp.feature_list
    tc = getattr(dset, "task_config", None)
    if tc is not None and getattr(tc, "feature_list", None) is not None:
        return tc.feature_list
    raise RuntimeError("could not locate tableshift FeatureList on the dataset")


def _is_categorical(feature, series: pd.Series) -> bool:
    if feature is not None:
        if feature.kind == _cat_dtype():
            return True
        if getattr(feature, "value_mapping", None):
            return True
    return not pd.api.types.is_numeric_dtype(series)


def build_numeric_frames_and_codebook(splits: dict[str, pd.DataFrame], feature_list):
    """splits: {"train"/"test_id"/"test_ood": DataFrame with a `label` column
    plus raw feature columns}. Returns (numeric_splits, codebook)."""
    feats_by_name = {f.name: f for f in feature_list.features if not f.is_target}
    feature_cols = [c for c in splits["train"].columns if c != "label"]

    out = {k: pd.DataFrame(index=df.index) for k, df in splits.items()}
    codebook: dict[str, dict] = {}

    for col in feature_cols:
        feat = feats_by_name.get(col)
        joined = pd.concat([splits[k][col] for k in ("train", "test_id", "test_ood")],
                           ignore_index=True)
        name_ext = (getattr(feat, "name_extended", None) or col) if feat else col
        desc = (getattr(feat, "description", None) or "") if feat else ""

        if _is_categorical(feat, joined):
            vm = getattr(feat, "value_mapping", None) if feat else None
            # factorize usually sends NaN -> -1, but on a mixed-object column it
            # can keep NaN as a real level -- so force every originally-NA row
            # to -1 explicitly, and also fold any NaN level factorize returned.
            na_mask = joined.map(_is_missing).to_numpy()
            raw_codes, uniques = pd.factorize(joined)
            raw_codes = np.where(na_mask, -1, raw_codes)

            values: dict[int, str] = {}
            remap: dict[int, int] = {}
            next_code = 0
            for i, u in enumerate(uniques):
                if _is_missing(u):
                    remap[i] = -1
                    continue
                remap[i] = next_code
                values[next_code] = _map_value(u, vm)
                next_code += 1
            codes = np.array([-1 if c == -1 else remap[c] for c in raw_codes], dtype="int64")
            if (codes == -1).any():
                values[next_code] = _map_value(np.nan, vm)
                codes = np.where(codes == -1, next_code, codes)

            codebook[col] = {"name_extended": name_ext, "description": desc,
                             "kind": "categorical", "values": values}
            kind = "categorical"
        else:
            num = pd.to_numeric(joined, errors="coerce")
            if col in _ZERO_CODED_NUMERIC:
                num = num.replace(_ZERO_CODED_NUMERIC[col], 0.0)
            codes = num.to_numpy()
            codebook[col] = {"name_extended": name_ext, "description": desc,
                             "kind": "numeric", "values": None}
            kind = "numeric"

        # scatter the joined column back to the three splits, in order
        pos = 0
        for k in ("train", "test_id", "test_ood"):
            n = len(splits[k])
            seg = codes[pos:pos + n]
            pos += n
            if kind == "categorical":
                out[k][col] = pd.Series(seg, index=splits[k].index).astype("int64")
            else:
                out[k][col] = pd.Series(seg, index=splits[k].index).astype("float64")

    for k in ("train", "test_id", "test_ood"):
        lbl = pd.to_numeric(splits[k]["label"], errors="coerce")
        if lbl.isna().any():
            raise ValueError(f"{k}: {int(lbl.isna().sum())} rows have a "
                             "missing/non-numeric label")
        out[k]["label"] = lbl.round().astype("int64").to_numpy()

    return out, codebook


def extract_dataset(dataset_name: str) -> None:
    _patch_xport_underscore_fields()
    _patch_domain_label_passthrough()
    _patch_acs_incremental_year_loading()
    from tableshift import get_dataset
    from tableshift.core.features import PreprocessorConfig

    # Disable BOTH transforms so get_pandas() returns features in their
    # natural units with their original column names -- no StandardScaler,
    # no one-hot. NOT passthrough_columns="all" (that path silently skips
    # name handling) and NOT categorical_features="map_values" (that path is
    # strict and raises on any value absent from a Feature.value_mapping;
    # VCF9201/VCF9202/OCCP/ST have no mapping at all).
    dset = get_dataset(
        dataset_name,
        cache_dir=str(TABLESHIFT_DOWNLOAD_CACHE),
        preprocessor_config=PreprocessorConfig(
            numeric_features="passthrough",
            categorical_features="passthrough",
        ),
    )
    out_dir = RAW_CACHE_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_list = _feature_list(dset)
    try:
        feature_list.to_jsonl(str(out_dir / "feature_list.jsonl"))
    except Exception as e:  # provenance only -- never fail the extraction on it
        logging.warning(f"{dataset_name}: could not write feature_list.jsonl ({e})")

    raw_splits: dict[str, pd.DataFrame] = {}
    for out_name, tableshift_split in [("train", "train"),
                                       ("test_id", "id_test"),
                                       ("test_ood", "ood_test")]:
        X, y, _, _ = dset.get_pandas(tableshift_split)
        df = X.copy()
        df["label"] = y.values
        raw_splits[out_name] = df.reset_index(drop=True)

    numeric_splits, codebook = build_numeric_frames_and_codebook(raw_splits, feature_list)

    for out_name, df in numeric_splits.items():
        non_num = df.select_dtypes(exclude="number").columns.tolist()
        assert not non_num, f"{dataset_name}/{out_name}: non-numeric columns {non_num}"
        out_path = out_dir / f"{out_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"{dataset_name}/{out_name}: {len(df)} rows, {df.shape[1] - 1} features -> {out_path}")

    with open(out_dir / "codebook.json", "w") as f:
        json.dump(codebook, f, indent=2)
    n_cat = sum(1 for v in codebook.values() if v["kind"] == "categorical")
    print(f"{dataset_name}: codebook.json written ({n_cat} categorical / "
          f"{len(codebook) - n_cat} numeric features)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets = sys.argv[1:] or CANDIDATE_DATASETS
    for name in datasets:
        extract_dataset(name)
    print(f"\nDone. src/data/tableshift_loader.py can now read from {RAW_CACHE_DIR} "
          "without `tableshift` installed.")
