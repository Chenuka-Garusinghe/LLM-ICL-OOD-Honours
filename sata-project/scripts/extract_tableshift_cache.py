"""One-time TableShift raw-data extraction.

Run this with a SEPARATE Python environment that has `tableshift` installed
(clone github.com/mlfoundations/tableshift and `pip install -e . --no-deps`,
then install its runtime deps manually) -- NOT this project's main .venv.

Why isolated: TableShift hard-pins numpy==1.23.5 / ray==2.2 and its `xport`
dependency breaks on pandas>=3, all of which conflict with this project's
modern stack (torch 2.x, transformers, current pandas/numpy/sklearn). Fighting that
version conflict once, in a throwaway environment, is far cheaper than
carrying tableshift's constraints into the main project's environment
permanently.

Usage (from that isolated environment, at the project root):
    python scripts/extract_tableshift_cache.py [dataset_name ...]
    # no args -> extracts all of CANDIDATE_DATASETS

Writes data/tableshift_raw_cache/{dataset_name}/{train,test_id,test_ood}.parquet
(feature columns + a `label` column). Once this has run, the main project
environment never needs `tableshift` installed again --
src/data/tableshift_loader.py only ever reads these cached parquet files,
it does not import `tableshift` at runtime.

`anes` is TableShift's one OfflineDataSource and won't auto-download; see
Notebook 01's Step 2 markdown for the manual-download instructions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "tableshift_raw_cache"
TABLESHIFT_DOWNLOAD_CACHE = PROJECT_ROOT / "data" / "tableshift_cache"

CANDIDATE_DATASETS = ["acsincome", "acspubcov", "brfss_diabetes", "anes"]


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


def extract_dataset(dataset_name: str) -> None:
    _patch_xport_underscore_fields()
    _patch_domain_label_passthrough()
    _patch_acs_incremental_year_loading()
    from tableshift import get_dataset

    dset = get_dataset(dataset_name, cache_dir=str(TABLESHIFT_DOWNLOAD_CACHE))
    out_dir = RAW_CACHE_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for out_name, tableshift_split in [("train", "train"), ("test_id", "id_test"), ("test_ood", "ood_test")]:
        X, y, _, _ = dset.get_pandas(tableshift_split)
        out = X.copy()
        out["label"] = y.values
        out_path = out_dir / f"{out_name}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"{dataset_name}/{out_name}: {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    datasets = sys.argv[1:] or CANDIDATE_DATASETS
    for name in datasets:
        extract_dataset(name)
    print(f"\nDone. src/data/tableshift_loader.py can now read from {RAW_CACHE_DIR} "
          "without `tableshift` installed.")
