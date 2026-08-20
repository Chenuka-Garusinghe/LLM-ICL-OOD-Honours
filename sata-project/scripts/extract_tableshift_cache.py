"""One-time TableShift raw-data extraction.

Run this with a SEPARATE Python environment that has `tableshift` installed
(clone github.com/mlfoundations/tableshift and `pip install -e . --no-deps`,
then install its runtime deps manually) -- NOT this project's main .venv.

Why isolated: TableShift hard-pins numpy==1.23.5 / ray==2.2 and its `xport`
dependency breaks on pandas>=3, all of which conflict with this project's
modern stack (torch 2.x, vllm, current pandas/numpy/sklearn). Fighting that
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
    """
    import collections

    try:
        import xport.v56 as _xport_v56
    except ImportError:
        return

    if getattr(_xport_v56.namedtuple, "_extract_script_patch", False):
        return

    _original_namedtuple = collections.namedtuple

    def _namedtuple_with_rename(typename, field_names, **kwargs):
        kwargs.setdefault("rename", True)
        return _original_namedtuple(typename, field_names, **kwargs)

    _namedtuple_with_rename._extract_script_patch = True
    _xport_v56.namedtuple = _namedtuple_with_rename


def extract_dataset(dataset_name: str) -> None:
    _patch_xport_underscore_fields()
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
