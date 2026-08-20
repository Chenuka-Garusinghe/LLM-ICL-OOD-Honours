#!/bin/bash
# Run this ON GADI, inside the Jupyter environment (e.g. an NCI ARE JupyterLab
# session, or a terminal within one) -- NOT from your Mac. Assumes the project
# files are already present at PROJECT_DIR (via git clone or a prior rsync).
#
# Does, in order:
#   1. Build (or reuse) a venv and install requirements.txt.
#   2. Register that venv as a Jupyter kernel so it's selectable from a notebook.
#   3. Check whether data/tableshift_raw_cache/ has all expected parquet files;
#      if anything is missing (most likely `anes`), print the manual-download
#      instructions instead of failing silently.
#
# Safe to re-run: reuses an existing venv and skips re-extraction work that's
# already done.
#
# CONFIG -- check these before running:
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Run `module avail python3` and pick the newest 3.1x line before running this
# -- don't just trust this default.
PYTHON_MODULE="python3/3.11.7"
KERNEL_NAME="sata-project"
KERNEL_DISPLAY_NAME="Python (sata-project)"

CANDIDATE_DATASETS=(acsincome acspubcov brfss_diabetes anes)
RAW_CACHE_DIR="${PROJECT_DIR}/data/tableshift_raw_cache"

cd "${PROJECT_DIR}"

echo "== Step 1/3: venv + deps (${PROJECT_DIR}) =="
module load "${PYTHON_MODULE}"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Python: $(python3 --version), venv at ${PROJECT_DIR}/.venv"

echo "== Step 2/3: register Jupyter kernel =="
python -m ipykernel install --user --name="${KERNEL_NAME}" --display-name="${KERNEL_DISPLAY_NAME}"
echo "Kernel '${KERNEL_DISPLAY_NAME}' registered -- select it from the notebook's kernel picker."

echo "== Step 3/3: check TableShift raw cache =="
missing=()
for name in "${CANDIDATE_DATASETS[@]}"; do
  if [ ! -f "${RAW_CACHE_DIR}/${name}/train.parquet" ]; then
    missing+=("${name}")
  fi
done

if [ ${#missing[@]} -eq 0 ]; then
  echo "All expected parquet files present under ${RAW_CACHE_DIR} -- nothing to extract."
else
  echo "MISSING datasets: ${missing[*]}"
  echo "These need scripts/extract_tableshift_cache.py run from a SEPARATE, throwaway"
  echo "environment with tableshift installed (its numpy==1.23.5/ray==2.2 pins conflict"
  echo "with this project's stack) -- see that script's docstring."
  if printf '%s\n' "${missing[@]}" | grep -qx anes; then
    echo
    echo "NOTE: 'anes' additionally requires a MANUAL download -- it is TableShift's"
    echo "one OfflineDataSource and cannot auto-download. Register at electionstudies.org,"
    echo "download the Time Series Cumulative Data File, and place it as"
    echo "  data/tableshift_cache/anes_timeseries_cdf_csv_20220916.csv"
    echo "(tableshift hardcodes this exact filename/date -- a newer release works fine"
    echo "renamed to match, as long as its VCF* columns are present). See Notebook"
    echo "01_tableshift_setup.ipynb, Step 2, for the full rationale."
  fi
fi

echo
echo "Done. Open a notebook in Jupyter and select the '${KERNEL_DISPLAY_NAME}' kernel."
