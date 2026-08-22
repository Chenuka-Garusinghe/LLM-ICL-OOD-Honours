#!/bin/bash
# Run this ON GADI, inside the Jupyter environment (e.g. an NCI ARE JupyterLab
# session, or a terminal within one) -- NOT from your Mac. Assumes the project
# files are already present at PROJECT_DIR (via git clone or a prior rsync).
#
# Does, in order:
#   1. Build (or reuse) a venv on /scratch and install requirements.txt.
#   2. Register that venv as a Jupyter kernel so it's selectable from a notebook.
#   3. Check whether the TableShift raw cache has all expected parquet files;
#      if anything is missing (most likely `anes`), print the manual-download
#      instructions instead of failing silently.
#
# Safe to re-run: reuses an existing venv and skips re-extraction work that's
# already done.
#
# --- WHY THIS VERSION IS DIFFERENT ---
# /home on Gadi is capped at a fixed 10 GB (not part of your project SU/storage
# allocation, so an admin/project lead can't raise it -- see `quota -s`). A venv
# with an ML stack plus pip's download cache blows past that on their own. This
# version puts everything that GROWS (venv, pip cache, HF cache, extracted
# parquet data) on /scratch instead, which has a ~1 TB project allocation.
# PROJECT_DIR (the git repo itself: code/notebooks/configs) stays wherever it
# already is -- that part is small and fine to leave in home.
#
# NOTE: /scratch is purged of files that go untouched for a long period (NCI
# policy, currently ~100 days of inactivity) -- it's working space, not
# archival storage. Copy anything you can't afford to lose (final checkpoints,
# final results tables) to /g/data periodically; don't treat /scratch as
# permanent.
#
# ONE-TIME CLEANUP if you're re-running this after a previous run that used
# the old home-based .venv: that old copy at
#   ${PROJECT_DIR}/.venv
# is now orphaned and still counting against your /home quota. Remove it
# yourself once you've confirmed the new scratch-based venv works:
#   rm -rf "${PROJECT_DIR}/.venv"
#
# CONFIG -- check these before running:
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Which project's SU/storage allocation should the scratch space (and thus
# the venv, caches, and extracted data) live under? Pick the one you actually
# want charged/quota'd -- check with `lquota` if unsure. Override at call time
# with: SCRATCH_PROJECT=pp49 ./setup_sata_project.sh
SCRATCH_PROJECT="${SCRATCH_PROJECT:-aa36}"
SCRATCH_BASE="/scratch/${SCRATCH_PROJECT}/${USER}/sata-project"
VENV_DIR="${SCRATCH_BASE}/.venv"
CACHE_DIR="${SCRATCH_BASE}/.cache"
RAW_CACHE_DIR="${SCRATCH_BASE}/tableshift_raw_cache"
TABLESHIFT_CACHE_DIR="${SCRATCH_BASE}/tableshift_cache"   # separate dir the
                                                            # anes manual
                                                            # download goes in

mkdir -p "${SCRATCH_BASE}" "${CACHE_DIR}" "${RAW_CACHE_DIR}" "${TABLESHIFT_CACHE_DIR}" "${SCRATCH_BASE}/tmp"

# Redirect every cache-y env var off /home *before* anything installs or
# downloads, so nothing sneaks back into ~/.cache.
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export XDG_CACHE_HOME="${CACHE_DIR}"           # covers puccinialin (the Rust
                                                # toolchain fetcher some
                                                # packages trigger during
                                                # build) and most other tools
export HF_HOME="${CACHE_DIR}/huggingface"      # transformers/datasets model
                                                # + dataset downloads -- these
                                                # get large fast for ICL work
export TMPDIR="${SCRATCH_BASE}/tmp"

# Run `module avail python3` and pick the newest 3.1x line before running this
# -- don't just trust this default.
PYTHON_MODULE="python3/3.11.7"
KERNEL_NAME="sata-project"
KERNEL_DISPLAY_NAME="Python (sata-project)"
CANDIDATE_DATASETS=(acsincome acspubcov brfss_diabetes anes)

cd "${PROJECT_DIR}"

echo "== Step 1/3: venv + deps (venv on scratch: ${VENV_DIR}) =="
if [ -d "${PROJECT_DIR}/.venv" ]; then
  echo "NOTE: found an old venv at ${PROJECT_DIR}/.venv (in /home) -- it's no"
  echo "      longer used by this script and is safe to remove once the new"
  echo "      scratch-based venv below is confirmed working:"
  echo "        rm -rf '${PROJECT_DIR}/.venv'"
fi
module load "${PYTHON_MODULE}"
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
echo "Python: $(python3 --version), venv at ${VENV_DIR}"

echo "== Step 2/3: register Jupyter kernel =="
python -m ipykernel install --user --name="${KERNEL_NAME}" --display-name="${KERNEL_DISPLAY_NAME}"
echo "Kernel '${KERNEL_DISPLAY_NAME}' registered -- select it from the notebook's kernel picker."

echo "== Step 3/3: check TableShift raw cache =="
# Data now lives on scratch too. Keep a symlink at the old in-repo path so any
# existing code/notebooks using the relative path
# "data/tableshift_raw_cache/..." keep working unchanged.
mkdir -p "${PROJECT_DIR}/data"
if [ -e "${PROJECT_DIR}/data/tableshift_raw_cache" ] && [ ! -L "${PROJECT_DIR}/data/tableshift_raw_cache" ]; then
  echo "NOTE: ${PROJECT_DIR}/data/tableshift_raw_cache already exists and is a"
  echo "      real directory, not a symlink -- leaving it alone rather than"
  echo "      overwriting it. If it has data in it, move that data under"
  echo "      ${RAW_CACHE_DIR} yourself, then delete the old directory and"
  echo "      re-run this script so the symlink can be created."
elif [ ! -e "${PROJECT_DIR}/data/tableshift_raw_cache" ]; then
  ln -s "${RAW_CACHE_DIR}" "${PROJECT_DIR}/data/tableshift_raw_cache"
fi
# Same treatment for the separate tableshift_cache dir (where the manual
# anes CSV goes).
if [ ! -e "${PROJECT_DIR}/data/tableshift_cache" ]; then
  ln -s "${TABLESHIFT_CACHE_DIR}" "${PROJECT_DIR}/data/tableshift_cache"
fi

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
    echo "  ${TABLESHIFT_CACHE_DIR}/anes_timeseries_cdf_csv_20220916.csv"
    echo "(also reachable via the repo as data/tableshift_cache/anes_timeseries_cdf_csv_20220916.csv,"
    echo "which is symlinked there for you)"
    echo "(tableshift hardcodes this exact filename/date -- a newer release works fine"
    echo "renamed to match, as long as its VCF* columns are present). See Notebook"
    echo "01_tableshift_setup.ipynb, Step 2, for the full rationale."
  fi
fi
echo
echo "Done. Open a notebook in Jupyter and select the '${KERNEL_DISPLAY_NAME}' kernel."