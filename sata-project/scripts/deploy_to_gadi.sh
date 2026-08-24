#!/bin/bash
# Run this ON GADI, inside the Jupyter environment (e.g. an NCI ARE JupyterLab
# session, or a terminal within one) -- NOT from your Mac. Assumes the project
# files are already present at PROJECT_DIR (via git clone or a prior rsync).
#
# Does, in order:
#   1. Check that the shared container (torch/vllm/transformers base image +
#      project-extras overlay) is present and importable.
#   2. Write a Jupyter kernelspec that launches that container as the kernel,
#      so it's selectable from a notebook in this JupyterLab session.
#   3. Check whether the TableShift raw cache has all expected parquet files;
#      if anything is missing (most likely `anes`), print the manual-download
#      instructions instead of failing silently.
#
# Safe to re-run: steps are idempotent.
#
# --- WHY THIS VERSION IS DIFFERENT (no venv, no pip install) ---
# A previous version of this script built a per-student venv from
# requirements.txt on /scratch. A raw `pip install` of vllm + torch +
# transformers costs ~99,000 inodes -- too close to our shared 202,000-inode
# scratch quota (aa36/pp49 combined, shared across the whole research group),
# and duplicated per student it would blow that quota outright.
#
# Instead, the whole torch/vllm/transformers stack lives in ONE prebuilt
# Apptainer image (a single .sif file, near-zero inode cost), plus a second
# read-only overlay (.sqfs) layering the handful of extra packages this
# project needs that aren't in the base vllm image (pandas, scikit-learn,
# xgboost, matplotlib, seaborn, sentence-transformers, pyarrow, datasets,
# ipykernel, nbconvert, jupyter_client). Both files are shared, group-readable,
# and built once -- see scripts/../containers/README.md (documented at
# CONTAINER_DIR below) for how to rebuild/update them.
#
# NOTE on the overlay: it was built with `pip install --target=... -c
# <container's own `pip freeze`>` so it could NOT silently upgrade
# torch/transformers/numpy away from the versions already verified to work
# with vllm in the base image. A naive `pip install --target` (no constraints)
# WILL happily resolve and duplicate a *different*, incompatible torch build
# into the overlay -- if you ever rebuild sata-extras.sqfs, freeze the base
# image's exact versions first and pass them as a constraints file, or you'll
# silently break vllm.
#
# NOTE: /scratch is purged of files that go untouched for a long period (NCI
# policy, currently ~100 days of inactivity) -- it's working space, not
# archival storage. Copy anything you can't afford to lose (final checkpoints,
# final results tables) to /g/data periodically; don't treat /scratch as
# permanent.
#
# ONE-TIME CLEANUP if you're re-running this after a previous run that used
# the old venv-based approach: that old venv at
#   ${PROJECT_DIR}/.venv   (or the scratch-based one from an even older
#                            version of this script)
# is now orphaned. Safe to remove once you've confirmed notebooks run fine
# against the container:
#   rm -rf "${PROJECT_DIR}/.venv"
#
# CONFIG -- check these before running:
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Shared container files (built once, reused by the whole lab -- see
# containers/README.md next to them for provenance/rebuild instructions).
# Override at call time if you've copied these somewhere else:
#   CONTAINER_DIR=/g/data/aa36/containers ./deploy_to_gadi.sh
CONTAINER_DIR="${CONTAINER_DIR:-/scratch/aa36/cg3543/containers}"
SIF="${CONTAINER_DIR}/sata-vllm.sif"
EXTRAS_SQFS="${CONTAINER_DIR}/sata-extras.sqfs"
EXTRAS_PYTHONPATH="/opt/sata-extra/site-packages"

# Which project's SU/storage allocation should the *data caches* (TableShift
# parquet cache, HF dataset cache) live under? Pick the one you actually want
# charged/quota'd -- check with `lquota` if unsure. Override at call time
# with: SCRATCH_PROJECT=pp49 ./deploy_to_gadi.sh
# (This is independent of CONTAINER_DIR above -- the container is a
# lab-wide shared artifact; your data caches are your own.)
SCRATCH_PROJECT="${SCRATCH_PROJECT:-aa36}"
SCRATCH_BASE="/scratch/${SCRATCH_PROJECT}/${USER}/sata-project"
CACHE_DIR="${SCRATCH_BASE}/.cache"
RAW_CACHE_DIR="${SCRATCH_BASE}/tableshift_raw_cache"
TABLESHIFT_CACHE_DIR="${SCRATCH_BASE}/tableshift_cache"   # separate dir the
                                                            # anes manual
                                                            # download goes in

mkdir -p "${SCRATCH_BASE}" "${CACHE_DIR}" "${RAW_CACHE_DIR}" "${TABLESHIFT_CACHE_DIR}"

export HF_HOME="${CACHE_DIR}/huggingface"      # transformers/datasets model
                                                # + dataset downloads -- these
                                                # get large fast for ICL work

KERNEL_NAME="sata-project"
KERNEL_DISPLAY_NAME="Python (sata-project, container)"
CANDIDATE_DATASETS=(acsincome acspubcov brfss_diabetes anes)

cd "${PROJECT_DIR}"

echo "== Step 1/3: check the shared container is present =="
if [ ! -f "${SIF}" ]; then
  echo "ERROR: base container not found at ${SIF}" >&2
  echo "       This is a shared, prebuilt artifact -- it should not need" >&2
  echo "       rebuilding per-student. If it's genuinely missing, see" >&2
  echo "       ${CONTAINER_DIR}/README.md for how it was built" >&2
  echo "       (apptainer pull docker://vllm/vllm-openai:v0.11.0)." >&2
  exit 1
fi
if [ ! -f "${EXTRAS_SQFS}" ]; then
  echo "ERROR: extras overlay not found at ${EXTRAS_SQFS}" >&2
  echo "       See ${CONTAINER_DIR}/README.md for how to rebuild it -- and" >&2
  echo "       read the constraints-file warning at the top of this script" >&2
  echo "       first, or a rebuild can silently break vllm's torch pin." >&2
  exit 1
fi
module load apptainer
# Resolve the absolute binary path (not just relying on `apptainer` being on
# PATH) -- the kernel.json below gets launched by the ARE JupyterLab SERVER
# process, which was started with whatever environment ARE gave it, NOT a
# fresh shell with `module load apptainer` applied. A bare "apptainer" in
# argv fails there with "[Errno 2] No such file or directory: 'apptainer'"
# even though `module load apptainer` works fine in an interactive terminal.
APPTAINER_BIN="$(command -v apptainer)"
echo "Container OK: ${SIF}"
echo "Extras overlay OK: ${EXTRAS_SQFS}"
echo "Apptainer binary: ${APPTAINER_BIN}"

echo "== Step 2/3: register Jupyter kernel (launches the container) =="
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}"
mkdir -p "${KERNEL_DIR}"
cat > "${KERNEL_DIR}/kernel.json" << EOF
{
 "argv": [
  "${APPTAINER_BIN}", "exec", "--nv",
  "--pwd", "${PROJECT_DIR}/notebooks",
  "--overlay", "${EXTRAS_SQFS}",
  "--env", "PYTHONPATH=${EXTRAS_PYTHONPATH}",
  "${SIF}",
  "python3", "-m", "ipykernel_launcher", "-f", "{connection_file}"
 ],
 "display_name": "${KERNEL_DISPLAY_NAME}",
 "language": "python"
}
EOF
echo "Kernel '${KERNEL_DISPLAY_NAME}' registered at ${KERNEL_DIR} -- select it from the notebook's kernel picker."
# NOTE: this kernel.json wraps the FULL apptainer exec call in its argv,
# because JupyterLab (the process discovering and launching this kernel) runs
# OUTSIDE the container. Don't simplify this to a bare python3 invocation --
# see gadi_run_notebook.pbs for the different (simpler) argv used when
# nbconvert itself is already running inside the container.
#
# NOTE on --pwd: every notebook's first cell does
# `PROJECT_ROOT = Path.cwd().parent; sys.path.insert(0, str(PROJECT_ROOT))`
# to make `import src...` work -- this only resolves correctly if the kernel
# process's cwd is PROJECT_DIR/notebooks. Depending on how ARE's JupyterLab
# server is configured, the cwd it hands to a spawned kernel isn't always the
# specific notebook's own directory (some Jupyter server versions default to
# the server's root_dir instead). `--pwd` here pins it explicitly rather than
# hoping ARE gets it right -- if this weren't set, cells doing `import src...`
# would fail with "ModuleNotFoundError: No module named 'src'".

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


# NOTE: checked via the in-repo path (data/tableshift_raw_cache), not the
# SCRATCH_BASE-derived RAW_CACHE_DIR above -- extract_tableshift_cache.py
# hardcodes PROJECT_ROOT/data/tableshift_raw_cache as its output dir
# regardless of SCRATCH_PROJECT, and if that path is already a real,
# pre-populated directory (not a symlink -- see the check above), this is
# where the actual data lives. Checking RAW_CACHE_DIR directly would report a
# false "missing" for data that's already present under a different SCRATCH_PROJECT
# than whatever this was last run with.
CHECK_DIR="${PROJECT_DIR}/data/tableshift_raw_cache"
missing=()
for name in "${CANDIDATE_DATASETS[@]}"; do
  if [ ! -f "${CHECK_DIR}/${name}/train.parquet" ]; then
    missing+=("${name}")
  fi
done
if [ ${#missing[@]} -eq 0 ]; then
  echo "All expected parquet files present under ${CHECK_DIR} -- nothing to extract."
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
