#!/bin/bash
# Run this FROM A TERMINAL ATTACHED TO THE COLAB KERNEL (e.g. the VS Code
# "Connect to Jupyter Server" terminal, or a `!bash deploy_to_colab.sh` cell) --
# not from your Mac. This is the Colab counterpart to deploy_to_gadi.sh, but
# the two environments differ enough that this is a separate script, not a
# variant of that one:
#
#   - Gadi:  persistent HPC allocation, shared /scratch, inode-quota'd -->
#            everything lives in a prebuilt Apptainer container + overlay.
#   - Colab: ephemeral VM, wiped on every disconnect, no quota pressure -->
#            plain `pip install` into the kernel's own env is fine, and there
#            is no kernel to register -- the "Python 3 (A100)" kernel already
#            exists and VS Code is already attached to it.
#
# Does, in order:
#   1. Check Google Drive is mounted at /content/drive (must be done from a
#      notebook cell first -- drive.mount() needs interactive auth and can't
#      run from a plain bash script).
#   2. git clone (or pull, if already present) the project into PROJECT_DIR.
#   3. pip install -r requirements.txt into the kernel's environment.
#   4. Point HF_HOME at Drive so model weights persist across sessions --
#      an 8B model download is too large to eat every fresh runtime.
#   5. Symlink data/tableshift_raw_cache (and data/tableshift_cache, for the
#      manual anes CSV) from Drive, and check the expected parquet files are
#      there -- same check as deploy_to_gadi.sh, and the same reason: `tableshift`
#      still can't be pip-installed alongside torch/vllm here either (its
#      numpy==1.23.5/ray==2.2 pins conflict). Extraction has to happen in a
#      SEPARATE environment (your Mac, or Gadi) and the resulting parquet
#      files copied into the Drive folder this script points at.
#   6. Symlink data/real and data/synthetic from Drive too -- these are where
#      Notebook 01/04 WRITE their output (train_pool/test_id/test_ood parquet
#      etc.), and everything downstream (Notebooks 02, 03, 07, 08) reads from
#      them. Left pointing at the ephemeral clone, a session wipe mid-pipeline
#      loses that output the same way the raw cache got lost once already.
#
# Safe to re-run: steps are idempotent. The data/real and data/synthetic
# symlinking (step 6) only auto-replaces the directory if it holds nothing but
# the tracked .gitkeep placeholder (i.e. a fresh clone) -- if real output has
# already landed there on ephemeral storage, it's left alone with a note
# rather than silently touched.
#
# CONFIG -- check these before running:
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Chenuka-Garusinghe/LLM-ICL-OOD-Honours.git}"
REPO_CLONE_DIR="${REPO_CLONE_DIR:-/content/LLM-ICL-OOD-Honours}"
PROJECT_DIR="${PROJECT_DIR:-${REPO_CLONE_DIR}/sata-project}"

# Everything under here must survive a runtime disconnect -- point it at a
# real folder in your Drive. Populate tableshift_raw_cache/ (and
# tableshift_cache/ for anes) here yourself before running, by copying over
# the output of scripts/extract_tableshift_cache.py from wherever you ran it.
DRIVE_BASE="${DRIVE_BASE:-/content/drive/MyDrive/sata-project}"
CACHE_DIR="${DRIVE_BASE}/.cache"
RAW_CACHE_DIR="${DRIVE_BASE}/tableshift_raw_cache"
TABLESHIFT_CACHE_DIR="${DRIVE_BASE}/tableshift_cache"
REAL_OUT_DIR="${DRIVE_BASE}/real"
SYNTHETIC_OUT_DIR="${DRIVE_BASE}/synthetic"

CANDIDATE_DATASETS=(acsincome acspubcov brfss_diabetes anes)

echo "== Step 1/5: check Google Drive is mounted =="
if [ ! -d "/content/drive/MyDrive" ]; then
  echo "ERROR: /content/drive/MyDrive not found." >&2
  echo "       Mount Drive from a NOTEBOOK CELL first (needs interactive auth" >&2
  echo "       -- can't be done from this script):" >&2
  echo "         from google.colab import drive" >&2
  echo "         drive.mount('/content/drive')" >&2
  echo "       Then re-run this script." >&2
  exit 1
fi
mkdir -p "${DRIVE_BASE}" "${CACHE_DIR}" "${RAW_CACHE_DIR}" "${TABLESHIFT_CACHE_DIR}" "${REAL_OUT_DIR}" "${SYNTHETIC_OUT_DIR}"
echo "Drive OK: ${DRIVE_BASE}"

echo "== Step 2/5: get the project onto the VM =="
if [ -d "${REPO_CLONE_DIR}/.git" ]; then
  echo "Repo already present at ${REPO_CLONE_DIR} -- pulling latest."
  git -C "${REPO_CLONE_DIR}" pull --ff-only
else
  git clone "${REPO_URL}" "${REPO_CLONE_DIR}"
fi

echo "== Step 3/5: install Python dependencies into the kernel env =="
# Colab's base image already ships a CUDA-matched torch build; letting the
# resolver pull vllm's own torch/transformers pins on top of it is expected to
# adjust versions -- that's normal here (unlike Gadi, there's no shared
# container to keep byte-for-byte stable across a whole lab).
# uv resolves and installs this stack much faster than pip -- bootstrap it via
# pip (small, no extra deps) if it's not already on PATH, then use it for the
# real install. --system targets the kernel's own interpreter directly, same
# as the pip call this replaces -- there's no venv here to install into.
command -v uv >/dev/null 2>&1 || pip install -q uv
uv pip install -q --system -r "${PROJECT_DIR}/requirements.txt"

export HF_HOME="${CACHE_DIR}/huggingface"
echo "HF_HOME set to ${HF_HOME} for this shell -- see note below about the kernel process."

echo "== Step 4/5: check TableShift raw cache =="
mkdir -p "${PROJECT_DIR}/data"
if [ -e "${PROJECT_DIR}/data/tableshift_raw_cache" ] && [ ! -L "${PROJECT_DIR}/data/tableshift_raw_cache" ]; then
  echo "NOTE: ${PROJECT_DIR}/data/tableshift_raw_cache already exists and is a"
  echo "      real directory, not a symlink -- leaving it alone."
elif [ ! -e "${PROJECT_DIR}/data/tableshift_raw_cache" ]; then
  ln -s "${RAW_CACHE_DIR}" "${PROJECT_DIR}/data/tableshift_raw_cache"
fi
if [ ! -e "${PROJECT_DIR}/data/tableshift_cache" ]; then
  ln -s "${TABLESHIFT_CACHE_DIR}" "${PROJECT_DIR}/data/tableshift_cache"
fi

missing=()
for name in "${CANDIDATE_DATASETS[@]}"; do
  if [ ! -f "${PROJECT_DIR}/data/tableshift_raw_cache/${name}/train.parquet" ]; then
    missing+=("${name}")
  fi
done
if [ ${#missing[@]} -eq 0 ]; then
  echo "All expected parquet files present under ${RAW_CACHE_DIR} -- nothing to extract."
else
  echo "MISSING datasets: ${missing[*]}"
  echo "Run scripts/extract_tableshift_cache.py from a SEPARATE, throwaway environment"
  echo "(your Mac or Gadi -- NOT this Colab runtime, same numpy/ray conflict applies here"
  echo "too), then copy its output into ${RAW_CACHE_DIR}/<dataset_name>/ via Drive."
  if printf '%s\n' "${missing[@]}" | grep -qx anes; then
    echo
    echo "NOTE: 'anes' also needs the manually-downloaded Time Series Cumulative Data"
    echo "File from electionstudies.org, placed as"
    echo "  ${TABLESHIFT_CACHE_DIR}/anes_timeseries_cdf_csv_20220916.csv"
    echo "See Notebook 01_tableshift_setup.ipynb, Step 2, for the full rationale."
  fi
fi

echo "== Step 5/5: point data/real and data/synthetic at Drive =="
for pair in "data/real:${REAL_OUT_DIR}" "data/synthetic:${SYNTHETIC_OUT_DIR}"; do
  rel="${pair%%:*}"
  target="${pair##*:}"
  path="${PROJECT_DIR}/${rel}"
  if [ -L "${path}" ]; then
    echo "${path} already symlinked -- OK."
    continue
  fi
  if [ -e "${path}" ]; then
    # A fresh clone leaves only the tracked .gitkeep placeholder here -- safe
    # to swap for the Drive symlink. Anything else means output already
    # landed on this (ephemeral) disk; leave it alone rather than delete it.
    contents="$(find "${path}" -mindepth 1 2>/dev/null)"
    if [ -z "${contents}" ] || [ "${contents}" = "${path}/.gitkeep" ]; then
      rm -rf "${path}"
      ln -s "${target}" "${path}"
      echo "${path} -> ${target}"
    else
      echo "NOTE: ${path} already has real content on this VM's disk -- leaving it"
      echo "      alone. Move it to ${target} yourself, delete ${path}, and re-run"
      echo "      this script to symlink it."
    fi
  else
    ln -s "${target}" "${path}"
    echo "${path} -> ${target}"
  fi
done

echo
echo "Done."
echo
echo "IMPORTANT: this script ran in its OWN shell process -- exporting HF_HOME here"
echo "does NOT propagate to the notebook's Python kernel. Also, unlike a local Jupyter"
echo "server, Colab's kernel cwd is NOT automatically the notebook's own folder. Add a"
echo "cell at the TOP of the notebook, before the existing 'PROJECT_ROOT = Path.cwd()...'"
echo "cell, doing:"
echo "  import os"
echo "  os.environ['HF_HOME'] = '${CACHE_DIR}/huggingface'"
echo "  os.chdir('${PROJECT_DIR}/notebooks')"
