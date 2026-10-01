#!/bin/bash
# Runpod pod setup (used for P4 on 1 October 2026; hiccups/17). Run on the pod after the
# project has been copied to /root/sata-project (without .env). Builds the pinned venv
# (torch 2.14.0+cu130 needs a CUDA 13 host) and downloads both models into RAM disk, in
# parallel. A stopped pod loses its container disk and /dev/shm, so rerun after a restart.
set -u
export HF_HOME=/dev/shm/hf
cd /root/sata-project
log() { echo "$(date -u +%H:%M:%S) $*" >> /root/setup.log; }
log "setup start"
for repo in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B hf-internal-testing/tiny-random-gpt2; do
    ( uvx --from "huggingface_hub[hf_xet]==1.32.0" hf download "$repo" > "/root/dl_$(basename $repo).log" 2>&1; log "download $repo exit $?" ) &
done
uv venv /root/venv --python 3.12 > /root/venv.log 2>&1
UV_HTTP_TIMEOUT=300 uv pip install --python /root/venv/bin/python -r requirements.txt >> /root/venv.log 2>&1
log "venv exit $?"
wait
/root/venv/bin/python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__)" >> /root/setup.log 2>&1
log "SETUP DONE"
