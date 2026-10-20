#!/bin/bash
# P5 robustness check (12 October 2026): is SATA's null on Qwen-Instruct down to drawing its
# demonstrations (sata.selection: sample, hiccups/21) rather than the plan's top-k rule?
# Runs SATA-v2's top-k checkpoint (models/sata_v2_topk.pt) with top-k selection, 1 seed
# (top-k is deterministic), on Qwen-Instruct under 3 namings:
#   H1  the LLM validation suite sata_llmval (8 tasks), comparable to h200/p5/llmval.parquet;
#   H2  the evaluation grid (24 tasks), comparable to h200/p5/grid_sata.parquet.
# Usage on the pod: bash scripts/run_topk_pod.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/root/venv/bin/python}
export HF_HOME=${HF_HOME:-/dev/shm/hf} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
R=results/v3/synthetic
L=$R/p5/logs_topk
mkdir -p $L
QI=qwen2.5-7b-instruct
TOPK="--models $QI --namings abstract,aligned,flipped --order seed --strategies sata --seeds 1 --sata-selection topk --sata-checkpoints sata=models/sata_v2_topk.pt"
step() { echo "$(date -u '+%F %T') $1" >> $L/steps.log; }
merge() { "$PY" scripts/merge_shards.py "$@" >> $L/steps.log 2>&1; }

[ -f models/sata_v2_topk.pt ] || { step "missing models/sata_v2_topk.pt"; exit 1; }
step "H start: SATA top-k, LLM validation then the evaluation grid"
for i in 0 1 2 3 4 5 6 7; do
    (
        "$PY" scripts/run_synth_grid.py --run p5/llmval_topk --suite sata_llmval $TOPK \
            --shard $i/8 --out $R/p5/llmval_topk.s$i.parquet || exit 1
        "$PY" scripts/run_synth_grid.py --run p5/grid_topk $TOPK \
            --shard $i/8 --out $R/p5/grid_topk.s$i.parquet
    ) > $L/H_$i.log 2>&1 &
    sleep 3
done
wait
merge $R/p5/llmval_topk.parquet $R/p5/llmval_topk.s{0,1,2,3,4,5,6,7}.parquet
merge $R/p5/grid_topk.parquet $R/p5/grid_topk.s{0,1,2,3,4,5,6,7}.parquet
step "H done; ALL DONE"
