#!/bin/bash
# P6 mechanistic probes on one H200, one session (docs/research_plan.md, P6 implementation).
#   A  logit-lens margins at every layer for Qwen-Instruct on the 24 evaluation tasks, 3 namings,
#      seed 0: zero_shot, random (with residuals), label_diversity, counter_prior, sata, sata_mi.
#   V  the P6 gate, then steering vectors from A's random residuals on the even-indexed tasks.
#   B  steering on the odd-indexed tasks with random demonstrations, aligned and flipped names:
#      layers 8/12/16/20/24 x alpha 1/2/4, plus a random direction of the same norm at alpha 4.
#      The vectors are about 1% of the residual norm, so a second pass adds alpha 8/16/32 (random
#      direction at 32):  STAGES=B ALPHAS="8 16 32" RAND_ALPHA=32 bash scripts/run_p6_pod.sh
# Needs results/v3/synthetic/h200/p3/pool_prior_<model>.parquet and h200/p5/zero_shot_eval.parquet.
# Usage on the pod: bash scripts/run_p6_pod.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/root/venv/bin/python}
export HF_HOME=${HF_HOME:-/dev/shm/hf} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
R=results/v3/synthetic
P6=$R/p6
L=$P6/logs
mkdir -p $L $P6/A $P6/B
QI=qwen2.5-7b-instruct
N3=abstract,aligned,flipped
step() { echo "$(date -u '+%F %T') $1" >> $L/steps.log; }
merge() { "$PY" scripts/merge_shards.py "$@" >> $L/steps.log 2>&1; }
COMMON="--models $QI --order seed --seeds 1 --probe-dir $P6/A"
PRIORS="--pool-prior $R/h200/p3/pool_prior_$QI.parquet --query-prior $R/h200/p5/zero_shot_eval.parquet"

STAGES=${STAGES:-A V B}
ALPHAS=${ALPHAS:-1 2 4}
RAND_ALPHA=${RAND_ALPHA:-4}
stage_A() {
step "A start: layer margins"
for i in 0 1 2 3 4 5 6 7; do
    (
        "$PY" scripts/run_synth_grid.py --run p6/A --namings $N3 $COMMON --strategies random --keep-residuals \
            --shard $i/8 --out $P6/A/grid.s$i.parquet || exit 1
        "$PY" scripts/run_synth_grid.py --run p6/A --namings $N3 $COMMON $PRIORS \
            --strategies zero_shot,label_diversity,counter_prior,sata,sata_mi --shard $i/8 --out $P6/A/grid.s$i.parquet
    ) > $L/A_$i.log 2>&1 &
    sleep 3
done
wait
merge $P6/A/grid.parquet $P6/A/grid.s{0,1,2,3,4,5,6,7}.parquet
step "A done"
}

stage_V() {
"$PY" scripts/p6_probes.py gate --probe-dir $P6/A >> $L/steps.log 2>&1
"$PY" scripts/p6_probes.py vectors --probe-dir $P6/A --out models/p6_steer.npz >> $L/steps.log 2>&1 \
    || { step "vectors failed"; exit 1; }
step "V done"
}

stage_B() {
ODD=$(seq -f "eval_%04g" 1 2 23 | paste -sd, -)
CONDS=""
for layer in 8 12 16 20 24; do
    for a in $ALPHAS; do CONDS="$CONDS v:$layer:$a"; done
    CONDS="$CONDS rand:$layer:$RAND_ALPHA"
done
step "B start: steering ($(echo $CONDS | wc -w) conditions)"
for i in 0 1 2 3 4 5 6 7; do
    (
        for c in $CONDS; do
            IFS=: read key layer a <<< "$c"
            tag=${key}_L${layer}_a${a}
            "$PY" scripts/run_synth_grid.py --run p6/B_$tag --models $QI --namings aligned,flipped --order seed --seeds 1 \
                --task-ids $ODD --strategies random --probe-dir $P6/B --condition $tag \
                --steer models/p6_steer.npz:$key:$layer:$a --shard $i/8 --out $P6/B/grid_$tag.s$i.parquet || exit 1
        done
    ) > $L/B_$i.log 2>&1 &
    sleep 3
done
wait
step "B done (alphas $ALPHAS); ALL DONE"
}

for stage in $STAGES; do stage_$stage; done
