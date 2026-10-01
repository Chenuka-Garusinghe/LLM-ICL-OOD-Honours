#!/bin/bash
# P4 on one large GPU, as run on a Runpod H200 on 1 October 2026 (hiccups/17).
# The whole P4 workload runs on one device, so every P4 comparison is within one
# device. It is split across 8 worker processes that share the GPU, each with its
# own copy of the model: at batch size 1 one process leaves the GPU mostly idle.
# Workers after stage A run as clients of NVIDIA MPS, which lets their kernels
# run concurrently (about 160 against 84 forward passes per second here). Each
# worker writes its own shard; shards are merged at the end of each stage
# (scripts/merge_shards.py). Every worker is resumable: rerunning skips finished units.
#
#   A  pool priors (Qwen-Instruct 5 shards, Qwen-base 2 shards), the P2
#      cache-agreement benchmark on this GPU, and Qwen-base's P2 checks
#   B  the Qwen-Instruct grid (4 shards: core strategies and counter_prior,
#      the random-label control, similarity), RQ3 on 3 seeds (3 shards) and
#      Qwen-base's reduced grid (only if its label mass is valid)
#   C  counter_prior_matched (exploratory, hiccups/16) for both models
#   D  covariate probes (exploratory, hiccups/18): covariate_scale and the f8-neutral
#      copies of the id, covariate and covariate_scale queries, on every demonstration set
#
# Pod setup: the venv from requirements.txt (torch 2.14.0+cu130 needs a CUDA 13
# host), both models downloaded to HF_HOME, no .env on the pod.
# Usage on the pod:  bash scripts/run_p4_pod.sh A B C
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/root/venv/bin/python}
export HF_HOME=${HF_HOME:-/dev/shm/hf} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
R=results/v3/synthetic
LOGS=$R/p4/logs
mkdir -p "$LOGS" "$R/p3" "$R/p4"
QI=qwen2.5-7b-instruct
QB=qwen2.5-7b-base
CORE=zero_shot,random,label_diversity,feature_range,rule_diversity,counter_spurious,counter_prior
step() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $1" >> "$LOGS/steps.log"; }
merge() { "$PY" scripts/merge_shards.py "$@" >> "$LOGS/steps.log" 2>&1; }

start_mps() {
    export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
}

stage_A() {
    step "stage A start"
    for i in 0 1 2 3 4; do
        "$PY" scripts/p3_pool_priors.py --models $QI --namings abstract,aligned,flipped --shard $i/5 \
            --out $R/p3/pool_prior_$QI.s$i.parquet > "$LOGS/A_prior_qi_$i.log" 2>&1 &
    done
    for i in 0 1; do
        "$PY" scripts/p3_pool_priors.py --models $QB --namings abstract,flipped --shard $i/2 \
            --out $R/p3/pool_prior_$QB.s$i.parquet > "$LOGS/A_prior_qb_$i.log" 2>&1 &
    done
    (
        "$PY" scripts/p2_benchmark.py --model $QI > "$LOGS/A_benchmark.log" 2>&1
        "$PY" scripts/run_synth_grid.py --run p4/base_p2 --suite pilot --models $QB --envs id \
            --strategies zero_shot,label_diversity --label-modes gold,shuffled --k 8 > "$LOGS/A_base_p2.log" 2>&1
        "$PY" scripts/p4_base_check.py >> "$LOGS/A_base_p2.log" 2>&1
    ) &
    wait
    merge $R/p3/pool_prior_$QI.parquet $R/p3/pool_prior_$QI.s{0,1,2,3,4}.parquet
    merge $R/p3/pool_prior_$QB.parquet $R/p3/pool_prior_$QB.s{0,1}.parquet
    step "stage A done"
}

grid_shard() {  # shard i of 4: core strategies and counter_prior, random-label control, similarity
    local i=$1 out=$R/p4/grid.s$1.parquet
    local G="scripts/run_synth_grid.py --run p4/grid --models $QI --namings abstract,aligned,flipped --order seed --shard $i/4 --out $out"
    "$PY" $G --strategies $CORE && "$PY" $G --strategies label_diversity --label-modes shuffled \
        && "$PY" $G --strategies similarity --seeds 1
}

stage_B() {
    start_mps
    step "stage B start (MPS)"
    for i in 0 1 2 3; do grid_shard $i > "$LOGS/B_grid_$i.log" 2>&1 & sleep 3; done
    for i in 0 1 2; do
        "$PY" scripts/run_rq3.py --run p4/rq3.s$i --models $QI --seeds 3 --shard $i/3 > "$LOGS/B_rq3_$i.log" 2>&1 & sleep 3
    done
    if grep -q "valid answer format: True" "$LOGS/A_base_p2.log"; then
        "$PY" scripts/run_synth_grid.py --run p4/grid_base --models $QB --namings abstract,flipped --order seed \
            --strategies zero_shot,random,counter_spurious,counter_prior > "$LOGS/B_base.log" 2>&1 &
    else
        step "base reduced grid skipped (label mass below 0.9; p4/base_p2_gate.json)"
    fi
    wait
    merge $R/p4/grid.parquet $R/p4/grid.s{0,1,2,3}.parquet
    merge $R/p4/rq3.parquet $R/p4/rq3.s{0,1,2}.parquet
    merge $R/p4/rq3_rankings.parquet $R/p4/rq3.s{0,1,2}_rankings.parquet
    step "stage B done"
}

stage_C() {
    start_mps
    step "stage C start (MPS): counter_prior_matched"
    for i in 0 1 2 3; do
        "$PY" scripts/run_synth_grid.py --run p4/grid_cpm --models $QI --namings abstract,aligned,flipped --order seed \
            --strategies counter_prior_matched --shard $i/4 --out $R/p4/grid_cpm.s$i.parquet > "$LOGS/C_cpm_$i.log" 2>&1 &
        sleep 3
    done
    for i in 0 1; do
        "$PY" scripts/run_synth_grid.py --run p4/grid_base_cpm --models $QB --namings abstract,flipped --order seed \
            --strategies counter_prior_matched --shard $i/2 --out $R/p4/grid_base_cpm.s$i.parquet > "$LOGS/C_base_cpm_$i.log" 2>&1 &
        sleep 3
    done
    wait
    merge $R/p4/grid_cpm.parquet $R/p4/grid_cpm.s{0,1,2,3}.parquet
    merge $R/p4/grid_base_cpm.parquet $R/p4/grid_base_cpm.s{0,1}.parquet
    step "stage C done"
}

PROBE_ENVS=covariate_scale,id_f8neutral,covariate_f8neutral,covariate_scale_f8neutral

stage_D() {  # covariate probes (hiccups/18): the new query sets only, on every existing demonstration set
    start_mps
    step "stage D start (MPS): covariate probes"
    for i in 0 1 2 3 4 5; do
        (
            G="scripts/run_synth_grid.py --run p4/grid_probe --models $QI --namings abstract,aligned,flipped --order seed --envs $PROBE_ENVS --shard $i/6 --out $R/p4/grid_probe.s$i.parquet"
            "$PY" $G --strategies $CORE,counter_prior_matched && "$PY" $G --strategies similarity --seeds 1
        ) > "$LOGS/D_probe_$i.log" 2>&1 &
        sleep 3
    done
    for i in 0 1; do
        "$PY" scripts/run_synth_grid.py --run p4/grid_base_probe --models $QB --namings abstract,flipped --order seed \
            --envs $PROBE_ENVS --strategies zero_shot,random,counter_spurious,counter_prior,counter_prior_matched \
            --shard $i/2 --out $R/p4/grid_base_probe.s$i.parquet > "$LOGS/D_base_probe_$i.log" 2>&1 &
        sleep 3
    done
    wait
    merge $R/p4/grid_probe.parquet $R/p4/grid_probe.s{0,1,2,3,4,5}.parquet
    merge $R/p4/grid_base_probe.parquet $R/p4/grid_base_probe.s{0,1}.parquet
    step "stage D done"
}

for stage in "$@"; do
    case $stage in
        A) stage_A ;;
        B) stage_B ;;
        C) stage_C ;;
        D) stage_D ;;
        *) echo "usage: $0 A|B|C|D ..."; exit 1 ;;
    esac
done
