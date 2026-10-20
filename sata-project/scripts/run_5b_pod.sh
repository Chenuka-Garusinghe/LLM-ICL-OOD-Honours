#!/bin/bash
# P5 5b (SATA-MI) on one H200, one session (docs/research_plan.md, 5b implementation).
#   G1  Qwen-Instruct's pool priors and the queries' zero-shot margins for the SATA-MI
#       suites (sata_mi_train 300 tasks, sata_mi_val 60), 3 namings, 8 MPS workers.
#   G2  gate S3a-MI (CPU; picks lambda), then sata_mi and sata_names train in parallel.
#   G3  sata_mi and sata_names on Qwen-Instruct: the LLM validation on sata_llmval, then the
#       evaluation grid (3 namings, 3 seeds). Queries' priors come from earlier zero-shot runs
#       (h200/p5/zero_shot_llmval.parquet, zero_shot_eval.parquet), pool priors from P3/stage F.
# Usage on the pod: bash scripts/run_5b_pod.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/root/venv/bin/python}
export HF_HOME=${HF_HOME:-/dev/shm/hf} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
R=results/v3/synthetic
MI=$R/p5/mi
L=$R/p5/logs5b
mkdir -p $MI $L
QI=qwen2.5-7b-instruct
N3=abstract,aligned,flipped
step() { echo "$(date -u '+%F %T') $1" >> $L/steps.log; }
merge() { "$PY" scripts/merge_shards.py "$@" >> $L/steps.log 2>&1; }

step "G1 start: pool priors and zero-shot margins for the SATA-MI suites"
for i in 0 1 2 3 4 5 6 7; do
    (
        for s in sata_mi_val sata_mi_train; do
            "$PY" scripts/p3_pool_priors.py --suite $s --models $QI --namings $N3 --shard $i/8 \
                --out $MI/pool_prior_$s.s$i.parquet || exit 1
            "$PY" scripts/run_synth_grid.py --run p5/mi/zero_shot_$s --suite $s --models $QI --namings $N3 \
                --strategies zero_shot --shard $i/8 --out $MI/zero_shot_$s.s$i.parquet || exit 1
        done
    ) > $L/G1_$i.log 2>&1 &
    sleep 3
done
wait
for s in sata_mi_val sata_mi_train; do
    merge $MI/pool_prior_$s.parquet $MI/pool_prior_$s.s{0,1,2,3,4,5,6,7}.parquet
    merge $MI/zero_shot_$s.parquet $MI/zero_shot_$s.s{0,1,2,3,4,5,6,7}.parquet
done
step "G1 done"

"$PY" scripts/p5_mi.py gate > $L/G2_gate.log 2>&1 || { step "gate failed to run"; exit 1; }
step "G2 gate: $(tail -1 $L/G2_gate.log)"
for v in sata_mi sata_names; do
    SATA_N_JOBS=10 "$PY" scripts/p5_mi.py train --variant $v --device cuda --epochs ${EPOCHS:-30} > $L/G2_train_$v.log 2>&1 &
    sleep 2
done
wait
step "G2 training done"

step "G3 start: LLM validation, then the evaluation grid"
for i in 0 1 2 3 4 5 6 7; do
    (
        "$PY" scripts/run_synth_grid.py --run p5/llmval_mi --suite sata_llmval --models $QI --namings $N3 --order seed \
            --strategies sata_mi,sata_names --pool-prior $R/h200/p5/pool_prior_${QI}_llmval.parquet \
            --query-prior $R/h200/p5/zero_shot_llmval.parquet --shard $i/8 --out $R/p5/llmval_mi.s$i.parquet || exit 1
        "$PY" scripts/run_synth_grid.py --run p5/grid_mi --models $QI --namings $N3 --order seed \
            --strategies sata_mi,sata_names --pool-prior $R/h200/p3/pool_prior_$QI.parquet \
            --query-prior $R/h200/p5/zero_shot_eval.parquet --shard $i/8 --out $R/p5/grid_mi.s$i.parquet
    ) > $L/G3_$i.log 2>&1 &
    sleep 3
done
wait
merge $R/p5/llmval_mi.parquet $R/p5/llmval_mi.s{0,1,2,3,4,5,6,7}.parquet
merge $R/p5/grid_mi.parquet $R/p5/grid_mi.s{0,1,2,3,4,5,6,7}.parquet
step "G3 done; ALL DONE"
