#!/bin/bash
# P5 on one H200 (11 October 2026): the rest of the remaining work in one pod session.
#   1. Stage E (notebook 03.1) of run_p4_pod.sh, 8 MPS workers, in the background.
#   2. At the same time, the three SATA training runs resume from their last finished
#      epoch (models/*.resume.pt, carried over from the Mac) on the GPU, as MPS clients.
#   3. When both are done, stage F (SATA's LLM validation and evaluation grid).
# Each training run's per-epoch data comes from its seeded stream, so the resumed runs
# continue the same training; only floating-point details differ by device.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/root/venv/bin/python}
export HF_HOME=${HF_HOME:-/dev/shm/hf} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export CUDA_MPS_PIPE_DIRECTORY=/root/mps/pipe CUDA_MPS_LOG_DIRECTORY=/root/mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" results/v3/synthetic/p5/logs
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
L=results/v3/synthetic/p5/logs
echo "$(date -u '+%F %T') start" >> $L/p5_pod.log
bash scripts/run_p4_pod.sh E > $L/stage_E.out 2>&1 &
E_PID=$!
T_PIDS=()
for v in sata sata_agnostic sata_designed; do
    SATA_N_JOBS=6 "$PY" scripts/p5_train_sata.py --variant $v --device cuda >> results/v3/synthetic/p5/train_$v.log 2>&1 &
    T_PIDS+=($!)
    sleep 2
done
wait "${T_PIDS[@]}"
echo "$(date -u '+%F %T') training done" >> $L/p5_pod.log
wait $E_PID
echo "$(date -u '+%F %T') stage E done" >> $L/p5_pod.log
bash scripts/run_p4_pod.sh F > $L/stage_F.out 2>&1
echo "$(date -u '+%F %T') stage F done; ALL DONE" >> $L/p5_pod.log
