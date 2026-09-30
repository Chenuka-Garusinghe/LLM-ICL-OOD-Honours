#!/bin/sh
# P2 pilot (docs/research_plan.md): run the learnability grid on the pilot
# suite and evaluate the gate, then benchmark both instruct models (throughput
# and cached-versus-full agreement). The gate comes first because it decides k.
# Resumable: the grid skips units already saved, and a benchmark is skipped
# when its JSON exists. Logs go to results/v3/synthetic/p2/logs/.
#
# Usage (from the project root; keep the Mac awake and memory free):
#   caffeinate -i sh scripts/run_p2.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGS=results/v3/synthetic/p2/logs
mkdir -p "$LOGS"

"$PY" scripts/run_synth_grid.py --run p2_learnability --suite pilot \
    --models qwen2.5-7b-instruct,llama-3.1-8b-instruct --envs id \
    --strategies zero_shot,label_diversity --label-modes gold,shuffled --k 8,16,32 \
    >> "$LOGS/learnability_grid.log" 2>&1

"$PY" scripts/p2_learnability_gate.py > "$LOGS/learnability_gate.log" 2>&1

for model in qwen2.5-7b-instruct llama-3.1-8b-instruct; do
    if [ ! -f "results/v3/synthetic/p2/benchmark_$model.json" ]; then
        "$PY" scripts/p2_benchmark.py --model "$model" > "$LOGS/benchmark_$model.log" 2>&1
    fi
done
echo "P2 done"
