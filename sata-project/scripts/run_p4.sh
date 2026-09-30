#!/bin/sh
# P4 (docs/research_plan.md) as resumable stages; finished units are skipped,
# so rerunning a stage continues where it stopped. Logs: results/v3/synthetic/p4/logs/.
#
#   abstract      abstract names: zero-shot and the naming-free strategies (3 seeds),
#                 the random-label control, similarity (1 seed), and the measured
#                 zero-shot prior on every pool row
#   named_core    zero-shot and the naming-free strategies under aligned and flipped names
#   named_priors  the pool priors under aligned and flipped names
#   prior         counter_prior under every naming (needs the pool priors)
#   rq3           RQ3 on seed 0; rq3_more adds seeds 1 and 2
#   named_extra   the random-label control and similarity under aligned and flipped names
#   base          Qwen2.5-7B base: P2 checks at k = 8, pool priors, reduced grid
#
# Usage (project root; keep the Mac awake and memory free):
#   caffeinate -dimsu sh scripts/run_p4.sh abstract named_core named_priors prior rq3 named_extra base rq3_more
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
LOGS=results/v3/synthetic/p4/logs
mkdir -p "$LOGS"
QI=qwen2.5-7b-instruct
QB=qwen2.5-7b-base
CORE=zero_shot,random,label_diversity,feature_range,rule_diversity,counter_spurious
step() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> "$LOGS/steps.log"; }
grid() {  # namings, then extra arguments
    n=$1; shift
    "$PY" scripts/run_synth_grid.py --run p4/grid --models $QI --namings "$n" --order seed "$@" >> "$LOGS/grid.log" 2>&1
}
priors() { "$PY" scripts/p3_pool_priors.py --models "$1" --namings "$2" >> "$LOGS/priors.log" 2>&1; }
extra() {
    step "grid $1: random-label control"; grid "$1" --strategies label_diversity --label-modes shuffled
    step "grid $1: similarity (1 seed)";  grid "$1" --strategies similarity --seeds 1
}

for stage in "$@"; do
    case $stage in
        abstract)
            step "grid abstract: core strategies"; grid abstract --strategies $CORE
            extra abstract
            step "pool priors abstract"; priors $QI abstract ;;
        named_core)   step "grid aligned,flipped: core strategies"; grid aligned,flipped --strategies $CORE ;;
        named_priors) step "pool priors aligned,flipped"; priors $QI aligned,flipped ;;
        prior)        step "grid: counter_prior"; grid abstract,aligned,flipped --strategies counter_prior ;;
        rq3)          step "rq3: seed 0"; "$PY" scripts/run_rq3.py --run p4/rq3 --models $QI --seeds 1 >> "$LOGS/rq3.log" 2>&1 ;;
        rq3_more)     step "rq3: seeds 1-2"; "$PY" scripts/run_rq3.py --run p4/rq3 --models $QI --seeds 3 >> "$LOGS/rq3.log" 2>&1 ;;
        named_extra)  extra aligned,flipped ;;
        base)
            step "base: P2 checks at k = 8 (pilot)"
            "$PY" scripts/run_synth_grid.py --run p4/base_p2 --suite pilot --models $QB --envs id \
                --strategies zero_shot,label_diversity --label-modes gold,shuffled --k 8 >> "$LOGS/base.log" 2>&1
            if "$PY" scripts/p4_base_check.py >> "$LOGS/base.log" 2>&1; then
                step "base: pool priors"; priors $QB abstract,flipped
                step "base: reduced grid"
                "$PY" scripts/run_synth_grid.py --run p4/grid_base --models $QB --namings abstract,flipped --order seed \
                    --strategies zero_shot,random,counter_spurious,counter_prior >> "$LOGS/base.log" 2>&1
            else
                step "base: reduced grid skipped (label mass below 0.9; p4/base_p2_gate.json)"
            fi ;;
        *) echo "unknown stage $stage"; exit 1 ;;
    esac
    step "stage $stage done"
done
