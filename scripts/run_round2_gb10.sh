#!/usr/bin/env bash
# Runs a foundation model's round-2 searches and refits on a GB10, size by size.
#
#   scripts/run_round2_gb10.sh METHOD "REGIME:SIZE ..." MIN_COMPLETE \
#     [THREADS] [JOBS] [MIN_FREE_GB] [FIT_FREE_GB]
#
# For each size in turn (give a regime's sizes smallest first, so that each
# search starts from the best of the one before): the search is resumed
# until it holds MIN_COMPLETE completed trials, which --total alone cannot
# promise since it also counts queued, pruned and failed trials; its best
# setting is exported; the size's production records made with another
# setting are moved to results/runs_r1/ (python -m fm4pz.stale); and the
# seeds without a current record are fitted again, each once FIT_FREE_GB of
# memory is free (scripts/run_produce_gb10.sh; by default MIN_FREE_GB). Each
# resumption of a search waits for MIN_FREE_GB, so that it never pushes a box
# shared with other jobs into the out-of-memory killer. A search runs as one process with
# THREADS CPU threads; a fit scores with JOBS processes. Progress goes to
# results/logs/round2_METHOD.log.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" FM4PZ_ROUND=2 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS="${4:-4}" OPENBLAS_NUM_THREADS="${4:-4}" \
  MKL_NUM_THREADS="${4:-4}" FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
method=$1
jobs=$2
min_complete=$3
threads=${4:-4}
workers=${5:-4}
min_free=${6:-40}
fit_free=${7:-$min_free}
mkdir -p "$FM4PZ_RESULTS/logs"
log="$FM4PZ_RESULTS/logs/round2_$method.log"

available_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }

counts() {  # REGIME SIZE -> "TRIALS COMPLETE"
  .venv/bin/python - "$method" "$1" "$2" <<'EOF'
import sys

import optuna
from optuna.storages import journal

from fm4pz import tune

optuna.logging.set_verbosity(optuna.logging.ERROR)
method, regime, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = tune.journal_path(method, regime, size)
if not path.exists():
    print(0, 0)
    raise SystemExit
study = optuna.load_study(
    study_name=f"{method}-{regime}-{size}",
    storage=optuna.storages.JournalStorage(journal.JournalFileBackend(str(path))),
)
states = [trial.state for trial in study.trials]
print(len(states), states.count(optuna.trial.TrialState.COMPLETE))
EOF
}

for job in $jobs; do
  IFS=: read -r regime size <<< "$job"
  for attempt in $(seq 1 40); do
    read -r n_trials n_complete < <(counts "$regime" "$size")
    (( n_complete >= min_complete )) && break
    while (( $(available_kb) < min_free * 1024 * 1024 )); do
      sleep 300
    done
    total=$(( n_trials + min_complete - n_complete ))
    echo "$(date -Is) search $method $regime $size: $n_complete complete" \
      "of $n_trials, to --total $total (attempt $attempt)" >> "$log"
    .venv/bin/python -m fm4pz.tune "$method" "$regime" "$size" \
      --total "$total" --workers 1 --threads "$threads" \
      >> "$FM4PZ_RESULTS/logs/tune_${method}_${regime}_${size}.r2.log" 2>&1
  done
  read -r n_trials n_complete < <(counts "$regime" "$size")
  .venv/bin/python -m fm4pz.tune "$method" "$regime" "$size" --export \
    > /dev/null 2>> "$log"
  echo "$(date -Is) searched $method $regime $size: $n_complete complete" \
    "of $n_trials" >> "$log"
  .venv/bin/python -m fm4pz.stale "$method" "$regime" "$size" >> "$log" 2>&1
  runs=""
  for seed in 0 1 2 3 4; do
    runs+="$regime:$size:$seed "
  done
  scripts/run_produce_gb10.sh "$method" "$runs" "$workers" "$fit_free"
  echo "$(date -Is) produced $method $regime $size" >> "$log"
done
echo "$(date -Is) ALL DONE $method $jobs" >> "$log"
