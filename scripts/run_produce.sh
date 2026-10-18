#!/usr/bin/env bash
# Runs a baseline's production fits, seeds 0-4 at each size, one after another.
#
#   scripts/run_produce.sh METHOD "REGIME:SIZE ..." JOBS
#
# Each fit uses the setting its search exported, refits on the seed's
# subsample and scores the test set across JOBS forked workers. A run whose
# record exists is skipped, so the script can be rerun after an interruption.
# Progress goes to results/logs/produce_METHOD.log.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 FM4PZ_THREADS="${FM4PZ_THREADS:-1}" \
  FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
method=$1
jobs=$2
workers=$3
mkdir -p "$FM4PZ_RESULTS/logs"
log="$FM4PZ_RESULTS/logs/produce_$method.log"
for job in $jobs; do
  IFS=: read -r regime size <<< "$job"
  for seed in 0 1 2 3 4; do
    record="$FM4PZ_RESULTS/runs/$method/${regime}_n${size}_s${seed}.json"
    if [[ -f "$record" ]]; then
      continue
    fi
    echo "$(date -Is) start $method $regime $size s$seed" >> "$log"
    .venv/bin/python -m fm4pz.produce "$method" "$regime" "$size" \
      --seed "$seed" --jobs "$workers" \
      >> "$FM4PZ_RESULTS/logs/produce_${method}.out" 2>&1
    echo "$(date -Is) done  $method $regime $size s$seed exit=$?" >> "$log"
  done
done
echo "$(date -Is) ALL DONE" >> "$log"
