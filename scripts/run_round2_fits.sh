#!/usr/bin/env bash
# Fits a foundation model's sizes once their round-2 searches are done.
#
#   scripts/run_round2_fits.sh METHOD "REGIME:SIZE ..." [JOBS] [FIT_FREE_GB]
#
# The companion of scripts/run_round2_gb10.sh run with FITS=0: it takes the
# sizes in whatever order their searches finish, waiting until that
# launcher has logged a size's search as done (at its completed trials,
# with its setting exported), and for each one moves the
# size's outdated records aside (python -m fm4pz.stale), and fits the seeds
# without a current record, each once FIT_FREE_GB of memory is free
# (scripts/run_produce_gb10.sh). A size's fits thus overlap the next size's
# search. Progress goes to results/logs/round2_METHOD.log.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" FM4PZ_ROUND=2 \
  FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
method=$1
jobs=$2
workers=${3:-4}
fit_free=${4:-40}
log="$FM4PZ_RESULTS/logs/round2_$method.log"
pending=($jobs)
while (( ${#pending[@]} )); do
  ready=""
  for job in "${pending[@]}"; do
    IFS=: read -r regime size <<< "$job"
    if grep -q "searched $method $regime $size: " "$log" 2>/dev/null; then
      ready=$job
      break
    fi
  done
  if [[ -z "$ready" ]]; then
    sleep 300
    continue
  fi
  IFS=: read -r regime size <<< "$ready"
  .venv/bin/python -m fm4pz.stale "$method" "$regime" "$size" >> "$log" 2>&1
  runs=""
  for seed in 0 1 2 3 4; do
    runs+="$regime:$size:$seed "
  done
  scripts/run_produce_gb10.sh "$method" "$runs" "$workers" "$fit_free"
  echo "$(date -Is) produced $method $regime $size" >> "$log"
  rest=()
  for job in "${pending[@]}"; do
    [[ "$job" != "$ready" ]] && rest+=("$job")
  done
  pending=("${rest[@]}")
done
echo "$(date -Is) FITS DONE $method $jobs" >> "$log"
