#!/usr/bin/env bash
# Fits a foundation model's sizes once their round-2 searches are done.
#
#   scripts/run_round2_fits.sh METHOD "REGIME:SIZE ..." [JOBS] [FIT_FREE_GB]
#
# The companion of scripts/run_round2_gb10.sh run with FITS=0: for each size
# in turn, it waits until that launcher has logged the size's search as
# done (at its completed trials, with its setting exported), moves the
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
for job in $jobs; do
  IFS=: read -r regime size <<< "$job"
  until grep -q "searched $method $regime $size: " "$log" 2>/dev/null; do
    sleep 300
  done
  .venv/bin/python -m fm4pz.stale "$method" "$regime" "$size" >> "$log" 2>&1
  runs=""
  for seed in 0 1 2 3 4; do
    runs+="$regime:$size:$seed "
  done
  scripts/run_produce_gb10.sh "$method" "$runs" "$workers" "$fit_free"
  echo "$(date -Is) produced $method $regime $size" >> "$log"
done
echo "$(date -Is) FITS DONE $method $jobs" >> "$log"
