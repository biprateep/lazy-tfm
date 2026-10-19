#!/usr/bin/env bash
# Runs a foundation model's production fits on one GPU, seeds 0-4 per size.
#
#   scripts/run_produce_gpu.sh GPU METHOD "REGIME:SIZE ..." JOBS
#
# Like run_produce.sh, but on GPU number GPU (CUDA_VISIBLE_DEVICES): the
# model fits and predicts there while JOBS single-threaded processes put the
# predictions on the grid and score them. A run whose record exists is
# skipped, so the script can be rerun after an interruption. With WAIT=1 in
# the environment, each size waits until its search has exported a setting,
# so that the fits can follow searches still running. Progress goes to
# results/logs/produce_METHOD.log.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  FM4PZ_THREADS=1 FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
method=$2
jobs=$3
workers=$4
mkdir -p "$FM4PZ_RESULTS/logs"
log="$FM4PZ_RESULTS/logs/produce_$method.log"
for job in $jobs; do
  IFS=: read -r regime size <<< "$job"
  best="$FM4PZ_RESULTS/tuning/$method/${regime}_n${size}.best.json"
  while [[ "${WAIT:-0}" == 1 && ! -f "$best" ]]; do
    sleep 60
  done
  for seed in 0 1 2 3 4; do
    record="$FM4PZ_RESULTS/runs/$method/${regime}_n${size}_s${seed}.json"
    if [[ -f "$record" ]]; then
      continue
    fi
    echo "$(date -Is) start $method $regime $size s$seed gpu=$1" >> "$log"
    .venv/bin/python -m fm4pz.produce "$method" "$regime" "$size" \
      --seed "$seed" --jobs "$workers" \
      >> "$FM4PZ_RESULTS/logs/produce_${method}.out" 2>&1
    status=$?
    echo "$(date -Is) done  $method $regime $size s$seed exit=$status" >> "$log"
  done
done
echo "$(date -Is) ALL DONE $method $jobs" >> "$log"
