#!/usr/bin/env bash
# Runs production fits one at a time on a box with one GPU and shared memory.
#
#   scripts/run_produce_gb10.sh METHOD "REGIME:SIZE:SEED ..." JOBS MIN_FREE_GB
#
# For a GB10, whose GPU and CPUs share one memory: before each fit the
# script waits until at least MIN_FREE_GB of memory is available, so that a
# fit neither starves other jobs on the box nor is killed for want of
# memory; JOBS processes score its predictions. A run whose record exists
# is skipped. Progress goes to results/logs/produce_METHOD.log.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" FM4PZ_ROUND=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  FM4PZ_THREADS=1 FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
method=$1
runs=$2
workers=$3
min_free_kb=$(($4 * 1024 * 1024))
mkdir -p "$FM4PZ_RESULTS/logs"
log="$FM4PZ_RESULTS/logs/produce_$method.log"
available() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
for run in $runs; do
  IFS=: read -r regime size seed <<< "$run"
  record="$FM4PZ_RESULTS/runs/$method/${regime}_n${size}_s${seed}.json"
  if [[ -f "$record" ]]; then
    continue
  fi
  while (( $(available) < min_free_kb )); do
    sleep 300
  done
  echo "$(date -Is) start $method $regime $size s$seed $(hostname)" >> "$log"
  .venv/bin/python -m fm4pz.produce "$method" "$regime" "$size" \
    --seed "$seed" --jobs "$workers" \
    >> "$FM4PZ_RESULTS/logs/produce_${method}.out" 2>&1
  status=$?
  echo "$(date -Is) done  $method $regime $size s$seed exit=$status" >> "$log"
done
echo "$(date -Is) ALL DONE $method $runs" >> "$log"
