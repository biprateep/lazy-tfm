#!/usr/bin/env bash
# Runs the calibration experiment's fits one at a time on a GB10.
#
#   scripts/run_calibration_gb10.sh METHOD "SIZE ..." "K ..." "SEED ..." JOBS MIN_FREE_GB
#
# For every calibration size K (outermost, so every curve grows together),
# biased training size and seed: the biased training set takes K galaxies of
# the representative calibration sample and is refitted with the setting
# tuned for the biased size alone (fm4pz.produce --calibration). K = 0 is the
# biased production run and is skipped, as is any run whose record exists.
# Before each fit the script waits for MIN_FREE_GB of memory. Progress goes
# to results/logs/calibration_METHOD.log.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  FM4PZ_THREADS=1 FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
method=$1
sizes=$2
calibrations=$3
seeds=$4
workers=$5
min_free_kb=$(($6 * 1024 * 1024))
mkdir -p "$FM4PZ_RESULTS/logs"
log="$FM4PZ_RESULTS/logs/calibration_$method.log"
available() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
for k in $calibrations; do
  (( k == 0 )) && continue
  for size in $sizes; do
    for seed in $seeds; do
      record="$FM4PZ_RESULTS/calibration/$method/biased_cal${k}_n${size}_s${seed}.json"
      [[ -f "$record" ]] && continue
      while (( $(available) < min_free_kb )); do
        sleep 120
      done
      echo "$(date -Is) start $method biased $size c$k s$seed" >> "$log"
      .venv/bin/python -m fm4pz.produce "$method" biased "$size" \
        --seed "$seed" --jobs "$workers" --calibration "$k" \
        >> "$FM4PZ_RESULTS/logs/calibration_${method}.out" 2>&1
      status=$?
      echo "$(date -Is) done  $method biased $size c$k s$seed exit=$status" >> "$log"
    done
  done
done
echo "$(date -Is) ALL DONE $method" >> "$log"
