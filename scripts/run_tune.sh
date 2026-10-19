#!/usr/bin/env bash
# Runs a baseline's hyperparameter searches one after another, detached-safe.
#
#   scripts/run_tune.sh METHOD "REGIME:SIZE:TRIALS ..." WORKERS
#
# Each search runs WORKERS single-threaded processes that share its Optuna
# journal under results/tuning/, until it holds TRIALS trials in all (so a
# search cut short resumes where it stopped), and exports its best setting.
# Progress goes to results/logs/tune_METHOD.log.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
method=$1
jobs=$2
workers=$3
mkdir -p "$FM4PZ_RESULTS/logs"
log="$FM4PZ_RESULTS/logs/tune_$method.log"
for job in $jobs; do
  IFS=: read -r regime size trials <<< "$job"
  echo "$(date -Is) start $method $regime $size trials=$trials" >> "$log"
  .venv/bin/python -m fm4pz.tune "$method" "$regime" "$size" \
    --total "$trials" --workers "$workers" --threads 1 \
    >> "$FM4PZ_RESULTS/logs/tune_${method}_${regime}_${size}.log" 2>&1
  status=$?
  echo "$(date -Is) done  $method $regime $size exit=$status" >> "$log"
done
echo "$(date -Is) ALL DONE" >> "$log"
