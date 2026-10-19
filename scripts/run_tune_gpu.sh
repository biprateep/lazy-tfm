#!/usr/bin/env bash
# Runs a foundation model's searches on one GPU, one after another.
#
#   scripts/run_tune_gpu.sh GPU METHOD "REGIME:SIZE:TRIALS ..." [THREADS]
#
# Like run_tune.sh, but with a single process on GPU number GPU
# (CUDA_VISIBLE_DEVICES), since a foundation model's fit is GPU work. Each
# search runs until it holds TRIALS trials in all, so a search cut short
# resumes where it stopped, and exports its best setting. Give the searches
# of one regime in increasing size, so that each starts from the best
# setting of the one before. Progress goes to results/logs/tune_METHOD.log.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" FM4PZ_ROUND=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS="${4:-4}" OPENBLAS_NUM_THREADS="${4:-4}" \
  MKL_NUM_THREADS="${4:-4}" FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
method=$2
jobs=$3
mkdir -p "$FM4PZ_RESULTS/logs"
log="$FM4PZ_RESULTS/logs/tune_$method.log"
for job in $jobs; do
  IFS=: read -r regime size trials <<< "$job"
  echo "$(date -Is) start $method $regime $size trials=$trials gpu=$1" >> "$log"
  .venv/bin/python -m fm4pz.tune "$method" "$regime" "$size" \
    --total "$trials" --workers 1 --threads "${4:-4}" \
    >> "$FM4PZ_RESULTS/logs/tune_${method}_${regime}_${size}.log" 2>&1
  status=$?
  echo "$(date -Is) done  $method $regime $size exit=$status" >> "$log"
done
echo "$(date -Is) ALL DONE $method $jobs" >> "$log"
