#!/usr/bin/env bash
# Fits round-2 sizes on one GPU, sharing each size's seeds with other lanes.
#
#   scripts/run_round2_fitlane.sh GPU METHOD "REGIME:SIZE ..." [JOBS]
#
# Like scripts/run_round2_fits.sh, it takes the sizes in the order their
# searches are logged done, moves a size's outdated records aside, and fits
# its seeds; but it fits through scripts/run_produce_gpu.sh, whose lock
# per seed lets several lanes, on several GPUs or nodes, split one size's
# five seeds between them.
set -u
cd "$(dirname "$0")/.."
export FM4PZ_ROUND=2 PYTHONPATH="$PWD/src" \
  FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
gpu=$1
method=$2
pending=($3)
workers=${4:-8}
log="$FM4PZ_RESULTS/logs/round2_$method.log"
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
    sleep 120
    continue
  fi
  IFS=: read -r regime size <<< "$ready"
  .venv/bin/python -m fm4pz.stale "$method" "$regime" "$size" >> "$log" 2>&1
  scripts/run_produce_gpu.sh "$gpu" "$method" "$regime:$size" "$workers"
  rest=()
  for job in "${pending[@]}"; do
    [[ "$job" != "$ready" ]] && rest+=("$job")
  done
  pending=("${rest[@]}")
done
echo "$(date -Is) FITLANE DONE gpu=$gpu $method $3" >> "$log"
