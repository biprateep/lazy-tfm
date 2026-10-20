#!/usr/bin/env bash
# Rebuilds the ledger from the files on this host and the others.
#
#   scripts/ledger.sh [HOST ...]      # default: tara
#
# Copies each host's search journals, exported settings, run records (not
# their arrays) and logs under results/remote/HOST/, then writes
# results/ledger.csv and results/ledger.md from all of them.
set -u
cd "$(dirname "$0")/.."
results="${FM4PZ_RESULTS:-$PWD/results}"
hosts=("$@")
(( ${#hosts[@]} )) || hosts=(tara)
for host in "${hosts[@]}"; do
  remote="code/photo-z/fm4pz/paper/results"
  local_copy="$results/remote/$host"
  mkdir -p "$local_copy"
  rsync -a --include '*/' --include '*.journal' --include '*.best.json' \
    --exclude '*' "$host:$remote/tuning/" "$local_copy/tuning/"
  rsync -a --include '*/' --include '*.json' --exclude '*' \
    "$host:$remote/runs/" "$local_copy/runs/"
  rsync -a "$host:$remote/logs/" "$local_copy/logs/"
done
FM4PZ_RESULTS="$results" .venv/bin/python -m fm4pz.ledger
