#!/usr/bin/env bash
# Rebuilds the common ledger from every result tree on every host.
#
#   scripts/ledger.sh [HOST:RESULTS_DIR ...]
#
# Copies each source's search journals, exported settings, run records (not
# their arrays) and logs under results/remote/<host>_<tree>/, then writes
# results/ledger.csv and results/ledger.md from all of them and from this
# host's own results. With no arguments it reads the trees where the runs of
# both sessions land: tara's paper/ and paper-tfm/ checkouts. This ledger is
# the one record of the experiments' state that every session reads.
set -u
cd "$(dirname "$0")/.."
results="${FM4PZ_RESULTS:-$PWD/results}"
sources=("$@")
(( ${#sources[@]} )) || sources=(
  "tara:code/photo-z/fm4pz/paper/results"
  "tara:code/photo-z/fm4pz/paper-tfm/results"
)
for source in "${sources[@]}"; do
  host=${source%%:*}
  remote=${source#*:}
  tree=$(basename "$(dirname "$remote")")
  local_copy="$results/remote/${host}_${tree}"
  mkdir -p "$local_copy"
  rsync -a --include '*/' --include '*.journal' --include '*.best.json' \
    --exclude '*' "$host:$remote/tuning/" "$local_copy/tuning/" 2>/dev/null
  rsync -a --include '*/' --include '*.json' --exclude '*' \
    "$host:$remote/runs/" "$local_copy/runs/" 2>/dev/null
  rsync -a "$host:$remote/logs/" "$local_copy/logs/" 2>/dev/null
done
FM4PZ_RESULTS="$results" .venv/bin/python -m fm4pz.ledger
