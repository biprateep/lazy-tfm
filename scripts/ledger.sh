#!/usr/bin/env bash
# Rebuilds the common ledger from every result tree on every host.
#
#   scripts/ledger.sh [[HOST:]RESULTS_DIR ...]
#
# Copies each source's search journals, exported settings, run and
# calibration records (not their arrays) and logs under
# results/remote/<host>_<tree>/, then writes results/ledger.csv, .md and .json
# from all of them and from this host's own results. A source without HOST is a directory on this host. With no
# arguments it reads $FM4PZ_LEDGER_SOURCES (space-separated; set and empty
# for this host's results alone), by default tara's paper/ and paper-tfm/
# checkouts. This ledger is the one record of the
# experiments' state that every session reads.
set -u
cd "$(dirname "$0")/.."
results="${FM4PZ_RESULTS:-$PWD/results}"
sources=("$@")
(( ${#sources[@]} )) || read -r -a sources <<< "${FM4PZ_LEDGER_SOURCES-\
tara:code/photo-z/fm4pz/paper/results tara:code/photo-z/fm4pz/paper-tfm/results}"
for source in "${sources[@]}"; do
  if [[ $source == *:* ]]; then
    host=${source%%:*}
  else
    host=$(hostname -s)
  fi
  tree=$(basename "$(dirname "${source#*:}")")
  local_copy="$results/remote/${host}_${tree}"
  mkdir -p "$local_copy"
  rsync -a --include '*/' --include '*.journal' --include '*.best.json' \
    --exclude '*' "$source/tuning/" "$local_copy/tuning/" 2>/dev/null
  rsync -a --include '*/' --include '*.json' --exclude '*' \
    "$source/runs/" "$local_copy/runs/" 2>/dev/null
  rsync -a --include '*/' --include '*.json' --exclude '*' \
    "$source/calibration/" "$local_copy/calibration/" 2>/dev/null
  rsync -a "$source/logs/" "$local_copy/logs/" 2>/dev/null
done
FM4PZ_RESULTS="$results" .venv/bin/python -m fm4pz.ledger
