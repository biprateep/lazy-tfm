#!/usr/bin/env bash
# Runs every foundation-model search and production fit on one 4-GPU node.
#
#   scripts/run_tfms.sh
#
# Each lane below is one sequence on one GPU: a model's searches in
# increasing training size, then its five production fits per size. A
# search resumes from its journal and stops at its planned number of trials
# (queued and failed trials included); a fit whose record exists is skipped.
# So the script can be started again on a new node after an allocation ends,
# once the stale locks of interrupted fits are removed, which it does first.
# LimiX-2's representative fits are the longest, so every lane that finishes
# early joins them. Logs go to results/logs/.
set -u
cd "$(dirname "$0")/.."
export FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
rm -rf "$FM4PZ_RESULTS"/runs/*/*.lock

# Trials per search, for the regime's sizes in order: what each search
# holds once finished, so that starting the script again adds none. (A
# search's first run counts its queued warm starts towards its total and
# then runs them, so the first searches of each model hold fewer trials
# than they were started with; the later totals were reached by topping
# them up, when it was clear that a model's fits were cheap enough.)
declare -A TRIALS=(
  [tabpfn]="25 25 25 20 20 20 20"
  [kumo]="13 10 16 16 16 16 16"
  [limix]="12 9 7 14 14 14 14"
)
declare -A SIZES=(
  [representative]="1000 5000 10000 15000 20000 30000 43486"
  [biased]="1000 5000 10000 15000 20000 30000 35011"
)

searches() {  # METHOD REGIME -> "REGIME:SIZE:TRIALS ..."
  local -a sizes trials
  read -r -a sizes <<< "${SIZES[$2]}"
  read -r -a trials <<< "${TRIALS[$1]}"
  local out="" k
  for k in "${!sizes[@]}"; do
    out+="$2:${sizes[$k]}:${trials[$k]} "
  done
  echo "$out"
}

fits() {  # REGIME -> "REGIME:SIZE ..."
  local out="" size
  for size in ${SIZES[$1]}; do
    out+="$1:$size "
  done
  echo "$out"
}

lane() {  # GPU METHOD REGIME
  scripts/run_tune_gpu.sh "$1" "$2" "$(searches "$2" "$3")" 4
  scripts/run_produce_gpu.sh "$1" "$2" "$(fits "$3")" 8
  WAIT=1 scripts/run_produce_gpu.sh "$1" limix "$(fits representative)" 8
}

lane 0 tabpfn biased > /dev/null 2>&1 &
lane 0 kumo biased > /dev/null 2>&1 &
lane 1 tabpfn representative > /dev/null 2>&1 &
lane 1 limix biased > /dev/null 2>&1 &
lane 2 limix representative > /dev/null 2>&1 &
lane 3 kumo representative > /dev/null 2>&1 &
wait
