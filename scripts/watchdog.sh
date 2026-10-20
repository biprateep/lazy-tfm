#!/usr/bin/env bash
# Stops a hyperparameter search that has stopped making progress.
#
#   scripts/watchdog.sh [HOURS]
#
# Every ten minutes, each running `python -m fm4pz.tune METHOD REGIME SIZE`
# whose Optuna journal has not been written for HOURS hours (default 3) is
# stopped together with its worker processes, and the stop is logged to
# results/logs/watchdog.log. The launcher that started the search then moves
# on, and rerunning it later resumes the search (run_tune.sh passes --total).
set -u
cd "$(dirname "$0")/.."
limit=$(( ${1:-3} * 3600 ))
results="${FM4PZ_RESULTS:-$PWD/results}"
mkdir -p "$results/logs"
while true; do
  for pid in $(pgrep -f "python -m fm4pz.tune [a-z]+ [a-z]+ [0-9]+"); do
    read -r -a argv < <(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    [[ ${#argv[@]} -ge 6 ]] || continue
    method=${argv[3]} regime=${argv[4]} size=${argv[5]}
    round=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
      | sed -n 's/^FM4PZ_ROUND=//p')
    suffix=".journal"
    [[ -n "$round" && "$round" != "1" ]] && suffix=".r$round.journal"
    journal="$results/tuning/$method/${regime}_n${size}${suffix}"
    [[ -f "$journal" ]] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$journal") ))
    if (( age > limit )); then
      echo "$(date -Is) stopping $method $regime $size (pid $pid):" \
        "journal silent for $(( age / 60 )) min" >> "$results/logs/watchdog.log"
      pkill -TERM -P "$pid"
      kill -TERM "$pid"
    fi
  done
  sleep 600
done
