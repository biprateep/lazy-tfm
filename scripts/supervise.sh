#!/usr/bin/env bash
# Runs a launcher again until its log says it finished.
#
#   scripts/supervise.sh LOG "DONE TEXT" COMMAND [ARG ...]
#
# A launcher killed partway (by the out-of-memory killer on a shared box, or
# an interruption) resumes where its searches and fits stopped, since every
# step of the launchers is resumable. This loop restarts it, five minutes
# after it exits, until LOG holds DONE TEXT.
set -u
cd "$(dirname "$0")/.."
log=$1
done_text=$2
shift 2
until grep -qF -- "$done_text" "$log" 2>/dev/null; do
  "$@"
  sleep 300
done
