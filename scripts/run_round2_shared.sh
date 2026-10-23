#!/usr/bin/env bash
# Runs one round-2 search with a worker on each of several GPUs of this node.
#
#   scripts/run_round2_shared.sh METHOD REGIME:SIZE MIN_COMPLETE "GPU ..." \
#     [THREADS] [FIRST_WORKER]
#
# The workers share the search's Optuna journal. Its trials left running by
# an earlier host are first marked failed, once, while no worker runs; each
# worker then adds one trial at a time (--trials, which never marks another
# worker's running trial failed) until the search holds MIN_COMPLETE
# completed trials. When every worker has stopped, the best setting is
# exported and the search is logged as done, so that
# scripts/run_round2_fits.sh or run_round2_fitlane.sh can fit it. Workers
# are numbered from FIRST_WORKER, which seeds their samplers apart.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src" FM4PZ_ROUND=2 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS="${5:-8}" OPENBLAS_NUM_THREADS="${5:-8}" \
  MKL_NUM_THREADS="${5:-8}" FM4PZ_RESULTS="${FM4PZ_RESULTS:-$PWD/results}"
method=$1
IFS=: read -r regime size <<< "$2"
min_complete=$3
gpus=$4
threads=${5:-8}
first=${6:-100}
log="$FM4PZ_RESULTS/logs/round2_$method.log"
out="$FM4PZ_RESULTS/logs/tune_${method}_${regime}_${size}.r2.log"

counts() {  # -> "TRIALS COMPLETE"
  .venv/bin/python - "$method" "$regime" "$size" <<'PY'
import sys

import optuna
from optuna.storages import journal

from fm4pz import tune

optuna.logging.set_verbosity(optuna.logging.ERROR)
method, regime, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = tune.journal_path(method, regime, size)
if not path.exists():
    print(0, 0)
    raise SystemExit
study = optuna.load_study(
    study_name=f"{method}-{regime}-{size}",
    storage=optuna.storages.JournalStorage(journal.JournalFileBackend(str(path))),
)
states = [trial.state for trial in study.trials]
print(len(states), states.count(optuna.trial.TrialState.COMPLETE))
PY
}

.venv/bin/python -c "from fm4pz import tune; tune.warm_start('$method', '$regime', $size); print(tune._settle('$method', '$regime', $size))" >> "$out" 2>&1
echo "$(date -Is) shared search $method $regime $size on GPUs $gpus" >> "$log"
k=$first
for gpu in $gpus; do
  (
    while true; do
      read -r _ n_complete < <(counts)
      (( n_complete >= min_complete )) && break
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m fm4pz.tune "$method" \
        "$regime" "$size" --trials 1 --workers 1 --threads "$threads" \
        --first-worker "$k" >> "$out" 2>&1
    done
  ) &
  k=$((k + 1))
done
wait
read -r n_trials n_complete < <(counts)
.venv/bin/python -m fm4pz.tune "$method" "$regime" "$size" --export \
  > /dev/null 2>> "$log"
echo "$(date -Is) searched $method $regime $size: $n_complete complete" \
  "of $n_trials" >> "$log"
