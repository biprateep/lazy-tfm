# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""The paper's steps for one regime: search, then production.

:func:`search_all` runs the hyperparameter search of every method at every
training size, in increasing size so that each search can start from the
setting the next smaller size chose, and writes the chosen setting beside the
search (:mod:`fm4pz.tune`). :func:`produce_all` fits each chosen setting on
the training subsets of every seed and scores the test set
(:mod:`fm4pz.produce`); in the biased regime it can also add representative
calibration galaxies to each training set. Both skip what is already done, so
an interrupted step resumes where it stopped.

Typical usage example:

  pipeline.search_all("representative", config.METHODS, workers=16)
  pipeline.produce_all("representative", config.METHODS, jobs=8)
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import os

import optuna
from optuna.storages import journal

from fm4pz import baselines
from fm4pz import config
from fm4pz import produce
from fm4pz import tune

__all__ = [
    "completed_trials",
    "produce_all",
    "search_all",
    "search_one",
    "sizes_of",
]

# A foundation model's search is resumed at most this many times to reach
# config.MIN_COMPLETE completed trials.
_MAX_RESUMES = 20

# Threads of one foundation-model search process, which runs on the GPU.
_GPU_THREADS = 4


def sizes_of(regime: str) -> tuple[int, ...]:
    """The training sizes of a regime's scaling curve."""
    return (
        config.REPRESENTATIVE_SIZES
        if regime == "representative"
        else config.BIASED_SIZES
    )


def completed_trials(method: str, regime: str, n_train: int) -> int:
    """How many trials of a search have completed, pruned ones excluded.

    Args:
        method: A method of :data:`fm4pz.config.METHODS`.
        regime: ``"representative"`` or ``"biased"``.
        n_train: The training size.

    Returns:
        The completed trials; 0 if the search has not begun.
    """
    path = tune.journal_path(method, regime, n_train)
    if not path.exists():
        return 0
    study = optuna.load_study(
        study_name=f"{method}-{regime}-{n_train}",
        storage=optuna.storages.JournalStorage(
            journal.JournalFileBackend(str(path))
        ),
    )
    states = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    return len(states)


def search_one(
    method: str, regime: str, n_train: int, *, workers: int
) -> dict[str, object]:
    """Runs one search to its budget and exports the chosen setting.

    A bespoke estimator's search runs to its budget of
    :data:`fm4pz.config.SEARCH_TRIALS` with ``workers`` processes; a
    foundation model's runs in one process on the GPU and is resumed until
    :data:`fm4pz.config.MIN_COMPLETE` trials have completed.

    Args:
        method: A method of :data:`fm4pz.config.METHODS`.
        regime: ``"representative"`` or ``"biased"``.
        n_train: The training size.
        workers: Processes of a bespoke estimator's search.

    Returns:
        The exported setting, as ``tune --export`` writes it.
    """
    common = [method, regime, str(n_train)]
    budget = config.SEARCH_TRIALS.get(method, {}).get(regime)
    if budget is not None:
        tune.main([*common, "--total", str(budget), "--workers", str(workers)])
    else:
        target = config.MIN_COMPLETE
        total = target + 10
        for _ in range(_MAX_RESUMES):
            tune.main(
                [
                    *common,
                    "--total",
                    str(total),
                    "--workers",
                    "1",
                    "--threads",
                    str(_GPU_THREADS),
                ]
            )
            done = completed_trials(method, regime, n_train)
            if done >= target:
                break
            # Pruned trials do not count; ask for twice the shortfall.
            total += 2 * (target - done)
    path = tune.best_path(method, regime, n_train)
    best: dict[str, object] = json.loads(path.read_text())
    return best


def search_all(
    regime: str,
    methods: Sequence[str],
    sizes: Sequence[int] | None = None,
    *,
    workers: int,
) -> None:
    """Runs every method's search at every size of a regime, smallest first.

    Args:
        regime: ``"representative"`` or ``"biased"``.
        methods: Methods of :data:`fm4pz.config.METHODS`.
        sizes: Training sizes; by default the regime's scaling curve.
        workers: Processes of a bespoke estimator's search.
    """
    for method in methods:
        for n_train in sorted(sizes or sizes_of(regime)):
            best = search_one(method, regime, n_train, workers=workers)
            print(
                f"{method} {regime} {n_train}: CV loss "
                f"{best['cv_loss']:.3f} from {best['n_complete']} trials",
                flush=True,
            )


def produce_all(
    regime: str,
    methods: Sequence[str],
    sizes: Sequence[int] | None = None,
    seeds: Sequence[int] = config.SEEDS,
    calibrations: Sequence[int] = (0,),
    *,
    jobs: int,
) -> None:
    """Runs every production fit of a regime that has no record yet.

    Args:
        regime: ``"representative"`` or ``"biased"``.
        methods: Methods of :data:`fm4pz.config.METHODS`.
        sizes: Training sizes; by default the regime's scaling curve.
        seeds: The subsamples.
        calibrations: Calibration galaxies added to each biased training
            set; 0 is the plain biased run.
        jobs: Processes scoring the test set.
    """
    for method in methods:
        on_gpu = baselines.get(method).on_gpu
        os.environ["FM4PZ_THREADS"] = "1"
        for n_calibration in calibrations:
            for n_train in sizes or sizes_of(regime):
                for seed in seeds:
                    path = (
                        produce.calibration_path(
                            method, n_train, n_calibration, seed
                        )
                        if n_calibration
                        else produce.run_path(method, regime, n_train, seed)
                    )
                    if path.with_suffix(".json").exists():
                        continue
                    record = produce.produce(
                        method,
                        regime,
                        n_train,
                        seed,
                        jobs,
                        n_calibration=n_calibration,
                    )
                    device = "GPU" if on_gpu else "CPU"
                    print(
                        f"{method} {regime} {n_train} +{n_calibration} "
                        f"s{seed} ({device}): CDE loss "
                        f"{record['metrics']['cde_loss']:.3f}",
                        flush=True,
                    )
