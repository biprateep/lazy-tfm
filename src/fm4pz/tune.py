# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Runs the hyperparameter search of one baseline at one training size.

The search is 5-fold cross-validation on the seed-0 training subset of the
size (the subset of the submitted paper), recorded in an Optuna journal under
``$FM4PZ_RESULTS/tuning/<method>/<regime>_n<size>.journal``. Several
processes may run one search at once (``--workers``); each adds ``--trials``
trials. ``--export`` writes the best setting to ``best.json`` beside it.

Typical usage example:

  python -m fm4pz.tune gpz representative 1000 --trials 60 --workers 4
  python -m fm4pz.tune gpz representative 1000 --export
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import multiprocessing
import os
import pathlib
import platform
from typing import Any

import optuna
from optuna.storages import journal

from fm4pz import baselines
from fm4pz import config
from fm4pz import cv
from fm4pz import splits

__all__ = [
    "best_path",
    "journal_path",
    "latest_best_path",
    "main",
    "search",
    "search_round",
]


def search_round() -> int:
    """The round of searches being run: ``$FM4PZ_ROUND``, by default 1.

    A later round searches a revised space in journals of its own, starting
    from the best settings of the round before; round 1 keeps its files.
    """
    return int(os.environ.get("FM4PZ_ROUND", "1"))


def journal_path(
    method: str, regime: str, n_train: int, round_: int | None = None
) -> pathlib.Path:
    """The journal file of one search, of ``round_`` (default: the current)."""
    round_ = search_round() if round_ is None else round_
    suffix = ".journal" if round_ == 1 else f".r{round_}.journal"
    return (
        config.results_dir()
        / "tuning"
        / method
        / f"{regime}_n{n_train}{suffix}"
    )


def best_path(
    method: str, regime: str, n_train: int, round_: int | None = None
) -> pathlib.Path:
    """Where the best setting of one search is exported."""
    return journal_path(method, regime, n_train, round_).with_suffix(
        ".best.json"
    )


def latest_best_path(method: str, regime: str, n_train: int) -> pathlib.Path:
    """The exported best setting of the latest round that has one.

    Raises:
        FileNotFoundError: If no round has exported a setting.
    """
    for round_ in range(9, 0, -1):
        path = best_path(method, regime, n_train, round_)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no search of {method} {regime} {n_train} has exported a setting"
    )


def search(
    method: str, regime: str, n_train: int, n_trials: int, worker: int
) -> None:
    """Adds ``n_trials`` trials to one search, as one worker process.

    Args:
        method: A baseline name, one of :data:`fm4pz.baselines.NAMES`.
        regime: ``"representative"`` or ``"biased"``.
        n_train: The training size.
        n_trials: Trials to add.
        worker: This worker's index, which seeds its sampler.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    split = splits.load(regime, n_train=n_train, seed=0)
    cv.run_study(
        baselines.get(method),
        split,
        n_trials=n_trials,
        storage=journal_path(method, regime, n_train),
        seed=worker,
    )


def export(method: str, regime: str, n_train: int) -> dict[str, object]:
    """Writes the best setting of a finished search.

    Args:
        method: A baseline name.
        regime: ``"representative"`` or ``"biased"``.
        n_train: The training size.

    Returns:
        What was written: :func:`fm4pz.cv.best_params` plus provenance.
    """
    storage = optuna.storages.JournalStorage(
        journal.JournalFileBackend(str(journal_path(method, regime, n_train)))
    )
    study = optuna.load_study(
        study_name=f"{method}-{regime}-{n_train}", storage=storage
    )
    best = cv.best_params(study)
    best["trial_params"] = dict(study.best_trial.params)
    states = [trial.state.name for trial in study.trials]
    best.update(
        {
            "round": search_round(),
            "method": method,
            "regime": regime,
            "n_train": n_train,
            "train_checksum": splits.checksum(
                splits.train_rows(regime, n_train, 0)
            ),
            "n_complete": states.count("COMPLETE"),
            "n_pruned": states.count("PRUNED"),
            "n_failed": states.count("FAIL"),
            "host": platform.node(),
        }
    )
    best_path(method, regime, n_train).write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n"
    )
    return best


def warm_start(method: str, regime: str, n_train: int) -> int:
    """Queues the first settings of a search that has not started.

    They are the method's own :meth:`~fm4pz.cv.Method.warm_starts` and the
    best setting of the next smaller size of the regime, when its search has
    finished. A later round also starts from the earlier round's five best
    settings at this size and, for a method whose
    :attr:`~fm4pz.cv.Method.cross_size_warm_starts` is true, from its best
    setting at every other size. Each setting goes through the method's
    :meth:`~fm4pz.cv.Method.adapt_warm_start` first.

    Args:
        method: A baseline name.
        regime: ``"representative"`` or ``"biased"``.
        n_train: The training size.

    Returns:
        How many settings were queued (0 if the search had begun).
    """
    baseline = baselines.get(method)
    queued: list[Mapping[str, Any] | None] = list(baseline.warm_starts())
    sizes = (
        config.REPRESENTATIVE_SIZES
        if regime == "representative"
        else config.BIASED_SIZES
    )
    if search_round() > 1:
        queued += _top_settings(
            journal_path(method, regime, n_train, search_round() - 1),
            f"{method}-{regime}-{n_train}",
        )
        if baseline.cross_size_warm_starts:
            queued += [
                _trial_params(
                    best_path(method, regime, size, search_round() - 1)
                )
                for size in sizes
                if size != n_train
            ]
    smaller = [size for size in sizes if size < n_train]
    if smaller:
        queued.append(_trial_params(best_path(method, regime, smaller[-1])))
    settings = [
        adapted
        for params in queued
        if params is not None
        and (adapted := baseline.adapt_warm_start(params, n_train)) is not None
    ]
    study = cv.open_study(
        baseline,
        splits.load(regime, n_train=n_train, seed=0),
        storage=journal_path(method, regime, n_train),
        enqueue=settings,
    )
    return len(settings) if len(study.trials) == len(settings) else 0


def _trial_params(path: pathlib.Path) -> dict[str, Any] | None:
    """The Optuna parameters of an exported setting, if it was exported."""
    if not path.exists():
        return None
    exported = json.loads(path.read_text())
    return exported.get("trial_params")


def _top_settings(
    storage: pathlib.Path, study_name: str, count: int = 5
) -> list[dict[str, object]]:
    """The parameters of the best finished trials of an earlier search.

    Args:
        storage: The earlier search's journal.
        study_name: Its study's name.
        count: How many settings to return.

    Returns:
        Their Optuna parameters, best first; empty if there is no journal.
    """
    if not storage.exists():
        return []
    study = optuna.load_study(
        study_name=study_name,
        storage=optuna.storages.JournalStorage(
            journal.JournalFileBackend(str(storage))
        ),
    )
    done = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    done.sort(key=lambda trial: float(trial.values[0]))
    return [dict(trial.params) for trial in done[:count]]


def _settle(method: str, regime: str, n_train: int) -> int:
    """Marks a search's abandoned trials failed; returns its trial count.

    A trial still marked running when no worker is (its process was
    killed, as when an allocation ends) would stay running forever; it is
    recorded as failed, so that resuming the search counts it and does not
    wait for it. Call it only when no worker of the search is running.

    Args:
        method: A baseline name.
        regime: ``"representative"`` or ``"biased"``.
        n_train: The training size.

    Returns:
        How many trials the search holds, the failed ones included.
    """
    storage = optuna.storages.JournalStorage(
        journal.JournalFileBackend(str(journal_path(method, regime, n_train)))
    )
    study = optuna.load_study(
        study_name=f"{method}-{regime}-{n_train}", storage=storage
    )
    for trial in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    ):
        storage.set_trial_state_values(
            trial._trial_id,  # noqa: SLF001 - Optuna's own storage key.
            state=optuna.trial.TrialState.FAIL,
        )
    return len(study.trials)


def _worker(args: tuple[str, str, int, int, int, int]) -> None:
    """Runs one worker process of a search, with its thread budget."""
    method, regime, n_train, n_trials, worker, threads = args
    os.environ["FM4PZ_THREADS"] = str(threads)
    search(method, regime, n_train, n_trials, worker)


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line; see the module docstring.

    Args:
        argv: The arguments; ``None`` for ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("method", choices=sorted(baselines.NAMES))
    parser.add_argument("regime", choices=config.REGIMES)
    parser.add_argument("n_train", type=int)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--first-worker", type=int, default=0)
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help=(
            "top the search up to this many trials in all, spread over the "
            "workers, instead of adding --trials per worker"
        ),
    )
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args(argv)
    if args.export:
        best = export(args.method, args.regime, args.n_train)
        print(json.dumps(best, indent=2, sort_keys=True))
        return 0
    warm_start(args.method, args.regime, args.n_train)
    per_worker = args.trials
    if args.total is not None:
        remaining = args.total - _settle(args.method, args.regime, args.n_train)
        if remaining <= 0:
            export(args.method, args.regime, args.n_train)
            return 0
        per_worker = -(-remaining // args.workers)
    jobs = [
        (
            args.method,
            args.regime,
            args.n_train,
            per_worker,
            args.first_worker + k,
            args.threads,
        )
        for k in range(args.workers)
    ]
    if args.workers == 1:
        _worker(jobs[0])
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.workers) as pool:
            pool.map(_worker, jobs)
    export(args.method, args.regime, args.n_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
