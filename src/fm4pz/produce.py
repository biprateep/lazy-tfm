# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Fits a baseline's best setting on one subsample and scores the test set.

The setting is the one :mod:`fm4pz.tune` chose by cross-validation at that
training size. The baseline is refitted on the whole subsample of the given
seed, and the test set is predicted in chunks, each turned into masses on the
common grid and scored at once, so that only per-object scores are kept: the
masses of 390,990 galaxies on 4,000 bins would take 12.5 GB. The scores go to
``$FM4PZ_RESULTS/runs/<method>/<regime>_n<size>_s<seed>.npz`` and the run's
record, its metrics included, to the ``.json`` beside it.

Typical usage example:

  python -m fm4pz.produce gpz representative 1000 --seed 3 --jobs 8
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib.metadata
import json
import multiprocessing
import os
import pathlib
import platform
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from fm4pz import baselines
from fm4pz import config
from fm4pz import scoring
from fm4pz import splits
from fm4pz import tune

__all__ = ["main", "produce", "run_path"]

# Test rows predicted and scored at a time.
_CHUNK = 10_000

# What a forked worker reads; set by produce() before the pool starts.
_STATE: dict[str, Any] = {}


def run_path(method: str, regime: str, n_train: int, seed: int) -> pathlib.Path:
    """The per-object scores of one run; its record is the ``.json`` beside."""
    return (
        config.results_dir()
        / "runs"
        / method
        / f"{regime}_n{n_train}_s{seed}.npz"
    )


def _score_chunk(rows: npt.NDArray[np.int64]) -> dict[str, np.ndarray]:
    """Predicts and scores some test rows, in a forked worker."""
    method, model, post, test = (
        _STATE["method"],
        _STATE["model"],
        _STATE["post"],
        _STATE["test"],
    )
    chunk = test.take(rows)
    masses = method.masses(method.predict(model, chunk), post)
    return scoring.score(chunk.redshift, masses)


def _record(
    best: dict[str, Any], split: splits.Split, metrics: dict[str, Any]
) -> dict[str, Any]:
    """The record of a run: its setting, data, metrics and provenance."""
    return {
        "regime": split.regime,
        "n_train": split.n_train,
        "seed": split.seed,
        "params": best["params"],
        "post": best["post"],
        "cv_loss": best["cv_loss"],
        "train_checksum": split.train_checksum,
        "metrics": metrics,
        "threads": int(os.environ.get("FM4PZ_THREADS", "1")),
        "host": platform.node(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("lazy-tfm", "numpy", "scipy", "xgboost", "flexcode")
        },
    }


def produce(
    method_name: str, regime: str, n_train: int, seed: int, jobs: int
) -> dict[str, Any]:
    """Runs one production fit and scores the test set.

    Args:
        method_name: A baseline name.
        regime: ``"representative"`` or ``"biased"``.
        n_train: The training size.
        seed: The subsample's seed, which also seeds the method.
        jobs: Worker processes predicting test chunks.

    Returns:
        The run's record, as written to its ``.json``.
    """
    best = json.loads(tune.best_path(method_name, regime, n_train).read_text())
    method = baselines.get(method_name)
    split = splits.load(regime, n_train=n_train, seed=seed)
    start = time.perf_counter()
    model = method.fit(
        best["params"], split.train, split.train.redshift, seed=seed
    )
    fitted = time.perf_counter()
    _STATE.update(
        method=method, model=model, post=best["post"], test=split.test
    )
    chunks = np.array_split(
        np.arange(len(split.test.redshift)),
        max(1, -(-len(split.test.redshift) // _CHUNK)),
    )
    if jobs == 1:
        parts = [_score_chunk(rows) for rows in chunks]
    else:
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            parts = pool.map(_score_chunk, chunks)
    _STATE.clear()
    scored = time.perf_counter()
    per_object = {
        key: np.concatenate([part[key] for part in parts]) for key in parts[0]
    }
    z_true = split.test.redshift
    record = _record(best, split, scoring.summary(z_true, per_object))
    record.update(
        method=method_name,
        fit_seconds=fitted - start,
        predict_seconds=scored - fitted,
        jobs=jobs,
    )
    path = run_path(method_name, regime, n_train, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"z_true": z_true, **per_object}
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    record = produce(
        args.method, args.regime, args.n_train, args.seed, args.jobs
    )
    print(json.dumps(record["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
