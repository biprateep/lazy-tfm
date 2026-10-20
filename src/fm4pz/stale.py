# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Moves aside the production records fitted with an outdated setting.

A record is current when its fit parameters and post-processing are those
of the latest exported setting of its search
(:func:`fm4pz.tune.latest_best_path`).
After a later round's search has chosen another setting, the records of the
earlier one are moved, with their arrays, from ``runs/<method>/`` to
``runs_r<round - 1>/<method>/``, so that :mod:`fm4pz.produce` fits those
seeds again and the records stay on disk.

Typical usage example:

  FM4PZ_ROUND=2 python -m fm4pz.stale limix representative 30000
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import pathlib

from fm4pz import config
from fm4pz import produce
from fm4pz import tune

__all__ = ["main", "retire"]


def retire(method: str, regime: str, n_train: int) -> list[pathlib.Path]:
    """Moves aside the records of one size that are not current.

    Args:
        method: A method name.
        regime: ``"representative"`` or ``"biased"``.
        n_train: The training size.

    Returns:
        Where each moved record now is.
    """
    best = json.loads(
        tune.latest_best_path(method, regime, n_train).read_text()
    )
    archive = config.results_dir() / f"runs_r{tune.search_round() - 1}" / method
    moved = []
    for seed in config.SEEDS:
        arrays = produce.run_path(method, regime, n_train, seed)
        record = arrays.with_suffix(".json")
        if not record.exists():
            continue
        run = json.loads(record.read_text())
        if run["params"] == best["params"] and run["post"] == best["post"]:
            continue
        archive.mkdir(parents=True, exist_ok=True)
        for path in (record, arrays):
            if path.exists():
                moved.append(path.rename(archive / path.name))
    return moved


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line; see the module docstring.

    Args:
        argv: The arguments; ``None`` for ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("method")
    parser.add_argument("regime", choices=config.REGIMES)
    parser.add_argument("n_train", type=int)
    args = parser.parse_args(argv)
    for path in retire(args.method, args.regime, args.n_train):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
