#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Step 2 of the biased experiments: the production runs and the matrix.

For every method, biased training size and seed (0 to 4), the setting
chosen in step 1 is refitted on that seed's training subset and the
representative hold-out (19,383 galaxies) is scored on the common grid. For
the calibration matrix, each biased training set is also refitted, with the
same setting, after adding 1,000, 2,000, 5,000 or 10,000 galaxies of the
representative calibration sample. A plain run writes its record to
``results/runs/<method>/biased_n<size>_s<seed>.json``, a run with
calibration galaxies to
``results/calibration/<method>/biased_cal<k>_n<size>_s<seed>.json``, each
with its per-galaxy scores in the ``.npz`` beside it, which step 3 reads.
Runs with a record are skipped, so the step resumes where it stopped. The
foundation models need a GPU.

Typical usage example:

  uv run python biased/produce.py
  uv run python biased/produce.py --calibrations 0
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
import pathlib

from fm4pz import config
from fm4pz import pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGIME = "biased"


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the production fits; see the module docstring.

    Args:
        argv: The arguments; ``None`` for ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--methods", nargs="+", default=list(config.METHODS))
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(config.SEEDS)
    )
    parser.add_argument(
        "--calibrations",
        nargs="+",
        type=int,
        default=list(config.MATRIX_CALIBRATION_SIZES),
        help="calibration galaxies added; 0 is the plain biased run",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="processes scoring the test set"
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("FM4PZ_RESULTS", str(ROOT / "results"))
    pipeline.produce_all(
        REGIME,
        args.methods,
        args.sizes,
        args.seeds,
        args.calibrations,
        jobs=args.jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
