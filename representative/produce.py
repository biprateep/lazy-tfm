#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Step 2 of the representative experiments: the production runs.

For every method, training size and seed (0 to 4), the setting chosen in
step 1 is refitted on that seed's training subset and the whole DC1 test
file (390,990 galaxies) is scored on the common grid. Each run writes its
record (setting, metrics, timings) to
``results/runs/<method>/representative_n<size>_s<seed>.json`` and its
per-galaxy scores to the ``.npz`` beside it, which step 3 reads. Runs with a
record are skipped, so the step resumes where it stopped. The foundation
models need a GPU.

Typical usage example:

  uv run python representative/produce.py
  uv run python representative/produce.py --methods tabpfn --seeds 0 1
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
import pathlib

from fm4pz import config
from fm4pz import pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGIME = "representative"


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
        "--jobs", type=int, default=4, help="processes scoring the test set"
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("FM4PZ_RESULTS", str(ROOT / "results"))
    pipeline.produce_all(
        REGIME, args.methods, args.sizes, args.seeds, jobs=args.jobs
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
