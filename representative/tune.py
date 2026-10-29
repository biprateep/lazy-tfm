#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Step 1 of the representative experiments: the hyperparameter searches.

For every method and training size (1,000 to 43,486), a search by 5-fold
cross-validation on the training rows of seed 0 chooses the method's
hyperparameters and features by the CDE loss on the common grid; the test set
is never used. A bespoke estimator's search runs to its trial budget of
``fm4pz.config.SEARCH_TRIALS``; a foundation model's runs on the GPU until 40
trials have completed. Each search writes its Optuna journal and the chosen
setting to ``results/tuning/<method>/representative_n<size>.journal`` and
``.best.json``, which step 2 reads. A search cut short resumes where it
stopped.

Typical usage example:

  uv run python representative/tune.py
  uv run python representative/tune.py --methods cmnn gpz --workers 16
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
    """Runs the searches; see the module docstring.

    Args:
        argv: The arguments; ``None`` for ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--methods", nargs="+", default=list(config.METHODS))
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes of a bespoke estimator's search",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("FM4PZ_RESULTS", str(ROOT / "results"))
    pipeline.search_all(REGIME, args.methods, args.sizes, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
