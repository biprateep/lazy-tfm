#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Step 3 of the representative experiments: Figure 1.

Reads the production records of step 2 and draws the representative scaling
curves: the CDE loss on the common grid and the scatter of the mode against
the training size, as the mean and standard deviation over the five seeds,
for the three bespoke estimators and the three foundation models. Writes
``figures/unbiased_scaling.png`` and prints the numbers behind it.

Typical usage example:

  uv run python representative/plot.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
import pathlib

import matplotlib.pyplot as plt

from fm4pz import config
from fm4pz import figures
from fm4pz import plotstyle

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGIME = "representative"


def main(argv: Sequence[str] | None = None) -> int:
    """Draws the figure; see the module docstring.

    Args:
        argv: The arguments; ``None`` for ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "figures")
    args = parser.parse_args(argv)
    os.environ.setdefault("FM4PZ_RESULTS", str(ROOT / "results"))
    figures.use_paper_style()
    summary = figures.scaling_summary(figures.load_runs(config.results_dir()))
    summary = summary[summary["regime"] == REGIME]
    print(summary.round(4).to_string(index=False))
    args.out.mkdir(parents=True, exist_ok=True)
    figure = figures.scaling_figure(summary, REGIME)
    plotstyle.save(figure, str(args.out / "unbiased_scaling"))
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
