#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Step 3 of the biased experiments: the appendix figure and the matrices.

Reads the production records of step 2 and draws the biased scaling curves
(the appendix figure, as Figure 1 but trained on the biased sets) and the
five calibration matrices: LimiX-2 against CMNN (the main text), LimiX-2
against GPz and FlexZBoost, and TabPFN-3.5 and Kumo against CMNN (the
appendix). Writes ``figures/biased_scaling.png`` and
``figures/matrix_<first>_vs_<second>.png``, with each matrix's means and
standard deviations in ``results/matrix/matrix_<first>_vs_<second>.csv``.

Typical usage example:

  uv run python biased/plot.py
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
REGIME = "biased"


def main(argv: Sequence[str] | None = None) -> int:
    """Draws the figures; see the module docstring.

    Args:
        argv: The arguments; ``None`` for ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "figures")
    args = parser.parse_args(argv)
    os.environ.setdefault("FM4PZ_RESULTS", str(ROOT / "results"))
    results = config.results_dir()
    figures.use_paper_style()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = figures.scaling_summary(figures.load_runs(results))
    summary = summary[summary["regime"] == REGIME]
    print(summary.round(4).to_string(index=False))
    figure = figures.scaling_figure(summary, REGIME)
    plotstyle.save(figure, str(args.out / "biased_scaling"))
    plt.close(figure)

    runs = figures.load_matrix(results)
    tables = results / "matrix"
    tables.mkdir(parents=True, exist_ok=True)
    for first, second in figures.MATRIX_PAIRS:
        name = f"matrix_{first}_vs_{second}"
        figure = figures.matrix_figure(runs, first, second)
        plotstyle.save(figure, str(args.out / name))
        plt.close(figure)
        table = figures.matrix_table(runs, first, second)
        table.to_csv(tables / f"{name}.csv")
        gap = table.loc["difference_mean"].to_numpy()
        print(f"{name}: difference {gap.min():.2f} to {gap.max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
