# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calibration matrices
#
# The paper's calibration matrices: a tabular foundation model and a bespoke
# estimator on one grid of biased training size (columns) and representative
# calibration galaxies added to it (rows), and their difference. Each cell is
# the mean test CDE loss over the five seeds, which draw the same training
# rows for every method, so the difference is paired. The runs are those of
# `fm4pz.produce --calibration` under `$FM4PZ_RESULTS/calibration/`, with the
# biased production runs under `runs/` as the column without calibration.
# Every figure has a CSV beside it with the means and standard deviations.

# %%
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
import json
import os
import pathlib

from matplotlib import colors
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotstyle
import seaborn as sns

from fm4pz import config

# %% [markdown]
# ## Configuration
#
# The pairs mirror the submitted paper's: the best foundation model and the
# best bespoke estimator on the full biased set in the main text, then the
# best foundation model against the other two bespoke estimators and the
# other two foundation models against the best bespoke one. The colour maps
# are the submitted paper's (seaborn's mako and rocket).

# %%
RESULTS = pathlib.Path(os.environ.get("FM4PZ_RESULTS", "../results"))
NEURIPS_TEXT_PT = 397.48499
NAMES = {
    "flexzboost": "FlexZBoost",
    "cmnn": "CMNN",
    "gpz": "GPz",
    "tabpfn": "TabPFN-3.5",
    "limix": "LimiX-2",
    "kumo": "Kumo",
}
PAIRS = (
    ("limix", "cmnn"),
    ("limix", "gpz"),
    ("limix", "flexzboost"),
    ("tabpfn", "cmnn"),
    ("kumo", "cmnn"),
)
VALUE_CMAP = sns.color_palette("mako", as_cmap=True)
DIFFERENCE_CMAP = sns.color_palette("rocket", as_cmap=True)
DIVERGING_CMAP = sns.color_palette("vlag", as_cmap=True)
# Seven numbers across a third of the text width fit only below the house
# sizes; the cell labels alone take this size, as in the submitted paper.
CELL_SIZE = 6

plotstyle.register_journal(
    "neurips", column_pt=NEURIPS_TEXT_PT, text_pt=NEURIPS_TEXT_PT
)
plotstyle.use_style(journal="neurips")
plotstyle.verify_style()

# %% [markdown]
# ## The runs
#
# One row per run of the matrix: every method, biased size, calibration size
# and seed of `fm4pz.config`. All 1,050 must be there.

# %%
rows = []
for method in config.MATRIX_METHODS:
    for n_train in config.BIASED_SIZES:
        for n_calibration in config.MATRIX_CALIBRATION_SIZES:
            for seed in config.SEEDS:
                path = (
                    RESULTS
                    / "runs"
                    / method
                    / f"biased_n{n_train}_s{seed}.json"
                    if n_calibration == 0
                    else RESULTS
                    / "calibration"
                    / method
                    / f"biased_cal{n_calibration}_n{n_train}_s{seed}.json"
                )
                metrics = json.loads(path.read_text())["metrics"]
                rows.append(
                    {
                        "method": method,
                        "n_train": n_train,
                        "n_calibration": n_calibration,
                        "seed": seed,
                        "cde_loss": metrics["cde_loss"],
                    }
                )
runs = pd.DataFrame(rows)
print(f"{len(runs)} runs")
means = runs.pivot_table(
    index=["method", "n_calibration"], columns="n_train", values="cde_loss"
)
print(means.round(2).to_string())

# %% [markdown]
# ## The figures


# %%
def grid(method: str, statistic: str = "mean") -> pd.DataFrame:
    """One method's matrix: calibration size (rows) by biased size (columns).

    Args:
        method: A method of :data:`fm4pz.config.MATRIX_METHODS`.
        statistic: ``"mean"`` or ``"std"`` over the seeds.

    Returns:
        The test CDE loss, rows in increasing calibration size.
    """
    mine = runs[runs["method"] == method]
    return mine.pivot_table(
        index="n_calibration",
        columns="n_train",
        values="cde_loss",
        aggfunc=statistic,
    )


def paired_difference(first: str, second: str) -> tuple[pd.DataFrame, ...]:
    """The first method's loss minus the second's, seed by seed.

    Args:
        first: The method whose loss is taken first.
        second: The method subtracted.

    Returns:
        A tuple (mean, std) of the differences over the seeds, each shaped
        as :func:`grid`.
    """
    keys = ["n_train", "n_calibration", "seed"]
    pair = runs[runs["method"] == first].merge(
        runs[runs["method"] == second], on=keys, suffixes=("_1", "_2")
    )
    pair["difference"] = pair["cde_loss_1"] - pair["cde_loss_2"]
    stats = [
        pair.pivot_table(
            index="n_calibration",
            columns="n_train",
            values="difference",
            aggfunc=statistic,
        )
        for statistic in ("mean", "std")
    ]
    return stats[0], stats[1]


def rank_norm(values: npt.NDArray[np.float64]) -> colors.FuncNorm:
    """A colour scale by rank, as the submitted paper's shared scale was.

    The two panels' values barely overlap, so a linear scale would leave
    each panel nearly one colour; by rank, each value takes its place among
    all of them.

    Args:
        values: The values the scale must cover.

    Returns:
        The normalisation, linear in rank between the smallest and largest.
    """
    knots = np.unique(values[np.isfinite(values)])
    ranks = np.linspace(0.0, 1.0, knots.size)
    return colors.FuncNorm(
        (
            lambda x: np.interp(x, knots, ranks),
            lambda y: np.interp(y, ranks, knots),
        ),
        vmin=knots[0],
        vmax=knots[-1],
    )


def readable_on(cmap: colors.Colormap, level: float) -> str:
    """Black or white, whichever reads on a colour of the map."""
    red, green, blue, _ = cmap(level)
    return "k" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.5 else "w"


def size_label(size: int) -> str:
    """A training or calibration size in thousands, as the axes show it."""
    return f"{size / 1000:.0f}"


def value_panel(
    axis: plt.Axes,
    values: pd.DataFrame,
    *,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    title: str,
    ring: tuple[int, int, str] | None,
) -> plt.cm.ScalarMappable:
    """Draws one matrix with its values written in the cells.

    Args:
        axis: The axis to draw on.
        values: Calibration size (rows) by biased size (columns).
        cmap: The colour map.
        norm: The colour normalisation.
        title: The panel's title.
        ring: A cell to ring as (row, column, colour), the colour ``"auto"``
            for black or white, whichever reads on the cell; None for none.

    Returns:
        The image, for a colour bar.
    """
    array = values.to_numpy(dtype=float)
    image = axis.imshow(
        array, cmap=cmap, norm=norm, aspect="auto", origin="lower"
    )
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            level = float(norm(array[row, column]))
            axis.text(
                column,
                row,
                f"{array[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=CELL_SIZE,
                color=readable_on(cmap, level),
            )
    if ring is not None:
        row, column, edge = ring
        if edge == "auto":
            edge = readable_on(cmap, float(norm(array[row, column])))
        axis.add_patch(
            patches.Rectangle(
                (column - 0.5, row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=edge,
                linewidth=1.5,
                zorder=5,
            )
        )
    axis.set_xticks(range(array.shape[1]))
    axis.set_xticklabels([size_label(n) for n in values.columns])
    axis.set_yticks(range(array.shape[0]))
    axis.set_yticklabels([size_label(k) for k in values.index])
    axis.set_xticks(np.arange(-0.5, array.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, array.shape[0], 1), minor=True)
    axis.grid(which="minor", color="w", linewidth=0.6)
    axis.grid(which="major", visible=False)
    axis.tick_params(which="both", length=0)
    axis.set_title(title, fontsize=plotstyle.NORMAL_SIZE)
    return image


def pair_figure(first: str, second: str) -> plt.Figure:
    """Two methods on one colour scale, and the first minus the second.

    Rings: round the first method with the least data (smallest biased set,
    no calibration) and the second with the most (full biased set, all
    calibration galaxies), the comparison the figure invites; red round the
    cell where the first is furthest ahead.

    Args:
        first: A foundation model.
        second: A bespoke estimator.

    Returns:
        The figure.
    """
    one, two = grid(first), grid(second)
    difference, _ = paired_difference(first, second)
    both = np.concatenate([one.to_numpy().ravel(), two.to_numpy().ravel()])
    shared = rank_norm(both)
    delta = difference.to_numpy()
    if delta.min() < 0 < delta.max():
        span = float(np.abs(delta).max())
        delta_cmap = DIVERGING_CMAP
        delta_norm: colors.Normalize = colors.TwoSlopeNorm(
            vcenter=0.0, vmin=-span, vmax=span
        )
    else:
        delta_cmap = DIFFERENCE_CMAP
        delta_norm = colors.Normalize(vmin=delta.min(), vmax=delta.max())
    peak = np.unravel_index(int(np.argmin(delta)), delta.shape)

    fig = plt.figure(
        figsize=plotstyle.figsize("text", 0.4), layout="constrained"
    )
    spec = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.08, 1])
    axes = [
        fig.add_subplot(spec[0, 0]),
        fig.add_subplot(spec[0, 1]),
        fig.add_subplot(spec[0, 3]),
    ]
    image = value_panel(
        axes[0],
        one,
        cmap=VALUE_CMAP,
        norm=shared,
        title=NAMES[first],
        ring=(0, 0, "auto"),
    )
    value_panel(
        axes[1],
        two,
        cmap=VALUE_CMAP,
        norm=shared,
        title=NAMES[second],
        ring=(two.shape[0] - 1, two.shape[1] - 1, "auto"),
    )
    image_delta = value_panel(
        axes[2],
        difference,
        cmap=delta_cmap,
        norm=delta_norm,
        title=f"{NAMES[first]} $-$ {NAMES[second]}",
        ring=(int(peak[0]), int(peak[1]), "r"),
    )
    axes[0].set_ylabel(r"Calibration Set Size [$10^3$]")
    axes[1].set_yticklabels([])
    axes[1].set_xlabel(r"Training Set Size [$10^3$]")
    axes[1].xaxis.set_label_coords(0.0, -0.18)
    axes[2].set_xlabel(r"Training Set Size [$10^3$]")
    bar = fig.colorbar(
        image, ax=axes[:2], location="bottom", fraction=0.07, aspect=40
    )
    ticks = np.quantile(both, np.linspace(0.0, 1.0, 6))
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"${tick:.1f}$" for tick in ticks])
    bar.set_label(r"$\mathcal{L}_{\rm CDE}$")
    bar_delta = fig.colorbar(
        image_delta, ax=axes[2], location="bottom", fraction=0.07, aspect=20
    )
    bar_delta.set_label(r"$\Delta\mathcal{L}_{\rm CDE}$")
    return fig


for first, second in PAIRS:
    figure = pair_figure(first, second)
    plotstyle.save(figure, f"./figs/matrix_{first}_vs_{second}")
    plt.close(figure)
    difference, spread = paired_difference(first, second)
    table = pd.concat(
        {
            f"{first}_mean": grid(first),
            f"{first}_std": grid(first, "std"),
            f"{second}_mean": grid(second),
            f"{second}_std": grid(second, "std"),
            "difference_mean": difference,
            "difference_std": spread,
        },
        names=["quantity", "n_calibration"],
    )
    table.to_csv(f"./figs/matrix_{first}_vs_{second}.csv")
    print(
        f"{first} - {second}: difference {difference.to_numpy().min():.2f}"
        f" to {difference.to_numpy().max():.2f}"
    )
