# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""The paper's figures, drawn from the production records.

:func:`scaling_figure` draws a regime's scaling curves (Figure 1 for the
representative regime, the appendix figure for the biased one): the CDE loss
on the common grid and the scatter of the mode against the training size,
as the mean and standard deviation over the five subsamples.
:func:`matrix_figure` draws a calibration matrix: a foundation model and a
bespoke estimator on one grid of biased training size and calibration
galaxies added, and their difference, paired seed by seed. The figures are
sized for the NeurIPS text width and drawn in the house style of
:mod:`fm4pz.plotstyle`.

Typical usage example:

  figures.use_paper_style()
  runs = figures.load_runs(config.results_dir())
  fig = figures.scaling_figure(figures.scaling_summary(runs), "biased")
"""

from __future__ import annotations

import json
import pathlib

from matplotlib import colors
from matplotlib import lines
from matplotlib import patches
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from fm4pz import config
from fm4pz import plotstyle

__all__ = [
    "MATRIX_PAIRS",
    "NAMES",
    "load_matrix",
    "load_runs",
    "matrix_figure",
    "matrix_table",
    "scaling_figure",
    "scaling_summary",
    "use_paper_style",
]

#: The NeurIPS class's text width, in points.
NEURIPS_TEXT_PT = 397.48499

#: Display names of the methods.
NAMES = {
    "flexzboost": "FlexZBoost",
    "cmnn": "CMNN",
    "gpz": "GPz",
    "tabpfn": "TabPFN-3.5",
    "limix": "LimiX-2",
    "kumo": "Kumo",
}

# The scaling curves' style: (colour, marker, family) by method. Bespoke
# estimators are dashed, foundation models solid.
_STYLES = {
    "flexzboost": ("C0", "v", "Bespoke"),
    "cmnn": ("C1", "D", "Bespoke"),
    "gpz": ("C2", "P", "Bespoke"),
    "limix": ("C3", "o", "TFM"),
    "tabpfn": ("C4", "s", "TFM"),
    "kumo": ("C5", "^", "TFM"),
}

# Candidate ticks of the scatter axis, in units of 10^-2.
_SIGMA_TICKS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)

#: The calibration matrices: the best foundation model and the best bespoke
#: estimator on the full biased set (the main text), then the best
#: foundation model against the other two bespoke estimators and the other
#: two foundation models against the best bespoke one (the appendix).
MATRIX_PAIRS = (
    ("limix", "cmnn"),
    ("limix", "gpz"),
    ("limix", "flexzboost"),
    ("tabpfn", "cmnn"),
    ("kumo", "cmnn"),
)

# Seven numbers across a third of the text width fit only below the house
# sizes; the matrix cell labels alone take this size, as in the submitted
# paper.
_CELL_SIZE = 6


def use_paper_style() -> None:
    """Applies the house style at the NeurIPS text width, and checks it."""
    plotstyle.register_journal(
        "neurips", column_pt=NEURIPS_TEXT_PT, text_pt=NEURIPS_TEXT_PT
    )
    plotstyle.use_style(journal="neurips")
    plotstyle.verify_style()


def load_runs(results: pathlib.Path) -> pd.DataFrame:
    """The production records of the scaling curves, one row per run.

    Args:
        results: The results directory.

    Returns:
        Columns ``method``, ``regime``, ``n_train``, ``seed``, ``cde_loss``
        and ``sigma_iqr``.
    """
    rows = []
    for path in sorted(results.glob("runs/*/*.json")):
        record = json.loads(path.read_text())
        rows.append(
            {
                "method": path.parent.name,
                "regime": record["regime"],
                "n_train": record["n_train"],
                "seed": record["seed"],
                "cde_loss": record["metrics"]["cde_loss"],
                "sigma_iqr": record["metrics"]["sigma_iqr"],
            }
        )
    return pd.DataFrame(rows)


def scaling_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Each method's mean and spread over the seeds, per regime and size.

    Args:
        runs: The output of :func:`load_runs`.

    Returns:
        One row per regime, method and size: ``cde_mean``, ``cde_std``,
        ``sigma_mean``, ``sigma_std`` and ``n_seeds``.
    """
    return (
        runs.groupby(["regime", "method", "n_train"])
        .agg(
            cde_mean=("cde_loss", "mean"),
            cde_std=("cde_loss", "std"),
            sigma_mean=("sigma_iqr", "mean"),
            sigma_std=("sigma_iqr", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )


def _legend_entries(
    axis: plt.Axes,
) -> tuple[list[lines.Line2D], list[str]]:
    """A two-row legend's entries, one row per family led by its name.

    The legend fills its columns first, so the families are interleaved.
    """
    by_label = dict(zip(*axis.get_legend_handles_labels()[::-1], strict=True))
    rows = []
    for family in ("Bespoke", "TFM"):
        row = [(lines.Line2D([], [], linestyle="none"), family)]
        row += [
            (by_label[NAMES[method]], NAMES[method])
            for method, (_, _, method_family) in _STYLES.items()
            if method_family == family
        ]
        rows.append(row)
    interleaved: list[tuple[object, str]] = []
    for column in zip(*rows, strict=True):
        interleaved.extend(column)
    handles, labels = zip(*interleaved, strict=True)
    return list(handles), list(labels)


def scaling_figure(summary: pd.DataFrame, regime: str) -> plt.Figure:
    """Draws one regime's two-panel scaling figure.

    Args:
        summary: The output of :func:`scaling_summary`.
        regime: ``"representative"`` or ``"biased"``.

    Returns:
        The figure: CDE loss (left) and the scatter of the mode on a
        logarithmic axis (right) against the training size.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=plotstyle.grid_figsize(1, 2, "text", panel_aspect=0.8),
        layout="constrained",
    )
    rows = summary[summary["regime"] == regime]
    for method, (colour, marker, family) in _STYLES.items():
        curve = rows[rows["method"] == method].sort_values("n_train")
        style = {
            "color": colour,
            "marker": marker,
            "markersize": 3.5,
            "linestyle": "--" if family == "Bespoke" else "-",
            "linewidth": 1.2,
            "capsize": 1.5,
            "elinewidth": 0.8,
            "label": NAMES[method],
        }
        axes[0].errorbar(
            curve["n_train"], curve["cde_mean"], yerr=curve["cde_std"], **style
        )
        axes[1].errorbar(
            curve["n_train"],
            100 * curve["sigma_mean"],
            yerr=100 * curve["sigma_std"],
            **style,
        )
    sizes = np.sort(rows["n_train"].unique())
    shown = [size for size in sizes if size not in (15_000, 30_000)]
    for axis in axes:
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(ticker.FixedLocator(shown))
        axis.xaxis.set_major_formatter(
            ticker.FixedFormatter([f"{size // 1000}k" for size in shown])
        )
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.set_xlabel("Training Set Size")
        # A grid at the major ticks, as in the submitted figures.
        axis.grid(
            which="major", color="k", alpha=0.15, linestyle="--", linewidth=0.6
        )
    axes[0].set_ylabel(r"$\mathcal{L}_{\rm CDE}$")
    axes[1].set_yscale("log")
    sigma = 100 * rows["sigma_mean"]
    axes[1].yaxis.set_major_locator(
        ticker.FixedLocator(
            [
                tick
                for tick in _SIGMA_TICKS
                if 0.95 * sigma.min() <= tick <= 1.05 * sigma.max()
            ]
        )
    )
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    axes[1].yaxis.set_minor_formatter(ticker.NullFormatter())
    axes[1].set_ylabel(r"$\sigma_{\rm IQR}$ [$10^{-2}$]")
    # A little room below the lowest points, so the full-size points clear
    # the bottom ticks (in log units on the scatter panel).
    low, high = axes[0].get_ylim()
    axes[0].set_ylim(low - 0.04 * (high - low), high)
    low, high = axes[1].get_ylim()
    axes[1].set_ylim(low * (low / high) ** 0.04, high)
    handles, labels = _legend_entries(axes[0])
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="outside lower center",
        ncols=len(labels) // 2,
        handlelength=1.8,
        columnspacing=1.0,
        fontsize=plotstyle.SMALL_SIZE,
    )
    return fig


def load_matrix(results: pathlib.Path) -> pd.DataFrame:
    """The calibration matrix's runs, one row each; all must exist.

    The column without calibration is the biased production run itself.

    Args:
        results: The results directory.

    Returns:
        Columns ``method``, ``n_train``, ``n_calibration``, ``seed`` and
        ``cde_loss``.

    Raises:
        FileNotFoundError: If a run of the matrix has no record yet.
    """
    rows = []
    for method in config.METHODS:
        for n_train in config.BIASED_SIZES:
            for n_calibration in config.MATRIX_CALIBRATION_SIZES:
                for seed in config.SEEDS:
                    path = (
                        results
                        / "runs"
                        / method
                        / f"biased_n{n_train}_s{seed}.json"
                        if n_calibration == 0
                        else results
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
    return pd.DataFrame(rows)


def _grid(runs: pd.DataFrame, method: str, statistic: str) -> pd.DataFrame:
    """One method's matrix: calibration size (rows) by biased size."""
    return runs[runs["method"] == method].pivot_table(
        index="n_calibration",
        columns="n_train",
        values="cde_loss",
        aggfunc=statistic,
    )


def _difference(
    runs: pd.DataFrame, first: str, second: str, statistic: str
) -> pd.DataFrame:
    """The first method's loss minus the second's, seed by seed."""
    keys = ["n_train", "n_calibration", "seed"]
    pair = runs[runs["method"] == first].merge(
        runs[runs["method"] == second], on=keys, suffixes=("_1", "_2")
    )
    pair["difference"] = pair["cde_loss_1"] - pair["cde_loss_2"]
    return pair.pivot_table(
        index="n_calibration",
        columns="n_train",
        values="difference",
        aggfunc=statistic,
    )


def matrix_table(runs: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
    """A matrix figure's numbers: means and spreads over the seeds.

    Args:
        runs: The output of :func:`load_matrix`.
        first: A foundation model.
        second: A bespoke estimator.

    Returns:
        Both methods' mean and standard deviation, and those of their
        paired difference, each calibration size (rows) by biased size,
        stacked under a ``quantity`` level.
    """
    return pd.concat(
        {
            f"{first}_mean": _grid(runs, first, "mean"),
            f"{first}_std": _grid(runs, first, "std"),
            f"{second}_mean": _grid(runs, second, "mean"),
            f"{second}_std": _grid(runs, second, "std"),
            "difference_mean": _difference(runs, first, second, "mean"),
            "difference_std": _difference(runs, first, second, "std"),
        },
        names=["quantity", "n_calibration"],
    )


def _rank_norm(values: npt.NDArray[np.float64]) -> colors.FuncNorm:
    """A colour scale linear in rank, as the submitted paper's shared one.

    The two panels' values barely overlap, so a linear scale would leave
    each panel nearly one colour; by rank, each value takes its place among
    all of them.
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


def _readable_on(cmap: colors.Colormap, level: float) -> str:
    """Black or white, whichever reads on a colour of the map."""
    red, green, blue, _ = cmap(level)
    return "k" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.5 else "w"


def _value_panel(
    axis: plt.Axes,
    values: pd.DataFrame,
    *,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    title: str,
    ring: tuple[int, int, str],
) -> plt.cm.ScalarMappable:
    """Draws one matrix, values in its cells, one cell ringed.

    ``ring`` is (row, column, colour); ``"auto"`` takes black or white,
    whichever reads on the cell.
    """
    array = values.to_numpy(dtype=float)
    image = axis.imshow(
        array, cmap=cmap, norm=norm, aspect="auto", origin="lower"
    )
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            axis.text(
                column,
                row,
                f"{array[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=_CELL_SIZE,
                color=_readable_on(cmap, float(norm(array[row, column]))),
            )
    row, column, edge = ring
    if edge == "auto":
        edge = _readable_on(cmap, float(norm(array[row, column])))
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
    axis.set_xticklabels([f"{n / 1000:.0f}" for n in values.columns])
    axis.set_yticks(range(array.shape[0]))
    axis.set_yticklabels([f"{k / 1000:.0f}" for k in values.index])
    axis.set_xticks(np.arange(-0.5, array.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, array.shape[0], 1), minor=True)
    axis.grid(which="minor", color="w", linewidth=0.6)
    axis.grid(which="major", visible=False)
    axis.tick_params(which="both", length=0)
    axis.set_title(title, fontsize=plotstyle.NORMAL_SIZE)
    return image


def matrix_figure(runs: pd.DataFrame, first: str, second: str) -> plt.Figure:
    """Two methods' calibration matrices and their difference.

    The first two panels share one colour scale (seaborn's mako, by rank);
    the difference has its own (rocket, or vlag when its sign changes).
    Rings: round the first method with the least data (smallest biased
    set, no calibration) and the second with the most (full biased set, all
    calibration galaxies); red round the cell where the first is furthest
    ahead.

    Args:
        runs: The output of :func:`load_matrix`.
        first: A foundation model.
        second: A bespoke estimator.

    Returns:
        The figure.
    """
    one = _grid(runs, first, "mean")
    two = _grid(runs, second, "mean")
    difference = _difference(runs, first, second, "mean")
    both = np.concatenate([one.to_numpy().ravel(), two.to_numpy().ravel()])
    shared = _rank_norm(both)
    delta = difference.to_numpy()
    delta_norm: colors.Normalize
    if delta.min() < 0 < delta.max():
        span = float(np.abs(delta).max())
        delta_cmap = sns.color_palette("vlag", as_cmap=True)
        delta_norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span)
    else:
        delta_cmap = sns.color_palette("rocket", as_cmap=True)
        delta_norm = colors.Normalize(vmin=delta.min(), vmax=delta.max())
    value_cmap = sns.color_palette("mako", as_cmap=True)
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
    image = _value_panel(
        axes[0],
        one,
        cmap=value_cmap,
        norm=shared,
        title=NAMES[first],
        ring=(0, 0, "auto"),
    )
    _value_panel(
        axes[1],
        two,
        cmap=value_cmap,
        norm=shared,
        title=NAMES[second],
        ring=(two.shape[0] - 1, two.shape[1] - 1, "auto"),
    )
    image_delta = _value_panel(
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
