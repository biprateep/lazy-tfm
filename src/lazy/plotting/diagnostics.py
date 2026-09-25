# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The standard diagnostic figures, drawn from PDFs and truth.

These are the plots that decide whether a conditional density estimator is any
good (photo-z's standard set; the calibration plots apply to any target), and
the reason they live in the library rather than in a notebook is that the
comparison only means anything when every method is drawn the same way, on the
same axes, at the same limits. Each function draws into an axes you supply (or
makes one), returns it, and changes no global state -- call
:func:`lazy.plotting.use_style` once in your preamble to get the publication
rcParams::

    from lazy import plotting

    plotting.use_style()
    fig = plotting.diagnostic_panel(z_true, grid.centers, pdfs)

Two families of diagnostic, and they answer different questions:

* **Point accuracy** -- :func:`plot_zphot_ztrue`, :func:`plot_residuals`. How
  close is the single value you would quote? Scatter, bias, outliers.
* **Calibration** -- :func:`plot_pit`, :func:`plot_pit_qq`,
  :func:`plot_coverage`. Are the *widths* honest? A model can have excellent
  point accuracy and badly wrong error bars, and only these plots show it.

:func:`plot_nz` is a third thing again: whether the stacked PDFs recover the
distribution of the target over the sample -- for redshift, the n(z) that
cosmological analyses actually consume.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib import axes as mpl_axes
from matplotlib import colors as mcolors
from matplotlib import figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lazy import grid
from lazy import metrics
from lazy.plotting import style

__all__ = [
    "diagnostic_panel",
    "plot_coverage",
    "plot_nz",
    "plot_pdfs",
    "plot_pit",
    "plot_pit_qq",
    "plot_residuals",
    "plot_zphot_ztrue",
]

_OUTLIER_FLOOR = 0.06


def _axes(ax: mpl_axes.Axes | None, **kwargs: Any) -> mpl_axes.Axes:
    """Returns ``ax``, or the axes of a new figure of ``style.figsize``."""
    if ax is not None:
        return ax
    return plt.subplots(figsize=style.figsize(**kwargs))[1]


def plot_zphot_ztrue(
    z_true: npt.ArrayLike,
    z_pred: npt.ArrayLike,
    *,
    ax: mpl_axes.Axes | None = None,
    bins: int = 200,
    z_max: float | None = None,
    outlier_lines: bool = True,
    cmap: str | None = None,
    **kwargs: Any,
) -> mpl_axes.Axes:
    """Plots predicted against true values as a log-density image.

    A scatter plot of a survey-sized sample is a black blob, so this is a 2D
    histogram on a log colour scale -- which is also the only way the outlier
    islands (catastrophic failures at the wrong value) stay visible against
    the main locus.

    Args:
        z_true: True values, shape (n_objects,).
        z_pred: Point estimates, shape (n_objects,).
        ax: The axes to draw into; a new column-width square figure if None.
        bins: Number of histogram bins along each axis.
        z_max: Upper limit of both axes; the largest value if None.
        outlier_lines: Draw the DC1 outlier boundary
            ``|z_pred - z_true| = 0.06 (1 + z_true)``, so the fraction of
            points outside it is readable by eye.
        cmap: Colormap name; the rcParams default if None.
        **kwargs: Passed to ``ax.hist2d``.

    Returns:
        The axes drawn into.
    """
    z_true = np.asarray(z_true, dtype=float)
    z_pred = np.asarray(z_pred, dtype=float)
    ax = _axes(ax, width="column", aspect="square")
    hi = float(z_max if z_max is not None else max(z_true.max(), z_pred.max()))
    ax.hist2d(
        z_true,
        z_pred,
        bins=bins,
        range=[[0, hi], [0, hi]],
        norm=mcolors.LogNorm(),
        cmap=cmap or plt.rcParams["image.cmap"],
        **kwargs,
    )
    style.one_to_one(ax, 0.0, hi)
    if outlier_lines:
        edge = np.array([0.0, hi])
        for sign in (+1, -1):
            ax.plot(
                edge, edge + sign * _OUTLIER_FLOOR * (1 + edge), "k:", lw=0.8
            )
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(r"$z_{\rm true}$")
    ax.set_ylabel(r"$z_{\rm phot}$")
    return ax


def plot_residuals(
    z_true: npt.ArrayLike,
    z_pred: npt.ArrayLike,
    *,
    ax: mpl_axes.Axes | None = None,
    n_bins: int = 20,
    quantiles: tuple[float, float, float] = (16.0, 50.0, 84.0),
) -> mpl_axes.Axes:
    """Plots the scaled residual against the true value.

    The scaled residual is ``(z_phot - z_true) / (1 + z_true)``. The running
    median and its 16th-84th percentile band say where the bias lives: a
    model can have a fine global bias and still be systematically high at
    low z and low at high z, which this shows and a single
    number hides.

    Args:
        z_true: True values, shape (n_objects,).
        z_pred: Point estimates, shape (n_objects,).
        ax: The axes to draw into; a new column-width figure if None.
        n_bins: Number of equal-count bins in the true value.
        quantiles: Lower edge, centre line and upper edge of the band, as
            percentiles between 0 and 100.

    Returns:
        The axes drawn into.
    """
    z_true = np.asarray(z_true, dtype=float)
    ez = (np.asarray(z_pred, dtype=float) - z_true) / (1.0 + z_true)
    ax = _axes(ax, width="column", aspect="tall")
    edges = np.quantile(z_true, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    index = np.clip(
        np.searchsorted(edges, z_true, side="right") - 1, 0, len(centers) - 1
    )
    stats = np.full((len(centers), 3), np.nan)
    for i in range(len(centers)):
        rows = index == i
        if rows.any():
            stats[i] = np.percentile(ez[rows], quantiles)
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.fill_between(
        centers, stats[:, 0], stats[:, 2], alpha=0.3, lw=0, color="C0"
    )
    ax.plot(centers, stats[:, 1], color="C0")
    ax.set_xlabel(r"$z_{\rm true}$")
    ax.set_ylabel(r"$(z_{\rm phot} - z_{\rm true}) / (1 + z_{\rm true})$")
    return ax


def plot_pit(
    pit: npt.ArrayLike,
    *,
    ax: mpl_axes.Axes | None = None,
    n_bins: int = 20,
    label: str | None = None,
    **kwargs: Any,
) -> mpl_axes.Axes:
    """Plots a histogram of the PIT values against the expected uniform.

    The shape names the failure: a U means the PDFs are too narrow
    (over-confident), a dome means they are too wide, a slope means they are
    biased, and a spike at 0 or 1 counts catastrophic outliers.

    Args:
        pit: Probability integral transform of each object, in [0, 1],
            shape (n_objects,).
        ax: The axes to draw into; a new column-width figure if None.
        n_bins: Number of histogram bins on [0, 1].
        label: Legend label.
        **kwargs: Passed to ``ax.hist``.

    Returns:
        The axes drawn into.
    """
    pit = np.asarray(pit, dtype=float)
    ax = _axes(ax, width="column", aspect="golden")
    ax.hist(
        pit,
        bins=n_bins,
        range=(0, 1),
        density=True,
        histtype="step",
        label=label,
        **kwargs,
    )
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    return ax


def plot_pit_qq(
    pit: npt.ArrayLike,
    *,
    ax: mpl_axes.Axes | None = None,
    label: str | None = None,
    **kwargs: Any,
) -> mpl_axes.Axes:
    """Quantile-quantile plot of the PIT sample against U(0, 1).

    Harder to misread than the histogram: perfect calibration is the diagonal,
    above it means too wide, below means too narrow, and the deviation is in
    the same units as the probability itself.

    Args:
        pit: Probability integral transform of each object, in [0, 1],
            shape (n_objects,).
        ax: The axes to draw into; a new column-width square figure if None.
        label: Legend label.
        **kwargs: Passed to ``ax.plot``.

    Returns:
        The axes drawn into.
    """
    pit = np.asarray(pit, dtype=float)
    ax = _axes(ax, width="column", aspect="square")
    ordered = np.sort(pit)
    uniform = (np.arange(1, ordered.size + 1) - 0.5) / ordered.size
    ax.plot(uniform, ordered, label=label, **kwargs)
    style.one_to_one(ax, 0.0, 1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("uniform quantile")
    ax.set_ylabel("PIT quantile")
    return ax


def plot_coverage(
    pit: npt.ArrayLike,
    *,
    ax: mpl_axes.Axes | None = None,
    label: str | None = None,
    **kwargs: Any,
) -> mpl_axes.Axes:
    """Empirical against nominal coverage of central credible intervals.

    For each nominal level ``q``, the fraction of objects whose true value
    falls inside the central ``q`` credible interval of its own PDF. This is
    the plot to quote when someone asks "if I take your 68% interval, how often
    is it right?" -- the answer should be 68% of the time, i.e. the diagonal.

    Args:
        pit: Probability integral transform of each object, in [0, 1],
            shape (n_objects,).
        ax: The axes to draw into; a new column-width square figure if None.
        label: Legend label.
        **kwargs: Passed to ``ax.plot``.

    Returns:
        The axes drawn into.
    """
    pit = np.asarray(pit, dtype=float)
    ax = _axes(ax, width="column", aspect="square")
    nominal = np.linspace(0.0, 1.0, 101)
    empirical = np.array(
        [np.mean(np.abs(pit - 0.5) <= 0.5 * q) for q in nominal]
    )
    ax.plot(nominal, empirical, label=label, **kwargs)
    style.one_to_one(ax, 0.0, 1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    return ax


def plot_nz(
    z_grid: npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    z_true: npt.ArrayLike | None = None,
    ax: mpl_axes.Axes | None = None,
    label: str | None = "stacked PDFs",
    truth_label: str | None = "truth",
    **kwargs: Any,
) -> mpl_axes.Axes:
    """Plots the sample distribution of the target: stacked PDFs against truth.

    Stacking is only an estimator of N(z) under assumptions that photo-z PDFs
    rarely satisfy exactly, but it is what most analyses do, so how badly it
    fails is worth knowing.

    Args:
        z_grid: Grid centres, shape (n_grid,).
        pdfs: PDFs on ``z_grid``, shape (n_objects, n_grid).
        z_true: True values, shape (n_objects,). If given, histogrammed
            on the same grid for comparison.
        ax: The axes to draw into; a new column-width figure if None.
        label: Legend label of the stacked PDFs.
        truth_label: Legend label of the true histogram.
        **kwargs: Passed to ``ax.plot`` for the stacked PDFs.

    Returns:
        The axes drawn into.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    pdfs = np.asarray(pdfs, dtype=float)
    ax = _axes(ax, width="column", aspect="golden")
    ax.plot(z_grid, pdfs.mean(axis=0), label=label, **kwargs)
    if z_true is not None:
        edges = grid.Grid.from_centers(z_grid).edges
        ax.hist(
            np.asarray(z_true, dtype=float),
            bins=edges.tolist(),
            density=True,
            histtype="step",
            color="k",
            lw=0.8,
            label=truth_label,
        )
    ax.set_xlim(z_grid[0], z_grid[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$n(z)$")
    return ax


def plot_pdfs(
    z_grid: npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    z_true: npt.ArrayLike | None = None,
    indices: npt.ArrayLike | None = None,
    n_objects: int = 6,
    random_state: int = 0,
    axes: mpl_axes.Axes
    | Sequence[mpl_axes.Axes]
    | npt.NDArray[np.object_]
    | None = None,
) -> npt.NDArray[np.object_]:
    """Plots a handful of individual PDFs, with their true values marked.

    Summary statistics hide multimodality; this is where you see it. By
    default ``n_objects`` objects are drawn at random (reproducibly), or pass
    ``indices`` to pick them yourself.

    Args:
        z_grid: Grid centres, shape (n_grid,).
        pdfs: PDFs on ``z_grid``, shape (n_objects, n_grid).
        z_true: True values, shape (n_objects,), marked as dashed
            vertical lines if given.
        indices: Rows of ``pdfs`` to draw; ``n_objects`` random rows if None.
        n_objects: Number of objects drawn at random when ``indices`` is None.
        random_state: Seed of the random draw.
        axes: Axes to draw into, one per object; a new text-width grid of
            up to three columns if None. Spare axes are hidden.

    Returns:
        The flattened array of axes.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    pdfs = np.asarray(pdfs, dtype=float)
    if indices is None:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(
            len(pdfs), size=min(n_objects, len(pdfs)), replace=False
        )
    indices = np.atleast_1d(np.asarray(indices, dtype=int))
    if axes is None:
        ncols = min(3, len(indices))
        nrows = int(np.ceil(len(indices) / ncols))
        _, axes = plt.subplots(
            nrows,
            ncols,
            figsize=style.figsize(width="text", aspect=0.33 * nrows),
            squeeze=False,
            sharex=True,
        )
    axes_array = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    for ax, row in zip(axes_array, indices, strict=False):
        ax.plot(z_grid, pdfs[row], color="C0")
        if z_true is not None:
            ax.axvline(
                float(np.asarray(z_true, dtype=float)[row]),
                color="k",
                ls="--",
                lw=0.8,
            )
        ax.set_xlim(z_grid[0], z_grid[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$z$")
    for ax in axes_array[len(indices) :]:
        ax.set_visible(False)
    return axes_array


def diagnostic_panel(
    z_true: npt.ArrayLike,
    z_grid: npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    point: str = "z_peak",
    label: str | None = None,
) -> mpl_figure.Figure:
    """Draws the four-panel summary of one estimator.

    The panels cover accuracy, bias, calibration and N(z): z_phot-z_true, the
    residual trend, the PIT Q-Q and the stacked N(z). The figure is returned
    open; closing it is up to the caller.

    Args:
        z_true: True values, shape (n_objects,).
        z_grid: Grid centres, shape (n_grid,).
        pdfs: PDFs on ``z_grid``, shape (n_objects, n_grid).
        point: Point estimate to use, a key of
            :func:`lazy.metrics.grid_point_estimates`.
        label: Legend label and figure title.

    Returns:
        The new :class:`matplotlib.figure.Figure`.
    """
    z_true = np.asarray(z_true, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    z_pred = metrics.grid_point_estimates(z_grid, pdfs)[point]
    _, _, pit = metrics.evaluate_grid_pdfs(z_true, z_grid, pdfs, point=point)

    fig, axes = plt.subplots(
        2, 2, figsize=style.figsize(width="text", aspect=0.85)
    )
    plot_zphot_ztrue(z_true, z_pred, ax=axes[0, 0])
    plot_residuals(z_true, z_pred, ax=axes[0, 1])
    plot_pit_qq(pit, ax=axes[1, 0], label=label)
    plot_nz(z_grid, pdfs, z_true=z_true, ax=axes[1, 1])
    axes[1, 1].legend(loc="upper right")
    if label:
        fig.suptitle(label)
    fig.tight_layout()
    return fig
