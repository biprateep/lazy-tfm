"""The standard photo-z diagnostic figures, drawn from PDFs and truth.

These are the plots that decide whether a photo-z estimator is any good, and
the reason they live in the library rather than in a notebook is that the
comparison only means anything when every method is drawn the same way, on the
same axes, at the same limits. Each function draws into an axes you supply (or
makes one), returns it, and changes no global state -- call
:func:`lazy.plotting.use_style` once in your preamble to get the publication
rcParams::

    from lazy.plotting import use_style, diagnostic_panel

    use_style()
    fig = diagnostic_panel(z_true, grid.centers, pdfs)

Two families of diagnostic, and they answer different questions:

* **Point accuracy** -- :func:`plot_zphot_ztrue`, :func:`plot_residuals`. How
  close is the single redshift you would quote? Scatter, bias, outliers.
* **Calibration** -- :func:`plot_pit`, :func:`plot_pit_qq`,
  :func:`plot_coverage`. Are the *widths* honest? A model can have excellent
  point accuracy and badly wrong error bars, and only these plots show it.

:func:`plot_nz` is a third thing again: whether the stacked PDFs recover the
redshift distribution of the sample, which is what cosmological analyses
actually consume.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from lazy.plotting.style import figsize, one_to_one

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


def _axes(ax, **kwargs):
    if ax is not None:
        return ax
    return plt.subplots(figsize=figsize(**kwargs))[1]


def plot_zphot_ztrue(
    z_true: ArrayLike,
    z_pred: ArrayLike,
    *,
    ax=None,
    bins: int = 200,
    z_max: float | None = None,
    outlier_lines: bool = True,
    cmap: str | None = None,
    **kwargs,
):
    """Predicted against true redshift as a log-density image.

    A scatter plot of a survey-sized sample is a black blob, so this is a 2D
    histogram on a log colour scale -- which is also the only way the outlier
    islands (catastrophic failures at the wrong redshift) stay visible against
    the main locus.

    ``outlier_lines`` draws the DC1 outlier boundary
    ``|z_pred - z_true| = 0.06 (1 + z_true)``, so the fraction of points
    outside it is readable by eye.
    """
    from matplotlib.colors import LogNorm

    z_true = np.asarray(z_true, dtype=float)
    z_pred = np.asarray(z_pred, dtype=float)
    ax = _axes(ax, width="column", aspect="square")
    hi = float(z_max if z_max is not None else max(z_true.max(), z_pred.max()))
    ax.hist2d(
        z_true,
        z_pred,
        bins=bins,
        range=[[0, hi], [0, hi]],
        norm=LogNorm(),
        cmap=cmap or plt.rcParams["image.cmap"],
        **kwargs,
    )
    one_to_one(ax, 0.0, hi)
    if outlier_lines:
        edge = np.array([0.0, hi])
        for sign in (+1, -1):
            ax.plot(edge, edge + sign * _OUTLIER_FLOOR * (1 + edge), "k:", lw=0.8)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(r"$z_{\rm true}$")
    ax.set_ylabel(r"$z_{\rm phot}$")
    return ax


def plot_residuals(
    z_true: ArrayLike,
    z_pred: ArrayLike,
    *,
    ax=None,
    n_bins: int = 20,
    quantiles: tuple[float, float, float] = (16.0, 50.0, 84.0),
):
    """Scaled residual ``(z_phot - z_true) / (1 + z_true)`` against true redshift.

    The running median and its 16th-84th percentile band say where the bias
    lives: a model can have a fine global bias and still be systematically high
    at low redshift and low at high redshift, which this shows and a single
    number hides.
    """
    z_true = np.asarray(z_true, dtype=float)
    ez = (np.asarray(z_pred, dtype=float) - z_true) / (1.0 + z_true)
    ax = _axes(ax, width="column", aspect="tall")
    edges = np.quantile(z_true, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    index = np.clip(np.searchsorted(edges, z_true, side="right") - 1, 0, len(centers) - 1)
    stats = np.full((len(centers), 3), np.nan)
    for i in range(len(centers)):
        rows = index == i
        if rows.any():
            stats[i] = np.percentile(ez[rows], quantiles)
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.fill_between(centers, stats[:, 0], stats[:, 2], alpha=0.3, lw=0, color="C0")
    ax.plot(centers, stats[:, 1], color="C0")
    ax.set_xlabel(r"$z_{\rm true}$")
    ax.set_ylabel(r"$(z_{\rm phot} - z_{\rm true}) / (1 + z_{\rm true})$")
    return ax


def plot_pit(pit: ArrayLike, *, ax=None, n_bins: int = 20, label: str | None = None, **kwargs):
    """Histogram of the PIT values against the uniform distribution they should follow.

    The shape names the failure: a U means the PDFs are too narrow
    (over-confident), a dome means they are too wide, a slope means they are
    biased, and a spike at 0 or 1 counts catastrophic outliers.
    """
    pit = np.asarray(pit, dtype=float)
    ax = _axes(ax, width="column", aspect="golden")
    ax.hist(pit, bins=n_bins, range=(0, 1), density=True, histtype="step", label=label, **kwargs)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    return ax


def plot_pit_qq(pit: ArrayLike, *, ax=None, label: str | None = None, **kwargs):
    """Quantile-quantile plot of the PIT sample against U(0, 1).

    Harder to misread than the histogram: perfect calibration is the diagonal,
    above it means too wide, below means too narrow, and the deviation is in
    the same units as the probability itself.
    """
    pit = np.asarray(pit, dtype=float)
    ax = _axes(ax, width="column", aspect="square")
    ordered = np.sort(pit)
    uniform = (np.arange(1, ordered.size + 1) - 0.5) / ordered.size
    ax.plot(uniform, ordered, label=label, **kwargs)
    one_to_one(ax, 0.0, 1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("uniform quantile")
    ax.set_ylabel("PIT quantile")
    return ax


def plot_coverage(pit: ArrayLike, *, ax=None, label: str | None = None, **kwargs):
    """Empirical against nominal coverage of central credible intervals.

    For each nominal level ``q``, the fraction of galaxies whose true redshift
    falls inside the central ``q`` credible interval of its own PDF. This is
    the plot to quote when someone asks "if I take your 68% interval, how often
    is it right?" -- the answer should be 68% of the time, i.e. the diagonal.
    """
    pit = np.asarray(pit, dtype=float)
    ax = _axes(ax, width="column", aspect="square")
    nominal = np.linspace(0.0, 1.0, 101)
    empirical = np.array([np.mean(np.abs(pit - 0.5) <= 0.5 * q) for q in nominal])
    ax.plot(nominal, empirical, label=label, **kwargs)
    one_to_one(ax, 0.0, 1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    return ax


def plot_nz(
    z_grid: ArrayLike,
    pdfs: ArrayLike,
    *,
    z_true: ArrayLike | None = None,
    ax=None,
    label: str | None = "stacked PDFs",
    truth_label: str | None = "truth",
    **kwargs,
):
    """The sample redshift distribution: stacked PDFs against the true histogram.

    Stacking is only an estimator of N(z) under assumptions that photo-z PDFs
    rarely satisfy exactly, but it is what most analyses do, so how badly it
    fails is worth knowing. ``z_true``, if given, is histogrammed on the same
    grid for comparison.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    pdfs = np.asarray(pdfs, dtype=float)
    ax = _axes(ax, width="column", aspect="golden")
    ax.plot(z_grid, pdfs.mean(axis=0), label=label, **kwargs)
    if z_true is not None:
        from lazy.grid import RedshiftGrid

        edges = RedshiftGrid.from_centers(z_grid).edges
        ax.hist(
            np.asarray(z_true, dtype=float),
            bins=edges,
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
    z_grid: ArrayLike,
    pdfs: ArrayLike,
    *,
    z_true: ArrayLike | None = None,
    indices: ArrayLike | None = None,
    n: int = 6,
    random_state: int = 0,
    axes=None,
):
    """A handful of individual PDFs, with their true redshifts marked.

    Summary statistics hide multimodality; this is where you see it. By default
    ``n`` galaxies are drawn at random (reproducibly), or pass ``indices`` to
    pick them yourself.

    Returns the array of axes.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    pdfs = np.asarray(pdfs, dtype=float)
    if indices is None:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(pdfs), size=min(n, len(pdfs)), replace=False)
    indices = np.atleast_1d(np.asarray(indices, dtype=int))
    if axes is None:
        ncols = min(3, len(indices))
        nrows = int(np.ceil(len(indices) / ncols))
        _, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize(width="text", aspect=0.33 * nrows),
            squeeze=False,
            sharex=True,
        )
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, indices, strict=False):
        ax.plot(z_grid, pdfs[row], color="C0")
        if z_true is not None:
            ax.axvline(float(np.asarray(z_true, dtype=float)[row]), color="k", ls="--", lw=0.8)
        ax.set_xlim(z_grid[0], z_grid[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$z$")
    for ax in axes[len(indices) :]:
        ax.set_visible(False)
    return axes


def diagnostic_panel(
    z_true: ArrayLike,
    z_grid: ArrayLike,
    pdfs: ArrayLike,
    *,
    point: str = "z_peak",
    label: str | None = None,
):
    """The four-panel summary of one estimator: accuracy, bias, calibration, N(z).

    Panels are z_phot-z_true, the residual trend, the PIT Q-Q and the stacked
    N(z). Returns the :class:`matplotlib.figure.Figure`.
    """
    from lazy.metrics import evaluate_grid_pdfs, grid_point_estimates

    z_true = np.asarray(z_true, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    z_pred = grid_point_estimates(z_grid, pdfs)[point]
    _, _, pit = evaluate_grid_pdfs(z_true, z_grid, pdfs, point=point)

    fig, axes = plt.subplots(2, 2, figsize=figsize(width="text", aspect=0.85))
    plot_zphot_ztrue(z_true, z_pred, ax=axes[0, 0])
    plot_residuals(z_true, z_pred, ax=axes[0, 1])
    plot_pit_qq(pit, ax=axes[1, 0], label=label)
    plot_nz(z_grid, pdfs, z_true=z_true, ax=axes[1, 1])
    axes[1, 1].legend(loc="upper right")
    if label:
        fig.suptitle(label)
    fig.tight_layout()
    return fig
