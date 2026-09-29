# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Density metrics, as the LSST DESC PZ Data Challenge (DC1) defined them.

Every definition here is the one used to produce the published DC1 numbers,
read off ``LSSTDESC/PZDC1paper/metric_scripts/individual_metrics.py`` rather
than paraphrased from the text, so that our columns and Schmidt et al. (2020)
Tables 2, 3 and B1 are directly comparable.

The PDF metrics (CDE loss, PIT and its goodness-of-fit statistics) apply to
any continuous target. The point metrics are photo-z's by default: they scale
every residual by ``1 + z_true``, which suits redshift and little else. For
another target pass ``scale="none"`` to :func:`point_metrics`,
:func:`summarize` or ``evaluate``: every point statistic is then of the plain
residual ``z_pred - z_true``, in the target's units, including the outlier
rates, whose thresholds (the 0.06 floor, the fixed 0.15) are then in those
units too. The tables say which in their ``scale`` column.

Point estimates (Appendix B1), with the default ``scale="1+z"``::

    ez        = (z_phot - z_true) / (1 + z_true)
    bias      = median(ez)
    sigma_IQR = (Q75 - Q25) / 1.349
    sigma_MAD = 1.4826 * median(|ez - median(ez)|)
    outlier   = fraction(|ez| > max(0.06, 3 * sigma_IQR))

PDF metrics on a grid of bin centres (:class:`lazy.grid.Grid`)::

    CDE loss  = mean_i [ trapz(p_i^2, z) - 2 p_i(z_nearest to z_true_i) ]
    PIT_i     = integral of the gridded (linearly interpolated) p_i from 0
                to z_true_i
    KS, CvM   = one-sample goodness-of-fit statistics of the PIT sample
                against U(0, 1) (DC1 used ``skgof``; ``scipy.stats`` gives
                the identical statistics)
    AD        = DC1 discards PIT values outside ``(vmin, vmax)`` -- the
                statistic diverges at 0 and 1 -- and runs a one-sample
                Anderson-Darling test of the survivors against
                ``U(vmin, vmax)``. DC1 reports two cuts; :data:`AD_CUTS`
                holds them.

The nearest-grid-point likelihood (rather than linear interpolation) is also
what the Cal-PIT reference implementation uses, and is exact for the
piecewise-constant densities this project produces.

Every function that takes ``z_grid`` accepts either the bin centres or the
:class:`lazy.grid.Grid` itself. Pass the grid: its normalisation convention
then comes with it (``bin_edges`` is taken from
:attr:`~lazy.grid.Grid.histogram_edges`), so densities from a model's
histogram-normalised native grid are scored exactly rather than by the
trapezoid rule over their centres.

Typical usage example:

  pdfs = model.predict_proba(X_test)
  table = metrics.summarize(z_true, model.grid_, pdfs, label="tabfm")
"""

from __future__ import annotations

import dataclasses
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from lazy import _typing
from lazy import grid as grid_lib

#: How the point metrics scale a residual: ``"1+z"`` divides it by
#: ``1 + z_true`` (photo-z's convention, DC1's numbers), ``"none"`` leaves
#: ``z_pred - z_true`` as it is.
Scale: TypeAlias = Literal["1+z", "none"]
#: The accepted values of ``scale``.
SCALES: tuple[str, ...] = ("1+z", "none")

OUTLIER_FLOOR = 0.06
DC1_OUTLIER_THRESHOLD = 0.15
PIT_EXTREME = 1e-4
# The two Anderson-Darling cut ranges DC1 tabulates (AD1, AD2); the script
# also computed (0.1, 0.9), which the paper does not quote.
AD_CUTS = ((0.05, 0.95), (0.01, 0.99))


@dataclasses.dataclass(frozen=True)
class PointMetrics:
    """The DC1 Appendix B1 point-estimate statistics for one sample.

    All statistics are of the residual ``ez``: by default the scaled
    ``(z_phot - z_true) / (1 + z_true)``, or the plain ``z_phot - z_true``
    when :attr:`scale` is ``"none"``, the thresholds then in the target's
    units.

    Attributes:
        n: Number of objects.
        bias: Median of ``ez``.
        sigma_mad: ``1.4826 * median(|ez - median(ez)|)``.
        sigma_iqr: Interquartile range of ``ez`` divided by 1.349.
        outlier_rate: Fraction of objects with
            ``|ez| > outlier_threshold``.
        outlier_threshold: ``max(0.06, 3 * sigma_iqr)``.
        outlier_rate_015: Fraction of objects with ``|ez| > 0.15``.
        median_abs_ez: Median of ``|ez|``.
        scale: The residual's scaling, ``"1+z"`` or ``"none"``.
    """

    n: int
    bias: float
    sigma_mad: float
    sigma_iqr: float
    outlier_rate: float
    outlier_threshold: float
    outlier_rate_015: float
    median_abs_ez: float
    scale: str = "1+z"

    def as_dict(self) -> dict[str, float | int | str]:
        """Returns the metrics as a plain dict, for building a table row."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PDFMetrics:
    """The DC1 conditional-density and PIT statistics for one sample.

    Attributes:
        n: Number of objects.
        cde_loss: Conditional-density-estimate loss (see :func:`cde_loss`).
        pit_ks: Kolmogorov-Smirnov statistic of the PIT against U(0, 1).
        pit_ks_pvalue: Its p-value.
        pit_cvm: Cramer-von Mises statistic of the PIT against U(0, 1).
        pit_ad1: DC1 Anderson-Darling statistic for the first cut of
            :data:`AD_CUTS`.
        pit_ad2: The same for the second cut.
        pit_rmse: RMS difference between the sorted PIT values and the
            uniform quantiles.
        pit_kl: KL divergence of the binned PIT histogram from uniform.
        pit_outlier_rate: Fraction of PIT values within :data:`PIT_EXTREME`
            of 0 or 1.
    """

    n: int
    cde_loss: float
    pit_ks: float
    pit_ks_pvalue: float
    pit_cvm: float
    pit_ad1: float
    pit_ad2: float
    pit_rmse: float
    pit_kl: float
    pit_outlier_rate: float

    def as_dict(self) -> dict[str, float | int]:
        """Returns the metrics as a plain dict, for building a table row."""
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------


def scaled_residual(
    z_true: npt.ArrayLike, z_pred: npt.ArrayLike
) -> _typing.FloatArray:
    """The residual ``(z_phot - z_true) / (1 + z_true)``.

    This ``ez`` is the residual every point metric is built on. Dividing by
    ``1 + z`` is what makes a 0.05 error at z = 0.2 and at z = 1.5
    comparable: photometric redshift errors scale with the observed
    wavelength shift, not with z itself.

    Args:
        z_true: True values, finite, shape (n,).
        z_pred: Point estimates, finite, shape (n,).

    Returns:
        The scaled residuals ``ez``, shape (n,).
    """
    truth, pred = _check_pair(z_true, z_pred)
    return (pred - truth) / (1.0 + truth)


def point_metrics(
    z_true: npt.ArrayLike,
    z_pred: npt.ArrayLike,
    *,
    scale: Scale = "1+z",
) -> PointMetrics:
    """Bias, scatter and outlier rate of a set of point estimates.

    ``sigma_iqr`` and ``sigma_mad`` are both robust widths; they differ when
    the residual distribution has heavy tails, and reporting both is what DC1
    does. The outlier threshold is ``max(0.06, 3 * sigma_IQR)``, so a model
    with tiny scatter is not credited for having few "outliers" merely
    because its own threshold shrank.

    Args:
        z_true: True values, finite, shape (n,).
        z_pred: Point estimates, finite, shape (n,).
        scale: ``"1+z"`` (the default, photo-z's and DC1's convention) for
            residuals scaled by ``1 + z_true``; ``"none"`` for the plain
            ``z_pred - z_true`` of any other target, every statistic and
            both outlier thresholds then in the target's units.

    Returns:
        The statistics.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}: {scale=}")
    if scale == "1+z":
        ez = scaled_residual(z_true, z_pred)
    else:
        truth, pred = _check_pair(z_true, z_pred)
        ez = pred - truth
    median = float(np.median(ez))
    q25, q75 = np.percentile(ez, [25.0, 75.0])
    sigma_iqr = float((q75 - q25) / 1.349)
    threshold = max(OUTLIER_FLOOR, 3.0 * sigma_iqr)
    return PointMetrics(
        n=int(ez.size),
        bias=median,
        sigma_mad=float(1.4826 * np.median(np.abs(ez - median))),
        sigma_iqr=sigma_iqr,
        outlier_rate=float(np.mean(np.abs(ez) > threshold)),
        outlier_threshold=float(threshold),
        outlier_rate_015=float(np.mean(np.abs(ez) > DC1_OUTLIER_THRESHOLD)),
        median_abs_ez=float(np.median(np.abs(ez))),
        scale=scale,
    )


def _check_pair(
    z_true: npt.ArrayLike, z_pred: npt.ArrayLike
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """True values and point estimates as equal-length finite 1-D arrays."""
    truth = np.asarray(z_true, dtype=float)
    pred = np.asarray(z_pred, dtype=float)
    if truth.shape != pred.shape or truth.ndim != 1:
        raise ValueError("z_true and z_pred must be equal-length 1D arrays")
    if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
        raise ValueError("non-finite values supplied")
    return truth, pred


# --------------------------------------------------------------------------
# Grid PDFs
# --------------------------------------------------------------------------


def normalize_grid_pdfs(
    z_grid: grid_lib.Grid | npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    bin_edges: npt.ArrayLike | None = None,
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Clip to non-negative, then renormalize each row to unit trapezoid mass.

    Trapezoid is the DC1/``qp`` convention
    (``Ensemble.evaluate(..., norm=True)``) and the one :func:`cde_loss`
    integrates with, so normalizing any other way would leave
    ``trapz(p) != 1`` inside the loss. Non-finite values are treated as zero,
    and a row with no mass left becomes a uniform density.

    Args:
        z_grid: Strictly increasing bin centres, shape (g,) with g >= 2, or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        pdfs: Densities at those centres, shape (n, g).
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.

    Returns:
        A tuple (grid, density): the centres as a float array, shape (g,),
        and the normalised densities, shape (n, g).
    """
    z_grid, bin_edges = _split_grid(z_grid, bin_edges)
    grid = np.asarray(z_grid, dtype=float)
    density = np.array(pdfs, dtype=float)
    if grid.ndim != 1 or density.ndim != 2 or density.shape[1] != grid.size:
        raise ValueError("expected z_grid shape (g,) and pdfs shape (n, g)")
    if grid.size < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError(
            "z_grid must be strictly increasing with at least two points"
        )
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(density, 0.0, None, out=density)
    if bin_edges is not None:
        widths = _bin_widths(bin_edges, grid.size)
        mass = density @ widths
        bad = ~(mass > 0)
        if bad.any():
            density[bad] = 1.0
            mass = density @ widths
        return grid, density / mass[:, None]
    mass = np.trapezoid(density, grid, axis=1)
    bad = ~(mass > 0)
    if bad.any():
        density[bad] = 1.0
        mass = np.trapezoid(density, grid, axis=1)
    return grid, density / mass[:, None]


def normalization_error(
    z_grid: grid_lib.Grid | npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    bin_edges: npt.ArrayLike | None = None,
) -> _typing.FloatArray:
    """``|trapz(p, z) - 1|`` per row: the check callers assert on.

    Args:
        z_grid: Bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        pdfs: Densities at those centres, shape (n, g).
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.

    Returns:
        The absolute normalisation error of each row, shape (n,).
    """
    z_grid, bin_edges = _split_grid(z_grid, bin_edges)
    grid = np.asarray(z_grid, dtype=float)
    if bin_edges is not None:
        widths = _bin_widths(bin_edges, grid.size)
        return np.abs(np.asarray(pdfs, dtype=float) @ widths - 1.0)
    return np.abs(
        np.trapezoid(np.asarray(pdfs, dtype=float), grid, axis=1) - 1.0
    )


def grid_cdf(
    z_grid: grid_lib.Grid | npt.ArrayLike,
    density: _typing.FloatArray,
    *,
    bin_edges: npt.ArrayLike | None = None,
) -> _typing.FloatArray:
    """Cumulative mass at each grid point, by the trapezoid rule of the norm.

    Args:
        z_grid: Bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        density: Densities at those centres, shape (n, g).
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.

    Returns:
        The cumulative mass below each centre, shape (n, g). By trapezoid
        column 0 is zero; for a histogram grid each entry is the mass below
        the bin's left edge plus half its own.
    """
    z_grid, bin_edges = _split_grid(z_grid, bin_edges)
    grid = np.asarray(z_grid, dtype=float)
    density = np.asarray(density, dtype=float)
    if bin_edges is not None:
        mass = density * _bin_widths(bin_edges, grid.size)
        return np.cumsum(mass, axis=1) - 0.5 * mass
    increments = 0.5 * (density[:, 1:] + density[:, :-1]) * np.diff(grid)
    return np.column_stack(
        (np.zeros(len(density)), np.cumsum(increments, axis=1))
    )


def z_peak(
    z_grid: grid_lib.Grid | npt.ArrayLike, density: npt.ArrayLike
) -> _typing.FloatArray:
    """DC1 ``z_PEAK``: the mode of the PDF.

    Args:
        z_grid: Bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        density: Densities at those centres, shape (n, g).

    Returns:
        The centre of each row's highest bin, shape (n,).
    """
    z_grid, _ = _split_grid(z_grid, None)
    grid = np.asarray(z_grid, dtype=float)
    return grid[np.argmax(np.asarray(density, dtype=float), axis=1)]


def z_weight(
    z_grid: grid_lib.Grid | npt.ArrayLike,
    density: npt.ArrayLike,
    frac: float = 0.05,
    *,
    bin_edges: npt.ArrayLike | None = None,
) -> _typing.FloatArray:
    """DC1 ``z_WEIGHT``: the main-peak weighted mean of Dahlen et al. (2013).

    The main peak is the contiguous run of grid points containing the mode
    over which ``p(z) >= frac * p(z_PEAK)``; ``z_WEIGHT`` is the
    probability-weighted mean of ``z`` over that run.

    Args:
        z_grid: Bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        density: Densities at those centres, shape (n, g).
        frac: Fraction of the peak density that bounds the main peak.
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.

    Returns:
        The main-peak weighted mean of each row, shape (n,). A peak one grid
        point wide falls back to the mode.
    """
    z_grid, bin_edges = _split_grid(z_grid, bin_edges)
    grid = np.asarray(z_grid, dtype=float)
    dens = np.asarray(density, dtype=float)
    n, g = dens.shape
    idx = np.arange(g)
    rows = np.arange(n)
    peak = np.argmax(dens, axis=1)
    below = dens < (frac * dens[rows, peak])[:, None]
    left = (
        np.maximum.accumulate(np.where(below, idx, -1), axis=1)[rows, peak] + 1
    )
    right = (
        np.minimum.accumulate(np.where(below, idx, g)[:, ::-1], axis=1)[
            :, ::-1
        ][rows, peak]
        - 1
    )
    inside = (idx[None, :] >= left[:, None]) & (idx[None, :] <= right[:, None])
    weights = dens * inside
    if bin_edges is not None:
        weights = weights * _bin_widths(bin_edges, g)
        mass = weights.sum(axis=1)
        weighted = weights @ grid
    else:
        mass = np.trapezoid(weights, grid, axis=1)
        weighted = np.trapezoid(weights * grid[None, :], grid, axis=1)
    # A peak one grid point wide integrates to zero mass; fall back to the mode.
    out = grid[peak].astype(float)
    np.divide(weighted, mass, out=out, where=mass > 0)
    return out


def grid_point_estimates(
    z_grid: grid_lib.Grid | npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    bin_edges: npt.ArrayLike | None = None,
) -> dict[str, _typing.FloatArray]:
    """The DC1 point estimators (``peak``, ``weight``) plus mean and median.

    The densities are normalised first (see :func:`normalize_grid_pdfs`).

    Args:
        z_grid: Strictly increasing bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        pdfs: Densities at those centres, shape (n, g).
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.

    Returns:
        A dict with keys ``"z_peak"``, ``"z_weight"``, ``"z_mean"`` and
        ``"z_median"``, each an array of shape (n,).
    """
    z_grid, bin_edges = _split_grid(z_grid, bin_edges)
    if bin_edges is not None:
        return _histogram_point_estimates(z_grid, pdfs, bin_edges)
    grid, density = normalize_grid_pdfs(z_grid, pdfs)
    cdf = grid_cdf(grid, density)
    rows = np.arange(len(density))
    right = np.clip(np.argmax(cdf >= 0.5, axis=1), 1, grid.size - 1)
    left = right - 1
    span = cdf[rows, right] - cdf[rows, left]
    frac = np.divide(
        0.5 - cdf[rows, left], span, out=np.zeros(len(density)), where=span > 0
    )
    z_mean: _typing.FloatArray = np.asarray(
        np.trapezoid(density * grid[None, :], grid, axis=1), dtype=np.float64
    )
    return {
        "z_peak": z_peak(grid, density),
        "z_weight": z_weight(grid, density),
        "z_mean": z_mean,
        "z_median": grid[left] + frac * (grid[right] - grid[left]),
    }


def evaluate_grid_at_truth(
    z_true: npt.ArrayLike,
    z_grid: grid_lib.Grid | npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    bin_edges: npt.ArrayLike | None = None,
) -> tuple[
    _typing.FloatArray,
    _typing.FloatArray,
    _typing.FloatArray,
    _typing.FloatArray,
]:
    """Returns ``(grid, density, pdf_at_truth, pit)``.

    ``pdf_at_truth`` is the density at the grid point *nearest* the truth
    (the DC1 and Cal-PIT convention, exact for a piecewise-constant density);
    ``pit`` is the integral of the linearly interpolated density from the
    grid's left edge to the truth, which is what ``qp``'s gridded
    ``integrate`` computes.

    Args:
        z_true: Finite true values, one per PDF, shape (n,).
        z_grid: Strictly increasing bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        pdfs: Densities at those centres, shape (n, g).
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.
            Then the density at the truth is its bin's, zero outside the
            edges, and the PIT is the exact piecewise-linear CDF.

    Returns:
        A tuple (grid, density, pdf_at_truth, pit): the centres, shape (g,);
        the normalised densities, shape (n, g); the density at the truth,
        shape (n,), zero for a truth more than half a bin outside the grid;
        and the PIT values, shape (n,), in [0, 1].
    """
    z_grid, bin_edges = _split_grid(z_grid, bin_edges)
    truth = np.asarray(z_true, dtype=float)
    grid, density = normalize_grid_pdfs(z_grid, pdfs, bin_edges=bin_edges)
    if truth.ndim != 1 or truth.size != len(density):
        raise ValueError("z_true must have one value per PDF")
    if not np.isfinite(truth).all():
        raise ValueError("z_true contains non-finite values")
    if bin_edges is not None:
        pdf_at_truth, pit = _histogram_at_truth(truth, bin_edges, density)
        return grid, density, pdf_at_truth, pit
    rows = np.arange(truth.size)
    cdf = grid_cdf(grid, density)
    right = np.clip(
        np.searchsorted(grid, truth, side="right"), 1, grid.size - 1
    )
    left = right - 1
    frac = np.clip((truth - grid[left]) / (grid[right] - grid[left]), 0.0, 1.0)
    nearest = np.clip(
        np.searchsorted(0.5 * (grid[1:] + grid[:-1]), truth, side="right"),
        0,
        grid.size - 1,
    )
    pdf_at_truth = np.where(
        (truth < grid[0] - 0.5 * (grid[1] - grid[0]))
        | (truth > grid[-1] + 0.5 * (grid[-1] - grid[-2])),
        0.0,
        density[rows, nearest],
    )
    pit = cdf[rows, left] + frac * (cdf[rows, right] - cdf[rows, left])
    pit = np.where(truth <= grid[0], 0.0, np.where(truth >= grid[-1], 1.0, pit))
    return grid, density, pdf_at_truth, pit


def cde_loss(
    z_true: npt.ArrayLike,
    z_grid: grid_lib.Grid | npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    bin_edges: npt.ArrayLike | None = None,
) -> float:
    """The conditional-density-estimate loss; lower is better.

    ``E[integral p^2] - 2 E[p(z_true)]`` is the L2 distance between the
    estimated and true conditional densities, up to a constant that does not
    depend on the estimator -- so it ranks methods without knowing the
    truth's density. It is the single number this project optimises.

    Args:
        z_true: True values, one per PDF, shape (n,).
        z_grid: Strictly increasing bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        pdfs: Densities at those centres, shape (n, g); normalised here.
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.

    Returns:
        The loss, averaged over objects.
    """
    terms, _ = per_object_scores(z_true, z_grid, pdfs, bin_edges=bin_edges)
    return float(np.mean(terms))


def per_object_scores(
    z_true: npt.ArrayLike,
    z_grid: grid_lib.Grid | npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    bin_edges: npt.ArrayLike | None = None,
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Each object's contribution to the CDE loss, and its PIT.

    Every PDF metric is a statistic of these two per-object arrays, so they
    can be computed a block of rows at a time and joined:
    :func:`cde_loss` is the mean of the first, and :func:`summarize_scores`
    turns both, with the point estimates, into the :func:`summarize` table.

    Args:
        z_true: Finite true values, one per PDF, shape (n,).
        z_grid: Strictly increasing bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        pdfs: Densities at those centres, shape (n, g); normalised here.
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.

    Returns:
        A tuple (cde_terms, pit): ``integral p_i^2 - 2 p_i(z_true_i)`` per
        object, shape (n,), and the PIT values, shape (n,).
    """
    z_grid, bin_edges = _split_grid(z_grid, bin_edges)
    grid, density, pdf_at_truth, pit = evaluate_grid_at_truth(
        z_true, z_grid, pdfs, bin_edges=bin_edges
    )
    squared = _integral_of_square(grid, density, bin_edges)
    return squared - 2.0 * pdf_at_truth, pit


# --------------------------------------------------------------------------
# PIT statistics
# --------------------------------------------------------------------------


def anderson_darling_dc1(
    pit: npt.ArrayLike, vmin: float = 0.05, vmax: float = 0.95
) -> float:
    """DC1's Anderson-Darling statistic of the PIT sample.

    The statistic diverges at 0 and 1, so DC1 discards PIT values outside
    ``(vmin, vmax)`` and tests the survivors against ``U(vmin, vmax)``
    (``individual_metrics.EvaluateMetric.AD``, which calls
    ``skgof.ad_test``).

    Args:
        pit: PIT values, shape (n,).
        vmin: Lower cut; values at or below it are discarded.
        vmax: Upper cut; values at or above it are discarded.

    Returns:
        The statistic, or NaN when no value survives the cuts.
    """
    values = np.asarray(pit, dtype=float)
    kept = np.sort(values[(values > vmin) & (values < vmax)])
    n = kept.size
    if n == 0:
        return float("nan")
    u = np.clip((kept - vmin) / (vmax - vmin), 1e-15, 1.0 - 1e-15)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))))


def pit_statistics(pit: npt.ArrayLike, n_bins: int = 20) -> dict[str, float]:
    """Goodness-of-fit of the PIT sample against U(0, 1), several ways.

    A well-calibrated set of PDFs has uniform PIT values. Each statistic is
    sensitive to a different departure, which is why DC1 quotes all of them:
    KS to the largest CDF gap, CvM and AD to the whole shape (AD weighted
    towards the tails), KL to the binned histogram, and ``pit_outlier_rate``
    to the spikes at 0 and 1 that mark catastrophic failures.

    Args:
        pit: PIT values, non-empty, shape (n,).
        n_bins: Number of histogram bins on [0, 1] for the KL divergence.

    Returns:
        A dict with the :class:`PDFMetrics` PIT fields (``pit_ks``,
        ``pit_ks_pvalue``, ``pit_cvm``, ``pit_ad1``, ``pit_ad2``,
        ``pit_rmse``, ``pit_kl``, ``pit_outlier_rate``).
    """
    values = np.asarray(pit, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pit must be a non-empty 1D array")
    ks = stats.kstest(values, "uniform")
    cvm = stats.cramervonmises(values, "uniform")
    ordered = np.sort(values)
    ideal = (np.arange(1, values.size + 1) - 0.5) / values.size
    counts, _ = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    return {
        "pit_ks": float(ks.statistic),
        "pit_ks_pvalue": float(ks.pvalue),
        "pit_cvm": float(cvm.statistic),
        "pit_ad1": anderson_darling_dc1(values, *AD_CUTS[0]),
        "pit_ad2": anderson_darling_dc1(values, *AD_CUTS[1]),
        "pit_rmse": float(np.sqrt(np.mean((ordered - ideal) ** 2))),
        "pit_kl": float(
            np.sum(
                probabilities[nonzero] * np.log(probabilities[nonzero] * n_bins)
            )
        ),
        "pit_outlier_rate": float(
            np.mean((values < PIT_EXTREME) | (values > 1.0 - PIT_EXTREME))
        ),
    }


def pdf_metrics(
    z_true: npt.ArrayLike,
    z_grid: grid_lib.Grid | npt.ArrayLike,
    pdfs: npt.ArrayLike,
    *,
    bin_edges: npt.ArrayLike | None = None,
) -> tuple[PDFMetrics, _typing.FloatArray]:
    """Scores grid PDFs with the CDE loss and the PIT statistics.

    Args:
        z_true: True values, one per PDF, shape (n,).
        z_grid: Strictly increasing bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        pdfs: Densities at those centres, shape (n, g); normalised here.
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.

    Returns:
        A tuple (metrics, pit): the metric bundle and the per-object PIT
        values, shape (n,).
    """
    terms, pit = per_object_scores(z_true, z_grid, pdfs, bin_edges=bin_edges)
    return _pdf_metrics_from(terms, pit), pit


def _pdf_metrics_from(
    cde_terms: _typing.FloatArray, pit: _typing.FloatArray
) -> PDFMetrics:
    """The PDF metric bundle from the per-object scores."""
    return PDFMetrics(
        n=int(pit.size),
        cde_loss=float(np.mean(cde_terms)),
        **pit_statistics(pit),
    )


def evaluate_grid_pdfs(
    z_true: npt.ArrayLike,
    z_grid: grid_lib.Grid | npt.ArrayLike,
    pdfs: npt.ArrayLike,
    point: str = "z_peak",
    *,
    bin_edges: npt.ArrayLike | None = None,
    scale: Scale = "1+z",
) -> tuple[PointMetrics, PDFMetrics, _typing.FloatArray]:
    """Convenience: point metrics from a PDF reduction plus the PDF metrics.

    Args:
        z_true: True values, one per PDF, shape (n,).
        z_grid: Strictly increasing bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        pdfs: Densities at those centres, shape (n, g); normalised here.
        point: The reduction to score as the point estimate: a key of
            :func:`grid_point_estimates` (``"z_peak"``, ``"z_weight"``,
            ``"z_mean"`` or ``"z_median"``).
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.
        scale: How the point metrics scale residuals: ``"1+z"`` (the
            default, DC1's) or ``"none"``; see :func:`point_metrics`.

    Returns:
        A tuple (point_metrics, pdf_metrics, pit): the point statistics of
        the chosen reduction, the PDF metric bundle and the per-object PIT
        values, shape (n,).
    """
    z_grid, bin_edges = _split_grid(z_grid, bin_edges)
    estimates = grid_point_estimates(z_grid, pdfs, bin_edges=bin_edges)
    if point not in estimates:
        raise ValueError(f"point must be one of {sorted(estimates)}")
    distribution, pit = pdf_metrics(z_true, z_grid, pdfs, bin_edges=bin_edges)
    return (
        point_metrics(z_true, estimates[point], scale=scale),
        distribution,
        pit,
    )


def summarize(
    z_true: npt.ArrayLike,
    z_grid: grid_lib.Grid | npt.ArrayLike,
    pdfs: npt.ArrayLike,
    point: str = "z_peak",
    label: str | None = None,
    *,
    bin_edges: npt.ArrayLike | None = None,
    scale: Scale = "1+z",
) -> pd.DataFrame:
    """Every point and PDF metric as a one-row table.

    Built for stacking: ``pd.concat([summarize(..., label=name) for ...])``
    gives the comparison table that the tutorials and the paper both report.

    Args:
        z_true: True values, one per PDF, shape (n,).
        z_grid: Strictly increasing bin centres, shape (g,), or the
            :class:`~lazy.grid.Grid` itself (see the module docstring).
        pdfs: Densities at those centres, shape (n, g); normalised here.
        point: The point-estimate reduction, as in
            :func:`evaluate_grid_pdfs`.
        label: Model name for a leading ``model`` column; no such column
            when ``None``.
        bin_edges: The grid's bin edges, shape (g + 1,), to treat each
            density as constant across its bin (a ``"histogram"`` grid)
            instead of using the trapezoid rule over the centres.
            A histogram-normalised :class:`~lazy.grid.Grid` passed as
            ``z_grid`` supplies them; given with a Grid, they must be its
            edges.
        scale: How the point metrics scale residuals: ``"1+z"`` (the
            default, DC1's, photo-z's) or ``"none"`` for plain residuals of
            any other target; see :func:`point_metrics`.

    Returns:
        A one-row table: ``model`` (if labelled), ``point_estimate``,
        ``scale``, the other :class:`PointMetrics` fields and the
        :class:`PDFMetrics` fields other than ``n``.

    Examples:
        >>> z = np.array([0.5, 1.0])
        >>> grid = np.linspace(0.005, 1.995, 200)
        >>> pdfs = np.ones((2, 200))
        >>> summarize(z, grid, pdfs, label="uniform")["model"].tolist()
        ['uniform']
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}: {scale=}")
    z_grid, bin_edges = _split_grid(z_grid, bin_edges)
    estimates = grid_point_estimates(z_grid, pdfs, bin_edges=bin_edges)
    if point not in estimates:
        raise ValueError(f"point must be one of {sorted(estimates)}")
    cde_terms, pit = per_object_scores(
        z_true, z_grid, pdfs, bin_edges=bin_edges
    )
    return summarize_scores(
        z_true,
        estimates[point],
        cde_terms,
        pit,
        point=point,
        label=label,
        scale=scale,
    )


def summarize_scores(
    z_true: npt.ArrayLike,
    z_pred: npt.ArrayLike,
    cde_terms: npt.ArrayLike,
    pit: npt.ArrayLike,
    point: str = "z_peak",
    label: str | None = None,
    *,
    scale: Scale = "1+z",
) -> pd.DataFrame:
    """The :func:`summarize` table from per-object pieces.

    For data too large to hold as one ``(n, g)`` array: compute the point
    estimates and :func:`per_object_scores` a block of rows at a time,
    join them, and pass them here. The table is the one :func:`summarize`
    gives on the whole array.

    Args:
        z_true: Finite true values, shape (n,).
        z_pred: The point estimates, shape (n,).
        cde_terms: Each object's CDE-loss term, shape (n,).
        pit: Each object's PIT value, shape (n,).
        point: The name of the point estimate, for the table.
        label: Model name for a leading ``model`` column; no such column
            when ``None``.
        scale: How the point metrics scale residuals; see
            :func:`point_metrics`.

    Returns:
        The one-row table :func:`summarize` describes.
    """
    point_metrics_ = point_metrics(z_true, z_pred, scale=scale)
    pdf_metrics_ = _pdf_metrics_from(
        np.asarray(cde_terms, dtype=float), np.asarray(pit, dtype=float)
    )
    row: dict[str, object] = {}
    if label is not None:
        row["model"] = label
    row["point_estimate"] = point
    point_row = point_metrics_.as_dict()
    row["scale"] = point_row.pop("scale")
    row.update(point_row)
    row.update({k: v for k, v in pdf_metrics_.as_dict().items() if k != "n"})
    return pd.DataFrame([row])


# --------------------------------------------------------------------------
# Histogram grids: densities constant across each bin
# --------------------------------------------------------------------------


def _split_grid(
    z_grid: grid_lib.Grid | npt.ArrayLike, bin_edges: npt.ArrayLike | None
) -> tuple[npt.ArrayLike, npt.ArrayLike | None]:
    """Returns a tuple (centres, bin_edges) from a Grid or bare centres.

    A :class:`~lazy.grid.Grid` gives its centres and, when it is
    histogram-normalised, its edges, so its own convention scores it. Bare
    centres pass through with ``bin_edges`` as given.
    """
    if not isinstance(z_grid, grid_lib.Grid):
        return z_grid, bin_edges
    if bin_edges is None:
        return z_grid.centers, z_grid.histogram_edges
    edges = np.asarray(bin_edges, dtype=float)
    if edges.shape != z_grid.edges.shape or not np.allclose(
        edges, z_grid.edges, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "bin_edges differ from the edges of the Grid passed as z_grid; "
            "pass the Grid alone"
        )
    return z_grid.centers, edges


def _bin_widths(bin_edges: npt.ArrayLike, n_bins: int) -> _typing.FloatArray:
    """The widths of ``n_bins`` bins from their strictly increasing edges."""
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size != n_bins + 1:
        raise ValueError(
            f"bin_edges must have one more entry than the grid: "
            f"{edges.shape=}, {n_bins=}"
        )
    widths = np.diff(edges)
    if not np.all(widths > 0):
        raise ValueError("bin_edges must be strictly increasing")
    return widths


def _integral_of_square(
    grid: _typing.FloatArray,
    density: _typing.FloatArray,
    bin_edges: npt.ArrayLike | None,
) -> _typing.FloatArray:
    """The integral of p(z)^2 per row, by each grid convention."""
    if bin_edges is None:
        return np.asarray(np.trapezoid(density**2, grid, axis=1))
    return (density**2) @ _bin_widths(bin_edges, grid.size)


def _histogram_at_truth(
    truth: _typing.FloatArray,
    bin_edges: npt.ArrayLike,
    density: _typing.FloatArray,
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Returns a tuple (density at the truth, PIT) for a histogram grid.

    The density is that of the bin holding the truth, zero outside the
    edges; the PIT is the mass below the truth, linear within its bin. This
    is the exact scoring of a bar distribution.
    """
    edges = np.asarray(bin_edges, dtype=float)
    widths = _bin_widths(edges, density.shape[1])
    mass = density * widths
    below = np.cumsum(mass, axis=1) - mass
    rows = np.arange(truth.size)
    index = np.clip(
        np.searchsorted(edges, truth, side="right") - 1, 0, widths.size - 1
    )
    inside = (truth >= edges[0]) & (truth < edges[-1])
    pdf_at_truth = np.where(inside, density[rows, index], 0.0)
    fraction = np.clip((truth - edges[index]) / widths[index], 0.0, 1.0)
    pit = below[rows, index] + mass[rows, index] * fraction
    pit = np.where(
        truth < edges[0], 0.0, np.where(truth >= edges[-1], 1.0, pit)
    )
    return pdf_at_truth, pit


def _histogram_point_estimates(
    z_grid: npt.ArrayLike, pdfs: npt.ArrayLike, bin_edges: npt.ArrayLike
) -> dict[str, _typing.FloatArray]:
    """The point estimates of :func:`grid_point_estimates` on a histogram grid.

    ``z_peak`` is the centre of the densest bin, ``z_mean`` the exact mean of
    the piecewise-constant density, ``z_median`` the exact inverse of its
    CDF at one half, and ``z_weight`` the main-peak mean with bin masses as
    weights.
    """
    grid, density = normalize_grid_pdfs(z_grid, pdfs, bin_edges=bin_edges)
    edges = np.asarray(bin_edges, dtype=float)
    widths = _bin_widths(edges, grid.size)
    mass = density * widths
    cumulative = np.cumsum(mass, axis=1)
    rows = np.arange(len(density))
    index = np.clip(np.argmax(cumulative >= 0.5, axis=1), 0, grid.size - 1)
    below = cumulative[rows, index] - mass[rows, index]
    fraction = np.divide(
        0.5 - below,
        mass[rows, index],
        out=np.zeros(len(density)),
        where=mass[rows, index] > 0,
    )
    return {
        "z_peak": z_peak(grid, density),
        "z_weight": z_weight(grid, density, bin_edges=edges),
        "z_mean": mass @ grid,
        "z_median": edges[index] + np.clip(fraction, 0.0, 1.0) * widths[index],
    }
