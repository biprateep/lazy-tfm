"""Redshift grids: the output format every :mod:`lazy` estimator writes onto.

A conditional density estimate here is always a table: one row per galaxy, one
column per redshift bin, holding a *density* (not a probability) at the bin
centre. :class:`RedshiftGrid` owns the bin edges, so the same object gives you
the centres the metrics integrate over, the edges that mass-conserving
rebinning needs, and the normalisation convention.

The grid is an input, not a constant. Nothing in the library requires the LSST
DESC DC1 grid, or a uniform grid, or a particular number of bins -- in
particular the "at most ten classes" limit of TabFM's classifier constrains the
*hierarchy* it builds internally, never the grid you ask for output on
(:mod:`lazy.models.tabfm`). :data:`DC1_GRID` is provided because reproducing the
Data Challenge numbers requires exactly that grid.

Normalisation is trapezoidal over the bin centres. That is the ``qp`` and DC1
convention and it is what :func:`lazy.metrics.cde_loss` integrates with, so
normalising any other way would leave ``trapz(p) != 1`` inside the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["DC1_GRID", "RedshiftGrid", "as_grid"]


@dataclass(frozen=True)
class RedshiftGrid:
    """A binned redshift axis defined by its ``n_bins + 1`` edges.

    >>> grid = RedshiftGrid.linear(0.0, 2.0, 200)
    >>> grid.n_bins, float(grid.centers[0]), float(grid.centers[-1])
    (200, 0.005, 1.995)
    """

    edges: NDArray[np.float64]

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=float)
        if edges.ndim != 1 or edges.size < 3:
            raise ValueError("edges must be a 1D array of at least three values")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be strictly increasing")
        if not np.isfinite(edges).all():
            raise ValueError("edges must all be finite")
        edges.setflags(write=False)
        object.__setattr__(self, "edges", edges)

    # -- constructors ------------------------------------------------------

    @classmethod
    def linear(cls, z_min: float, z_max: float, n_bins: int) -> RedshiftGrid:
        """``n_bins`` equal-width bins spanning ``[z_min, z_max]``."""
        return cls(np.linspace(float(z_min), float(z_max), int(n_bins) + 1))

    @classmethod
    def from_edges(cls, edges: ArrayLike) -> RedshiftGrid:
        """A grid from explicit bin edges (any spacing)."""
        return cls(np.asarray(edges, dtype=float))

    @classmethod
    def from_centers(cls, centers: ArrayLike) -> RedshiftGrid:
        """A grid from bin centres, splitting the gaps and extrapolating the ends.

        Exact for a uniform grid: ``from_centers(g.centers) == g``.

        >>> grid = RedshiftGrid.linear(0.0, 2.0, 200)
        >>> RedshiftGrid.from_centers(grid.centers) == grid
        True
        """
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 1 or centers.size < 2:
            raise ValueError("centers must be a 1D array of at least two values")
        inner = 0.5 * (centers[1:] + centers[:-1])
        return cls(
            np.concatenate(
                [
                    [centers[0] - (inner[0] - centers[0])],
                    inner,
                    [centers[-1] + (centers[-1] - inner[-1])],
                ]
            )
        )

    # -- geometry ----------------------------------------------------------

    @property
    def centers(self) -> NDArray[np.float64]:
        """Bin centres: the redshifts a density row is tabulated at."""
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def widths(self) -> NDArray[np.float64]:
        """Bin widths."""
        return np.diff(self.edges)

    @property
    def n_bins(self) -> int:
        """Number of bins."""
        return int(self.edges.size - 1)

    @property
    def z_min(self) -> float:
        """Left edge of the first bin."""
        return float(self.edges[0])

    @property
    def z_max(self) -> float:
        """Right edge of the last bin."""
        return float(self.edges[-1])

    def __len__(self) -> int:
        return self.n_bins

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RedshiftGrid):
            return NotImplemented
        return self.edges.shape == other.edges.shape and bool(
            np.allclose(self.edges, other.edges, rtol=0, atol=1e-12)
        )

    def __hash__(self) -> int:
        return hash((self.n_bins, self.z_min, self.z_max))

    def __repr__(self) -> str:
        return f"RedshiftGrid(n_bins={self.n_bins}, z_min={self.z_min:g}, z_max={self.z_max:g})"

    # -- densities ---------------------------------------------------------

    def normalize(self, pdfs: ArrayLike) -> NDArray[np.float64]:
        """Clip to non-negative and rescale each row to unit trapezoid mass.

        Rows that carry no mass at all (all-zero, or all non-finite) become a
        uniform density rather than NaN, so a single degenerate galaxy cannot
        poison an array-wide metric.
        """
        from lazy.metrics import normalize_grid_pdfs

        return normalize_grid_pdfs(self.centers, pdfs)[1]

    def cdf(self, density: ArrayLike) -> NDArray[np.float64]:
        """Cumulative distribution at the bin centres, by trapezoid.

        The returned array has the same shape as ``density``; column ``j`` is
        the mass below ``centers[j]``, so column 0 is exactly zero.
        """
        from lazy.metrics import grid_cdf

        return grid_cdf(self.centers, np.asarray(density, dtype=float))

    def rebin(self, probs: ArrayLike, edges: ArrayLike) -> NDArray[np.float64]:
        """Density on this grid from bin *probabilities* on another set of edges.

        Each output bin is given the exact probability mass the input histogram
        places inside it. The input is piecewise constant, so its CDF is
        piecewise linear and evaluating that CDF at this grid's edges is exact.
        Point-sampling the input density at the output centres instead would
        silently drop input bins narrower than an output bin -- which is the
        normal case for the equal-mass quantile bins a classifier produces in
        the crowded part of N(z).

        Parameters
        ----------
        probs
            ``(n_rows, n_input_bins)`` non-negative masses; each row is
            renormalised to sum to one.
        edges
            ``(n_input_bins + 1,)`` edges the masses are defined on.
        """
        p = np.asarray(probs, dtype=float)
        in_edges = np.asarray(edges, dtype=float)
        if p.ndim != 2:
            raise ValueError("probs must be a 2D (n_rows, n_bins) array")
        if in_edges.ndim != 1 or in_edges.size != p.shape[1] + 1:
            raise ValueError("edges must have one more entry than probs has columns")
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
        widths = np.diff(in_edges)
        out_edges = self.edges
        cum = np.concatenate([np.zeros((len(p), 1)), np.cumsum(p, axis=1)], axis=1)
        idx = np.clip(np.searchsorted(in_edges, out_edges, side="right") - 1, 0, len(widths) - 1)
        frac = np.divide(
            out_edges - in_edges[idx],
            widths[idx],
            out=np.zeros(out_edges.size),
            where=widths[idx] > 0,
        )
        cdf_at_edges = cum[:, idx] + p[:, idx] * np.clip(frac, 0.0, 1.0)
        mass = np.diff(cdf_at_edges, axis=1)
        return np.clip(mass, 0.0, None) / self.widths

    def from_quantiles(self, values: ArrayLike, levels: ArrayLike) -> NDArray[np.float64]:
        """Density on this grid from per-row quantiles of the predictive CDF.

        ``values[i]`` are the redshifts at cumulative probabilities ``levels``.
        Those pairs define a CDF; evaluating it at this grid's edges and
        differencing gives each bin exactly the mass the quantiles place inside
        it. Differencing at the bin *centres* instead (``np.gradient``) would
        smear any feature narrower than a bin into its neighbours and would not
        conserve mass.
        """
        q = np.maximum.accumulate(np.asarray(values, dtype=np.float64), axis=1)
        alphas = np.asarray(levels, dtype=np.float64)
        if q.ndim != 2:
            raise ValueError("values must be a 2D (n_rows, n_quantiles) array")
        if alphas.ndim != 1 or alphas.size != q.shape[1]:
            raise ValueError("levels must have one entry per quantile column")
        cdf = np.empty((q.shape[0], self.edges.size))
        for i in range(q.shape[0]):
            cdf[i] = np.interp(self.edges, q[i], alphas, left=0.0, right=1.0)
        density = np.diff(cdf, axis=1) / self.widths
        return self.normalize(np.clip(density, 0.0, None))

    def bin_index(self, z: ArrayLike) -> NDArray[np.intp]:
        """Index of the bin each redshift falls in, clipped to the grid."""
        z = np.asarray(z, dtype=float)
        return np.clip(np.searchsorted(self.edges, z, side="right") - 1, 0, self.n_bins - 1)


#: The LSST DESC PZ Data Challenge output format: 200 bins of width 0.01 over
#: ``0 < z < 2``. Reproducing the published DC1 numbers requires this grid;
#: nothing else in the library does.
DC1_GRID = RedshiftGrid.linear(0.0, 2.0, 200)


def as_grid(grid: RedshiftGrid | ArrayLike | None) -> RedshiftGrid:
    """Coerce a user-supplied ``z_grid`` argument to a :class:`RedshiftGrid`.

    Accepts a :class:`RedshiftGrid`, an array of bin centres, or ``None`` for
    :data:`DC1_GRID`. This is what every estimator calls on its ``z_grid``
    parameter, so ``z_grid=np.linspace(0, 3, 300)`` works anywhere.

    >>> as_grid(None) is DC1_GRID
    True
    >>> as_grid(np.linspace(0.005, 2.995, 300)).n_bins
    300
    """
    if grid is None:
        return DC1_GRID
    if isinstance(grid, RedshiftGrid):
        return grid
    return RedshiftGrid.from_centers(grid)
