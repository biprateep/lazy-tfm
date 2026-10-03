# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Per-object distributions of the target, in the form each model produces.

A model's native answer is a *distribution*, not a table on some grid. The
bar-distribution models (TabPFN, LimiX) and TabFM's bin hierarchy give
probability masses in buckets; TabICL gives quantiles. Both are exactly a
piecewise-linear cumulative distribution function (CDF), so one small algebra
serves them all without approximation:

* :class:`HistogramDistribution` -- masses in buckets shared by every row,
  uniform within each bucket (qp's ``hist`` parameterisation);
* :class:`QuantileDistribution` -- values at fixed cumulative levels,
  linear in between, the tails beyond the outermost levels sitting on the
  outermost values (qp's ``quant``);
* :class:`MixtureDistribution` -- a weighted mixture of histograms whose
  buckets may differ, as bagged members and TabFM's dithers produce.

The method names are those of :mod:`scipy.stats` and LSST DESC's ``qp``:
``pdf``, ``cdf``, ``ppf``, ``sf``, ``rvs``, ``mean``, ``median``, ``mode``,
``std``, ``var``, ``interval``. ``pit`` gives each row's CDF at its own
value, ``F_i(y_i)``, without the full ``cdf(values)`` table.
:meth:`~HistogramDistribution.on_grid` gives the densities on a
:class:`~lazy.grid.Grid` exactly as ``predict_proba`` returns them, and
``to_qp`` / :func:`from_qp` convert to and from qp ensembles (install
``lazy-tfm[qp]``) for RAIL.

Typical usage example:

  dist = model.predict_distribution(X_test)
  low, median, high = dist.ppf([0.16, 0.5, 0.84]).T
  samples = dist.rvs(100, random_state=0)
  pit = dist.pit(z_test)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from lazy import _typing
from lazy import grid as grid_lib

__all__ = [
    "Distribution",
    "HistogramDistribution",
    "MixtureDistribution",
    "QuantileDistribution",
    "concatenate",
    "from_qp",
]

#: Optional per-object metadata carried alongside a distribution, as in qp:
#: each value has one entry per row.
Ancil: TypeAlias = Mapping[str, npt.NDArray[Any]]


def _freeze(array: npt.ArrayLike) -> _typing.FloatArray:
    """A read-only float64 copy of ``array``."""
    out = np.array(array, dtype=np.float64)
    out.flags.writeable = False
    return out


def _check_ancil(ancil: Ancil | None, n_rows: int) -> dict[str, Any] | None:
    """Validates that every metadata column has one entry per row."""
    if ancil is None:
        return None
    checked = {key: np.asarray(value) for key, value in ancil.items()}
    for key, value in checked.items():
        if len(value) != n_rows:
            raise ValueError(
                f"ancil[{key!r}] has {len(value)} entries for {n_rows} rows"
            )
    return checked


def _slice_ancil(ancil: dict[str, Any] | None, rows: Any) -> Any:
    """The metadata of the selected rows, one entry per selected row.

    A single integer row selects a one-row distribution, so its metadata is
    indexed by a one-element array to keep one entry per row rather than a
    bare scalar.
    """
    if ancil is None:
        return None
    if np.ndim(rows) == 0 and np.issubdtype(np.asarray(rows).dtype, np.integer):
        rows = np.atleast_1d(rows)
    return {key: value[rows] for key, value in ancil.items()}


def _as_row_values(values: npt.ArrayLike, n_rows: int) -> _typing.FloatArray:
    """One value per row as a float array, shape (n_rows,), without NaN."""
    out = np.asarray(values, dtype=np.float64)
    if out.shape != (n_rows,):
        raise ValueError(
            f"values must have shape ({n_rows},), one per row: {out.shape=}"
        )
    if np.isnan(out).any():
        raise ValueError("values must not contain NaN")
    return out


def _as_levels(levels: npt.ArrayLike) -> _typing.FloatArray:
    """Probability levels as a 1-D array inside [0, 1]."""
    out = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    if out.ndim != 1 or np.any((out < 0.0) | (out > 1.0)):
        raise ValueError(f"levels must be a 1-D array within [0, 1]: {levels=}")
    return out


class _Base:
    """Methods every distribution derives from its own ``cdf`` and ``ppf``."""

    ancil: dict[str, Any] | None

    def __len__(self) -> int:
        return self.npdf

    @property
    def npdf(self) -> int:
        """The number of rows (objects), as qp calls it."""
        raise NotImplementedError

    def cdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The CDF at ``values``, shape (n_rows, len(values))."""
        raise NotImplementedError

    def _ppf_rows(self, levels: _typing.FloatArray) -> _typing.FloatArray:
        """Per-row inverse CDF at per-row levels, shape (n_rows, k)."""
        raise NotImplementedError

    def _cdf_rows(self, values: _typing.FloatArray) -> _typing.FloatArray:
        """Each row's CDF at its own value, shape (n_rows,)."""
        raise NotImplementedError

    def pit(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The probability integral transform: each row's CDF at its value.

        Row ``i`` gives ``F_i(values[i])``, exactly, which is the diagonal of
        ``cdf(values)`` without the (n_rows, n_rows) table. With the true
        values, these are the PIT values that :mod:`lazy.metrics` computes
        on a grid, here free of any grid.

        Args:
            values: One value per row, e.g. the true targets, shape
                (n_rows,); infinite values give 0 or 1.

        Returns:
            The cumulative probabilities, within [0, 1], shape (n_rows,).
        """
        return self._cdf_rows(_as_row_values(values, self.npdf))

    def sf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The survival function 1 - CDF, shape (n_rows, len(values))."""
        return 1.0 - self.cdf(values)

    def ppf(self, levels: npt.ArrayLike) -> _typing.FloatArray:
        """The inverse CDF (quantile function), exact.

        Args:
            levels: Cumulative probabilities within [0, 1], shape (k,).

        Returns:
            The values at those levels, shape (n_rows, k).
        """
        levels = _as_levels(levels)
        return self._ppf_rows(np.broadcast_to(levels, (self.npdf, levels.size)))

    def quantiles(self, levels: npt.ArrayLike) -> _typing.FloatArray:
        """Alias of :meth:`ppf`."""
        return self.ppf(levels)

    def median(self) -> _typing.FloatArray:
        """The median of each row, shape (n_rows,)."""
        return self.ppf([0.5])[:, 0]

    def interval(self, confidence: float) -> _typing.FloatArray:
        """The central interval holding ``confidence`` of the probability.

        Args:
            confidence: A probability in (0, 1), e.g. 0.68.

        Returns:
            The lower and upper values, shape (n_rows, 2).
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must lie in (0, 1): {confidence=}")
        tail = 0.5 * (1.0 - confidence)
        return self.ppf([tail, 1.0 - tail])

    def std(self) -> _typing.FloatArray:
        """The standard deviation of each row, shape (n_rows,)."""
        return np.sqrt(np.maximum(self.var(), 0.0))

    def var(self) -> _typing.FloatArray:
        """The variance of each row, shape (n_rows,)."""
        raise NotImplementedError

    def rvs(
        self, size: int, random_state: int | np.random.Generator | None = None
    ) -> _typing.FloatArray:
        """Draws values from each row by inverse-transform sampling.

        Args:
            size: Draws per row.
            random_state: A seed or a NumPy Generator.

        Returns:
            The draws, shape (n_rows, size).
        """
        rng = np.random.default_rng(random_state)
        return self._ppf_rows(rng.uniform(size=(self.npdf, size)))


@dataclasses.dataclass(frozen=True, eq=False)
class HistogramDistribution(_Base):
    """Probability masses in buckets shared by all rows, uniform within each.

    This is a bar distribution (TabPFN, LimiX) or a histogram over bins
    (TabFM). Buckets may be of any widths; a zero-width bucket must carry no
    mass.

    Attributes:
        bins: Bucket edges, shape (n_buckets + 1,), non-decreasing, finite.
        masses: Non-negative weight in each bucket, shape (n_rows,
            n_buckets), stored as given; every method treats each row as
            renormalised to sum to one (an empty row as uniform over the
            buckets). :attr:`probabilities` gives the renormalised masses.
        ancil: Optional per-row metadata, each value of length n_rows.
    """

    bins: _typing.FloatArray
    masses: _typing.FloatArray
    ancil: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        bins = np.asarray(self.bins, dtype=np.float64)
        masses = np.atleast_2d(np.asarray(self.masses, dtype=np.float64))
        if bins.ndim != 1 or bins.size < 2 or not np.isfinite(bins).all():
            raise ValueError("bins must be a finite 1-D array of >= 2 edges")
        if np.any(np.diff(bins) < 0):
            raise ValueError("bins must be non-decreasing")
        if masses.shape[1] != bins.size - 1:
            raise ValueError(
                f"masses have {masses.shape[1]} buckets for {bins.size} edges"
            )
        if np.any(masses < 0) or not np.isfinite(masses).all():
            raise ValueError("masses must be finite and non-negative")
        object.__setattr__(self, "bins", _freeze(bins))
        object.__setattr__(self, "masses", _freeze(masses))
        object.__setattr__(
            self, "ancil", _check_ancil(self.ancil, masses.shape[0])
        )

    @property
    def npdf(self) -> int:
        """The number of rows (objects)."""
        return int(self.masses.shape[0])

    @property
    def widths(self) -> _typing.FloatArray:
        """The bucket widths, shape (n_buckets,)."""
        return np.diff(self.bins)

    @property
    def probabilities(self) -> _typing.FloatArray:
        """The masses renormalised to sum to one per row, (n_rows, n_buckets).

        An empty row is uniform in probability over the buckets.
        """
        total = self.masses.sum(axis=1, keepdims=True)
        empty = total[:, 0] <= 0
        uniform = np.full_like(self.masses, 1.0 / self.masses.shape[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(empty[:, None], uniform, self.masses / total)

    def __getitem__(self, rows: Any) -> HistogramDistribution:
        return HistogramDistribution(
            self.bins,
            np.atleast_2d(self.masses[rows]),
            _slice_ancil(self.ancil, rows),
        )

    def on_grid(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """Densities on ``grid`` as ``predict_proba`` gives them.

        Each bin gets the exact mass the buckets put in it
        (:meth:`lazy.grid.Grid.rebin`), divided by its width; the
        probability outside the grid is dropped and each row is normalized
        by the grid's convention (:meth:`lazy.grid.Grid.normalize`). For
        the masses before that normalization, use
        ``histogramize(grid.edges).masses``.

        Args:
            grid: The grid to put the densities on.

        Returns:
            Normalized densities, shape (n_rows, grid.n_bins).
        """
        return grid.normalize(self._bin_densities(grid))

    def _bin_densities(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """The mass in each bin of ``grid`` over its width, unnormalized.

        From :attr:`probabilities`, so that an empty row is uniform in
        probability over the buckets here as in every other method.
        """
        return grid.rebin(self.probabilities, self.bins)

    def pdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The density at ``values``, shape (n_rows, len(values)).

        The density inside a bucket is its mass over its width; zero outside
        the buckets and in zero-width buckets.
        """
        y = np.atleast_1d(np.asarray(values, dtype=np.float64))
        widths = self.widths
        index = np.searchsorted(self.bins, y, side="right") - 1
        inside = (index >= 0) & (index < widths.size)
        index = np.clip(index, 0, widths.size - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(
                widths[index] > 0,
                self.probabilities[:, index] / widths[index],
                0.0,
            )
        return np.where(inside, density, 0.0)

    def cdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The CDF at ``values``, shape (n_rows, len(values)); exact."""
        y = np.atleast_1d(np.asarray(values, dtype=np.float64))
        probabilities = self.probabilities
        cumulative = _cumulative(probabilities)
        index, fraction = self._locate(y)
        value = cumulative[:, index] + probabilities[:, index] * fraction
        value = np.where(y < self.bins[0], 0.0, value)
        return np.where(y >= self.bins[-1], 1.0, value)

    def _cdf_rows(self, values: _typing.FloatArray) -> _typing.FloatArray:
        probabilities = self.probabilities
        cumulative = _cumulative(probabilities)
        index, fraction = self._locate(values)
        rows = np.arange(self.npdf)
        value = cumulative[rows, index] + probabilities[rows, index] * fraction
        value = np.where(values < self.bins[0], 0.0, value)
        return np.where(values >= self.bins[-1], 1.0, value)

    def _locate(
        self, values: _typing.FloatArray
    ) -> tuple[_typing.IntArray, _typing.FloatArray]:
        """Returns a tuple (bucket, fraction of it below each value).

        Both have the shape of ``values``; the bucket is clipped to the
        buckets, the fraction to [0, 1], and a zero-width bucket counts as
        wholly below.
        """
        widths = self.widths
        index = np.clip(
            np.searchsorted(self.bins, values, side="right") - 1,
            0,
            widths.size - 1,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(
                widths[index] > 0,
                (values - self.bins[index]) / widths[index],
                1.0,
            )
        return index, np.clip(fraction, 0.0, 1.0)

    def _ppf_rows(self, levels: _typing.FloatArray) -> _typing.FloatArray:
        probabilities = self.probabilities
        n_buckets = probabilities.shape[1]
        # The support runs from the first bucket with mass to the last.
        carries = probabilities > 0
        first = np.argmax(carries, axis=1)
        last = n_buckets - 1 - np.argmax(carries[:, ::-1], axis=1)
        cumulative = np.cumsum(probabilities, axis=1)
        # Exactly one from the last bucket with mass on, so that a level
        # near one is not pushed past a sum rounded to just below it.
        cumulative[np.arange(n_buckets)[None, :] >= last[:, None]] = 1.0
        out = np.empty(levels.shape)
        for row in range(self.npdf):
            bucket = np.searchsorted(cumulative[row], levels[row], side="left")
            bucket = np.clip(bucket, 0, n_buckets - 1)
            mass = probabilities[row, bucket]
            below = cumulative[row, bucket] - mass
            with np.errstate(divide="ignore", invalid="ignore"):
                fraction = np.where(mass > 0, (levels[row] - below) / mass, 0.0)
            value = (
                self.bins[bucket]
                + np.clip(fraction, 0.0, 1.0) * (self.widths[bucket])
            )
            # The end levels are the ends of the support exactly, never an
            # edge of an empty bucket beyond it.
            value = np.where(levels[row] <= 0.0, self.bins[first[row]], value)
            out[row] = np.where(
                levels[row] >= 1.0, self.bins[last[row] + 1], value
            )
        return out

    def mean(self) -> _typing.FloatArray:
        """The mean of each row, shape (n_rows,)."""
        centres = 0.5 * (self.bins[:-1] + self.bins[1:])
        return self.probabilities @ centres

    def var(self) -> _typing.FloatArray:
        """The variance of each row, shape (n_rows,); buckets are uniform."""
        low, high = self.bins[:-1], self.bins[1:]
        second = (low**2 + low * high + high**2) / 3.0
        return self.probabilities @ second - self.mean() ** 2

    def mode(self) -> _typing.FloatArray:
        """The centre of each row's densest bucket, shape (n_rows,)."""
        widths = self.widths
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(widths > 0, self.probabilities / widths, -np.inf)
        best = np.argmax(density, axis=1)
        return 0.5 * (self.bins[best] + self.bins[best + 1])

    def histogramize(self, bins: npt.ArrayLike) -> HistogramDistribution:
        """The same distribution's masses over other bucket edges, exactly.

        Probability outside ``bins`` is dropped and each row renormalised.
        """
        target = np.asarray(bins, dtype=np.float64)
        cdf = self.cdf(target)
        return HistogramDistribution(target, np.diff(cdf, axis=1), self.ancil)

    def to_qp(self) -> Any:
        """This distribution as a ``qp.Ensemble`` (``hist``); needs qp.

        Zero-width buckets are merged away, since qp requires strictly
        increasing bins.
        """
        qp = _import_qp()
        keep = np.concatenate([[True], np.diff(self.bins) > 0])
        bins = self.bins[keep]
        cumulative = np.concatenate(
            [np.zeros((self.npdf, 1)), np.cumsum(self.probabilities, axis=1)],
            axis=1,
        )[:, keep]
        pdfs = np.diff(cumulative, axis=1) / np.diff(bins)
        ensemble = qp.Ensemble(qp.hist, data={"bins": bins, "pdfs": pdfs})
        if self.ancil is not None:
            ensemble.set_ancil(dict(self.ancil))
        return ensemble


@dataclasses.dataclass(frozen=True, eq=False)
class QuantileDistribution(_Base):
    """Values at fixed cumulative levels, linear in between.

    The CDF passes through each (``locs[i, j]``, ``quants[j]``) and is linear
    between consecutive knots, so the density is constant there. Beyond the
    outermost levels the remaining probability sits on the outermost values:
    ``quants[0]`` at ``locs[:, 0]`` and ``1 - quants[-1]`` at ``locs[:, -1]``.
    This is the convention :meth:`lazy.grid.Grid.from_quantiles` has
    always used, so densities on a grid are unchanged.

    Attributes:
        quants: The cumulative levels, shape (n_levels,), strictly
            increasing inside [0, 1].
        locs: The value at each level, shape (n_rows, n_levels), made
            non-decreasing on construction (crossing quantiles are sorted).
        ancil: Optional per-row metadata, each value of length n_rows.
    """

    quants: _typing.FloatArray
    locs: _typing.FloatArray
    ancil: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        quants = np.asarray(self.quants, dtype=np.float64)
        locs = np.atleast_2d(np.asarray(self.locs, dtype=np.float64))
        if quants.ndim != 1 or quants.size < 2:
            raise ValueError("quants must be a 1-D array of >= 2 levels")
        if np.any(np.diff(quants) <= 0) or quants[0] < 0 or quants[-1] > 1:
            raise ValueError("quants must increase strictly inside [0, 1]")
        if locs.shape[1] != quants.size:
            raise ValueError(
                f"locs have {locs.shape[1]} columns for {quants.size} levels"
            )
        if not np.isfinite(locs).all():
            raise ValueError("locs must be finite")
        object.__setattr__(self, "quants", _freeze(quants))
        object.__setattr__(
            self, "locs", _freeze(np.maximum.accumulate(locs, axis=1))
        )
        object.__setattr__(self, "ancil", _check_ancil(self.ancil, len(locs)))

    @property
    def npdf(self) -> int:
        """The number of rows (objects)."""
        return int(self.locs.shape[0])

    def __getitem__(self, rows: Any) -> QuantileDistribution:
        return QuantileDistribution(
            self.quants,
            np.atleast_2d(self.locs[rows]),
            _slice_ancil(self.ancil, rows),
        )

    @classmethod
    def average(
        cls,
        parts: Sequence[QuantileDistribution],
        weights: npt.ArrayLike | None = None,
    ) -> QuantileDistribution:
        """The weighted average of quantile functions at shared levels.

        This is how TabICL combines its own ensemble members, so bagged
        TabICL members are combined the same way.
        """
        quants = parts[0].quants
        if any(not np.array_equal(p.quants, quants) for p in parts):
            raise ValueError("every part must share the same levels")
        if weights is None:
            weight = np.full(len(parts), 1.0 / len(parts))
        else:
            weight = np.asarray(weights, dtype=np.float64)
            weight = weight / weight.sum()
        locs = np.tensordot(weight, np.stack([p.locs for p in parts]), axes=1)
        return cls(quants, locs, parts[0].ancil)

    def on_grid(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """Densities on ``grid`` as ``predict_proba`` gives them.

        Exactly :meth:`lazy.grid.Grid.from_quantiles`: the mass in each bin
        over its width, with the probability outside the grid dropped and
        each row normalized by the grid's convention.

        Args:
            grid: The grid to put the densities on.

        Returns:
            Normalized densities, shape (n_rows, grid.n_bins).
        """
        return grid.from_quantiles(self.locs, self.quants)

    def cdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The CDF at ``values``, shape (n_rows, len(values)); exact."""
        y = np.atleast_1d(np.asarray(values, dtype=np.float64))
        out = np.empty((self.npdf, y.size))
        for row in range(self.npdf):
            out[row] = np.interp(
                y, self.locs[row], self.quants, left=0.0, right=1.0
            )
        return out

    def _cdf_rows(self, values: _typing.FloatArray) -> _typing.FloatArray:
        out = np.empty(self.npdf)
        for row in range(self.npdf):
            out[row] = np.interp(
                values[row], self.locs[row], self.quants, left=0.0, right=1.0
            )
        return out

    def pdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The density at ``values``, shape (n_rows, len(values)).

        Constant between knots; zero outside the outermost values (where the
        tails sit as point masses) and on zero-width segments.
        """
        y = np.atleast_1d(np.asarray(values, dtype=np.float64))
        step = np.diff(self.quants)
        out = np.zeros((self.npdf, y.size))
        for row in range(self.npdf):
            locs = self.locs[row]
            segment = np.searchsorted(locs, y, side="right") - 1
            inside = (segment >= 0) & (segment < step.size)
            segment = np.clip(segment, 0, step.size - 1)
            width = locs[segment + 1] - locs[segment]
            with np.errstate(divide="ignore", invalid="ignore"):
                density = np.where(width > 0, step[segment] / width, 0.0)
            out[row] = np.where(inside, density, 0.0)
        return out

    def _ppf_rows(self, levels: _typing.FloatArray) -> _typing.FloatArray:
        out = np.empty(levels.shape)
        for row in range(self.npdf):
            out[row] = np.interp(levels[row], self.quants, self.locs[row])
        return out

    def _segment_moments(self) -> tuple[_typing.FloatArray, _typing.FloatArray]:
        """Returns a tuple (mean, second raw moment) per row."""
        low, high = self.locs[:, :-1], self.locs[:, 1:]
        step = np.diff(self.quants)
        head, tail = self.quants[0], 1.0 - self.quants[-1]
        first = (
            head * self.locs[:, 0]
            + (step * 0.5 * (low + high)).sum(axis=1)
            + tail * self.locs[:, -1]
        )
        second = (
            head * self.locs[:, 0] ** 2
            + (step * (low**2 + low * high + high**2) / 3.0).sum(axis=1)
            + tail * self.locs[:, -1] ** 2
        )
        return first, second

    def mean(self) -> _typing.FloatArray:
        """The mean of each row, shape (n_rows,)."""
        return self._segment_moments()[0]

    def var(self) -> _typing.FloatArray:
        """The variance of each row, shape (n_rows,)."""
        first, second = self._segment_moments()
        return second - first**2

    def mode(self) -> _typing.FloatArray:
        """The centre of each row's densest segment, shape (n_rows,)."""
        width = np.diff(self.locs, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(width > 0, np.diff(self.quants) / width, -np.inf)
        best = np.argmax(density, axis=1)
        rows = np.arange(self.npdf)
        return 0.5 * (self.locs[rows, best] + self.locs[rows, best + 1])

    def histogramize(self, bins: npt.ArrayLike) -> HistogramDistribution:
        """The masses this distribution puts in ``bins``, exactly."""
        target = np.asarray(bins, dtype=np.float64)
        cdf = self.cdf(target)
        return HistogramDistribution(target, np.diff(cdf, axis=1), self.ancil)

    def to_qp(self) -> Any:
        """This distribution as a ``qp.Ensemble`` (``quant``); needs qp.

        The two agree exactly at the quantile levels. Between them qp
        interpolates its quantile function its own way, and it extends the
        outermost quantiles to its own support, so its values between the
        levels and in the tails differ slightly from the piecewise-linear CDF
        used here. For an exact copy, export ``histogramize(bins).to_qp()``.
        """
        qp = _import_qp()
        ensemble = qp.Ensemble(
            qp.quant, data={"quants": self.quants, "locs": self.locs}
        )
        if self.ancil is not None:
            ensemble.set_ancil(dict(self.ancil))
        return ensemble


@dataclasses.dataclass(frozen=True, eq=False)
class MixtureDistribution(_Base):
    """A weighted mixture of histogram distributions with their own buckets.

    Bagged ensemble members and TabFM's dithered hierarchies each put their
    masses in different buckets, which cannot be added bucket by bucket; the
    mixture keeps them apart, so its densities on any grid are the weighted
    average of the components' (what averaging on the output grid always
    meant), and its CDF is exact on the union of all the components' edges.

    Attributes:
        components: The mixed distributions, all with the same rows.
        weights: Their weights, shape (n_components,), summing to one.
        ancil: Optional per-row metadata, each value of length n_rows.
    """

    components: tuple[HistogramDistribution, ...]
    weights: _typing.FloatArray
    ancil: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise ValueError("a mixture needs at least one component")
        if any(not isinstance(c, HistogramDistribution) for c in components):
            raise TypeError("mixture components must be HistogramDistribution")
        if len({c.npdf for c in components}) != 1:
            raise ValueError("every component must have the same rows")
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.shape != (len(components),) or np.any(weights < 0):
            raise ValueError("weights must be non-negative, one per component")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "weights", _freeze(weights / weights.sum()))
        object.__setattr__(
            self, "ancil", _check_ancil(self.ancil, components[0].npdf)
        )

    @classmethod
    def equal(
        cls,
        components: Sequence[HistogramDistribution],
        ancil: Ancil | None = None,
    ) -> MixtureDistribution:
        """An equal-weight mixture of ``components``."""
        return cls(
            tuple(components),
            np.full(len(components), 1.0 / len(components)),
            None if ancil is None else dict(ancil),
        )

    @property
    def npdf(self) -> int:
        """The number of rows (objects)."""
        return self.components[0].npdf

    @property
    def bins(self) -> _typing.FloatArray:
        """The union of the components' edges, shape (n_edges,)."""
        return np.unique(np.concatenate([c.bins for c in self.components]))

    def __getitem__(self, rows: Any) -> MixtureDistribution:
        return MixtureDistribution(
            tuple(c[rows] for c in self.components),
            self.weights,
            _slice_ancil(self.ancil, rows),
        )

    def to_histogram(self) -> HistogramDistribution:
        """The mixture as one histogram on the union of edges; exact."""
        bins = self.bins
        cdf = self.cdf(bins)
        return HistogramDistribution(bins, np.diff(cdf, axis=1), self.ancil)

    def _weighted(self, values: Sequence[_typing.FloatArray]) -> Any:
        """The weighted sum of one array per component."""
        total = self.weights[0] * values[0]
        for weight, value in zip(self.weights[1:], values[1:], strict=True):
            total = total + weight * value
        return total

    def on_grid(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """Densities on ``grid`` as ``predict_proba`` gives them.

        The weighted average of the mass each component puts in each bin,
        over the bin's width, with the probability outside the grid dropped
        and each row then normalized by the grid's convention. Normalizing
        after averaging, not before, is the mixture itself restricted to the
        grid, even when its components put different mass outside it.

        Args:
            grid: The grid to put the densities on.

        Returns:
            Normalized densities, shape (n_rows, grid.n_bins).
        """
        total = np.zeros((self.npdf, grid.n_bins))
        for weight, component in zip(
            self.weights, self.components, strict=True
        ):
            total += weight * component._bin_densities(grid)  # noqa: SLF001 - same module.
        return grid.normalize(total)

    def pdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The density at ``values``, shape (n_rows, len(values))."""
        return self._weighted([c.pdf(values) for c in self.components])

    def cdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The CDF at ``values``, shape (n_rows, len(values)); exact."""
        return self._weighted([c.cdf(values) for c in self.components])

    def _cdf_rows(self, values: _typing.FloatArray) -> _typing.FloatArray:
        return self._weighted(
            [c._cdf_rows(values) for c in self.components]  # noqa: SLF001 - same module.
        )

    def _ppf_rows(self, levels: _typing.FloatArray) -> _typing.FloatArray:
        # The union histogram is the same distribution, so its inverse is
        # this one's.
        return self.to_histogram()._ppf_rows(levels)  # noqa: SLF001 - same module.

    def mean(self) -> _typing.FloatArray:
        """The mean of each row, shape (n_rows,)."""
        return self._weighted([c.mean() for c in self.components])

    def var(self) -> _typing.FloatArray:
        """The variance of each row, shape (n_rows,)."""
        second = self._weighted(
            [c.var() + c.mean() ** 2 for c in self.components]
        )
        return second - self.mean() ** 2

    def mode(self) -> _typing.FloatArray:
        """The centre of each row's densest union bucket, shape (n_rows,)."""
        return self.to_histogram().mode()

    def histogramize(self, bins: npt.ArrayLike) -> HistogramDistribution:
        """The masses the mixture puts in ``bins``, exactly."""
        target = np.asarray(bins, dtype=np.float64)
        cdf = self.cdf(target)
        return HistogramDistribution(target, np.diff(cdf, axis=1), self.ancil)

    def to_qp(self) -> Any:
        """The mixture as a ``qp.Ensemble`` (``hist`` on the union edges)."""
        return self.to_histogram().to_qp()


def _cumulative(probabilities: _typing.FloatArray) -> _typing.FloatArray:
    """The mass below each bucket edge, shape (n_rows, n_buckets + 1)."""
    return np.concatenate(
        [np.zeros((len(probabilities), 1)), np.cumsum(probabilities, axis=1)],
        axis=1,
    )


#: Any per-object distribution this module defines.
Distribution: TypeAlias = (
    HistogramDistribution | QuantileDistribution | MixtureDistribution
)


def concatenate(parts: Sequence[Distribution]) -> Distribution:
    """Stacks the rows of distributions of the same kind and parameters.

    Args:
        parts: Distributions from consecutive chunks of rows: histograms
            with the same bins, quantiles at the same levels, or mixtures
            with the same weights and aligned components.

    Returns:
        One distribution holding every row, in order.
    """
    first = parts[0]
    ancil = _concatenate_ancil([p.ancil for p in parts])
    if isinstance(first, HistogramDistribution):
        pieces = [_same_kind(p, HistogramDistribution) for p in parts]
        if any(not np.array_equal(p.bins, first.bins) for p in pieces):
            raise ValueError("histograms must share their bins")
        masses = np.concatenate([p.masses for p in pieces])
        return HistogramDistribution(first.bins, masses, ancil)
    if isinstance(first, QuantileDistribution):
        pieces_q = [_same_kind(p, QuantileDistribution) for p in parts]
        if any(not np.array_equal(p.quants, first.quants) for p in pieces_q):
            raise ValueError("quantile distributions must share their levels")
        locs = np.concatenate([p.locs for p in pieces_q])
        return QuantileDistribution(first.quants, locs, ancil)
    pieces_m = [_same_kind(p, MixtureDistribution) for p in parts]
    components = tuple(
        concatenate([p.components[k] for p in pieces_m])
        for k in range(len(first.components))
    )
    return MixtureDistribution(
        components,  # type: ignore[arg-type]  # histograms stay histograms
        first.weights,
        ancil,
    )


def _same_kind(part: Distribution, kind: type[Any]) -> Any:
    """``part``, checked to be of the expected kind."""
    if not isinstance(part, kind):
        raise TypeError(
            f"cannot concatenate {type(part).__name__} with {kind.__name__}"
        )
    return part


def _concatenate_ancil(
    ancils: Sequence[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """The row-wise concatenation of every part's metadata, if all have it."""
    if any(a is None for a in ancils):
        return None
    keys = ancils[0].keys()  # type: ignore[union-attr]  # checked above
    return {k: np.concatenate([a[k] for a in ancils]) for k in keys}  # type: ignore[index]  # checked above


def from_qp(ensemble: Any) -> HistogramDistribution | QuantileDistribution:
    """Converts a ``qp.Ensemble`` of ``hist`` or ``quant`` PDFs; needs qp.

    Args:
        ensemble: A qp ensemble whose parameterisation is ``hist`` or
            ``quant``; convert others first with ``ensemble.convert_to``.

    Returns:
        The equivalent distribution, with the ensemble's ancil, if any.

    Raises:
        TypeError: If the ensemble is of another parameterisation.
    """
    qp = _import_qp()
    data = ensemble.objdata
    ancil = None if ensemble.ancil is None else dict(ensemble.ancil)
    if isinstance(ensemble.gen_obj, qp.hist_gen):
        bins = np.asarray(ensemble.metadata["bins"], dtype=np.float64).ravel()
        pdfs = np.asarray(data["pdfs"], dtype=np.float64)
        return HistogramDistribution(bins, pdfs * np.diff(bins), ancil)
    if isinstance(ensemble.gen_obj, qp.quant_gen):
        quants = np.asarray(ensemble.metadata["quants"], dtype=np.float64)
        locs = np.asarray(data["locs"], dtype=np.float64)
        return QuantileDistribution(quants.ravel(), locs, ancil)
    raise TypeError(
        "from_qp handles hist and quant ensembles; convert with "
        f"ensemble.convert_to(qp.hist, ...) first, not {type(ensemble.gen_obj)}"
    )


def _import_qp() -> Any:
    """Imports qp, naming the extra to install when it is missing."""
    try:
        import qp  # noqa: PLC0415 - optional extra, imported on use.
    except ImportError as error:
        raise ImportError(
            "qp interoperability needs qp: pip install 'lazy-tfm[qp]'"
        ) from error
    return qp
