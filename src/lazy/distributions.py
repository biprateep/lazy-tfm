# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Per-object distributions of the target, in the form each model produces.

A model's native answer is a *distribution*, not a table on some grid. The
bar-distribution models (TabPFN, LimiX) and TabFM's bin hierarchy give
probability masses in buckets; TabICL gives quantiles. Both are exactly a
piecewise-linear cumulative distribution function (CDF), so one small algebra
serves them all without approximation:

* :class:`HistogramDistribution` -- masses in buckets shared by every row,
  uniform within each bucket (qp's ``hist`` parameterization);
* :class:`QuantileDistribution` -- values at fixed cumulative levels,
  linear in between, each tail beyond the outermost levels spread evenly
  over one quantile spacing (qp's ``quant``);
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

The private functions that put a distribution on a grid, and compute its
mean, take numpy arrays or torch tensors (through ``array_api_compat``), so
that :mod:`lazy.models._differentiable` runs the same code on torch's
autograd graph. Gradients pass through the arithmetic on the masses and on
the quantile values. Every search for an index (a bucket, a knot, a tie)
runs in numpy on detached values, carries no gradient, and gathers from the
arrays with the indices it finds. The point masses of tied quantile values
are steps between fixed levels, so they carry none either.

Typical usage example:

  dist = model.predict_distribution(X_test)
  low, median, high = dist.ppf([0.16, 0.5, 0.84]).T
  samples = dist.rvs(100, random_state=0)
  pit = dist.pit(y_test)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import dataclasses
import operator
from typing import Any, TypeAlias

import array_api_compat
import numpy as np
import numpy.typing as npt

from lazy import _arrays
from lazy import _inputs
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

# How much denser than every segment between the quantiles a quantile
# distribution's tail must be to be its mode, relatively.
_MODE_RTOL = 1e-9

# A mixture's quantiles, mode and ``to_histogram`` come from its histogram
# on the union of the components' edges, which holds members x buckets
# values per row: a bagged model's 40,008 for eight members of 5,000
# buckets. The rows go a block at a time, so that a block's histogram holds
# about this many bytes; its components' CDFs take a few times that while
# it is built.
_MIXTURE_BLOCK_BYTES = 32 * 2**20

Ancil: TypeAlias = Mapping[str, npt.NDArray[Any]]
"""Optional per-object metadata carried alongside a distribution, as in qp: each
value has one entry per row.
"""


def _freeze(array: npt.ArrayLike) -> _typing.FloatArray:
    """A read-only float64 copy of ``array``."""
    out = np.array(array, dtype=np.float64)
    out.flags.writeable = False
    return out


def _normalized(weights: npt.ArrayLike, n: int) -> _typing.FloatArray:
    """``weights`` over their sum: n finite, non-negative values, sum > 0."""
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.shape != (n,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not 0 < weights.sum() < np.inf
    ):
        raise ValueError(
            f"weights must be {n} finite, non-negative values with a positive "
            f"sum: {weights=}"
        )
    return weights / weights.sum()


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


def _unit(values: Any) -> Any:
    """Cumulative probabilities clipped to [0, 1], numpy's or torch's.

    A cumulative sum of masses that add to one can round to just above one
    (or a weighted sum of CDFs to just outside [0, 1]); clipping keeps every
    CDF a probability and non-decreasing, so differences between its values
    are never negative masses.
    """
    return _arrays.clip(values, 0.0, 1.0)


def _closed_bucket(
    edges: _typing.FloatArray, values: _typing.FloatArray
) -> _typing.IntArray:
    """The bucket holding each value, ``edges[i] <= value < edges[i + 1]``.

    As in ``np.histogram``, the last edge closes the last bucket of non-zero
    width, so a value on it is inside. Values outside the edges give -1 or
    ``edges.size - 1``.
    """
    index = np.searchsorted(edges, values, side="right") - 1
    last = np.searchsorted(edges, edges[-1], side="left") - 1
    return np.where(values == edges[-1], last, index)


def _tail_gaps(
    gaps: _typing.FloatArray,
) -> tuple[_typing.IntArray, _typing.IntArray]:
    """The spacing each tail of a quantile row borrows as its width.

    The outermost spacing or, where the outermost values are tied, the
    nearest non-zero one; for a row of equal values, whose spacings are all
    zero, the first and the last.

    Args:
        gaps: The spacings of each row's values, shape (n_rows, n_gaps).

    Returns:
        A tuple (first, last) of spacing indices, each of shape (n_rows,).
    """
    wide = gaps > 0
    first = np.argmax(wide, axis=1)
    last = gaps.shape[1] - 1 - np.argmax(wide[:, ::-1], axis=1)
    return first, last


def _quantile_pieces(
    quants: _typing.FloatArray,
    locs: _typing.FloatArray,
    values: _typing.FloatArray,
) -> tuple[_typing.FloatArray, _typing.IntArray]:
    """Which piece of each quantile row's density holds each value.

    The pieces of :meth:`QuantileDistribution.pdf`: the segments between
    adjacent values, the outermost values belonging to the segments they
    end, and one tail on each side that carries probability, reaching one
    spacing (:func:`_tail_gaps`) beyond the outermost value, its far end
    included. A piece's density is its probability over its width.

    Args:
        quants: The levels, increasing, shape (n_levels,).
        locs: Each row's values, non-decreasing, shape (n_rows, n_levels).
        values: Where to look, one row of values per row, shape (n_rows,
            n_values).

    Returns:
        A tuple (mass, gap): the probability of the piece holding each
        value, 0 beyond the tails; and the index ``k`` of the spacing
        ``locs[:, k + 1] - locs[:, k]`` that is the piece's width; both of
        the shape of ``values``.
    """
    steps = np.diff(quants)
    gaps = np.diff(locs, axis=1)
    rows = np.arange(len(locs))
    segment = np.array(
        [
            _closed_bucket(row, row_values)
            for row, row_values in zip(locs, values)
        ],
        dtype=np.intp,
    ).reshape(values.shape)
    mass = np.where((segment >= 0) & (segment < steps.size), 1.0, 0.0)
    gap = np.clip(segment, 0, steps.size - 1)
    mass = mass * steps[gap]
    first, last = _tail_gaps(gaps)
    below = (gaps[rows, first] * (quants[0] > 0))[:, None]
    above = (gaps[rows, last] * (quants[-1] < 1))[:, None]
    lowest, highest = locs[:, :1], locs[:, -1:]
    in_below = (below > 0) & (values >= lowest - below) & (values < lowest)
    in_above = (above > 0) & (values > highest) & (values <= highest + above)
    mass = np.where(
        in_below, quants[0], np.where(in_above, 1 - quants[-1], mass)
    )
    gap = np.where(
        in_below, first[:, None], np.where(in_above, last[:, None], gap)
    )
    return mass, gap


def _tail_widths(quants: _typing.FloatArray, locs: Any) -> tuple[Any, Any]:
    """How far each quantile row's tails reach: a tuple (below, above).

    One quantile spacing (:func:`_tail_gaps`); zero for a row whose values
    are all equal and for a tail that carries no probability (a level of 0
    or 1).

    Args:
        quants: The levels, increasing, shape (n_levels,).
        locs: Each row's values, non-decreasing, shape (n_rows, n_levels), a
            numpy array or a torch tensor.

    Returns:
        A tuple (below, above), each of ``locs``'s kind and shape (n_rows,).
    """
    gaps = locs[:, 1:] - locs[:, :-1]
    first, last = _tail_gaps(_arrays.detached(gaps))
    rows = np.arange(len(first))
    return (
        gaps[rows, first] * float(quants[0] > 0),
        gaps[rows, last] * float(quants[-1] < 1),
    )


def _knots(quants: _typing.FloatArray, locs: Any) -> tuple[Any, Any]:
    """Returns a tuple (levels, locs): a quantile CDF's knots, tails included.

    The levels gain 0 below and 1 above unless they already reach them,
    and each row's values gain the ends of its tails.

    Args:
        quants: The levels, increasing, shape (n_levels,).
        locs: Each row's values, non-decreasing, shape (n_rows, n_levels), a
            numpy array or a torch tensor.

    Returns:
        The levels, a numpy array of shape (n_knots,), and the values, of
        ``locs``'s kind and shape (n_rows, n_knots).
    """
    xp = array_api_compat.array_namespace(locs)
    levels = quants
    below, above = _tail_widths(quants, locs)
    if levels[0] > 0:
        levels = np.concatenate([[0.0], levels])
        locs = xp.concat([(locs[:, 0] - below)[:, None], locs], axis=1)
    if levels[-1] < 1:
        levels = np.concatenate([levels, [1.0]])
        locs = xp.concat([locs, (locs[:, -1] + above)[:, None]], axis=1)
    return levels, locs


def _quantile_cdf(
    quants: _typing.FloatArray, locs: Any, values: _typing.FloatArray
) -> Any:
    """Each quantile row's CDF at its own values; exact.

    ``np.interp`` between the knots of :func:`_knots`; on a torch tensor,
    np.interp's own formula, which keeps the graph.

    Args:
        quants: The levels, increasing, shape (n_levels,).
        locs: Each row's values, non-decreasing, shape (n_rows, n_levels), a
            numpy array or a torch tensor.
        values: Where to evaluate each row, shape (n_rows, n_values).

    Returns:
        The cumulative probabilities, within [0, 1], of ``locs``'s kind and
        shape (n_rows, n_values).
    """
    xp = array_api_compat.array_namespace(locs)
    levels, knots = _knots(quants, locs)
    if array_api_compat.is_numpy_array(knots):
        # The same values, several times faster: np.interp's C loop finds
        # each knot from the one before.
        out = np.empty(values.shape)
        for row, row_values in enumerate(values):
            out[row] = np.interp(
                row_values, knots[row], levels, left=0.0, right=1.0
            )
        return _unit(out)
    fixed = _arrays.detached(knots)
    index = (
        np.array(
            [
                np.searchsorted(row, row_values, side="right")
                for row, row_values in zip(fixed, values, strict=True)
            ],
            dtype=np.intp,
        ).reshape(values.shape)
        - 1
    )
    last = levels.size - 1
    start = np.clip(index, 0, last - 1)
    rows = np.arange(len(fixed))[:, None]
    # As np.interp: 0 below the knots, 1 from the last on, and a segment of
    # no width at its level. On a knot, the formula gives the knot's own
    # level exactly, and keeps the gradient of the segment above it.
    exact = (
        (index < 0)
        | (index >= last)
        | (fixed[rows, start + 1] == fixed[rows, start])
    )
    level = np.where(
        index < 0, 0.0, np.where(index >= last, 1.0, levels[start])
    )
    low, high = knots[rows, start], knots[rows, start + 1]
    slope = xp.asarray(np.diff(levels)[start]) / xp.where(
        xp.asarray(exact), 1.0, high - low
    )
    # A copy: torch cannot wrap a read-only (broadcast) numpy array.
    value = slope * (xp.asarray(values.copy()) - low) + xp.asarray(
        levels[start]
    )
    return _unit(xp.where(xp.asarray(exact), xp.asarray(level), value))


def _quantile_atoms(
    quants: _typing.FloatArray,
    locs: _typing.FloatArray,
    values: _typing.FloatArray,
) -> _typing.FloatArray | None:
    """Each quantile row's point mass at ``values``, or None for none.

    A tail has no width only in a row of equal values, so every point mass
    is a tie between values; most rows have none to look for.

    Args:
        quants: The levels, increasing, shape (n_levels,).
        locs: Each row's values, non-decreasing, shape (n_rows, n_levels).
        values: Where to look, shape (n_values,).

    Returns:
        The point masses, shape (n_rows, n_values), or None.
    """
    if not (locs[:, 1:] == locs[:, :-1]).any():
        return None
    levels, knots = _knots(quants, locs)
    steps = np.diff(levels)
    tied = (np.diff(knots, axis=1) == 0) & (steps > 0)
    if not tied.any():
        return None
    out = np.zeros((len(locs), values.size))
    for row, knot in zip(*np.nonzero(tied), strict=True):
        out[row] += steps[knot] * (values == knots[row, knot])
    return out


def _quantile_bin_densities(
    quants: _typing.FloatArray, locs: Any, edges: _typing.FloatArray
) -> Any:
    """The mass of each quantile row in each bin over its width.

    Args:
        quants: The levels, increasing, shape (n_levels,).
        locs: Each row's values, non-decreasing, shape (n_rows, n_levels), a
            numpy array or a torch tensor.
        edges: The bin edges, strictly increasing, shape (n_bins + 1,).

    Returns:
        The unnormalized densities, of ``locs``'s kind and shape (n_rows,
        n_bins).
    """
    xp = array_api_compat.array_namespace(locs)
    cdf = _quantile_cdf(
        quants, locs, np.broadcast_to(edges, (len(locs), edges.size))
    )
    atoms = _quantile_atoms(quants, _arrays.detached(locs), edges[:-1])
    if atoms is not None:
        atoms = xp.asarray(atoms)
    masses = _masses_from(cdf, atoms)[0]
    return _arrays.update(masses, operator.truediv, xp.asarray(np.diff(edges)))


def _quantile_mean(quants: _typing.FloatArray, locs: Any) -> Any:
    """The mean of each quantile row, of ``locs``'s kind, shape (n_rows,)."""
    xp = array_api_compat.array_namespace(locs)
    levels, knots = _knots(quants, locs)
    step = np.diff(levels)
    return xp.sum(
        xp.asarray(step * 0.5) * (knots[:, :-1] + knots[:, 1:]), axis=1
    )


def _probabilities(masses: Any, widths: _typing.FloatArray) -> Any:
    """Histogram rows' masses renormalized to sum to one per row.

    An empty row is uniform in probability over the buckets of positive
    width, so that a zero-width bucket carries none; over every bucket if
    none has a width.

    Args:
        masses: Non-negative masses, shape (n_rows, n_buckets), a numpy
            array or a torch tensor.
        widths: The bucket widths, shape (n_buckets,).

    Returns:
        A new array of ``masses``'s kind and shape.
    """
    xp = array_api_compat.array_namespace(masses)
    total = xp.sum(masses, axis=1, keepdims=True)
    full = total > 0
    probabilities = masses / xp.where(full, total, 1.0)
    if bool(xp.all(full)):
        return probabilities
    wide = widths > 0
    weights = wide if wide.any() else np.ones_like(wide)
    return _arrays.put(probabilities, ~full, weights / weights.sum())


def _locate(
    bins: _typing.FloatArray, values: _typing.FloatArray
) -> tuple[_typing.IntArray, _typing.FloatArray]:
    """Returns a tuple (bucket, fraction of it below each value).

    Both have the shape of ``values``; the bucket is clipped to the
    buckets, the fraction to [0, 1], and a zero-width bucket counts as
    wholly below.
    """
    widths = np.diff(bins)
    index = np.clip(
        np.searchsorted(bins, values, side="right") - 1, 0, widths.size - 1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(
            widths[index] > 0, (values - bins[index]) / widths[index], 1.0
        )
    return index, np.clip(fraction, 0.0, 1.0, out=fraction)


def _histogram_cdf(
    bins: _typing.FloatArray, probabilities: Any, values: _typing.FloatArray
) -> Any:
    """Histogram rows' CDF at ``values``; exact.

    Args:
        bins: The bucket edges, non-decreasing, shape (n_buckets + 1,).
        probabilities: Each row's renormalized masses, shape (n_rows,
            n_buckets), a numpy array or a torch tensor.
        values: Where to evaluate, shape (n_values,).

    Returns:
        The cumulative probabilities, of ``probabilities``'s kind and shape
        (n_rows, n_values).
    """
    xp = array_api_compat.array_namespace(probabilities)
    index, fraction = _locate(bins, values)
    below = _mass_below(probabilities, slice(None), index)
    value = _arrays.update(
        probabilities[:, index], operator.mul, xp.asarray(fraction)
    )
    value = _arrays.update(value, operator.add, below)
    value = _arrays.put(value, values < bins[0], 0.0)
    return _unit(_arrays.put(value, values >= bins[-1], 1.0))


def _mass_below(probabilities: Any, rows: Any, index: Any) -> Any:
    """Histogram rows' mass in the buckets below ``index``.

    Gathered from the cumulative sum at ``index - 1`` and zeroed at bucket
    0, without the leading zero column that ``include_initial`` would copy
    the whole cumulative sum to add.

    Args:
        probabilities: Each row's renormalized masses, shape (n_rows,
            n_buckets), a numpy array or a torch tensor.
        rows: The rows to gather from, paired with ``index``, or
            ``slice(None)`` for every row at every index.
        index: The buckets, a numpy integer array.

    Returns:
        The masses, of ``probabilities``'s kind, shaped as the gather.
    """
    xp = array_api_compat.array_namespace(probabilities)
    below = xp.cumulative_sum(probabilities, axis=1)[rows, index - 1]
    return _arrays.put(below, index == 0, 0.0)


def _histogram_atoms(
    bins: _typing.FloatArray, probabilities: Any, values: _typing.FloatArray
) -> Any:
    """Histogram rows' point masses at ``values``, or None for none.

    Args:
        bins: The bucket edges, non-decreasing, shape (n_buckets + 1,).
        probabilities: Each row's renormalized masses, shape (n_rows,
            n_buckets), a numpy array or a torch tensor.
        values: Where to look, shape (n_values,).

    Returns:
        The mass of the zero-width buckets on each value, of
        ``probabilities``'s kind and shape (n_rows, n_values), or None.
    """
    zero = np.diff(bins) == 0
    if not zero.any():
        return None
    xp = array_api_compat.array_namespace(probabilities)
    at = values[:, None] == bins[:-1][zero]
    return probabilities[:, zero] @ xp.asarray(at.T, dtype=xp.float64)


def _histogram_bin_densities(
    bins: _typing.FloatArray, probabilities: Any, edges: _typing.FloatArray
) -> Any:
    """The mass of each histogram row in each bin over its width.

    On the buckets' own edges, each bin is one bucket: its mass over its
    width, without the cumulative sums of the CDF, which lose a small bin's
    precision.

    Args:
        bins: The bucket edges, non-decreasing, shape (n_buckets + 1,).
        probabilities: Each row's renormalized masses, shape (n_rows,
            n_buckets), a numpy array, overwritten on the buckets' own
            edges, or a torch tensor.
        edges: The bin edges, strictly increasing, shape (n_bins + 1,).

    Returns:
        The unnormalized densities, of ``probabilities``'s kind and shape
        (n_rows, n_bins).
    """
    xp = array_api_compat.array_namespace(probabilities)
    if np.array_equal(edges, bins):
        masses = probabilities
    else:
        cdf = _histogram_cdf(bins, probabilities, edges)
        atoms = _histogram_atoms(bins, probabilities, edges[:-1])
        masses = _masses_from(cdf, atoms)[0]
    return _arrays.update(masses, operator.truediv, xp.asarray(np.diff(edges)))


def _histogram_mean(bins: _typing.FloatArray, probabilities: Any) -> Any:
    """The mean of each histogram row, of ``probabilities``'s kind."""
    xp = array_api_compat.array_namespace(probabilities)
    return probabilities @ xp.asarray(0.5 * (bins[:-1] + bins[1:]))


def _masses_from(cdf: Any, atoms: Any) -> tuple[Any, Any]:
    """The mass in each bucket from the CDF at the bucket edges.

    The left limit of the CDF at every edge but the last, so that a point
    mass on an edge counts in the bucket above it.

    Args:
        cdf: The CDF at the edges, shape (n_rows, n_buckets + 1), a numpy
            array, overwritten, or a torch tensor.
        atoms: The point masses at every edge but the last, of ``cdf``'s
            kind and shape (n_rows, n_buckets), or None for none.

    Returns:
        A tuple ``(masses, outside)``: the non-negative masses, shape
        (n_rows, n_buckets), and the probability below the first edge plus
        that above the last, shape (n_rows,).
    """
    xp = array_api_compat.array_namespace(cdf)
    if atoms is not None:
        head = cdf[:, :-1]
        updated = _arrays.update(head, operator.sub, atoms)
        if updated is not head:
            cdf = xp.concat([updated, cdf[:, -1:]], axis=1)
        cdf = _unit(cdf)
    masses = _arrays.clip(cdf[:, 1:] - cdf[:, :-1], 0.0)
    return masses, cdf[:, 0] + (1.0 - cdf[:, -1])


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

    def _atoms(self, values: _typing.FloatArray) -> _typing.FloatArray | None:
        """Each row's point mass at ``values``, shape (n_rows, len(values)).

        None when the distribution has no point mass at all, so that callers
        skip the correction.
        """
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
            levels: Cumulative probabilities within [0, 1], a scalar or shape
                (k,).

        Returns:
            The values at those levels, shape (n_rows, k).

        Raises:
            ValueError: If ``levels`` is not a scalar or 1-D array within
                [0, 1].
        """
        levels = _inputs.check_levels(levels, name="levels")
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
        _inputs.check_coverage(confidence, name="confidence")
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

    def histogramize(self, bins: npt.ArrayLike) -> HistogramDistribution:
        """The masses this distribution puts in ``bins``, exactly.

        Each bucket holds ``bins[i] <= y < bins[i + 1]``, and the last also
        its upper edge, as :meth:`lazy.grid.Grid.bin_index` assigns values;
        a point mass on an edge goes to the bucket above it. Probability
        outside ``bins`` is dropped and each row renormalized.

        Args:
            bins: The bucket edges, shape (n_buckets + 1,), non-decreasing.

        Returns:
            The distribution's masses in those buckets, with its ``ancil``.
        """
        target = np.asarray(bins, dtype=np.float64)
        masses, _ = self._masses(target)
        return HistogramDistribution(target, masses, self.ancil)

    def _masses(
        self, edges: _typing.FloatArray
    ) -> tuple[_typing.FloatArray, _typing.FloatArray]:
        """The mass in each bucket of ``edges``, as :meth:`histogramize`'s.

        The CDF at the last edge, and its left limit at every other one, so
        that a point mass on an edge counts in the bucket above it.

        Args:
            edges: Non-decreasing bucket edges, shape (n_buckets + 1,).

        Returns:
            A tuple ``(masses, outside)``: the non-negative masses, shape
            (n_rows, n_buckets), and the probability below ``edges[0]``
            plus that above ``edges[-1]``, shape (n_rows,).
        """
        return _masses_from(self.cdf(edges), self._atoms(edges[:-1]))

    def _bin_densities(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """The mass in each bin of ``grid`` over its width, unnormalized."""
        raise NotImplementedError

    def _bin_densities_and_outside(
        self, grid: grid_lib.Grid
    ) -> tuple[_typing.FloatArray, _typing.FloatArray]:
        """:meth:`_bin_densities`, and each row's probability outside ``grid``.

        Both come from one evaluation of the CDF at the grid's edges
        (:meth:`_masses`), so the probability outside costs no second one.

        Args:
            grid: The grid to put the densities on.

        Returns:
            A tuple ``(density, outside)``: the unnormalized densities, shape
            (n_rows, grid.n_bins), and the probability below ``grid.y_min``
            plus that above ``grid.y_max``, shape (n_rows,).
        """
        masses, outside = self._masses(grid.edges)
        return _arrays.update(masses, operator.truediv, grid.widths), outside


@dataclasses.dataclass(frozen=True, eq=False)
class HistogramDistribution(_Base):
    """Probability masses in buckets shared by all rows, uniform within each.

    This is a bar distribution (TabPFN, LimiX) or a histogram over bins
    (TabFM). Buckets may be of any widths. The mass of a zero-width bucket
    is a point mass at its edge: the CDF steps there, and the quantiles, the
    moments and :meth:`on_grid` (in the bin above the edge) include it, but
    :meth:`pdf`, :meth:`mode` and :meth:`to_qp` do not, since it has no
    density.

    Attributes:
        bins: Bucket edges, shape (n_buckets + 1,), non-decreasing, finite.
        masses: Non-negative weight in each bucket, shape (n_rows,
            n_buckets), stored as given; every method treats each row as
            renormalized to sum to one (an empty row as uniform over the
            buckets of positive width). :attr:`probabilities` gives the
            renormalized masses.
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
        """The masses renormalized to sum to one per row, (n_rows, n_buckets).

        An empty row is uniform in probability over the buckets of positive
        width, so that a zero-width bucket carries none; over every bucket
        if none has a width. Each access returns a new array, which the
        caller may modify.
        """
        return _probabilities(self.masses, self.widths)

    def __getitem__(self, rows: Any) -> HistogramDistribution:
        return HistogramDistribution(
            self.bins,
            np.atleast_2d(self.masses[rows]),
            _slice_ancil(self.ancil, rows),
        )

    def on_grid(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """Densities on ``grid`` as ``predict_proba`` gives them.

        Each bin gets the exact mass the buckets put in it, as
        ``histogramize`` counts it, divided by its width (on the
        buckets' own edges, each bucket's own mass); the probability outside
        the grid is dropped and each row is normalized by the grid's
        convention (:meth:`lazy.grid.Grid.normalize`). For the masses before
        that normalization, use ``histogramize(grid.edges).masses``.

        Args:
            grid: The grid to put the densities on.

        Returns:
            Normalized densities, shape (n_rows, grid.n_bins).
        """
        return grid.normalize(self._bin_densities(grid))

    def _bin_densities(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """:func:`_histogram_bin_densities` of this distribution."""
        return _histogram_bin_densities(
            self.bins, self.probabilities, grid.edges
        )

    def pdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The density at ``values``, shape (n_rows, len(values)).

        The density inside a bucket is its mass over its width; zero outside
        the buckets and in zero-width buckets. The last edge belongs to the
        last bucket, as in ``np.histogram``.
        """
        y = np.atleast_1d(np.asarray(values, dtype=np.float64))
        widths = self.widths
        index = _closed_bucket(self.bins, y)
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
        return _histogram_cdf(self.bins, self.probabilities, y)

    def _cdf_rows(self, values: _typing.FloatArray) -> _typing.FloatArray:
        probabilities = self.probabilities
        index, fraction = _locate(self.bins, values)
        rows = np.arange(self.npdf)
        below = _mass_below(probabilities, rows, index)
        value = probabilities[rows, index]
        value *= fraction
        value += below
        value[values < self.bins[0]] = 0.0
        value[values >= self.bins[-1]] = 1.0
        return _unit(value)

    def _atoms(self, values: _typing.FloatArray) -> _typing.FloatArray | None:
        return _histogram_atoms(self.bins, self.probabilities, values)

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
        return _histogram_mean(self.bins, self.probabilities)

    def var(self) -> _typing.FloatArray:
        """The variance of each row, shape (n_rows,); buckets are uniform.

        About each row's mean, not as E[x^2] - mean^2, which cancels when
        the values sit far from zero: a bucket centered ``d`` from the mean
        and ``w`` wide contributes ``d^2 + w^2 / 12``.
        """
        probabilities = self.probabilities
        centres = 0.5 * (self.bins[:-1] + self.bins[1:])
        offset = centres - self.mean()[:, None]
        offset **= 2
        spread = probabilities @ (self.widths**2 / 12.0)
        return np.einsum("ij,ij->i", probabilities, offset) + spread

    def mode(self) -> _typing.FloatArray:
        """The center of each row's densest bucket, shape (n_rows,)."""
        widths = self.widths
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(widths > 0, self.probabilities / widths, -np.inf)
        best = np.argmax(density, axis=1)
        return 0.5 * (self.bins[best] + self.bins[best + 1])

    def to_qp(self) -> Any:
        """This distribution as a ``qp.Ensemble`` (``hist``); needs qp.

        Zero-width buckets are merged away, since qp requires strictly
        increasing bins.

        Raises:
            ImportError: If qp is not installed.
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
    outermost levels, each tail is spread evenly over one quantile spacing:
    the ``quants[0]`` below over ``[l_1 - (l_2 - l_1), l_1]`` and the
    ``1 - quants[-1]`` above over ``[l_n, l_n + (l_n - l_{n-1})]``, where
    ``l_j`` are the row's values. The CDF is therefore piecewise linear
    through two more knots, at the levels 0 and 1, and has no point masses,
    which on a fine grid would be one-bin spikes that a mode or a peak
    picks. For levels evenly spaced from the spacing itself (TabICL's and
    Kumo Tabular's 0.001, ..., 0.999), these are the knots qp's ``quant``
    adds by extrapolating linearly to 0 and 1. Where the outermost values
    are tied, a tail is spread over the nearest non-zero spacing in the
    row instead; only a row whose values are all equal keeps its tails as
    point masses on that value.

    Attributes:
        quants: The cumulative levels, shape (n_levels,), strictly
            increasing inside [0, 1].
        locs: The value at each level, shape (n_rows, n_levels), made
            non-decreasing on construction (a value below the one before it
            is raised to it).
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

        Args:
            parts: One distribution per member, at the same levels and with
                the same rows.
            weights: One finite, non-negative weight per part with a positive
                sum, normalized here; ``None`` for equal weights.

        Returns:
            The weighted average, with the first part's ``ancil``.
        """
        quants = parts[0].quants
        if any(not np.array_equal(p.quants, quants) for p in parts):
            raise ValueError("every part must share the same levels")
        if weights is None:
            weight = np.full(len(parts), 1.0 / len(parts))
        else:
            weight = _normalized(weights, len(parts))
        locs = np.tensordot(weight, np.stack([p.locs for p in parts]), axes=1)
        return cls(quants, locs, parts[0].ancil)

    @classmethod
    def trimmed_average(
        cls,
        parts: Sequence[QuantileDistribution],
        trim_fraction: float,
    ) -> QuantileDistribution:
        """The trimmed mean of quantile functions at shared levels.

        At every row and level, the parts' values are sorted, the lowest and
        the highest ``int(trim_fraction * len(parts))`` are dropped, and the
        rest are averaged. This is how Kumo Tabular combines its ensemble
        members, so that one member far from the rest moves the answer less.

        Args:
            parts: One distribution per member, at the same levels and with
                the same rows.
            trim_fraction: The fraction of parts dropped from each tail, in
                [0, 0.5); 0 is :meth:`average` with equal weights.

        Returns:
            The trimmed mean, with the first part's ``ancil``.

        Raises:
            ValueError: If the parts' levels differ or ``trim_fraction`` is
                outside [0, 0.5).

        Examples:
            >>> parts = [
            ...     QuantileDistribution([0.25, 0.75], [[value, value + 1]])
            ...     for value in (0.0, 1.0, 9.0)
            ... ]
            >>> QuantileDistribution.trimmed_average(parts, 0.34).locs
            array([[1., 2.]])
        """
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError(
                f"trim_fraction must be in [0, 0.5): {trim_fraction=}"
            )
        quants = parts[0].quants
        if any(not np.array_equal(p.quants, quants) for p in parts):
            raise ValueError("every part must share the same levels")
        locs = np.stack([p.locs for p in parts])
        cut = int(trim_fraction * len(parts))
        if cut > 0:
            locs = np.sort(locs, axis=0)[cut : len(parts) - cut]
        return cls(quants, locs.mean(axis=0), parts[0].ancil)

    def on_grid(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """Densities on ``grid`` as ``predict_proba`` gives them.

        The mass in each bin, as ``histogramize`` counts it, tails
        included, over the bin's width, with the probability outside the
        grid dropped and each row normalized by the grid's convention.

        Args:
            grid: The grid to put the densities on.

        Returns:
            Normalized densities, shape (n_rows, grid.n_bins).
        """
        return grid.normalize(self._bin_densities(grid))

    def _bin_densities(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """:func:`_quantile_bin_densities` of this distribution."""
        return _quantile_bin_densities(self.quants, self.locs, grid.edges)

    def _atoms(self, values: _typing.FloatArray) -> _typing.FloatArray | None:
        return _quantile_atoms(self.quants, self.locs, values)

    def cdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The CDF at ``values``, shape (n_rows, len(values)); exact."""
        y = np.atleast_1d(np.asarray(values, dtype=np.float64))
        return _quantile_cdf(
            self.quants, self.locs, np.broadcast_to(y, (self.npdf, y.size))
        )

    def _cdf_rows(self, values: _typing.FloatArray) -> _typing.FloatArray:
        return _quantile_cdf(self.quants, self.locs, values[:, None])[:, 0]

    def pdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The density at ``values``, shape (n_rows, len(values)).

        Constant between knots: in each tail, its probability over its
        width. Zero beyond the tails and on zero-width segments (a point
        mass has no density). The outermost values belong to the segments
        they end, and the ends of the tails to the tails.
        """
        y = np.atleast_1d(np.asarray(values, dtype=np.float64))
        mass, gap = _quantile_pieces(
            self.quants, self.locs, np.broadcast_to(y, (self.npdf, y.size))
        )
        width = np.take_along_axis(np.diff(self.locs, axis=1), gap, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((mass > 0) & (width > 0), mass / width, 0.0)

    def _ppf_rows(self, levels: _typing.FloatArray) -> _typing.FloatArray:
        knot_levels, locs = _knots(self.quants, self.locs)
        out = np.empty(levels.shape)
        for row in range(self.npdf):
            out[row] = np.interp(levels[row], knot_levels, locs[row])
        return out

    def mean(self) -> _typing.FloatArray:
        """The mean of each row, shape (n_rows,)."""
        return _quantile_mean(self.quants, self.locs)

    def var(self) -> _typing.FloatArray:
        """The variance of each row, shape (n_rows,).

        About each row's mean, not as E[x^2] - mean^2, which cancels when
        the values sit far from zero.
        """
        levels, locs = _knots(self.quants, self.locs)
        locs = locs - self.mean()[:, None]
        low, high = locs[:, :-1], locs[:, 1:]
        step = np.diff(levels)
        return (step * (low**2 + low * high + high**2) / 3.0).sum(axis=1)

    def mode(self) -> _typing.FloatArray:
        """The center of each row's densest segment, shape (n_rows,).

        A tail is the mode only where it is denser than every segment
        between the quantiles by more than rounding. With evenly spaced
        levels, a tail is as dense, to rounding, as the segment it borrows
        its width from, and the mode then stays between the quantiles.
        """
        rows = np.arange(self.npdf)
        width = np.diff(self.locs, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(width > 0, np.diff(self.quants) / width, -np.inf)
        best = np.argmax(density, axis=1)
        peak = density[rows, best]
        mode = 0.5 * (self.locs[rows, best] + self.locs[rows, best + 1])
        widths = np.stack(_tail_widths(self.quants, self.locs), axis=1)
        masses = np.array([self.quants[0], 1.0 - self.quants[-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            tails = np.where(widths > 0, masses / widths, -np.inf)
        centres = (
            self.locs[:, 0] - 0.5 * widths[:, 0],
            self.locs[:, -1] + 0.5 * widths[:, 1],
        )
        for side, centre in enumerate(centres):
            denser = tails[:, side] > peak * (1.0 + _MODE_RTOL)
            mode = np.where(denser, centre, mode)
            peak = np.where(denser, tails[:, side], peak)
        return mode

    def to_qp(self) -> Any:
        """This distribution as a ``qp.Ensemble`` (``quant``); needs qp.

        The ensemble gets the knots, the ends of the tails at the levels 0
        and 1 included, so that qp has no tails of its own to add. The two
        then agree exactly at the knots; between them qp interpolates its
        quantile function its own way, so its values there differ slightly
        from the piecewise-linear CDF used here. For an exact copy, export
        ``histogramize(bins).to_qp()``.

        Raises:
            ImportError: If qp is not installed.
        """
        qp = _import_qp()
        levels, locs = _knots(self.quants, self.locs)
        ensemble = qp.Ensemble(qp.quant, data={"quants": levels, "locs": locs})
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
        weights = _normalized(self.weights, len(components))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "weights", _freeze(weights))
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
        return self._with_components(
            tuple(c[rows] for c in self.components),
            _slice_ancil(self.ancil, rows),
        )

    def _with_components(
        self,
        components: tuple[HistogramDistribution, ...],
        ancil: dict[str, Any] | None,
    ) -> MixtureDistribution:
        """A mixture of ``components`` with this one's weights, bit for bit.

        Normalizing weights that already sum to one can move them by an ulp
        (six weights of 1/6 sum to just under one), so they are not
        normalized again.
        """
        out = MixtureDistribution(components, self.weights, ancil)
        object.__setattr__(out, "weights", self.weights)
        return out

    def to_histogram(self) -> HistogramDistribution:
        """The mixture as one histogram on the union of edges; exact.

        The rows are built a block at a time, so the components' CDFs
        exist for one block at once. The result takes ``8 * n_rows *
        n_edges`` bytes, held twice while the constructor copies it.
        """
        bins = self.bins
        masses = np.empty((self.npdf, bins.size - 1))
        for rows, block in self._histogram_blocks():
            masses[rows] = block.masses
        return HistogramDistribution(bins, masses, self.ancil)

    def _weighted(self, values: Iterable[_typing.FloatArray]) -> Any:
        """The weighted sum of one array per component, accumulated in turn.

        A generator of the arrays keeps only the running sum and one array.
        """
        return _arrays.weighted_sum(self.weights, values)

    def _histogram_blocks(
        self,
    ) -> Iterator[tuple[slice, HistogramDistribution]]:
        """The union histogram a block of rows at a time.

        Yields:
            Tuples (rows, histogram): the slice of rows a block covers, and
            those rows' histogram on the union of edges. No rows give one
            empty block.
        """
        bins = self.bins
        size = max(1, _MIXTURE_BLOCK_BYTES // (8 * bins.size))
        for start in range(0, max(self.npdf, 1), size):
            rows = slice(start, start + size)
            yield rows, self[rows].histogramize(bins)

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
        return grid.normalize(self._bin_densities(grid))

    def _bin_densities(self, grid: grid_lib.Grid) -> _typing.FloatArray:
        """The components' bin densities, weighted and unnormalized."""
        return _arrays.weighted_total(
            self.weights,
            (c._bin_densities(grid) for c in self.components),  # noqa: SLF001 - same module.
        )

    def pdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The density at ``values``, shape (n_rows, len(values))."""
        return self._weighted(c.pdf(values) for c in self.components)

    def cdf(self, values: npt.ArrayLike) -> _typing.FloatArray:
        """The CDF at ``values``, shape (n_rows, len(values)); exact."""
        return _unit(self._weighted(c.cdf(values) for c in self.components))

    def _cdf_rows(self, values: _typing.FloatArray) -> _typing.FloatArray:
        return _unit(
            self._weighted(
                c._cdf_rows(values)  # noqa: SLF001 - same module.
                for c in self.components
            )
        )

    def _atoms(self, values: _typing.FloatArray) -> _typing.FloatArray | None:
        atoms = [c._atoms(values) for c in self.components]  # noqa: SLF001 - same module.
        if all(a is None for a in atoms):
            return None
        zero = np.zeros((self.npdf, values.size))
        return self._weighted(zero if a is None else a for a in atoms)

    def _ppf_rows(self, levels: _typing.FloatArray) -> _typing.FloatArray:
        # The union histogram is the same distribution, so its inverse is
        # this one's.
        return np.concatenate(
            [
                block._ppf_rows(levels[rows])  # noqa: SLF001 - same module.
                for rows, block in self._histogram_blocks()
            ]
        )

    def mean(self) -> _typing.FloatArray:
        """The mean of each row, shape (n_rows,)."""
        return _arrays.weighted_total(
            self.weights, (c.mean() for c in self.components)
        )

    def var(self) -> _typing.FloatArray:
        """The variance of each row, shape (n_rows,).

        Each component's variance plus its mean's squared distance from the
        mixture's, which does not cancel when the values sit far from zero.
        """
        means = [c.mean() for c in self.components]
        mean = self._weighted(means)
        return self._weighted(
            c.var() + (m - mean) ** 2
            for c, m in zip(self.components, means, strict=True)
        )

    def mode(self) -> _typing.FloatArray:
        """The center of each row's densest union bucket, shape (n_rows,)."""
        return np.concatenate(
            [block.mode() for _, block in self._histogram_blocks()]
        )

    def to_qp(self) -> Any:
        """The mixture as a ``qp.Ensemble`` (``hist`` on the union edges).

        Raises:
            ImportError: If qp is not installed.
        """
        return self.to_histogram().to_qp()


Distribution: TypeAlias = (
    HistogramDistribution | QuantileDistribution | MixtureDistribution
)
"""Any per-object distribution this module defines."""


def concatenate(parts: Sequence[Distribution]) -> Distribution:
    """Stacks the rows of distributions of the same kind and parameters.

    Args:
        parts: Distributions from consecutive chunks of rows: histograms
            with the same bins, quantiles at the same levels, or mixtures
            with the same weights and aligned components.

    Returns:
        One distribution holding every row, in order.

    Raises:
        TypeError: If the parts are of different kinds.
        ValueError: If their bins, levels, weights or numbers of
            components differ.
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
    for part in pieces_m:
        if len(part.components) != len(first.components):
            raise ValueError(
                "mixtures must have the same number of components: "
                f"{len(part.components)} != {len(first.components)}"
            )
        if not np.array_equal(part.weights, first.weights):
            raise ValueError(
                "mixtures must share their weights: "
                f"{part.weights} != {first.weights}"
            )
    components = tuple(
        concatenate([p.components[k] for p in pieces_m])
        for k in range(len(first.components))
    )
    return pieces_m[0]._with_components(components, ancil)  # noqa: SLF001 - same module.


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
        ensemble: A qp ensemble whose parameterization is ``hist`` or
            ``quant``; convert others first with ``ensemble.convert_to``.

    Returns:
        The equivalent distribution, with the ensemble's ancil, if any.

    Raises:
        ImportError: If qp is not installed.
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
