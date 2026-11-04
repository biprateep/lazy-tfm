# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Invariants of the distributions and the grid, on generated inputs.

Seeded (``derandomize=True``), so every run draws the same examples. The
offsets and scales reach where float64 rounding bites: offsets up to 1e7 with
spreads down to 1e-4.
"""

import hypothesis
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np

import lazy
from lazy import distributions

SETTINGS = hypothesis.settings(
    max_examples=100,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)


def _resolution(points, value):
    """The probability float64 resolves at ``value``, among sorted ``points``.

    A value is held to an ulp of the largest point, which the narrower of the
    two segments beside it turns into an error in probability.
    """
    edges = np.unique(points)
    if edges.size == 1:
        return np.inf  # A point mass: no CDF inverts it.
    i = np.clip(np.searchsorted(edges, value), 1, edges.size - 1)
    gap = edges[i] - edges[i - 1]
    if i + 1 < edges.size:
        gap = min(gap, edges[i + 1] - edges[i])
    return 4 * np.spacing(np.abs(edges).max()) / gap


@st.composite
def histograms(draw):
    """Histograms of two rows, some buckets of zero width, maybe one empty.

    The support has a width: a histogram of zero-width buckets only would be
    a point mass.
    """
    n = draw(st.integers(1, 12))
    offset = draw(st.sampled_from([0.0, 1.0, -50.0, 1e4, 1e6]))
    scale = draw(st.sampled_from([1e-3, 1.0, 10.0]))
    widths = draw(
        hnp.arrays(np.float64, n, elements=st.just(0.0) | st.floats(1e-6, 5.0))
    )
    bins = offset + scale * np.concatenate([[0.0], np.cumsum(widths)])
    hypothesis.assume(bins[-1] > bins[0])
    masses = draw(hnp.arrays(np.float64, (2, n), elements=st.floats(0.0, 10.0)))
    masses[:, np.diff(bins) == 0] = 0.0  # A zero-width bucket carries none.
    if draw(st.booleans()):
        masses[1] = 0.0
    return distributions.HistogramDistribution(bins, masses)


@st.composite
def quantile_distributions(draw):
    """Quantile distributions of two rows, some knots tied."""
    k = draw(st.integers(2, 9))
    levels = np.sort(
        draw(
            hnp.arrays(
                np.float64, k, elements=st.floats(0.001, 0.999), unique=True
            )
        )
    )
    offset = draw(st.sampled_from([0.0, 2.0, -300.0, 1e5, 1e7]))
    scale = draw(st.sampled_from([1e-4, 1.0, 50.0]))
    steps = draw(hnp.arrays(np.float64, (2, k), elements=st.floats(0.0, 3.0)))
    locs = offset + scale * np.cumsum(steps, axis=1)
    return distributions.QuantileDistribution(levels, locs)


@SETTINGS
@hypothesis.given(histograms(), st.floats(0.0, 1.0))
def test_a_histograms_cdf_inverts_its_ppf(dist, level):
    values = dist.ppf([level])[:, 0]
    atol = 1e-9 + np.array([_resolution(dist.bins, v) for v in values])
    assert np.all(np.abs(dist.pit(values) - level) <= atol)


@SETTINGS
@hypothesis.given(histograms())
def test_a_histograms_moments_lie_within_its_support(dist):
    low, high = dist.bins[0], dist.bins[-1]
    slack = 1e-9 * max(1.0, abs(low), abs(high))
    mean, var = dist.mean(), dist.var()
    assert np.all(var >= 0.0)
    assert np.all((mean >= low - slack) & (mean <= high + slack))
    assert np.all(np.sqrt(var) <= (high - low) / 2 + slack)


@SETTINGS
@hypothesis.given(quantile_distributions(), st.floats(0.0, 1.0))
def test_a_quantile_distributions_cdf_inverts_its_ppf(dist, level):
    hypothesis.assume(np.ptp(dist.locs) > 0)
    values = dist.ppf([level])[:, 0]
    pit = dist.pit(values)
    atol = 1e-6 + np.array(
        [
            _resolution(locs, v)
            for locs, v in zip(dist.locs, values, strict=True)
        ]
    )
    # Tied knots make an atom, at which the CDF jumps past the level.
    assert np.all(pit >= level - atol)
    untied = np.all(np.diff(dist.locs, axis=1) > 0, axis=1)
    assert np.all(np.abs(pit - level)[untied] <= atol[untied])


@SETTINGS
@hypothesis.given(quantile_distributions())
def test_histogramizing_on_a_covering_grid_keeps_the_mass(dist):
    low, high = dist.ppf([0.0]).min(), dist.ppf([1.0]).max()
    pad = 1e-6 * max(1.0, abs(low), abs(high)) + 0.01 * (high - low) + 1e-9
    histogram = dist.histogramize(np.linspace(low - pad, high + pad, 301))
    np.testing.assert_allclose(histogram.masses.sum(axis=1), 1.0)


@SETTINGS
@hypothesis.given(histograms(), st.integers(2, 40))
def test_densities_on_a_covering_grid_keep_the_mass(dist, n_bins):
    low, high = dist.bins[0], dist.bins[-1]
    # Room for n_bins strictly increasing edges.
    hypothesis.assume(high - low > 1e-9 * n_bins * max(1.0, abs(high)))
    grid = lazy.Grid.linear(low, high, n_bins, normalization="histogram")
    density = dist._bin_densities(grid)
    np.testing.assert_allclose(density @ grid.widths, 1.0)
