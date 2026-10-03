# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Exactness of the distribution algebra, and agreement with the grid code."""

import dataclasses

import numpy as np
import pytest

import lazy
from lazy import distributions

LEVELS = np.array([0.01, 0.16, 0.5, 0.84, 0.99])


@pytest.fixture
def histogram():
    rng = np.random.default_rng(7)
    bins = np.concatenate([[0.0], np.cumsum(rng.uniform(0.01, 0.2, 30))])
    masses = rng.gamma(0.7, size=(6, 30))
    masses[2, :] = 0.0  # an empty row
    return distributions.HistogramDistribution(bins, masses)


@pytest.fixture
def quantiles():
    rng = np.random.default_rng(8)
    quants = np.linspace(0.02, 0.98, 25)
    locs = np.sort(rng.normal(1.0, 0.3, (5, 25)), axis=1)
    locs[1, 5:9] = locs[1, 5]  # a zero-width segment
    return distributions.QuantileDistribution(quants, locs)


def _integral_moments(dist, lo, hi, n=400_001):
    """Mean and variance by integrating the CDF numerically."""
    z = np.linspace(lo, hi, n)
    survival = 1.0 - dist.cdf(z)
    below = dist.cdf(z)
    mean = lo + np.trapezoid(survival, z, axis=1)
    second = (
        lo**2
        + 2 * np.trapezoid(z * survival, z, axis=1)
        - 0 * np.trapezoid(below, z, axis=1)
    )
    return mean, second - mean**2


# -- histograms --------------------------------------------------------------


def test_histogram_on_grid_is_exactly_the_normalized_rebinning(histogram):
    grid = lazy.Grid.linear(0.0, 3.0, 57)
    np.testing.assert_array_equal(
        histogram.on_grid(grid),
        grid.normalize(grid.rebin(histogram.probabilities, histogram.bins)),
    )


def test_histogram_cdf_at_the_edges_is_the_cumulative_mass(histogram):
    cdf = histogram.cdf(histogram.bins)
    expected = np.concatenate(
        [np.zeros((6, 1)), np.cumsum(histogram.probabilities, axis=1)], axis=1
    )
    np.testing.assert_allclose(cdf, expected, atol=1e-15)
    assert (histogram.cdf([-1.0]) == 0).all()
    assert (histogram.cdf([99.0]) == 1).all()


def test_histogram_ppf_inverts_the_cdf(histogram):
    z = histogram.ppf(LEVELS)
    for row in range(histogram.npdf):
        np.testing.assert_allclose(
            histogram[row].cdf(z[row])[0], LEVELS, atol=1e-12
        )


def test_an_empty_histogram_row_is_uniform_in_probability(histogram):
    np.testing.assert_allclose(histogram.probabilities[2], 1 / 30)


def test_histogram_moments_match_integration(histogram):
    mean, var = _integral_moments(histogram, 0.0, histogram.bins[-1])
    np.testing.assert_allclose(histogram.mean(), mean, rtol=1e-6)
    np.testing.assert_allclose(histogram.var(), var, rtol=1e-4)


def test_histogram_pdf_integrates_to_one(histogram):
    z = np.linspace(-0.5, histogram.bins[-1] + 0.5, 200_001)
    np.testing.assert_allclose(
        np.trapezoid(histogram.pdf(z), z, axis=1), 1.0, atol=2e-3
    )


def test_histogram_mode_is_in_the_densest_bucket(histogram):
    density = histogram.probabilities / histogram.widths
    best = density.argmax(axis=1)
    centres = 0.5 * (histogram.bins[:-1] + histogram.bins[1:])
    np.testing.assert_allclose(histogram.mode(), centres[best])


def test_histogram_samples_follow_the_cdf(histogram):
    draws = histogram.rvs(20_000, random_state=1)
    empirical = (draws[0, :, None] <= histogram.bins[None, 10:20]).mean(axis=0)
    np.testing.assert_allclose(
        empirical, histogram[0].cdf(histogram.bins[10:20])[0], atol=0.02
    )


def test_histogramize_conserves_mass_inside_the_new_bins(histogram):
    coarse = histogram.histogramize(histogram.bins[::3])
    np.testing.assert_allclose(
        coarse.cdf(coarse.bins), histogram.cdf(coarse.bins), atol=1e-12
    )


def test_zero_width_buckets_are_allowed():
    dist = distributions.HistogramDistribution(
        [0.0, 1.0, 1.0, 2.0], [[0.5, 0.0, 0.5]]
    )
    np.testing.assert_allclose(dist.ppf([0.25, 0.75]), [[0.5, 1.5]])
    np.testing.assert_allclose(dist.pdf([0.5, 1.0, 1.5]), [[0.5, 0.5, 0.5]])


@pytest.mark.parametrize(
    ("bins", "masses", "match"),
    [
        ([0.0, 2.0, 1.0], [[1.0, 1.0]], "non-decreasing"),
        ([0.0, 1.0], [[1.0, 1.0]], "buckets for"),
        ([0.0, 1.0, 2.0], [[1.0, -0.1]], "non-negative"),
    ],
)
def test_histogram_validation(bins, masses, match):
    with pytest.raises(ValueError, match=match):
        distributions.HistogramDistribution(bins, masses)


# -- quantiles ---------------------------------------------------------------


def test_quantile_on_grid_is_exactly_from_quantiles(quantiles):
    grid = lazy.Grid.linear(0.0, 2.0, 80)
    np.testing.assert_array_equal(
        quantiles.on_grid(grid),
        grid.from_quantiles(quantiles.locs, quantiles.quants),
    )


def test_quantile_ppf_returns_the_knots_and_inverts_the_cdf(quantiles):
    np.testing.assert_array_equal(
        quantiles.ppf(quantiles.quants), quantiles.locs
    )
    inner = np.array([0.1, 0.33, 0.5, 0.9])
    z = quantiles.ppf(inner)
    for row in (0, 2, 3):
        np.testing.assert_allclose(quantiles[row].cdf(z[row])[0], inner)


def test_quantile_tails_are_point_masses_on_the_outer_knots(quantiles):
    np.testing.assert_array_equal(
        quantiles.ppf([0.0, 0.01, 0.99, 1.0]),
        quantiles.locs[:, [0, 0, -1, -1]],
    )
    below = quantiles.cdf(quantiles.locs[:, 0] - 1e-9)
    assert (np.diag(below) == 0).all()


def test_quantile_moments_match_integration(quantiles):
    lo, hi = quantiles.locs.min() - 0.1, quantiles.locs.max() + 0.1
    mean, var = _integral_moments(quantiles, lo, hi)
    np.testing.assert_allclose(quantiles.mean(), mean, rtol=1e-6)
    np.testing.assert_allclose(quantiles.var(), var, rtol=1e-4)


def test_quantile_average_averages_the_quantile_functions(quantiles):
    shifted = distributions.QuantileDistribution(
        quantiles.quants, quantiles.locs + 0.2
    )
    both = distributions.QuantileDistribution.average([quantiles, shifted])
    np.testing.assert_allclose(both.locs, quantiles.locs + 0.1)


def test_crossing_quantiles_are_sorted():
    dist = distributions.QuantileDistribution([0.25, 0.5, 0.75], [[1, 0.5, 2]])
    np.testing.assert_array_equal(dist.locs, [[1.0, 1.0, 2.0]])


# -- mixtures ----------------------------------------------------------------


@pytest.fixture
def mixture(histogram):
    other = distributions.HistogramDistribution(
        np.linspace(0.3, 4.0, 12),
        np.random.default_rng(9).uniform(size=(6, 11)),
    )
    return distributions.MixtureDistribution((histogram, other), [0.3, 0.7])


def test_mixture_on_grid_is_the_normalized_weighted_average(mixture):
    # The grid cuts the components' tails unequally, so normalizing the
    # average differs from averaging the normalized components.
    grid = lazy.Grid.linear(0.2, 3.0, 40)
    first, second = mixture.components
    masses = 0.3 * first.histogramize(grid.edges).masses
    masses += 0.7 * second.histogramize(grid.edges).masses
    np.testing.assert_allclose(
        mixture.on_grid(grid), grid.normalize(masses / grid.widths)
    )


def test_mixture_histogram_is_exact_on_the_union_of_edges(mixture):
    union = mixture.to_histogram()
    z = np.linspace(-0.5, 4.5, 1001)
    np.testing.assert_allclose(union.cdf(z), mixture.cdf(z), atol=1e-12)
    np.testing.assert_allclose(union.mean(), mixture.mean(), rtol=1e-12)
    np.testing.assert_allclose(union.var(), mixture.var(), rtol=1e-9)


def test_mixture_ppf_inverts_the_cdf(mixture):
    z = mixture.ppf(LEVELS)
    for row in range(mixture.npdf):
        np.testing.assert_allclose(
            mixture[row].cdf(z[row])[0], LEVELS, atol=1e-12
        )


def test_mixtures_take_histograms_only(histogram, quantiles):
    with pytest.raises(TypeError, match="HistogramDistribution"):
        distributions.MixtureDistribution((histogram, quantiles), [0.5, 0.5])


# -- on a grid, and per-row CDFs, for every kind ------------------------------

KINDS = ["histogram", "quantiles", "mixture"]


@pytest.mark.parametrize("name", KINDS)
@pytest.mark.parametrize("normalization", ["trapezoid", "histogram"])
def test_on_grid_is_normalized_by_the_grid_even_when_it_cuts_mass(
    request, name, normalization
):
    dist = request.getfixturevalue(name)
    grid = lazy.Grid.linear(0.4, 1.6, 50, normalization=normalization)
    assert (np.diff(dist.cdf([grid.y_min, grid.y_max])) < 1.0).any()
    pdfs = dist.on_grid(grid)
    np.testing.assert_allclose(
        lazy.metrics.normalization_error(grid, pdfs), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(grid.normalize(pdfs), pdfs, rtol=1e-12)


@pytest.mark.parametrize("name", KINDS)
def test_pit_is_the_diagonal_of_the_cdf(request, name):
    dist = request.getfixturevalue(name)
    rng = np.random.default_rng(11)
    values = rng.uniform(-0.5, 4.5, len(dist))
    values[0] = -np.inf
    values[1] = np.inf
    pit = dist.pit(values)
    assert pit.shape == (len(dist),)
    np.testing.assert_allclose(pit, np.diag(dist.cdf(values)), atol=1e-15)
    assert pit[0] == 0.0 and pit[1] == 1.0
    # Away from the quantile fixture's atom, the CDF inverts the ppf.
    median = dist.ppf([0.5])[:, 0]
    np.testing.assert_allclose(dist.pit(median), 0.5, atol=1e-12)


@pytest.mark.parametrize("name", KINDS)
def test_pit_matches_the_grid_metrics_on_a_fine_grid(request, name):
    dist = request.getfixturevalue(name)
    values = dist.ppf([0.37])[:, 0] + 0.01
    low, high = dist.ppf([0.0, 1.0]).T
    grid = lazy.Grid.linear(
        low.min() - 0.01, high.max() + 0.01, 20_000, normalization="histogram"
    )
    _, grid_pit = lazy.metrics.per_object_scores(
        values, grid, dist.on_grid(grid)
    )
    np.testing.assert_allclose(dist.pit(values), grid_pit, atol=1e-6)


def test_pit_is_exact_on_a_histogram_grid_of_its_own_buckets(histogram):
    grid = lazy.Grid.from_edges(histogram.bins, normalization="histogram")
    values = np.random.default_rng(12).uniform(0.0, histogram.bins[-1], 6)
    _, grid_pit = lazy.metrics.per_object_scores(
        values, grid, histogram.on_grid(grid)
    )
    np.testing.assert_allclose(histogram.pit(values), grid_pit, atol=1e-12)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.zeros(5), r"shape \(6,\)"),
        (np.zeros((6, 1)), r"shape \(6,\)"),
        (0.5, r"shape \(6,\)"),
        (np.r_[np.zeros(5), np.nan], "NaN"),
    ],
)
def test_pit_takes_one_value_per_row(histogram, values, match):
    with pytest.raises(ValueError, match=match):
        histogram.pit(values)


def test_pit_of_no_rows_is_empty(histogram):
    assert histogram[:0].pit(np.zeros(0)).shape == (0,)


# -- rows, metadata, concatenation ------------------------------------------


def test_slicing_and_concatenation_keep_rows_and_ancil(histogram):
    tagged = distributions.HistogramDistribution(
        histogram.bins, histogram.masses, {"id": np.arange(6)}
    )
    parts = [tagged[:2], tagged[2:5], tagged[5:]]
    joined = distributions.concatenate(parts)
    np.testing.assert_array_equal(joined.masses, tagged.masses)
    np.testing.assert_array_equal(joined.ancil["id"], np.arange(6))
    assert len(joined) == 6


def test_concatenating_mixtures_keeps_their_components(mixture):
    joined = distributions.concatenate([mixture[:3], mixture[3:]])
    np.testing.assert_allclose(joined.cdf([1.0]), mixture.cdf([1.0]))


def test_ancil_must_have_one_entry_per_row(histogram):
    with pytest.raises(ValueError, match="entries for 6 rows"):
        distributions.HistogramDistribution(
            histogram.bins, histogram.masses, {"id": np.arange(3)}
        )


def test_interval_is_central(quantiles):
    np.testing.assert_allclose(
        quantiles.interval(0.68), quantiles.ppf([0.16, 0.84])
    )


# -- qp interoperability -----------------------------------------------------


def test_histograms_round_trip_through_qp(histogram):
    pytest.importorskip("qp")
    back = distributions.from_qp(histogram.to_qp())
    np.testing.assert_allclose(back.bins, histogram.bins)
    np.testing.assert_allclose(
        back.probabilities, histogram.probabilities, atol=1e-12
    )


def test_qp_agrees_on_histogram_quantiles(histogram):
    pytest.importorskip("qp")
    ours = histogram[[0, 1, 3]]
    theirs = ours.to_qp().ppf(np.array([0.16, 0.5, 0.84]))
    np.testing.assert_allclose(theirs, ours.ppf([0.16, 0.5, 0.84]), atol=1e-6)


def test_qp_agrees_at_the_quantile_levels(quantiles):
    pytest.importorskip("qp")
    levels = quantiles.quants[2:-2:4]
    np.testing.assert_allclose(
        quantiles.to_qp().ppf(levels), quantiles.ppf(levels), atol=1e-9
    )


@pytest.mark.parametrize("row", [3, -1, np.int64(2)])
@pytest.mark.parametrize("name", ["histogram", "quantiles", "mixture"])
def test_an_integer_row_keeps_its_ancil(request, name, row):
    dist = request.getfixturevalue(name)
    n = len(dist)
    tagged = dataclasses.replace(
        dist, ancil={"id": np.arange(n), "pair": np.ones((n, 2))}
    )
    one = tagged[row]
    assert len(one) == 1
    np.testing.assert_array_equal(one.ancil["id"], [np.arange(n)[row]])
    assert one.ancil["pair"].shape == (1, 2)


def test_histogram_ppf_ends_at_the_support_despite_empty_edge_buckets():
    """Levels 0 and 1 are the edges of the buckets that carry mass."""
    masses = np.r_[0.0, 0.0, np.full(10, 0.1), 0.0, 0.0]
    dist = distributions.HistogramDistribution(np.arange(15.0), masses)
    np.testing.assert_array_equal(dist.ppf([0.0, 0.5, 1.0]), [[2.0, 7.0, 12.0]])
    samples = dist.rvs(2000, random_state=0)
    assert samples.min() >= 2.0 and samples.max() <= 12.0
