# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Histogram-normalised grids score bar distributions exactly.

The reference formulas are the paper's native scoring of a bar distribution
(lazy-paper/paperlib/native.py::native_scores): the integral of p^2 is the
sum of mass^2 / width, the density at the truth is its bucket's mass over
width, and the PIT is the mass below plus the fraction of the truth's bucket.
"""

import numpy as np
import pytest

import lazy
from lazy import distributions
from lazy import metrics


@pytest.fixture
def bar():
    """A bar distribution on strongly non-uniform buckets, and truths."""
    rng = np.random.default_rng(11)
    inner = np.linspace(0.0, 2.0, 81)
    edges = np.concatenate([[-3.0, -1.0], inner, [3.0, 6.0]])
    masses = rng.gamma(0.6, size=(40, edges.size - 1))
    masses /= masses.sum(axis=1, keepdims=True)
    truth = rng.uniform(-0.5, 2.5, 40)
    truth[0] = 7.0  # outside every bucket
    return edges, masses, truth


def _native_scores(edges, masses, truth):
    """The paper's exact per-object CDE terms, PIT and mode."""
    widths = np.diff(edges)
    rows = np.arange(len(truth))
    bucket = np.clip(
        np.searchsorted(edges, truth, "right") - 1, 0, widths.size - 1
    )
    inside = (truth >= edges[0]) & (truth < edges[-1])
    p_true = np.where(inside, masses[rows, bucket] / widths[bucket], 0.0)
    int_p2 = (masses**2 / widths).sum(axis=1)
    below = np.cumsum(masses, axis=1) - masses
    fraction = np.clip((truth - edges[bucket]) / widths[bucket], 0, 1)
    pit = np.where(
        truth < edges[0],
        0.0,
        np.where(
            truth >= edges[-1],
            1.0,
            below[rows, bucket] + masses[rows, bucket] * fraction,
        ),
    )
    return int_p2 - 2 * p_true, pit


def test_histogram_grid_normalises_to_unit_mass(bar):
    edges, masses, _ = bar
    grid = lazy.Grid.from_edges(edges, normalization="histogram")
    density = grid.normalize(3.0 * masses / grid.widths)
    np.testing.assert_allclose(density @ grid.widths, 1.0)
    np.testing.assert_allclose(density * grid.widths, masses, atol=1e-15)


def test_histogram_cde_loss_and_pit_match_the_native_scores(bar):
    edges, masses, truth = bar
    grid = lazy.Grid.from_edges(edges, normalization="histogram")
    density = masses / grid.widths
    expected_cde, expected_pit = _native_scores(edges, masses, truth)
    loss = metrics.cde_loss(truth, grid.centers, density, bin_edges=grid.edges)
    np.testing.assert_allclose(loss, expected_cde.mean(), rtol=1e-12)
    *_, pit = metrics.evaluate_grid_at_truth(
        truth, grid.centers, density, bin_edges=grid.edges
    )
    np.testing.assert_allclose(pit, expected_pit, atol=1e-12)


def test_histogram_point_estimates_agree_with_the_distribution(bar):
    edges, masses, _ = bar
    grid = lazy.Grid.from_edges(edges, normalization="histogram")
    estimates = metrics.grid_point_estimates(
        grid.centers, masses / grid.widths, bin_edges=grid.edges
    )
    dist = distributions.HistogramDistribution(edges, masses)
    np.testing.assert_allclose(estimates["mean"], dist.mean(), rtol=1e-12)
    np.testing.assert_allclose(estimates["median"], dist.median(), atol=1e-12)
    np.testing.assert_allclose(estimates["mode"], dist.mode())


def test_histogram_cdf_at_centres_is_exact(bar):
    edges, masses, _ = bar
    grid = lazy.Grid.from_edges(edges, normalization="histogram")
    dist = distributions.HistogramDistribution(edges, masses)
    np.testing.assert_allclose(
        grid.cdf(masses / grid.widths), dist.cdf(grid.centers), atol=1e-12
    )


def test_summarize_takes_bin_edges(bar):
    edges, masses, truth = bar
    grid = lazy.Grid.from_edges(edges, normalization="histogram")
    row = metrics.summarize(
        truth, grid.centers, masses / grid.widths, bin_edges=grid.edges
    )
    expected, _ = _native_scores(edges, masses, truth)
    np.testing.assert_allclose(row["cde_loss"].iloc[0], expected.mean())


def test_normalization_is_part_of_grid_identity():
    trapezoid = lazy.Grid.linear(0.0, 2.0, 10)
    histogram = lazy.Grid.linear(0.0, 2.0, 10, normalization="histogram")
    assert trapezoid != histogram
    assert trapezoid.histogram_edges is None
    np.testing.assert_array_equal(histogram.histogram_edges, histogram.edges)
    assert "histogram" in repr(histogram) and "histogram" not in repr(trapezoid)


def test_an_unknown_normalization_is_rejected():
    with pytest.raises(ValueError, match="normalization must be one of"):
        lazy.Grid.linear(0.0, 1.0, 4, normalization="simpson")


def test_bin_edges_must_match_the_grid():
    with pytest.raises(ValueError, match="one more entry"):
        metrics.normalize_grid_pdfs(
            np.array([0.5, 1.5]), np.ones((1, 2)), bin_edges=[0.0, 1.0]
        )
