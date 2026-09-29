# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import numpy as np
import pytest

import lazy


def test_dc1_grid_matches_the_challenge_output_format():
    assert lazy.DC1_GRID.n_bins == 200
    assert lazy.DC1_GRID.z_min == 0.0
    assert lazy.DC1_GRID.z_max == 2.0
    assert np.allclose(lazy.DC1_GRID.widths, 0.01)


def test_from_centers_round_trips_a_uniform_grid():
    assert lazy.Grid.from_centers(lazy.DC1_GRID.centers) == lazy.DC1_GRID


def test_edges_must_increase():
    with pytest.raises(ValueError, match="strictly increasing"):
        lazy.Grid(np.array([0.0, 1.0, 0.5]))


def test_edges_are_immutable():
    grid = lazy.Grid.linear(0.0, 1.0, 4)
    with pytest.raises(ValueError):
        grid.edges[0] = -1.0


def test_normalize_gives_unit_trapezoid_mass():
    grid = lazy.Grid.linear(0.0, 2.0, 200)
    pdfs = np.random.default_rng(0).uniform(size=(5, 200))
    mass = np.trapezoid(grid.normalize(pdfs), grid.centers, axis=1)
    assert np.allclose(mass, 1.0)


def test_normalize_rescues_a_zero_row_instead_of_returning_nan():
    grid = lazy.Grid.linear(0.0, 2.0, 200)
    pdfs = np.zeros((2, 200))
    pdfs[1] = 1.0
    out = grid.normalize(pdfs)
    assert np.isfinite(out).all()
    assert np.allclose(np.trapezoid(out, grid.centers, axis=1), 1.0)


def test_rebin_conserves_mass_onto_a_coarser_grid():
    """The whole point of rebin: no probability is created or lost."""
    fine = lazy.Grid.linear(0.0, 2.0, 1000)
    coarse = lazy.Grid.linear(0.0, 2.0, 50)
    probs = np.random.default_rng(0).uniform(size=(4, 1000))
    probs /= probs.sum(axis=1, keepdims=True)
    density = coarse.rebin(probs, fine.edges)
    assert np.allclose((density * coarse.widths).sum(axis=1), 1.0)


def test_rebin_keeps_mass_from_bins_narrower_than_the_output_bin():
    """Sampling the density at output centres drops them; integrating cannot.

    Equal-mass bins are routinely narrower than an output bin.
    """
    output = lazy.Grid.linear(0.0, 1.0, 2)  # two 0.5-wide bins
    # Two bins 500x narrower than the output bin, then two that align with it.
    narrow_edges = np.array([0.0, 0.001, 0.002, 0.5, 1.0])
    probs = np.array([[0.4, 0.4, 0.1, 0.1]])
    mass = output.rebin(probs, narrow_edges) * output.widths
    assert mass[0, 0] == pytest.approx(0.9)  # both narrow bins survive, whole
    assert mass[0, 1] == pytest.approx(0.1)
    assert mass.sum() == pytest.approx(1.0)


def test_rebin_rejects_mismatched_edges():
    grid = lazy.Grid.linear(0.0, 1.0, 4)
    with pytest.raises(ValueError, match="one more entry"):
        grid.rebin(np.ones((2, 5)), np.linspace(0, 1, 5))


def test_from_quantiles_recovers_a_known_uniform_distribution():
    grid = lazy.Grid.linear(0.0, 1.0, 100)
    levels = np.linspace(0.0, 1.0, 101)[1:-1]
    values = np.tile(levels, (3, 1))  # CDF of U(0, 1) is the identity
    density = grid.from_quantiles(values, levels)
    assert np.allclose(density, 1.0, atol=0.05)


def test_from_quantiles_conserves_mass_for_a_narrow_distribution():
    grid = lazy.Grid.linear(0.0, 2.0, 200)
    levels = np.linspace(0.0, 1.0, 101)[1:-1]
    values = np.tile(0.5 + 1e-3 * (levels - 0.5), (2, 1))  # all mass in one bin
    density = grid.from_quantiles(values, levels)
    assert np.trapezoid(density, grid.centers, axis=1) == pytest.approx(1.0)


def test_cdf_starts_at_zero_and_ends_at_one():
    grid = lazy.Grid.linear(0.0, 2.0, 200)
    density = grid.normalize(np.ones((3, 200)))
    cdf = grid.cdf(density)
    assert np.allclose(cdf[:, 0], 0.0)
    assert cdf[:, -1] == pytest.approx(1.0, abs=1e-12)


def test_as_grid_accepts_none_centers_and_grid():
    assert lazy.as_grid(None) is lazy.DC1_GRID
    assert lazy.as_grid(lazy.DC1_GRID) is lazy.DC1_GRID
    assert lazy.as_grid(np.linspace(0.05, 2.95, 30)).n_bins == 30


def test_bin_index_clips_outside_the_grid():
    grid = lazy.Grid.linear(0.0, 1.0, 10)
    assert grid.bin_index([-5.0, 0.05, 0.95, 5.0]).tolist() == [0, 0, 9, 9]


def test_from_centers_refuses_centres_it_would_move():
    """Midpoint edges of [0, 1, 3] would put the middle centre at 1.25."""
    with pytest.raises(ValueError, match="Grid.from_edges"):
        lazy.Grid.from_centers([0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="from_edges"):
        lazy.as_grid(np.geomspace(0.01, 3.0, 50))


def test_from_centers_names_unsorted_centres():
    with pytest.raises(ValueError, match="centers must be strictly increasing"):
        lazy.Grid.from_centers([0.0, 2.0, 1.0])


def test_from_centers_accepts_float_rounded_uniform_centres():
    centers = np.linspace(0.005, 2.995, 300)
    assert np.allclose(lazy.Grid.from_centers(centers).centers, centers)


@pytest.mark.parametrize(
    ("probs", "edges", "match"),
    [
        (np.ones((2, 4)), [0.0, 0.5, np.inf, 1.5, 2.0], "finite"),
        (np.ones((2, 4)), [0.0, 0.5, np.nan, 1.5, 2.0], "finite"),
        (np.ones((2, 4)), [0.0, 1.0, 0.5, 1.5, 2.0], "non-decreasing"),
        ([[1.0, -0.5, 1.0, 1.0]], [0.0, 0.5, 1.0, 1.5, 2.0], "non-negative"),
        ([[1.0, np.nan, 1.0, 1.0]], [0.0, 0.5, 1.0, 1.5, 2.0], "finite"),
    ],
)
def test_rebin_refuses_what_a_histogram_distribution_refuses(
    probs, edges, match
):
    with pytest.raises(ValueError, match=match):
        lazy.Grid.linear(0.0, 2.0, 10).rebin(probs, edges)


def test_rebin_accepts_zero_width_input_bins_without_mass():
    edges = [0.0, 0.5, 0.5, 1.5, 2.0]
    density = lazy.Grid.linear(0.0, 2.0, 10).rebin(
        [[1.0, 0.0, 1.0, 1.0]], edges
    )
    assert np.isfinite(density).all()


def test_cdf_column_zero_follows_the_normalisation_convention():
    """Zero by the trapezoid rule; half the first bin on a histogram grid."""
    density = np.random.default_rng(1).uniform(size=(3, 8))
    trapezoid = lazy.Grid.linear(0.0, 2.0, 8)
    np.testing.assert_array_equal(trapezoid.cdf(density)[:, 0], 0.0)
    histogram = lazy.Grid.linear(0.0, 2.0, 8, normalization="histogram")
    np.testing.assert_allclose(
        histogram.cdf(density)[:, 0], 0.5 * density[:, 0] * 0.25
    )
    assert "histogram grid" in lazy.Grid.cdf.__doc__
    assert "histogram grid" in lazy.BaseDensityRegressor.predict_cdf.__doc__
