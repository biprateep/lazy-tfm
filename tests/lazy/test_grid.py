import numpy as np
import pytest

from lazy.grid import as_grid
from lazy.grid import DC1_GRID
from lazy.grid import RedshiftGrid


def test_dc1_grid_matches_the_challenge_output_format():
    assert DC1_GRID.n_bins == 200
    assert DC1_GRID.z_min == 0.0
    assert DC1_GRID.z_max == 2.0
    assert np.allclose(DC1_GRID.widths, 0.01)


def test_from_centers_round_trips_a_uniform_grid():
    assert RedshiftGrid.from_centers(DC1_GRID.centers) == DC1_GRID


def test_edges_must_increase():
    with pytest.raises(ValueError, match="strictly increasing"):
        RedshiftGrid(np.array([0.0, 1.0, 0.5]))


def test_edges_are_immutable():
    grid = RedshiftGrid.linear(0.0, 1.0, 4)
    with pytest.raises(ValueError):
        grid.edges[0] = -1.0


def test_normalize_gives_unit_trapezoid_mass():
    grid = RedshiftGrid.linear(0.0, 2.0, 200)
    pdfs = np.random.default_rng(0).uniform(size=(5, 200))
    mass = np.trapezoid(grid.normalize(pdfs), grid.centers, axis=1)
    assert np.allclose(mass, 1.0)


def test_normalize_rescues_a_zero_row_instead_of_returning_nan():
    grid = RedshiftGrid.linear(0.0, 2.0, 200)
    pdfs = np.zeros((2, 200))
    pdfs[1] = 1.0
    out = grid.normalize(pdfs)
    assert np.isfinite(out).all()
    assert np.allclose(np.trapezoid(out, grid.centers, axis=1), 1.0)


def test_rebin_conserves_mass_onto_a_coarser_grid():
    """The whole point of rebin: no probability is created or lost."""
    fine = RedshiftGrid.linear(0.0, 2.0, 1000)
    coarse = RedshiftGrid.linear(0.0, 2.0, 50)
    probs = np.random.default_rng(0).uniform(size=(4, 1000))
    probs /= probs.sum(axis=1, keepdims=True)
    density = coarse.rebin(probs, fine.edges)
    assert np.allclose((density * coarse.widths).sum(axis=1), 1.0)


def test_rebin_keeps_mass_from_bins_narrower_than_the_output_bin():
    """Equal-mass bins are routinely narrower than an output bin; sampling the
    density at output centres would drop them, integrating cannot."""
    output = RedshiftGrid.linear(0.0, 1.0, 2)  # two 0.5-wide bins
    # Two bins 500x narrower than the output bin, then two that align with it.
    narrow_edges = np.array([0.0, 0.001, 0.002, 0.5, 1.0])
    probs = np.array([[0.4, 0.4, 0.1, 0.1]])
    mass = output.rebin(probs, narrow_edges) * output.widths
    assert mass[0, 0] == pytest.approx(0.9)  # both narrow bins survive, whole
    assert mass[0, 1] == pytest.approx(0.1)
    assert mass.sum() == pytest.approx(1.0)


def test_rebin_rejects_mismatched_edges():
    grid = RedshiftGrid.linear(0.0, 1.0, 4)
    with pytest.raises(ValueError, match="one more entry"):
        grid.rebin(np.ones((2, 5)), np.linspace(0, 1, 5))


def test_from_quantiles_recovers_a_known_uniform_distribution():
    grid = RedshiftGrid.linear(0.0, 1.0, 100)
    levels = np.linspace(0.0, 1.0, 101)[1:-1]
    values = np.tile(levels, (3, 1))  # CDF of U(0, 1) is the identity
    density = grid.from_quantiles(values, levels)
    assert np.allclose(density, 1.0, atol=0.05)


def test_from_quantiles_conserves_mass_for_a_narrow_distribution():
    grid = RedshiftGrid.linear(0.0, 2.0, 200)
    levels = np.linspace(0.0, 1.0, 101)[1:-1]
    values = np.tile(0.5 + 1e-3 * (levels - 0.5), (2, 1))  # all mass in one bin
    density = grid.from_quantiles(values, levels)
    assert np.trapezoid(density, grid.centers, axis=1) == pytest.approx(1.0)


def test_cdf_starts_at_zero_and_ends_at_one():
    grid = RedshiftGrid.linear(0.0, 2.0, 200)
    density = grid.normalize(np.ones((3, 200)))
    cdf = grid.cdf(density)
    assert np.allclose(cdf[:, 0], 0.0)
    assert cdf[:, -1] == pytest.approx(1.0, abs=1e-12)


def test_as_grid_accepts_none_centers_and_grid():
    assert as_grid(None) is DC1_GRID
    assert as_grid(DC1_GRID) is DC1_GRID
    assert as_grid(np.linspace(0.05, 2.95, 30)).n_bins == 30


def test_bin_index_clips_outside_the_grid():
    grid = RedshiftGrid.linear(0.0, 1.0, 10)
    assert grid.bin_index([-5.0, 0.05, 0.95, 5.0]).tolist() == [0, 0, 9, 9]
