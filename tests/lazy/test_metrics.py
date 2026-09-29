# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import numpy as np
import pytest
from scipy import stats

from lazy import metrics

GRID = np.linspace(0.0, 2.0, 401)


def _gaussian_grid_pdfs(centres, sigma):
    return stats.norm.pdf(GRID[None, :], loc=centres[:, None], scale=sigma)


def test_point_metrics_match_challenge_definitions():
    rng = np.random.default_rng(1)
    z_true = rng.uniform(0.1, 1.5, 20000)
    ez = rng.normal(0.01, 0.02, z_true.size)
    ez[:400] = 0.5  # gross outliers
    z_pred = z_true + ez * (1 + z_true)
    result = metrics.point_metrics(z_true, z_pred)
    assert result.bias == pytest.approx(0.01, abs=1e-3)
    assert result.sigma_mad == pytest.approx(0.02, abs=2e-3)
    assert result.sigma_iqr == pytest.approx(0.02, abs=2e-3)
    assert result.outlier_threshold == pytest.approx(
        max(0.06, 3 * result.sigma_iqr)
    )
    # injected outliers plus the N(0.01, 0.02) tail beyond |ez| > 0.06
    tail = stats.norm.sf((0.06 - 0.01) / 0.02) + stats.norm.cdf(
        (-0.06 - 0.01) / 0.02
    )
    assert result.outlier_rate == pytest.approx(0.02 + 0.98 * tail, abs=3e-3)
    assert result.outlier_rate_015 == pytest.approx(0.02, abs=1e-3)


def test_outlier_threshold_has_006_floor():
    z_true = np.linspace(0.2, 1.0, 1000)
    z_pred = z_true * (1 + 1e-4 * np.sign(np.sin(np.arange(1000))))
    assert metrics.point_metrics(z_true, z_pred).outlier_threshold == 0.06


def test_normalization_and_point_estimates():
    centres = np.array([0.3, 0.9, 1.5])
    _, density = metrics.normalize_grid_pdfs(
        GRID, 5.0 * _gaussian_grid_pdfs(centres, 0.05)
    )
    np.testing.assert_allclose(
        np.trapezoid(density, GRID, axis=1), 1.0, atol=1e-10
    )
    estimates = metrics.grid_point_estimates(GRID, density)
    for key in ("z_peak", "z_weight", "z_mean", "z_median"):
        np.testing.assert_allclose(estimates[key], centres, atol=0.006)


def test_calibrated_gaussians_give_uniform_pit_and_correct_cde_loss():
    rng = np.random.default_rng(2)
    sigma = 0.05
    centres = rng.uniform(0.4, 1.6, 20000)
    z_true = rng.normal(centres, sigma)
    pdfs = _gaussian_grid_pdfs(centres, sigma)
    scores, pit = metrics.pdf_metrics(z_true, GRID, pdfs)
    assert scores.pit_ks < 0.012
    assert scores.pit_ks_pvalue > 0.01
    assert scores.pit_rmse < 0.01
    assert scores.pit_kl < 0.005
    assert abs(scores.pit_cvm) < 0.5
    assert scores.pit_ad1 < 3.0 and scores.pit_ad2 < 5.0
    # E[int p^2] - 2 E[p(z_true)] for the true Gaussian model:
    expected = 1 / (2 * np.sqrt(np.pi) * sigma) - 2 / (
        2 * np.sqrt(np.pi) * sigma
    )
    assert scores.cde_loss == pytest.approx(expected, rel=0.02)
    assert np.all((pit >= 0) & (pit <= 1))


def test_cde_loss_prefers_correct_width():
    rng = np.random.default_rng(3)
    centres = rng.uniform(0.4, 1.6, 5000)
    z_true = rng.normal(centres, 0.05)
    right = metrics.cde_loss(z_true, GRID, _gaussian_grid_pdfs(centres, 0.05))
    too_wide = metrics.cde_loss(z_true, GRID, _gaussian_grid_pdfs(centres, 0.2))
    too_narrow = metrics.cde_loss(
        z_true, GRID, _gaussian_grid_pdfs(centres, 0.01)
    )
    assert right < too_wide and right < too_narrow


def test_anderson_darling_is_small_for_uniform_and_large_for_skewed():
    rng = np.random.default_rng(4)
    uniform = rng.uniform(size=20000)
    skewed = rng.beta(2.0, 5.0, size=20000)
    for cut in metrics.AD_CUTS:
        assert metrics.anderson_darling_dc1(uniform, *cut) < 3.0
        assert metrics.anderson_darling_dc1(skewed, *cut) > 100.0


def test_z_weight_ignores_a_secondary_peak():
    """z_WEIGHT averages only over the main peak.

    So a far secondary peak that drags the full-PDF mean must leave it alone.
    """
    grid = GRID
    main = stats.norm.pdf(grid, 0.5, 0.03)
    secondary = 0.3 * stats.norm.pdf(grid, 1.6, 0.03)
    _, dens = metrics.normalize_grid_pdfs(grid, (main + secondary)[None, :])
    est = metrics.grid_point_estimates(grid, dens)
    assert est["z_weight"][0] == pytest.approx(0.5, abs=0.01)
    assert est["z_mean"][0] > 0.7  # the secondary peak moves the mean


def test_cde_loss_uses_the_nearest_grid_point():
    """DC1 and Cal-PIT read the density at the nearest grid point.

    That is the exact bin value for a piecewise-constant density.
    """
    grid = np.array([0.0, 1.0, 2.0])
    pdfs = np.array([[0.0, 1.0, 0.0]])
    pdfs = metrics.normalize_grid_pdfs(grid, pdfs)[1]
    # 0.6 is nearest to grid point 1.0, so the likelihood is that bin's density
    at_truth = (
        np.trapezoid(pdfs[0] ** 2, grid)
        - metrics.cde_loss(np.array([0.6]), grid, pdfs)
    ) / 2
    assert at_truth == pytest.approx(pdfs[0, 1])


def test_a_non_finite_truth_is_refused_rather_than_scored():
    """A NaN truth used to match the last grid point and score finitely."""
    centers = np.linspace(0.005, 1.995, 200)
    pdfs = np.ones((2, 200))
    with pytest.raises(ValueError, match="non-finite"):
        metrics.cde_loss(np.array([0.5, np.nan]), centers, pdfs)
