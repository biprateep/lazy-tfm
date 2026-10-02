# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lazy import grid as grid_lib
from lazy import metrics

GRID = np.linspace(0.0, 2.0, 401)


def _gaussian_grid_pdfs(centres, sigma):
    return stats.norm.pdf(GRID[None, :], loc=centres[:, None], scale=sigma)


def test_point_metrics_match_challenge_definitions():
    rng = np.random.default_rng(1)
    y_true = rng.uniform(0.1, 1.5, 20000)
    ez = rng.normal(0.01, 0.02, y_true.size)
    ez[:400] = 0.5  # gross outliers
    y_pred = y_true + ez * (1 + y_true)
    result = metrics.point_metrics(y_true, y_pred, scale="1+y")
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
    y_true = np.linspace(0.2, 1.0, 1000)
    y_pred = y_true * (1 + 1e-4 * np.sign(np.sin(np.arange(1000))))
    assert metrics.point_metrics(y_true, y_pred).outlier_threshold == 0.06


def test_normalization_and_point_estimates():
    centres = np.array([0.3, 0.9, 1.5])
    _, density = metrics.normalize_grid_pdfs(
        GRID, 5.0 * _gaussian_grid_pdfs(centres, 0.05)
    )
    np.testing.assert_allclose(
        np.trapezoid(density, GRID, axis=1), 1.0, atol=1e-10
    )
    estimates = metrics.grid_point_estimates(GRID, density)
    for key in ("mode", "peak_mean", "mean", "median"):
        np.testing.assert_allclose(estimates[key], centres, atol=0.006)


def test_calibrated_gaussians_give_uniform_pit_and_correct_cde_loss():
    rng = np.random.default_rng(2)
    sigma = 0.05
    centres = rng.uniform(0.4, 1.6, 20000)
    y_true = rng.normal(centres, sigma)
    pdfs = _gaussian_grid_pdfs(centres, sigma)
    scores, pit = metrics.pdf_metrics(y_true, GRID, pdfs)
    assert scores.pit_ks < 0.012
    assert scores.pit_ks_pvalue > 0.01
    assert scores.pit_rmse < 0.01
    assert scores.pit_kl < 0.005
    assert abs(scores.pit_cvm) < 0.5
    assert scores.pit_ad1 < 3.0 and scores.pit_ad2 < 5.0
    # E[int p^2] - 2 E[p(y_true)] for the true Gaussian model:
    expected = 1 / (2 * np.sqrt(np.pi) * sigma) - 2 / (
        2 * np.sqrt(np.pi) * sigma
    )
    assert scores.cde_loss == pytest.approx(expected, rel=0.02)
    assert np.all((pit >= 0) & (pit <= 1))


def test_cde_loss_prefers_correct_width():
    rng = np.random.default_rng(3)
    centres = rng.uniform(0.4, 1.6, 5000)
    y_true = rng.normal(centres, 0.05)
    right = metrics.cde_loss(y_true, GRID, _gaussian_grid_pdfs(centres, 0.05))
    too_wide = metrics.cde_loss(y_true, GRID, _gaussian_grid_pdfs(centres, 0.2))
    too_narrow = metrics.cde_loss(
        y_true, GRID, _gaussian_grid_pdfs(centres, 0.01)
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
    assert est["peak_mean"][0] == pytest.approx(0.5, abs=0.01)
    assert est["mean"][0] > 0.7  # the secondary peak moves the mean


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


def _histogram_case():
    edges = np.concatenate([[0.0], np.cumsum(np.geomspace(0.002, 0.2, 40))])
    grid = grid_lib.Grid.from_edges(edges, normalization="histogram")
    rng = np.random.default_rng(3)
    pdfs = rng.gamma(2.0, size=(25, grid.n_bins))
    y_true = rng.uniform(grid.y_min, grid.y_max, 25)
    return grid, pdfs, y_true


def test_a_histogram_grid_brings_its_own_edges_to_every_metric():
    grid, pdfs, y_true = _histogram_case()
    edges = grid.edges
    pd.testing.assert_frame_equal(
        metrics.summarize(y_true, grid, pdfs),
        metrics.summarize(y_true, grid.centers, pdfs, bin_edges=edges),
    )
    assert metrics.cde_loss(y_true, grid, pdfs) == metrics.cde_loss(
        y_true, grid.centers, pdfs, bin_edges=edges
    )
    for got, want in zip(
        metrics.evaluate_grid_at_truth(y_true, grid, pdfs),
        metrics.evaluate_grid_at_truth(
            y_true, grid.centers, pdfs, bin_edges=edges
        ),
    ):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(
        metrics.normalize_grid_pdfs(grid, pdfs)[1], grid.normalize(pdfs)
    )
    density = grid.normalize(pdfs)
    np.testing.assert_array_equal(
        metrics.grid_cdf(grid, density),
        metrics.grid_cdf(grid.centers, density, bin_edges=edges),
    )
    np.testing.assert_allclose(
        metrics.normalization_error(grid, density), 0.0, atol=1e-12
    )
    np.testing.assert_array_equal(
        metrics.peak_mean(grid, density),
        metrics.peak_mean(grid.centers, density, bin_edges=edges),
    )
    np.testing.assert_array_equal(
        metrics.mode(grid, density), metrics.mode(grid.centers, density)
    )
    estimates = metrics.grid_point_estimates(grid, pdfs)
    expected = metrics.grid_point_estimates(grid.centers, pdfs, bin_edges=edges)
    for name, values in expected.items():
        np.testing.assert_array_equal(estimates[name], values)
    assert (
        metrics.pdf_metrics(y_true, grid, pdfs)[0]
        == (metrics.pdf_metrics(y_true, grid.centers, pdfs, bin_edges=edges)[0])
    )


def test_a_trapezoid_grid_scores_like_its_centres():
    grid = grid_lib.Grid.linear(0.0, 2.0, 100)
    rng = np.random.default_rng(4)
    pdfs = rng.gamma(2.0, size=(10, 100))
    y_true = rng.uniform(0.0, 2.0, 10)
    assert metrics.cde_loss(y_true, grid, pdfs) == metrics.cde_loss(
        y_true, grid.centers, pdfs
    )


def test_explicit_bin_edges_must_agree_with_the_grid():
    grid, pdfs, y_true = _histogram_case()
    trapezoid = grid_lib.Grid.from_edges(grid.edges)
    # Matching edges win, and score a trapezoid grid as a histogram.
    assert metrics.cde_loss(
        y_true, trapezoid, pdfs, bin_edges=grid.edges
    ) == metrics.cde_loss(y_true, grid, pdfs)
    with pytest.raises(ValueError, match="differ from the edges"):
        metrics.cde_loss(y_true, grid, pdfs, bin_edges=grid.edges * 1.01)


def test_point_metrics_leave_residuals_unscaled_by_default():
    y_true = np.array([0.0, 1.0, 3.0, 10.0])
    y_pred = np.array([0.1, 1.2, 3.1, 14.0])
    plain = metrics.point_metrics(y_true, y_pred)
    assert plain == metrics.point_metrics(y_true, y_pred, scale="none")
    scaled = metrics.point_metrics(y_true, y_pred, scale="1+y")
    assert scaled.scale == "1+y"
    assert scaled.bias == np.median((y_pred - y_true) / (1 + y_true))
    assert plain.scale == "none"
    residual = y_pred - y_true
    assert plain.bias == np.median(residual)
    assert plain.outlier_rate_015 == np.mean(np.abs(residual) > 0.15)
    assert plain.median_abs_error == np.median(np.abs(residual))
    with pytest.raises(ValueError, match="scale must be one of"):
        metrics.point_metrics(y_true, y_pred, scale="log")


def test_the_summary_table_records_the_residual_scale():
    grid, pdfs, y_true = _histogram_case()
    table = metrics.summarize(y_true, grid, pdfs, scale="1+y")
    assert table.columns[:2].tolist() == ["point_estimate", "scale"]
    assert table["scale"].iloc[0] == "1+y"
    plain = metrics.summarize(y_true, grid, pdfs)
    assert plain["scale"].iloc[0] == "none"
    y_pred = metrics.grid_point_estimates(grid, pdfs)["mode"]
    assert plain["bias"].iloc[0] == np.median(y_pred - y_true)
    pdf_columns = ["cde_loss", "pit_ks", "pit_ad1"]
    pd.testing.assert_frame_equal(plain[pdf_columns], table[pdf_columns])


FINE_GRID = np.linspace(0.0, 2.0, 4001)


def _gaussian_crps(y_true, mean, sigma):
    z = (y_true - mean) / sigma
    return sigma * (
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )


def test_crps_of_a_gaussian_matches_the_closed_form():
    rng = np.random.default_rng(5)
    sigma = 0.05
    centres = rng.uniform(0.5, 1.5, 200)
    y_true = centres + rng.normal(0.0, 2.0 * sigma, centres.size)
    pdfs = stats.norm.pdf(FINE_GRID[None, :], loc=centres[:, None], scale=sigma)
    expected = _gaussian_crps(y_true, centres, sigma)
    np.testing.assert_allclose(
        metrics.per_object_crps(y_true, FINE_GRID, pdfs),
        expected,
        rtol=1e-5,
        atol=1e-7,
    )
    assert metrics.crps(y_true, FINE_GRID, pdfs) == pytest.approx(
        expected.mean(), rel=1e-5
    )


def test_crps_of_a_point_mass_is_the_absolute_error():
    # All mass in one histogram bin of width 1e-6 at 1.0.
    edges = np.array([0.0, 1.0, 1.0 + 1e-6, 2.0])
    grid = grid_lib.Grid.from_edges(edges, normalization="histogram")
    pdfs = np.array([[0.0, 1.0, 0.0]] * 3)
    y_true = np.array([0.25, 1.75, 3.0])
    np.testing.assert_allclose(
        metrics.per_object_crps(y_true, grid, pdfs),
        np.abs(y_true - 1.0),
        atol=1e-6,
    )


def test_crps_is_exact_on_a_histogram_grid():
    # U(0, 3) as two unequal bins: the CDF is linear, and the CRPS at t is
    # ((t - a)^3 + (b - t)^3) / (3 L^2) inside, plus the distance outside.
    grid = grid_lib.Grid.from_edges([0.0, 1.0, 3.0], normalization="histogram")
    y_true = np.array([0.5, 2.2, -1.0, 4.0])
    inside = np.clip(y_true, 0.0, 3.0)
    expected = (inside**3 + (3.0 - inside) ** 3) / 27.0 + np.abs(
        y_true - inside
    )
    np.testing.assert_allclose(
        metrics.per_object_crps(y_true, grid, np.ones((4, 2))),
        expected,
        rtol=1e-14,
    )


def test_crps_on_a_random_histogram_matches_brute_force():
    grid, pdfs, y_true = _histogram_case()
    y_true[:3] = [-0.5, grid.y_max + 0.3, grid.edges[7]]
    density = grid.normalize(pdfs)
    cdf_at_edges = np.column_stack(
        (np.zeros(len(density)), np.cumsum(density * grid.widths, axis=1))
    )
    expected = []
    for row, truth in zip(cdf_at_edges, y_true):
        # Midpoint sums on a fine grid holding the step and every kink.
        fine = np.union1d(
            np.linspace(-1.0, grid.y_max + 1.0, 200001),
            np.append(grid.edges, truth),
        )
        middles = 0.5 * (fine[1:] + fine[:-1])
        cdf = np.interp(middles, grid.edges, row, left=0.0, right=1.0)
        step = (middles >= truth).astype(float)
        expected.append(np.sum((cdf - step) ** 2 * np.diff(fine)))
    np.testing.assert_allclose(
        metrics.per_object_crps(y_true, grid, pdfs), expected, rtol=1e-8
    )
    assert metrics.crps(y_true, grid, pdfs) == metrics.crps(
        y_true, grid.centers, pdfs, bin_edges=grid.edges
    )


def test_crps_prefers_the_correct_width():
    rng = np.random.default_rng(6)
    centres = rng.uniform(0.4, 1.6, 5000)
    y_true = rng.normal(centres, 0.05)
    right = metrics.crps(y_true, GRID, _gaussian_grid_pdfs(centres, 0.05))
    for sigma in (0.01, 0.2):
        wrong = metrics.crps(y_true, GRID, _gaussian_grid_pdfs(centres, sigma))
        assert right < wrong


def test_nll_of_a_gaussian_is_minus_its_log_density():
    sigma = 0.05
    centres = np.array([0.5, 1.0, 1.4])
    # Truths on grid points, where the nearest-point density is exact.
    y_true = FINE_GRID[[1100, 2050, 3100]]
    pdfs = stats.norm.pdf(FINE_GRID[None, :], loc=centres[:, None], scale=sigma)
    expected = -stats.norm.logpdf(y_true, centres, sigma)
    np.testing.assert_allclose(
        metrics.per_object_nll(y_true, FINE_GRID, pdfs), expected, rtol=1e-9
    )
    assert metrics.nll(y_true, FINE_GRID, pdfs) == pytest.approx(
        expected.mean(), rel=1e-9
    )


def test_nll_reads_the_density_the_cde_loss_reads():
    grid, pdfs, y_true = _histogram_case()
    _, _, at_truth, _ = metrics.evaluate_grid_at_truth(y_true, grid, pdfs)
    np.testing.assert_allclose(
        metrics.per_object_nll(y_true, grid, pdfs), -np.log(at_truth)
    )


def test_nll_clips_a_zero_density_at_the_floor():
    grid = grid_lib.Grid.linear(0.0, 2.0, 100, normalization="histogram")
    pdfs = np.ones((2, 100))
    y_true = np.array([1.0, 5.0])  # The second lies off the grid.
    scores = metrics.per_object_nll(y_true, grid, pdfs)
    assert scores[0] == pytest.approx(np.log(2.0))
    assert scores[1] == pytest.approx(-np.log(metrics.NLL_DENSITY_FLOOR))
    assert np.isfinite(metrics.nll(y_true, grid, pdfs))
    assert metrics.per_object_nll(y_true, grid, pdfs, floor=1e-3)[
        1
    ] == pytest.approx(-np.log(1e-3))
    with pytest.raises(ValueError, match="floor must be positive"):
        metrics.nll(y_true, grid, pdfs, floor=0.0)


def test_pinball_loss_matches_a_hand_computation():
    y_true = np.array([1.0, 2.0])
    quantiles = np.array([[0.5, 1.5], [2.5, 3.0]])
    levels = np.array([0.1, 0.9])
    # Residuals y - q: [[0.5, -0.5], [-0.5, -1.0]].
    # Losses: [[0.1 * 0.5, 0.1 * 0.5], [0.9 * 0.5, 0.1 * 1.0]].
    expected = (0.05 + 0.05 + 0.45 + 0.1) / 4
    assert metrics.pinball_loss(y_true, quantiles, levels) == pytest.approx(
        expected
    )
    # One level, as a 1-D array of quantiles and a scalar level.
    assert metrics.pinball_loss(y_true, quantiles[:, 0], 0.1) == pytest.approx(
        (0.05 + 0.45) / 2
    )
    assert metrics.pinball_loss(
        y_true, quantiles[:, 0], [0.1]
    ) == metrics.pinball_loss(y_true, quantiles[:, :1], [0.1])
    # At the median it is half the absolute error.
    assert metrics.pinball_loss(y_true, quantiles[:, 1], 0.5) == pytest.approx(
        0.5 * np.mean(np.abs(y_true - quantiles[:, 1]))
    )


def test_pinball_loss_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(7)
    y_true = rng.normal(0.0, 1.0, 200000)
    for tau in (0.1, 0.5, 0.84):
        true_quantile = stats.norm.ppf(tau)
        shifts = np.linspace(-0.5, 0.5, 21)
        losses = [
            metrics.pinball_loss(
                y_true, np.full(y_true.size, true_quantile + shift), tau
            )
            for shift in shifts
        ]
        assert abs(shifts[np.argmin(losses)]) < 0.06


def test_twice_the_mean_pinball_loss_over_levels_is_the_crps():
    sigma = 0.05
    centres = np.array([0.6, 1.0, 1.3])
    y_true = np.array([0.65, 0.9, 1.3])
    levels = (np.arange(20000) + 0.5) / 20000
    quantiles = stats.norm.ppf(
        levels[None, :], loc=centres[:, None], scale=sigma
    )
    pdfs = stats.norm.pdf(FINE_GRID[None, :], loc=centres[:, None], scale=sigma)
    assert 2 * metrics.pinball_loss(y_true, quantiles, levels) == pytest.approx(
        metrics.crps(y_true, FINE_GRID, pdfs), rel=1e-4
    )


def test_pinball_loss_validates_its_inputs():
    y_true = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="one column per level"):
        metrics.pinball_loss(y_true, np.zeros((2, 3)), [0.1, 0.9])
    with pytest.raises(ValueError, match="one row per truth"):
        metrics.pinball_loss(y_true, np.zeros((3, 2)), [0.1, 0.9])
    with pytest.raises(ValueError, match="within"):
        metrics.pinball_loss(y_true, np.zeros((2, 1)), [1.5])
    with pytest.raises(ValueError, match="non-finite"):
        metrics.pinball_loss(y_true, np.array([0.0, np.nan]), 0.5)
    with pytest.raises(ValueError, match="1D"):
        metrics.pinball_loss(np.zeros((2, 1)), np.zeros((2, 1)), [0.5])


def test_crps_and_nll_validate_their_inputs():
    pdfs = np.ones((2, GRID.size))
    with pytest.raises(ValueError, match="one value per PDF"):
        metrics.crps(np.array([0.5]), GRID, pdfs)
    with pytest.raises(ValueError, match="non-finite"):
        metrics.nll(np.array([0.5, np.inf]), GRID, pdfs)
    with pytest.raises(ValueError, match="shape"):
        metrics.crps(np.array([0.5, 1.0]), GRID, pdfs[:, :-1])


def test_the_summary_table_adds_crps_and_nll():
    grid, pdfs, y_true = _histogram_case()
    table = metrics.summarize(y_true, grid, pdfs)
    assert table.columns[-2:].tolist() == ["crps", "nll"]
    assert table["crps"].iloc[0] == metrics.crps(y_true, grid, pdfs)
    assert table["nll"].iloc[0] == metrics.nll(y_true, grid, pdfs)
    scores, _ = metrics.pdf_metrics(y_true, grid, pdfs)
    assert scores.crps == table["crps"].iloc[0]
    assert scores.nll == table["nll"].iloc[0]
    y_pred = metrics.grid_point_estimates(grid, pdfs)["mode"]
    cde_terms, pit = metrics.per_object_scores(y_true, grid, pdfs)
    pd.testing.assert_frame_equal(
        metrics.summarize_scores(
            y_true,
            y_pred,
            cde_terms,
            pit,
            crps_terms=metrics.per_object_crps(y_true, grid, pdfs),
            nll_terms=metrics.per_object_nll(y_true, grid, pdfs),
        ),
        table,
    )
    # Without the per-object terms the columns are there, as NaN.
    partial = metrics.summarize_scores(y_true, y_pred, cde_terms, pit)
    assert partial.columns.tolist() == table.columns.tolist()
    assert partial[["crps", "nll"]].isna().all(axis=None)
    with pytest.raises(ValueError, match="one value per object"):
        metrics.summarize_scores(
            y_true, y_pred, cde_terms, pit, crps_terms=np.zeros(3)
        )
