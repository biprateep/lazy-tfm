# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Smoke tests: the diagnostics must draw cleanly and honour a given axes."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import lazy
from lazy import metrics
from lazy import plotting


@pytest.fixture
def predictions():
    generator = np.random.default_rng(0)
    z_true = generator.uniform(0.1, 1.9, 300)
    centers = z_true + generator.normal(0.0, 0.05, 300)
    z = lazy.DC1_GRID.centers[None, :]
    pdfs = lazy.DC1_GRID.normalize(
        np.exp(-0.5 * ((z - centers[:, None]) / 0.08) ** 2)
    )
    _, _, pit = metrics.evaluate_grid_pdfs(z_true, lazy.DC1_GRID.centers, pdfs)
    return z_true, pdfs, pit


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_use_style_sets_the_house_rcparams():
    plotting.use_style()
    assert plt.rcParams["xtick.direction"] == "in"
    assert plt.rcParams["legend.frameon"] is False
    assert plt.rcParams["font.family"] == ["serif"]


def test_figsize_is_column_width_by_default():
    width, height = plotting.figsize()
    assert 3.3 < width < 3.4  # AASTeX column, in inches
    assert height < width


@pytest.mark.parametrize(
    "plot", [plotting.plot_pit, plotting.plot_pit_qq, plotting.plot_coverage]
)
def test_calibration_plots_draw_into_a_given_axes(predictions, plot):
    _, _, pit = predictions
    _, ax = plt.subplots()
    assert plot(pit, ax=ax) is ax
    assert ax.lines or ax.patches


def test_point_plots_draw(predictions):
    z_true, pdfs, _ = predictions
    z_pred = lazy.DC1_GRID.centers[np.argmax(pdfs, axis=1)]
    assert plotting.plot_zphot_ztrue(z_true, z_pred).get_xlabel()
    assert plotting.plot_residuals(z_true, z_pred).get_ylabel()


def test_nz_plot_overlays_the_truth(predictions):
    z_true, pdfs, _ = predictions
    ax = plotting.plot_nz(lazy.DC1_GRID.centers, pdfs, z_true=z_true)
    assert ax.lines
    assert ax.patches or len(ax.lines) >= 2


def test_pdf_examples_make_one_axes_per_galaxy(predictions):
    _, pdfs, _ = predictions
    axes = plotting.plot_pdfs(lazy.DC1_GRID.centers, pdfs, n=4)
    assert sum(bool(a.get_visible()) for a in axes) == 4


def test_diagnostic_panel_has_four_populated_axes(predictions):
    z_true, pdfs, _ = predictions
    fig = plotting.diagnostic_panel(
        z_true, lazy.DC1_GRID.centers, pdfs, label="dummy"
    )
    assert len(fig.axes) == 4
    assert fig._suptitle.get_text() == "dummy"


def test_coverage_is_monotone_and_ends_at_one(predictions):
    _, _, pit = predictions
    ax = plotting.plot_coverage(pit)
    empirical = ax.lines[0].get_ydata()
    assert (np.diff(empirical) >= -1e-12).all()
    assert empirical[-1] == pytest.approx(1.0)
