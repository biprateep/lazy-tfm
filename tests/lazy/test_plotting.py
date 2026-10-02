# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Smoke tests: the diagnostics must draw cleanly and honour a given axes."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import lazy
from lazy import grid as grid_lib
from lazy import metrics
from lazy import plotting


@pytest.fixture
def predictions():
    generator = np.random.default_rng(0)
    y_true = generator.uniform(0.1, 1.9, 300)
    centers = y_true + generator.normal(0.0, 0.05, 300)
    z = lazy.datasets.DC1_GRID.centers[None, :]
    pdfs = lazy.datasets.DC1_GRID.normalize(
        np.exp(-0.5 * ((z - centers[:, None]) / 0.08) ** 2)
    )
    _, _, pit = metrics.evaluate_grid_pdfs(
        y_true, lazy.datasets.DC1_GRID.centers, pdfs
    )
    return y_true, pdfs, pit


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
    y_true, pdfs, _ = predictions
    y_pred = lazy.datasets.DC1_GRID.centers[np.argmax(pdfs, axis=1)]
    assert plotting.plot_actual_vs_predicted(y_true, y_pred).get_xlabel()
    assert plotting.plot_residuals(y_true, y_pred).get_ylabel()


def test_nz_plot_overlays_the_truth(predictions):
    y_true, pdfs, _ = predictions
    ax = plotting.plot_stacked_pdfs(
        lazy.datasets.DC1_GRID.centers, pdfs, y_true=y_true
    )
    assert ax.lines
    assert ax.patches or len(ax.lines) >= 2


def test_pdf_examples_make_one_axes_per_galaxy(predictions):
    _, pdfs, _ = predictions
    axes = plotting.plot_pdfs(lazy.datasets.DC1_GRID.centers, pdfs, n_objects=4)
    assert sum(bool(a.get_visible()) for a in axes) == 4


def test_diagnostic_panel_has_four_populated_axes(predictions):
    y_true, pdfs, _ = predictions
    fig = plotting.diagnostic_panel(
        y_true, lazy.datasets.DC1_GRID.centers, pdfs, label="dummy"
    )
    assert len(fig.axes) == 4
    assert fig._suptitle.get_text() == "dummy"


def test_coverage_is_monotone_and_ends_at_one(predictions):
    _, _, pit = predictions
    ax = plotting.plot_coverage(pit)
    empirical = ax.lines[0].get_ydata()
    assert (np.diff(empirical) >= -1e-12).all()
    assert empirical[-1] == pytest.approx(1.0)


@pytest.fixture
def native():
    """Histogram PDFs on a wide, non-uniform grid, like a model's native one."""
    generator = np.random.default_rng(3)
    inner = np.linspace(-0.5, 2.5, 301)
    edges = np.concatenate(
        [-np.geomspace(60.0, 0.6, 40), inner, np.geomspace(2.6, 60.0, 40)]
    )
    native_grid = grid_lib.Grid.from_edges(edges, normalization="histogram")
    y_true = generator.uniform(0.1, 1.9, 300)
    centers = y_true + generator.normal(0.0, 0.05, 300)
    z = native_grid.centers[None, :]
    pdfs = native_grid.normalize(
        np.exp(-0.5 * ((z - centers[:, None]) / 0.08) ** 2)
    )
    return y_true, native_grid, pdfs


def test_nz_on_a_native_grid_is_histogram_exact_and_framed(native):
    y_true, native_grid, pdfs = native
    ax = plotting.plot_stacked_pdfs(native_grid, pdfs, y_true=y_true)
    lo, hi = ax.get_xlim()
    assert -0.5 < lo < 0.1 and 1.9 < hi < 2.5  # not the grid's +-60
    stacked, truth = ax.patches[0], ax.patches[1]
    values, edges = stacked.get_data()[:2]
    np.testing.assert_array_equal(edges, native_grid.edges)
    assert np.sum(values * np.diff(edges)) == pytest.approx(1.0)
    assert len(truth.get_xy()) < 200  # on the truth's own coarse bins


def test_bin_edges_stand_in_for_the_grid(native):
    y_true, native_grid, pdfs = native
    from_grid = plotting.plot_stacked_pdfs(native_grid, pdfs, y_true=y_true)
    from_edges = plotting.plot_stacked_pdfs(
        native_grid.centers, pdfs, y_true=y_true, bin_edges=native_grid.edges
    )
    assert from_grid.get_xlim() == from_edges.get_xlim()
    np.testing.assert_array_equal(
        from_grid.patches[0].get_data()[0], from_edges.patches[0].get_data()[0]
    )


@pytest.mark.parametrize(
    "plot", [plotting.plot_stacked_pdfs, plotting.plot_pdfs, "diagnostic_panel"]
)
def test_bare_non_uniform_centres_are_refused(native, plot):
    y_true, native_grid, pdfs = native
    with pytest.raises(ValueError, match="bin_edges"):
        if plot == "diagnostic_panel":
            plotting.diagnostic_panel(y_true, native_grid.centers, pdfs)
        else:
            plot(native_grid.centers, pdfs)


def test_the_panel_scores_a_native_grid_by_its_histograms(native):
    y_true, native_grid, pdfs = native
    fig = plotting.diagnostic_panel(y_true, native_grid, pdfs)
    _, _, pit = metrics.evaluate_grid_pdfs(
        y_true, native_grid.centers, pdfs, bin_edges=native_grid.edges
    )
    np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), np.sort(pit))
    assert fig.axes[3].get_xlim()[1] < 2.5


def test_pdf_examples_on_a_native_grid_frame_the_mass(native):
    y_true, native_grid, pdfs = native
    axes = plotting.plot_pdfs(native_grid, pdfs, y_true=y_true, indices=[0, 1])
    lo, hi = axes[0].get_xlim()
    assert hi - lo < 3.0
    assert lo <= min(y_true[:2]) and hi >= max(y_true[:2])


def test_the_point_plot_frames_any_range_and_skips_nan_pairs():
    z = np.linspace(-7.0, -5.0, 50)
    ax = plotting.plot_actual_vs_predicted(z, z)
    assert ax.get_xlim() == ax.get_ylim() == (-7.0, -5.0)
    ax = plotting.plot_actual_vs_predicted(
        z + 1e4, z + 1e4, outlier_lines=False
    )
    assert ax.get_xlim() == (1e4 - 7.0, 1e4 - 5.0)
    predicted = z.copy()
    predicted[3] = np.nan
    truth = z.copy()
    truth[0] = np.nan
    ax = plotting.plot_actual_vs_predicted(truth, predicted)
    assert ax.get_xlim() == (z[1], z[-1])
    with pytest.raises(ValueError, match="finite"):
        plotting.plot_actual_vs_predicted([np.nan], [1.0])


def test_residuals_are_plain_unless_scaled_by_one_plus_y():
    y_true = np.array([-1.0, 0.5, 1.0, 2.0])
    y_pred = y_true + 0.1
    ax = plotting.plot_residuals(y_true, y_pred, n_bins=2)
    np.testing.assert_allclose(ax.lines[-1].get_ydata(), 0.1)
    with pytest.raises(ValueError, match="scale='none'"):
        plotting.plot_residuals(y_true, y_pred, scale="1+y")
    ax = plotting.plot_residuals(y_true[1:], y_pred[1:], scale="1+y", n_bins=1)
    assert ax.lines[-1].get_ydata()[0] == pytest.approx(0.1 / 2.0)
    with pytest.raises(ValueError, match="scale must be"):
        plotting.plot_residuals(y_true, y_pred, scale="log")


def test_a_small_sample_is_not_a_blank_point_plot():
    z = np.array([0.2, 0.5, 0.9, 1.3])
    ax = plotting.plot_actual_vs_predicted(z, z + 0.01)
    mesh = ax.collections[0]
    assert (mesh.norm.vmin, mesh.norm.vmax) == (1.0, 2.0)
    counts = np.ma.masked_invalid(mesh.get_array())
    assert counts.count() == 4  # every object shows, the empty bins do not
    colours = mesh.to_rgba(counts.compressed())
    assert np.all(colours[:, 3] > 0)


@pytest.mark.parametrize("sheet", ["shipped", "fallback"])
def test_use_style_leaves_the_on_screen_dpi_alone(sheet, tmp_path):
    with plt.rc_context({"figure.dpi": 100}):
        if sheet == "shipped":
            plotting.use_style()
        else:
            plotting.use_style(style_file=tmp_path / "missing.mplstyle")
        assert plt.rcParams["figure.dpi"] == 100
        assert plt.rcParams["savefig.dpi"] == 300


def _without_the_paper_font(monkeypatch):
    monkeypatch.setattr(
        plotting.style.font_manager,
        "findfont",
        lambda properties: "/fonts/DejaVuSerif.ttf",
    )


def test_a_missing_paper_font_warns_rather_than_raises(monkeypatch):
    _without_the_paper_font(monkeypatch)
    with plt.rc_context():
        plotting.use_style()
        with pytest.warns(UserWarning, match="Nimbus Roman"):
            report = plotting.verify_style()
        assert report["problems"] and "DejaVu" in report["problems"][0]
        with pytest.raises(RuntimeError, match="Nimbus Roman"):
            plotting.verify_style(strict_font=True)


def test_a_wrong_rcparam_still_raises():
    with plt.rc_context():
        plotting.use_style()
        plt.rcParams["xtick.direction"] = "out"
        with (
            warnings.catch_warnings(),
            pytest.raises(RuntimeError, match="xtick.direction"),
        ):
            warnings.simplefilter("ignore")  # this machine's font, if any
            plotting.verify_style()


def test_one_pdf_can_be_drawn_on_its_own(predictions):
    y_true, pdfs, _ = predictions
    axes = plotting.plot_pdfs(lazy.datasets.DC1_GRID, pdfs[0], y_true=y_true[0])
    assert sum(bool(a.get_visible()) for a in axes) == 1
    assert axes[0].lines


@pytest.mark.parametrize(
    "arguments",
    [{"n_objects": 0}, {"n_objects": -2}, {"n_objects": 1.5}, {"indices": []}],
)
def test_asking_for_no_pdf_is_refused(predictions, arguments):
    _, pdfs, _ = predictions
    with pytest.raises(ValueError, match="n_objects|no PDF"):
        plotting.plot_pdfs(lazy.datasets.DC1_GRID, pdfs, **arguments)


@pytest.mark.parametrize(
    ("name", "written"),
    [
        ("model", "model.png"),
        ("model.v2", "model.v2.png"),
        ("model.pdf", "model.pdf"),
        ("model.SVG", "model.SVG"),
    ],
)
def test_save_adds_png_unless_the_suffix_is_a_format(tmp_path, name, written):
    fig = plt.figure()
    path = plotting.save(fig, tmp_path / "out" / name)
    assert path == tmp_path / "out" / written
    assert path.is_file()
