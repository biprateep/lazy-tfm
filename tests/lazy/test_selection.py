"""The spectroscopic selection function: who gets a redshift, and who is quietly dropped.

The library's own reproduction test -- that this port returns the same galaxies
as the runs the paper reports -- needs the real 14 MB HSC grid and the 1 GB DC1
catalogue, so it is not here. What is here is the behaviour that reproduction
rests on: the pixel lookup, the redshift ceiling, the per-pixel subsampling and
its determinism.
"""

import numpy as np
import pandas as pd
import pytest

from lazy.selection import grid_selection, selection_summary


@pytest.fixture
def raw():
    """Twenty galaxies in the bright pixel, twenty in the faint one."""
    return pd.DataFrame(
        {
            "I": np.repeat([21.0, 23.0], 20),
            "G": np.full(40, 21.5),
            "Z": np.full(40, 21.0),  # G - Z = 0.5 throughout
        }
    )


# -- the grid ---------------------------------------------------------------


def test_pixels_off_the_grid_are_reported_as_such(hsc_grid):
    px, py = hsc_grid.pixel([19.0, 21.0, 25.0], [0.5, 0.5, 0.5])
    assert list(px) == [-1, 0, 2]
    assert list(hsc_grid.inside(px, py)) == [False, True, False]


def test_the_ceiling_is_the_percentile_of_the_spectra_in_that_pixel(hsc_grid):
    """Column 0 is the bright pixel, column 1 the faint one; row 1 has no spectra."""
    ceiling = hsc_grid.max_specz(100.0)
    assert ceiling[0, 0] == pytest.approx(1.0)
    assert ceiling[0, 1] == pytest.approx(0.5)
    assert ceiling[1].tolist() == [0.0, 0.0]


def test_a_pixel_with_no_spectra_keeps_nothing(hsc_grid):
    """Zero is not "no cut" -- it is a ceiling every galaxy fails, which is the point."""
    raw = pd.DataFrame(
        {"I": [21.0], "G": [23.0], "Z": [21.0]}
    )  # colour 2.0: the empty row
    keep, _ = grid_selection(raw, np.array([0.1]), grid=hsc_grid)
    assert not keep.any()


def test_ceiling_tables_are_cached_per_percentile(hsc_grid):
    assert hsc_grid.max_specz(50.0) is hsc_grid.max_specz(50.0)
    assert not np.array_equal(
        hsc_grid.max_specz(50.0), hsc_grid.max_specz(100.0)
    )


# -- the selection ----------------------------------------------------------


def test_the_kept_fraction_follows_the_pixel_ratio(hsc_grid, raw):
    """Half the bright pixel, a tenth of the faint one -- that is the whole bias."""
    keep, _ = grid_selection(
        raw, np.full(40, 0.1), grid=hsc_grid, color_redshift_cut=False
    )
    assert keep[:20].sum() == 10
    assert keep[20:].sum() == 2


def test_the_redshift_ceiling_removes_galaxies_the_photometry_would_have_kept(
    hsc_grid, raw
):
    """The faint pixel's spectra stop at z = 0.5, so nothing beyond it can be targeted."""
    redshift = np.where(np.arange(40) < 20, 0.1, 0.9)
    keep, diagnostics = grid_selection(
        raw, redshift, grid=hsc_grid, percentile_cut=100.0
    )
    assert keep[:20].any()
    assert not keep[20:].any()
    assert diagnostics["z_ceiling"][20] == pytest.approx(0.5)
    assert not diagnostics["survives_z_cut"][20:].any()


def test_the_scaling_factor_cannot_keep_more_galaxies_than_there_are(
    hsc_grid, raw
):
    keep, _ = grid_selection(
        raw, np.full(40, 0.1), grid=hsc_grid, scaling_factor=1000.0
    )
    assert keep[:20].all()


def test_galaxies_at_zero_redshift_are_never_selected(hsc_grid, raw):
    keep, _ = grid_selection(raw, np.zeros(40), grid=hsc_grid)
    assert not keep.any()


def test_diagnostics_say_why_each_galaxy_was_reachable(hsc_grid, raw):
    _, diagnostics = grid_selection(raw, np.full(40, 0.1), grid=hsc_grid)
    assert list(diagnostics.columns) == ["ratio", "z_ceiling", "survives_z_cut"]
    assert diagnostics["ratio"][:20].eq(0.5).all()
    assert diagnostics["ratio"][20:].eq(0.1).all()


def test_the_same_seed_gives_the_same_galaxies(hsc_grid, raw):
    redshift = np.full(40, 0.1)
    first, _ = grid_selection(
        raw, redshift, grid=hsc_grid, color_redshift_cut=False
    )
    again, _ = grid_selection(
        raw, redshift, grid=hsc_grid, color_redshift_cut=False
    )
    other, _ = grid_selection(
        raw, redshift, grid=hsc_grid, color_redshift_cut=False, seed=7
    )
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_other_bands_can_define_the_grid_axes(hsc_grid, raw):
    renamed = raw.rename(columns={"I": "i_mag", "G": "g_mag", "Z": "z_mag"})
    keep, _ = grid_selection(
        renamed,
        np.full(40, 0.1),
        grid=hsc_grid,
        magnitude="i_mag",
        color=("g_mag", "z_mag"),
        color_redshift_cut=False,
    )
    assert keep.sum() == 12


def test_missing_columns_are_reported(hsc_grid, raw):
    with pytest.raises(KeyError, match="'Z'"):
        grid_selection(raw.drop(columns=["Z"]), np.full(40, 0.1), grid=hsc_grid)


def test_a_redshift_array_of_the_wrong_length_is_reported(hsc_grid, raw):
    with pytest.raises(ValueError, match="40 rows"):
        grid_selection(raw, np.full(3, 0.1), grid=hsc_grid)


# -- reporting --------------------------------------------------------------


def test_the_summary_reports_the_selected_fraction_per_bin(hsc_grid, raw):
    keep, _ = grid_selection(
        raw, np.full(40, 0.1), grid=hsc_grid, color_redshift_cut=False
    )
    summary = selection_summary(keep, raw["I"].to_numpy(), np.full(40, 0.1))
    bright = summary.loc[summary.bin == "i in [21.0, 22.0)"].iloc[0]
    faint = summary.loc[summary.bin == "i in [23.0, 24.0)"].iloc[0]
    assert bright.fraction == pytest.approx(0.5)
    assert faint.fraction == pytest.approx(0.1)


def test_empty_bins_do_not_divide_by_zero(hsc_grid):
    summary = selection_summary(
        np.array([True]), np.array([21.0]), np.array([0.1])
    )
    assert np.isfinite(summary.fraction).all()
    assert (summary.loc[summary.n_all == 0, "fraction"] == 0).all()
