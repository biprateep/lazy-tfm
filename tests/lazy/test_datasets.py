# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Where cached catalogues land, how they come back, and their feature views.

The same code has to run on a laptop, a shared login node and a compute node
with a node-local disk, so every part of the path is overridable and none of it
is hardcoded.
"""

import numpy as np
import pandas as pd
import pytest

from lazy import datasets

EXPECTED_COLUMNS = {
    "mag": 12,
    "mag-color": 12,
    "all": 42,
}


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real home directory or cache while testing."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LAZY_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    (tmp_path / "home").mkdir()
    return tmp_path


def test_default_is_under_the_users_cache(isolated_home):
    assert (
        datasets.data_home()
        == isolated_home / "home" / ".cache" / "lazy-photoz"
    )


def test_xdg_cache_home_is_honoured(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(isolated_home / "xdg"))
    assert datasets.data_home() == isolated_home / "xdg" / "lazy-photoz"


def test_an_empty_xdg_cache_home_falls_back(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert (
        datasets.data_home()
        == isolated_home / "home" / ".cache" / "lazy-photoz"
    )


def test_lazy_data_home_wins_over_xdg(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(isolated_home / "xdg"))
    monkeypatch.setenv("LAZY_DATA_HOME", str(isolated_home / "scratch"))
    assert datasets.data_home() == isolated_home / "scratch"


def test_the_argument_wins_over_everything(isolated_home, monkeypatch):
    monkeypatch.setenv("LAZY_DATA_HOME", str(isolated_home / "scratch"))
    assert (
        datasets.data_home(isolated_home / "explicit")
        == isolated_home / "explicit"
    )


def test_the_directory_is_created(isolated_home):
    assert datasets.data_home(isolated_home / "made" / "deeply").is_dir()


def test_a_missing_cache_is_reported_rather_than_downloaded(isolated_home):
    """`download_if_missing=False` is what an offline compute node passes."""
    with pytest.raises(FileNotFoundError, match="network access"):
        datasets.fetch_dc1(data_home=isolated_home, download_if_missing=False)


def test_catalog_repr_summarises_without_dumping_the_frame():
    catalog = datasets.Catalog(
        split="train",
        raw=pd.DataFrame({"U": np.zeros(3)}),
        redshift=np.zeros(3),
        object_id=np.arange(3),
    )
    assert repr(catalog) == "Catalog(split='train', n=3, columns=['U'])"


# -- catalogues built outside fetch_dc1 ------------------------------------


def test_a_dataframe_of_your_own_becomes_a_catalog(photometry):
    frame = photometry.assign(z=np.linspace(0.1, 1.9, len(photometry)))
    catalog = datasets.Catalog.from_frame(frame, redshift="z")
    assert len(catalog) == len(photometry)
    assert list(catalog.raw.columns) == list(datasets.RAW_COLUMNS)
    assert np.allclose(catalog.redshift, frame["z"])
    assert np.array_equal(catalog.object_id, np.arange(len(photometry)))


def test_truth_and_ids_can_come_as_arrays_instead_of_columns(photometry):
    ids = np.arange(100, 100 + len(photometry))
    catalog = datasets.Catalog.from_frame(
        photometry, redshift=np.zeros(len(photometry)), object_id=ids
    )
    assert np.array_equal(catalog.object_id, ids)
    assert list(catalog.raw.columns) == list(datasets.RAW_COLUMNS)


def test_id_columns_are_moved_out_of_the_features(photometry):
    frame = photometry.assign(z=0.5, objid=np.arange(len(photometry)))
    catalog = datasets.Catalog.from_frame(
        frame, redshift="z", object_id="objid"
    )
    assert "objid" not in catalog.raw.columns
    assert "z" not in catalog.raw.columns


def test_mismatched_lengths_are_reported(photometry):
    with pytest.raises(ValueError, match="rows"):
        datasets.Catalog.from_frame(photometry, redshift=np.zeros(3))


def test_a_catalog_of_your_own_builds_features_the_same_way(photometry):
    catalog = datasets.Catalog.from_frame(
        photometry, redshift=np.zeros(len(photometry))
    )
    assert catalog.features("mag-color").equals(
        datasets.build_features(photometry)
    )


# -- feature modes ----------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(datasets.FEATURE_MODES))
def test_every_mode_builds_the_documented_width(photometry, mode):
    X = datasets.build_features(photometry, mode)
    assert len(X) == len(photometry)
    assert X.shape[1] == EXPECTED_COLUMNS[mode]
    assert np.isfinite(X.to_numpy()).all()


def test_mag_is_the_photometry_untouched(photometry):
    X = datasets.build_features(photometry, "mag")
    assert list(X.columns) == list(datasets.RAW_COLUMNS)
    assert np.allclose(
        X.to_numpy(), photometry[list(datasets.RAW_COLUMNS)].to_numpy()
    )


def test_mag_color_keeps_the_reference_magnitude_and_the_adjacent_colours(
    photometry,
):
    X = datasets.build_features(photometry, "mag-color")
    assert list(X.columns) == ["I", "U-G", "G-R", "R-I", "I-Z", "Z-Y"] + [
        "IERR",
        "U-GERR",
        "G-RERR",
        "R-IERR",
        "I-ZERR",
        "Z-YERR",
    ]
    assert np.allclose(X["G-R"], photometry["G"] - photometry["R"])
    assert np.allclose(X["I"], photometry["I"])


def test_colour_errors_are_the_quadrature_sum_of_the_two_magnitude_errors(
    photometry,
):
    """This is the whole point of carrying colours rather than magnitudes."""
    X = datasets.build_features(photometry, "mag-color")
    assert np.allclose(
        X["G-RERR"], np.hypot(photometry["GERR"], photometry["RERR"])
    )


def test_all_carries_every_colour_and_every_error(photometry):
    X = datasets.build_features(photometry, "all")
    colours = [c for c in X.columns if "-" in c and not c.endswith("ERR")]
    assert len(colours) == 15
    assert list(X.columns[: len(datasets.BANDS)]) == list(datasets.BANDS)
    assert np.allclose(X["U-Y"], photometry["U"] - photometry["Y"])
    assert np.allclose(
        X["U-YERR"], np.hypot(photometry["UERR"], photometry["YERR"])
    )
    assert sum(c.endswith("ERR") for c in X.columns) == 21


def test_the_default_mode_is_mag_color(photometry):
    assert datasets.build_features(photometry).equals(
        datasets.build_features(photometry, "mag-color")
    )


def test_unknown_mode_names_the_alternatives(photometry):
    with pytest.raises(ValueError, match="mag-color"):
        datasets.build_features(photometry, "magic")


def test_missing_columns_are_reported(photometry):
    with pytest.raises(KeyError, match="UERR"):
        datasets.build_features(photometry.drop(columns=["UERR"]), "mag")


def test_alternative_band_sets_are_supported(photometry):
    X = datasets.build_features(photometry, "mag-color", bands=("G", "R", "I"))
    assert list(X.columns) == ["I", "G-R", "R-I", "IERR", "G-RERR", "R-IERR"]


def test_a_reference_band_outside_the_band_set_is_reported(photometry):
    with pytest.raises(ValueError, match="reference_band"):
        datasets.build_features(
            photometry, "mag-color", bands=("G", "R"), reference_band="I"
        )


# -- subsetting a catalogue -------------------------------------------------


def test_take_returns_the_named_rows_in_the_order_asked_for(photometry):
    catalog = datasets.Catalog.from_frame(
        photometry, redshift=np.arange(len(photometry), dtype=float)
    )
    subset = catalog.take([5, 1, 1])
    assert len(subset) == 3
    assert list(subset.redshift) == [5.0, 1.0, 1.0]
    assert np.allclose(subset.raw["I"], photometry["I"].to_numpy()[[5, 1, 1]])


def test_take_accepts_a_boolean_mask(photometry):
    catalog = datasets.Catalog.from_frame(
        photometry, redshift=np.arange(len(photometry), dtype=float)
    )
    mask = np.zeros(len(photometry), dtype=bool)
    mask[[2, 7]] = True
    assert list(catalog.take(mask).redshift) == [2.0, 7.0]


def test_take_carries_the_source_column_when_there_is_one(photometry):
    source = np.arange(len(photometry)) % 2
    catalog = datasets.Catalog(
        split="combined",
        raw=photometry,
        redshift=np.zeros(len(photometry)),
        object_id=np.arange(len(photometry)),
        source=source,
    )
    assert list(catalog.take([0, 1]).source) == [0, 1]
    assert catalog.take([0]).raw.index.tolist() == [0]


def test_take_labels_the_result_after_its_parent(photometry):
    catalog = datasets.Catalog.from_frame(
        photometry, redshift=np.zeros(len(photometry)), split="dc1"
    )
    assert catalog.take([0, 1]).split == "dc1[2]"


# -- the spectroscopic-selection split --------------------------------------


@pytest.fixture
def split(biasable_catalog, hsc_grid):
    """The toy analogue of the real split.

    100 biased, 50 calibration, the rest to test on.
    """
    return datasets.make_selection_split(
        biasable_catalog,
        grid=hsc_grid,
        n_train=100,
        n_calibration=50,
        color_redshift_cut=False,
    )


def test_the_training_set_is_the_size_that_was_asked_for(split):
    assert len(split.biased) == 100
    assert split.meta["history"][-1]["n_biased"] == 100


def test_the_two_parts_of_the_shuffle_account_for_every_galaxy(split):
    """The catalogue is cut in two.

    One part is what the selection draws from, the other what is held back.
    """
    assert (
        split.meta["n_pool"] + split.meta["n_holdout"]
        == split.meta["n_catalog"]
    )
    assert split.meta["n_pool_unused"] == split.meta["n_pool"] - len(
        split.biased
    )


def test_the_calibration_sample_comes_out_of_the_hold_out_not_on_top_of_it(
    split,
):
    """Stolen, not added.

    Otherwise galaxies a model was given are also scored.
    """
    assert len(split.calibration) == 50
    assert len(split.test) + len(split.calibration) == split.meta["n_holdout"]


def test_nothing_a_model_is_given_is_also_scored(split):
    scored = split.rows["test"]
    assert not np.intersect1d(split.rows["biased"], scored).size
    assert not np.intersect1d(split.rows["calibration"], scored).size
    assert not np.intersect1d(
        split.rows["calibration"], split.rows["biased"]
    ).size


def test_the_hold_out_is_drawn_from_the_whole_catalogue_by_default(
    split, biasable_catalog
):
    """No source file is privileged.

    The merged catalogue is shuffled and cut.
    """
    held = np.concatenate([split.rows["test"], split.rows["calibration"]])
    assert set(np.unique(biasable_catalog.source[held])) == {0, 1}


def test_the_hold_out_can_be_restricted_to_chosen_rows(
    biasable_catalog, hsc_grid
):
    """One source file only.

    That keeps a model trained on the other one scorable.
    """
    from_test_file = np.flatnonzero(biasable_catalog.source == 1)
    split = datasets.make_selection_split(
        biasable_catalog,
        grid=hsc_grid,
        n_train=100,
        n_calibration=50,
        holdout_rows=from_test_file,
        color_redshift_cut=False,
    )
    held = np.concatenate([split.rows["test"], split.rows["calibration"]])
    assert (biasable_catalog.source[held] == 1).all()


def test_every_row_index_points_back_at_the_parent_catalogue(
    split, biasable_catalog
):
    rows = split.rows["biased"]
    assert np.array_equal(
        split.biased.redshift, biasable_catalog.redshift[rows]
    )
    assert all(np.all(np.diff(index) > 0) for index in split.rows.values())


def test_asking_for_no_calibration_sample_leaves_the_whole_hold_out_to_test_on(
    biasable_catalog, hsc_grid
):
    split = datasets.make_selection_split(
        biasable_catalog,
        grid=hsc_grid,
        n_train=100,
        n_calibration=0,
        color_redshift_cut=False,
    )
    assert split.calibration is None
    assert "calibration" not in split.rows
    assert len(split.test) == split.meta["n_holdout"]


def test_the_split_is_reproducible_from_its_seed(biasable_catalog, hsc_grid):
    kwargs = dict(
        grid=hsc_grid, n_train=100, n_calibration=50, color_redshift_cut=False
    )
    first = datasets.make_selection_split(biasable_catalog, **kwargs)
    again = datasets.make_selection_split(biasable_catalog, **kwargs)
    other = datasets.make_selection_split(biasable_catalog, seed=99, **kwargs)
    assert np.array_equal(first.rows["test"], again.rows["test"])
    assert not np.array_equal(first.rows["test"], other.rows["test"])


def test_where_the_catalogue_is_cut_is_solved_for_rather_than_guessed(split):
    """The selection keeps a fixed fraction.

    So the cut is what sets the training-set size.
    """
    history = split.meta["history"]
    assert len(history) > 1
    assert abs(history[-1]["n_biased"] - 100) <= abs(
        history[0]["n_biased"] - 100
    )


def test_a_training_set_the_catalogue_cannot_supply_is_reported(
    biasable_catalog, hsc_grid
):
    """The ceiling is what the selection returns on everything.

    And the message says so.
    """
    with pytest.raises(RuntimeError, match="cannot supply more than"):
        datasets.make_selection_split(
            biasable_catalog,
            grid=hsc_grid,
            n_train=1_900,
            n_calibration=0,
            color_redshift_cut=False,
        )


def test_a_hold_out_too_small_for_the_calibration_sample_is_reported(
    biasable_catalog, hsc_grid
):
    with pytest.raises(ValueError, match="nothing to test on"):
        datasets.make_selection_split(
            biasable_catalog,
            grid=hsc_grid,
            n_train=100,
            n_calibration=5_000,
            color_redshift_cut=False,
        )


def test_the_summary_puts_every_subset_next_to_the_catalogue_it_came_from(
    split,
):
    summary = split.summary()
    assert list(summary.index) == ["biased", "calibration", "test", "catalog"]
    assert summary.loc["catalog", "n"] == 2_000
    assert summary.loc["test", "median_z"] == pytest.approx(
        summary.loc["catalog", "median_z"], abs=0.05
    )


def test_the_repr_says_how_big_each_piece_is(split):
    assert (
        repr(split) == "SelectionSplit(biased=100, calibration=50, test=1,750)"
    )


def test_a_missing_hsc_grid_is_reported_rather_than_downloaded(isolated_home):
    with pytest.raises(FileNotFoundError, match="network access"):
        datasets.fetch_hsc_grid(
            data_home=isolated_home, download_if_missing=False
        )


# -- the optional control ---------------------------------------------------


def test_no_control_is_drawn_unless_it_is_asked_for(split):
    assert split.unbiased is None
    assert "unbiased" not in split.rows
    assert split.meta["n_biased_also_unbiased"] is None


def test_the_control_matches_the_biased_set_in_size_and_nothing_else(
    biasable_catalog, hsc_grid
):
    """Without it "biased" and "smaller" cannot be told apart."""
    split = datasets.make_selection_split(
        biasable_catalog,
        grid=hsc_grid,
        n_train=100,
        n_calibration=50,
        control=True,
        color_redshift_cut=False,
    )
    assert len(split.unbiased) == len(split.biased)
    assert not np.intersect1d(split.rows["unbiased"], split.rows["test"]).size
    assert not np.intersect1d(
        split.rows["unbiased"], split.rows["calibration"]
    ).size


def test_asking_for_the_control_leaves_the_rest_of_the_split_alone(
    split, biasable_catalog, hsc_grid
):
    """It is drawn last.

    So it cannot perturb what the other runs were scored on.
    """
    with_control = datasets.make_selection_split(
        biasable_catalog,
        grid=hsc_grid,
        n_train=100,
        n_calibration=50,
        control=True,
        color_redshift_cut=False,
    )
    for name in ("biased", "calibration", "test"):
        assert np.array_equal(with_control.rows[name], split.rows[name]), name
