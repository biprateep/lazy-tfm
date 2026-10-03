# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Where cached catalogues land, how they come back, and their feature views.

The same code has to run on a laptop, a shared login node and a compute node
with a node-local disk, so every part of the path is overridable and none of it
is hardcoded.
"""

import concurrent.futures
import hashlib
import io
import os
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets as sklearn_datasets
from sklearn import utils as sklearn_utils
from sklearn.datasets import _openml as sklearn_openml

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
    monkeypatch.setattr(datasets, "_warned_legacy", set())
    return tmp_path


def test_default_is_under_the_users_cache(isolated_home):
    assert (
        datasets.data_home() == isolated_home / "home" / ".cache" / "lazy-tfm"
    )


def test_xdg_cache_home_is_honoured(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(isolated_home / "xdg"))
    assert datasets.data_home() == isolated_home / "xdg" / "lazy-tfm"


def test_an_empty_xdg_cache_home_falls_back(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert (
        datasets.data_home() == isolated_home / "home" / ".cache" / "lazy-tfm"
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


class _SlowResponse:
    """A stand-in for urlopen's response, served in small, slow pieces."""

    def __init__(self, payload, pause):
        self._stream = io.BytesIO(payload)
        self._pause = pause

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self, size):
        time.sleep(self._pause)
        return self._stream.read(min(size, 64))


def _serve(monkeypatch, payload, *, pause=0.0, checksum=None):
    """Serves ``payload`` for every URL; returns the timeouts asked for."""
    timeouts = []

    def urlopen(url, timeout=None):
        timeouts.append(timeout)
        return _SlowResponse(payload, pause)

    monkeypatch.setattr(datasets.urllib.request, "urlopen", urlopen)
    digest = checksum or hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(datasets._SHA256, "fake.bin", digest)
    monkeypatch.setitem(datasets._SIZES, "fake.bin", len(payload))
    return timeouts


def _fetch_fake(root):
    return datasets._cached_file(
        "fake.bin",
        "https://example.invalid/fake.bin",
        root=root,
        download_if_missing=True,
        hint="test",
    )


def test_parallel_downloads_to_one_cache_do_not_collide(
    isolated_home, monkeypatch
):
    payload = bytes(range(256)) * 8
    timeouts = _serve(monkeypatch, payload, pause=0.002)
    root = isolated_home / "shared"
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        paths = list(pool.map(_fetch_fake, [root, root]))
    assert paths[0] == paths[1] == root / "fake.bin"
    assert paths[0].read_bytes() == payload
    assert sorted(p.name for p in root.iterdir()) == ["fake.bin"]
    assert timeouts and all(t is not None and t > 0 for t in timeouts)


def test_a_corrupted_download_leaves_nothing_behind(isolated_home, monkeypatch):
    _serve(monkeypatch, b"not the file", checksum="0" * 64)
    root = isolated_home / "cache"
    with pytest.raises(OSError, match="checksum mismatch"):
        _fetch_fake(root)
    assert not list(root.iterdir())


def test_a_truncated_cached_file_is_refused_offline(isolated_home, monkeypatch):
    _serve(monkeypatch, b"the whole file")
    root = isolated_home / "cache"
    root.mkdir()
    (root / "fake.bin").write_bytes(b"the who")
    with pytest.raises(OSError, match="incomplete or corrupted"):
        datasets._cached_file(
            "fake.bin",
            "https://example.invalid/fake.bin",
            root=root,
            download_if_missing=False,
            hint="test",
        )


def test_a_truncated_cached_file_is_fetched_again(isolated_home, monkeypatch):
    _serve(monkeypatch, b"the whole file")
    root = isolated_home / "cache"
    root.mkdir()
    (root / "fake.bin").write_bytes(b"the who")
    assert _fetch_fake(root).read_bytes() == b"the whole file"


def test_every_published_file_has_a_size_and_a_checksum():
    assert set(datasets._SIZES) == set(datasets._SHA256)


def _legacy_file(isolated_home, payload):
    legacy = isolated_home / "home" / ".cache" / "lazy-photoz"
    legacy.mkdir(parents=True)
    (legacy / "fake.bin").write_bytes(payload)
    return legacy / "fake.bin"


def test_a_file_in_the_old_cache_is_used_with_one_warning(
    isolated_home, monkeypatch
):
    _serve(monkeypatch, b"the whole file")
    old = _legacy_file(isolated_home, b"the whole file")
    with pytest.warns(UserWarning, match="old cache directory") as record:
        assert _fetch_fake(None) == old
    assert record[0].filename == __file__
    assert "lazy-tfm" in str(record[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _fetch_fake(None) == old  # warned once


def test_the_old_cache_is_ignored_when_a_cache_is_chosen(
    isolated_home, monkeypatch
):
    timeouts = _serve(monkeypatch, b"the whole file")
    old = _legacy_file(isolated_home, b"the whole file")
    chosen = isolated_home / "chosen"
    assert _fetch_fake(chosen) == chosen / "fake.bin"
    monkeypatch.setenv("LAZY_DATA_HOME", str(isolated_home / "env"))
    assert _fetch_fake(None) == isolated_home / "env" / "fake.bin"
    assert len(timeouts) == 2  # both downloaded, neither used the old copy
    assert old.exists()


@pytest.mark.parametrize("mode", sorted(datasets.FEATURE_MODES))
def test_the_features_keep_the_input_index(photometry, mode):
    raw = photometry.set_axis([f"obj{i}" for i in range(len(photometry))])
    subset = raw.iloc[::3]
    features = datasets.build_features(subset, mode)
    pd.testing.assert_index_equal(features.index, subset.index)
    np.testing.assert_array_equal(features["IERR"], subset["IERR"])


# -- the demo dataset registry ------------------------------------------------

LIST_COLUMNS = [
    "n_rows",
    "n_features",
    "target",
    "description",
    "source",
    "license",
    "url",
]
# Each grade of diamonds from worst to best, and five diamonds' grades.
DIAMOND_GRADES = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}
DIAMOND_ROWS = {
    "cut": ["Ideal", "Fair", "Very Good", "Premium", "Good"],
    "color": ["E", "J", "D", "G", "I"],
    "clarity": ["SI2", "IF", "I1", "VS1", "VVS2"],
}
OPENML_NAMES = [
    name
    for name, row in datasets.list_datasets().iterrows()
    if row["source"].startswith("OpenML")
]


@pytest.fixture
def fake_openml(monkeypatch):
    """Replaces fetch_openml with a stand-in; returns the calls it saw.

    The stand-in answers with a frame of the registered width, its first
    column categorical, and a target named as registered, so nothing is
    downloaded. For diamonds, the first three columns are the grades, served
    as OpenML serves them: unordered, with the categories alphabetical.
    """
    table = datasets.list_datasets()
    by_id = {
        int(row["source"].split()[1]): (name, row)
        for name, row in table.iterrows()
        if row["source"].startswith("OpenML")
    }
    calls = []

    def fetch_openml(*, data_id, data_home, target_column, as_frame):
        calls.append(
            {
                "data_id": data_id,
                "data_home": data_home,
                "target_column": target_column,
                "as_frame": as_frame,
            }
        )
        name, row = by_id[data_id]
        n = 5
        frame = pd.DataFrame(
            {f"f{i}": np.arange(n, dtype=np.int64) for i in range(1, 99)}
        ).iloc[:, : row["n_features"] - 1]
        frame.insert(0, "kind", pd.Categorical(["a", "b", "a", "c", "b"]))
        if name == "diamonds":
            frame = frame.iloc[:, len(DIAMOND_ROWS) :]
            for column, values in reversed(DIAMOND_ROWS.items()):
                categories = sorted(DIAMOND_GRADES[column])
                frame.insert(
                    0, column, pd.Categorical(values, categories=categories)
                )
        target = pd.Series(np.arange(n), name=target_column)
        return sklearn_utils.Bunch(data=frame, target=target)

    monkeypatch.setattr(sklearn_datasets, "fetch_openml", fetch_openml)
    return calls


@pytest.fixture
def fake_dc1(monkeypatch, photometry):
    """Replaces fetch_dc1 with the small photometry fixture."""
    catalog = datasets.Catalog.from_frame(
        photometry, redshift=np.linspace(0.1, 1.9, len(photometry))
    )

    def fetch(*, split=False, **_):
        if not split:
            return catalog
        half = len(photometry) // 2
        return (
            datasets.Catalog.from_frame(
                photometry.iloc[:half], redshift=catalog.redshift[:half]
            ),
            datasets.Catalog.from_frame(
                photometry.iloc[half:], redshift=catalog.redshift[half:]
            ),
        )

    monkeypatch.setattr(datasets, "fetch_dc1", fetch)
    return catalog


def test_list_datasets_has_one_complete_row_per_dataset():
    table = datasets.list_datasets()
    assert list(table.columns) == LIST_COLUMNS
    assert table.index.name == "name"
    assert table.index.is_unique
    assert "dc1" in table.index and "yacht" in table.index
    assert (table["n_rows"] > 0).all() and (table["n_features"] > 0).all()
    for column in ("target", "description", "license", "url"):
        assert table[column].str.len().gt(0).all(), column
    assert table.loc["dc1", "source"] == "Zenodo 10975874"
    assert table.loc["chirp", "source"] == "Generated"
    for name in OPENML_NAMES:
        data_id = table.loc[name, "source"].removeprefix("OpenML ")
        assert table.loc[name, "url"] == f"https://www.openml.org/d/{data_id}"


@pytest.mark.parametrize("name", OPENML_NAMES)
def test_every_openml_dataset_loads_by_its_pinned_id(fake_openml, name):
    row = datasets.list_datasets().loc[name]
    dataset = datasets.load_dataset(name)
    (call,) = fake_openml
    assert f"OpenML {call['data_id']}" == row["source"]
    assert call["target_column"] == row["target"]
    assert call["as_frame"] is True
    assert dataset.name == name
    assert dataset.target == row["target"]
    assert dataset.X.shape[1] == row["n_features"]
    assert (dataset.source, dataset.license, dataset.url) == (
        row["source"],
        row["license"],
        row["url"],
    )
    assert dataset.description == row["description"]
    assert isinstance(dataset.citation, str)


def test_dataset_fields_and_dtypes(fake_openml):
    dataset = datasets.load_dataset("diamonds")
    assert isinstance(dataset.X, pd.DataFrame)
    assert isinstance(dataset.X["cut"].dtype, pd.CategoricalDtype)
    assert isinstance(dataset.y, np.ndarray)
    assert dataset.y.dtype == np.float64 and dataset.y.ndim == 1
    assert len(dataset) == len(dataset.X) == len(dataset.y)
    assert repr(dataset) == (
        "Dataset(name='diamonds', n_rows=5, n_features=9, target='price')"
    )
    with pytest.raises(AttributeError):
        dataset.name = "other"


def test_unordered_categories_keep_the_source_dtype(fake_openml):
    dataset = datasets.load_dataset("yacht")
    assert isinstance(dataset.X["kind"].dtype, pd.CategoricalDtype)
    assert not dataset.X["kind"].cat.ordered


def test_diamond_grades_are_ordered_from_worst_to_best(fake_openml):
    X = datasets.load_dataset("diamonds").X
    for column, order in DIAMOND_GRADES.items():
        assert X[column].cat.ordered, column
        assert list(X[column].cat.categories) == order, column
        assert list(X[column]) == DIAMOND_ROWS[column], column
    # The codes rank the grades, so a model sees them in order.
    assert list(X["cut"].cat.codes) == [4, 0, 2, 3, 1]
    assert X["color"].max() == "D"


def test_diamond_grades_the_source_does_not_have_are_reported(
    fake_openml, monkeypatch
):
    fetch = sklearn_datasets.fetch_openml

    def renamed(**kwargs):
        bunch = fetch(**kwargs)
        bunch.data["cut"] = bunch.data["cut"].cat.rename_categories(
            {"Fair": "Poor"}
        )
        return bunch

    monkeypatch.setattr(sklearn_datasets, "fetch_openml", renamed)
    with pytest.raises(ValueError, match="categor"):
        datasets.load_dataset("diamonds")


def test_return_x_y_gives_the_features_and_target(fake_openml):
    X, y = datasets.load_dataset("yacht", return_X_y=True)
    dataset = datasets.load_dataset("yacht")
    pd.testing.assert_frame_equal(X, dataset.X)
    np.testing.assert_array_equal(y, dataset.y)
    assert y.dtype == np.float64


def test_openml_downloads_are_cached_under_data_home(
    isolated_home, monkeypatch, fake_openml
):
    datasets.load_dataset("yacht")
    datasets.load_dataset("yacht", data_home=isolated_home / "explicit")
    monkeypatch.setenv("LAZY_DATA_HOME", str(isolated_home / "scratch"))
    datasets.load_dataset("yacht")
    # scikit-learn adds the "openml" subfolder itself.
    assert [call["data_home"] for call in fake_openml] == [
        str(isolated_home / "home" / ".cache" / "lazy-tfm"),
        str(isolated_home / "explicit"),
        str(isolated_home / "scratch"),
    ]


def test_dc1_is_the_mag_color_view_of_fetch_dc1(fake_dc1, fake_openml):
    dataset = datasets.load_dataset("dc1")
    assert not fake_openml
    pd.testing.assert_frame_equal(dataset.X, fake_dc1.features("mag-color"))
    np.testing.assert_array_equal(dataset.y, fake_dc1.redshift)
    assert dataset.target == "redshift"
    assert dataset.source == "Zenodo 10975874"
    expected = datasets.list_datasets().loc["dc1", "n_features"]
    assert dataset.X.shape[1] == expected


def test_an_unknown_dataset_lists_the_valid_names(fake_openml):
    with pytest.raises(ValueError, match="unknown dataset 'nope'") as error:
        datasets.load_dataset("nope")
    for name in datasets.list_datasets().index:
        assert repr(name) in str(error.value)
    assert not fake_openml


def _no_network(*args, **kwargs):
    raise AssertionError("the cached copy should have been read")


@pytest.mark.skipif(
    os.environ.get("LAZY_RUN_NETWORK_TESTS") != "1",
    reason="set LAZY_RUN_NETWORK_TESTS=1 to run tests that download data",
)
def test_yacht_downloads_once_then_loads_offline(isolated_home, monkeypatch):
    dataset = datasets.load_dataset("yacht", data_home=isolated_home)
    row = datasets.list_datasets().loc["yacht"]
    assert dataset.X.shape == (row["n_rows"], row["n_features"])
    assert (isolated_home / "openml").is_dir()
    monkeypatch.setattr(sklearn_openml, "urlopen", _no_network)
    again = datasets.load_dataset("yacht", data_home=isolated_home)
    pd.testing.assert_frame_equal(again.X, dataset.X)


def test_dc1_splits_into_the_challenge_train_and_test(fake_dc1, fake_openml):
    train, test = datasets.load_dataset("dc1", split=True)
    assert not fake_openml
    assert len(train) + len(test) == len(fake_dc1)
    np.testing.assert_array_equal(
        np.concatenate([train.y, test.y]), fake_dc1.redshift
    )
    X_train, X_test, y_train, y_test = datasets.load_dataset(
        "dc1", split=True, return_X_y=True
    )
    pd.testing.assert_frame_equal(X_train, train.X)
    pd.testing.assert_frame_equal(X_test, test.X)
    np.testing.assert_array_equal(y_train, train.y)
    np.testing.assert_array_equal(y_test, test.y)


def test_openml_datasets_have_no_split(fake_openml):
    with pytest.raises(ValueError, match="no train/test split"):
        datasets.load_dataset("yacht", split=True)
    assert not fake_openml


def test_only_the_chirp_takes_options(fake_openml):
    with pytest.raises(TypeError, match="noise"):
        datasets.load_dataset("yacht", noise=0.1)
    with pytest.raises(TypeError, match="colour"):
        datasets.load_dataset("chirp", colour="red")
    assert not fake_openml


def test_the_chirp_cuts_its_gaps_into_the_test_rows(fake_openml):
    train, test = datasets.load_dataset(
        "chirp",
        split=True,
        n_samples=1000,
        n_gaps=3,
        gap_width=0.06,
        random_state=0,
    )
    assert not fake_openml
    assert list(train.X.columns) == list(test.X.columns) == ["x"]
    assert len(train) + len(test) == 1000
    assert (train.name, train.target, train.source) == (
        "chirp",
        "y",
        "Generated",
    )
    # Three separate runs of test rows, each about 6% of the range wide.
    x = np.linspace(0.0, 10.0, 1000)
    in_gap = np.isin(x, test.X["x"])
    assert np.count_nonzero(np.diff(in_gap.astype(int)) == 1) == 3
    assert 3 * 55 <= len(test) <= 3 * 62
    # The test targets are the noise-free curve; the training ones are not.
    truth = (1 + 0.2 * x) * np.sin(2 * np.pi * (0.2 * x + 0.015 * x**2))
    np.testing.assert_array_equal(test.y, truth[in_gap])
    assert 0.15 < np.std(train.y - truth[~in_gap]) < 0.25


def test_the_chirp_is_reproducible_and_validates():
    first = datasets.load_dataset(
        "chirp", split=True, return_X_y=True, random_state=1
    )
    second = datasets.load_dataset(
        "chirp", split=True, return_X_y=True, random_state=1
    )
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    # Unsplit, the training rows come first, then the test rows.
    whole = datasets.load_dataset("chirp", random_state=1)
    np.testing.assert_array_equal(whole.y, np.concatenate([first[2], first[3]]))
    assert list(whole.X.index) == list(range(len(whole)))
    _, X_test, _, _ = datasets.load_dataset(
        "chirp", split=True, return_X_y=True, n_gaps=0, random_state=1
    )
    assert len(X_test) == 0
    with pytest.raises(ValueError, match="do not fit"):
        datasets.load_dataset("chirp", n_gaps=10, gap_width=0.2)
