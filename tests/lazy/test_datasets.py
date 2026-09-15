"""Where cached catalogues land, how they are handed back, and the feature views of them.

The same code has to run on a laptop, a shared login node and a compute node
with a node-local disk, so every part of the path is overridable and none of it
is hardcoded.
"""

import numpy as np
import pandas as pd
import pytest

from lazy.datasets import BANDS, FEATURE_MODES, RAW_COLUMNS, Catalog, data_home, fetch_dc1

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
    assert data_home() == isolated_home / "home" / ".cache" / "lazy-photoz"


def test_xdg_cache_home_is_honoured(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(isolated_home / "xdg"))
    assert data_home() == isolated_home / "xdg" / "lazy-photoz"


def test_an_empty_xdg_cache_home_falls_back(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert data_home() == isolated_home / "home" / ".cache" / "lazy-photoz"


def test_lazy_data_home_wins_over_xdg(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(isolated_home / "xdg"))
    monkeypatch.setenv("LAZY_DATA_HOME", str(isolated_home / "scratch"))
    assert data_home() == isolated_home / "scratch"


def test_the_argument_wins_over_everything(isolated_home, monkeypatch):
    monkeypatch.setenv("LAZY_DATA_HOME", str(isolated_home / "scratch"))
    assert data_home(isolated_home / "explicit") == isolated_home / "explicit"


def test_the_directory_is_created(isolated_home):
    assert data_home(isolated_home / "made" / "deeply").is_dir()


def test_a_missing_cache_is_reported_rather_than_downloaded(isolated_home):
    """`download_if_missing=False` is what a network-less compute node passes."""
    with pytest.raises(FileNotFoundError, match="network access"):
        fetch_dc1(data_home=isolated_home, download_if_missing=False)


def test_catalog_repr_summarises_without_dumping_the_frame():
    catalog = Catalog(
        split="train",
        raw=pd.DataFrame({"U": np.zeros(3)}),
        redshift=np.zeros(3),
        object_id=np.arange(3),
    )
    assert repr(catalog) == "Catalog(split='train', n=3, columns=['U'])"


# -- catalogues built outside fetch_dc1 ------------------------------------


def test_a_dataframe_of_your_own_becomes_a_catalog(photometry):
    frame = photometry.assign(z=np.linspace(0.1, 1.9, len(photometry)))
    catalog = Catalog.from_frame(frame, redshift="z")
    assert len(catalog) == len(photometry)
    assert list(catalog.raw.columns) == list(RAW_COLUMNS)
    assert np.allclose(catalog.redshift, frame["z"])
    assert np.array_equal(catalog.object_id, np.arange(len(photometry)))


def test_truth_and_ids_can_come_as_arrays_instead_of_columns(photometry):
    ids = np.arange(100, 100 + len(photometry))
    catalog = Catalog.from_frame(photometry, redshift=np.zeros(len(photometry)), object_id=ids)
    assert np.array_equal(catalog.object_id, ids)
    assert list(catalog.raw.columns) == list(RAW_COLUMNS)


def test_id_columns_are_moved_out_of_the_features(photometry):
    frame = photometry.assign(z=0.5, objid=np.arange(len(photometry)))
    catalog = Catalog.from_frame(frame, redshift="z", object_id="objid")
    assert "objid" not in catalog.raw.columns
    assert "z" not in catalog.raw.columns


def test_mismatched_lengths_are_reported(photometry):
    with pytest.raises(ValueError, match="rows"):
        Catalog.from_frame(photometry, redshift=np.zeros(3))


def test_a_catalog_of_your_own_builds_features_the_same_way(photometry):
    catalog = Catalog.from_frame(photometry, redshift=np.zeros(len(photometry)))
    assert catalog.features("mag-color").equals(Catalog.build_features(photometry))


# -- feature modes ----------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(FEATURE_MODES))
def test_every_mode_builds_the_documented_width(photometry, mode):
    X = Catalog.build_features(photometry, mode)
    assert len(X) == len(photometry)
    assert X.shape[1] == EXPECTED_COLUMNS[mode]
    assert np.isfinite(X.to_numpy()).all()


def test_mag_is_the_photometry_untouched(photometry):
    X = Catalog.build_features(photometry, "mag")
    assert list(X.columns) == list(RAW_COLUMNS)
    assert np.allclose(X.to_numpy(), photometry[list(RAW_COLUMNS)].to_numpy())


def test_mag_color_keeps_the_reference_magnitude_and_the_adjacent_colours(photometry):
    X = Catalog.build_features(photometry, "mag-color")
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


def test_colour_errors_are_the_quadrature_sum_of_the_two_magnitude_errors(photometry):
    """This is the whole point of carrying colours rather than magnitudes."""
    X = Catalog.build_features(photometry, "mag-color")
    assert np.allclose(X["G-RERR"], np.hypot(photometry["GERR"], photometry["RERR"]))


def test_all_carries_every_colour_and_every_error(photometry):
    X = Catalog.build_features(photometry, "all")
    colours = [c for c in X.columns if "-" in c and not c.endswith("ERR")]
    assert len(colours) == 15
    assert list(X.columns[: len(BANDS)]) == list(BANDS)
    assert np.allclose(X["U-Y"], photometry["U"] - photometry["Y"])
    assert np.allclose(X["U-YERR"], np.hypot(photometry["UERR"], photometry["YERR"]))
    assert sum(c.endswith("ERR") for c in X.columns) == 21


def test_the_default_mode_is_mag_color(photometry):
    assert Catalog.build_features(photometry).equals(Catalog.build_features(photometry, "mag-color"))


def test_unknown_mode_names_the_alternatives(photometry):
    with pytest.raises(ValueError, match="mag-color"):
        Catalog.build_features(photometry, "magic")


def test_missing_columns_are_reported(photometry):
    with pytest.raises(KeyError, match="UERR"):
        Catalog.build_features(photometry.drop(columns=["UERR"]), "mag")


def test_alternative_band_sets_are_supported(photometry):
    X = Catalog.build_features(photometry, "mag-color", bands=("G", "R", "I"))
    assert list(X.columns) == ["I", "G-R", "R-I", "IERR", "G-RERR", "R-IERR"]


def test_a_reference_band_outside_the_band_set_is_reported(photometry):
    with pytest.raises(ValueError, match="reference_band"):
        Catalog.build_features(photometry, "mag-color", bands=("G", "R"), reference_band="I")
