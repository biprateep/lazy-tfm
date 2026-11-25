# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""PLAsTiCC's Deep Drilling Field objects, loaded through lazy.datasets.

A fake Zenodo serves two small gzipped CSV files, laid out as the published
ones are, so the loader runs offline.
"""

import gzip
import hashlib
import io
import os

import numpy as np
import pytest

from lazy import _plasticc
from lazy import datasets

LIGHT_CURVES = "plasticc_test_lightcurves_01.csv.gz"
METADATA = "plasticc_test_metadata.csv.gz"
HEADER = (
    "object_id,ra,decl,ddf_bool,hostgal_specz,hostgal_photoz,"
    "hostgal_photoz_err,distmod,mwebv,target,true_target,true_submodel,"
    "true_z,true_distmod,true_lensdmu,true_vpec,true_rv,true_av,"
    "true_peakmjd,libid_cadence,tflux_u,tflux_g,tflux_r,tflux_i,tflux_z,"
    "tflux_y"
)
# Objects 7 and 30 are extragalactic, 13 galactic, and 99 outside the
# Deep Drilling Fields, so not in the file of light curves.
TRUE_Z = {30: 1.25, 13: 0.0, 7: 0.5, 99: 0.8}
COLUMNS = [
    "hostgal_photoz",
    "hostgal_photoz_err",
    "mwebv",
    "ra",
    "decl",
    "flux_u",
    "flux_g",
    "flux_r",
    "flux_i",
    "flux_z",
    "flux_y",
]


def _light_curves():
    """Points of objects 7, 13 and 30, bands interleaved, times shuffled.

    Each flux is ``1000 * object_id + 100 * passband + mjd - 60000``, so it
    names the object, the band and the time it belongs to.
    """
    rng = np.random.default_rng(0)
    lines = ["object_id,mjd,passband,flux,flux_err,detected_bool"]
    for object_id in (7, 13, 30):
        n = 4 + object_id % 5
        mjd = 60000 + rng.permutation(n * 6).astype(float)
        passband = rng.integers(0, 6, n * 6)
        for t, band in zip(mjd, passband, strict=True):
            flux = 1000 * object_id + 100 * band + t - 60000
            lines.append(f"{object_id},{t:.4f},{band},{flux},1.0,0")
    return "\n".join(lines) + "\n"


def _metadata():
    """One padded row per object, as the published file pads its columns."""
    lines = [HEADER]
    for object_id, z in TRUE_Z.items():
        values = [
            f"{object_id:>9}",
            f"{object_id + 0.5:8.4f}",  # ra
            f"{-object_id:8.4f}",  # decl
            str(int(object_id != 99)),
            f"{z:6.3f}",  # hostgal_specz
            f"{z + 0.01:6.3f}",  # hostgal_photoz
            " 0.050",
            "41.112",
            " 0.019",
            " 0",
            " 42",
            "1",
            f"{z:5.3f}",
            *["1.0"] * 5,  # true_distmod to true_av
            "60499.461",
            "  124",
            *["0."] * 6,
        ]
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


@pytest.fixture(name="zenodo")
def _fake_zenodo(monkeypatch, tmp_path):
    """Serves the two files; returns the URLs fetched.

    The metadata is read two rows at a time, so it spans several chunks.
    """
    files = {
        LIGHT_CURVES: gzip.compress(_light_curves().encode(), mtime=0),
        METADATA: gzip.compress(_metadata().encode(), mtime=0),
    }
    fetched = []

    def urlopen(url, timeout=None):
        fetched.append(url)
        name = url.rsplit("/", 1)[1]
        assert url == f"https://zenodo.org/records/2539456/files/{name}"
        return io.BytesIO(files[name])

    monkeypatch.setattr(datasets.urllib.request, "urlopen", urlopen)
    for name, payload in files.items():
        digest = hashlib.sha256(payload).hexdigest()
        monkeypatch.setitem(datasets._SHA256, name, digest)
        monkeypatch.setitem(datasets._SIZES, name, len(payload))
    monkeypatch.setattr(_plasticc, "_CHUNK_ROWS", 2)
    monkeypatch.setenv("LAZY_DATA_HOME", str(tmp_path / "home"))
    return fetched


def test_the_extragalactic_objects_load_without_the_truth_columns(zenodo):
    dataset = datasets.load_dataset("plasticc_ddf")
    assert list(dataset.X.columns) == COLUMNS
    assert dataset.target == "true_z"
    np.testing.assert_array_equal(dataset.y, [0.5, 1.25])
    assert dataset.y.dtype == np.float64
    np.testing.assert_array_equal(dataset.X["hostgal_photoz"], [0.51, 1.26])
    np.testing.assert_array_equal(dataset.X["ra"], [7.5, 30.5])
    assert dataset.modalities == {column: "time" for column in COLUMNS[5:]}
    row = datasets.list_datasets().loc["plasticc_ddf"]
    assert dataset.X.shape[1] == row["n_features"]
    assert (dataset.source, dataset.url) == (row["source"], row["url"])
    assert row["source"] == "Zenodo 2539456"


def test_each_flux_column_holds_its_bands_fluxes_in_order_of_time(zenodo):
    X = datasets.load_dataset("plasticc_ddf").X
    for object_id, (_, row) in zip((7, 30), X.iterrows(), strict=True):
        n_points = 0
        for band in range(6):
            flux = row[COLUMNS[5 + band]]
            assert flux.dtype == np.float64 and flux.ndim == 1
            n_points += flux.size
            offset = flux - 1000 * object_id - 100 * band
            assert ((offset >= 0) & (offset < 100)).all()
            assert (np.diff(offset) > 0).all()
        assert n_points == 6 * (4 + object_id % 5)


def test_a_second_load_reads_the_cache_and_downloads_nothing(zenodo):
    first = datasets.load_dataset("plasticc_ddf")
    assert len(zenodo) == 2
    for name in (LIGHT_CURVES, METADATA):
        os.remove(datasets.data_home() / name)
    second = datasets.load_dataset("plasticc_ddf")
    assert len(zenodo) == 2
    np.testing.assert_array_equal(first.y, second.y)
    for column in COLUMNS[5:]:
        for a, b in zip(first.X[column], second.X[column], strict=True):
            np.testing.assert_array_equal(a, b)


def test_a_file_that_is_not_the_published_one_is_refused(zenodo, monkeypatch):
    monkeypatch.setitem(datasets._SHA256, METADATA, "0" * 64)
    with pytest.raises(OSError, match="checksum mismatch"):
        datasets.load_dataset("plasticc_ddf")
    assert not list(datasets.data_home().glob("plasticc_ddf-*"))


def test_a_cached_file_of_the_right_size_but_other_bytes_is_refused(zenodo):
    datasets.load_dataset("plasticc_ddf")
    home = datasets.data_home()
    for cache in home.glob("plasticc_ddf-*"):
        cache.unlink()
    metadata = home / METADATA
    damaged = bytearray(metadata.read_bytes())
    damaged[-1] ^= 0xFF
    metadata.write_bytes(bytes(damaged))
    with pytest.raises(OSError, match="is not the published"):
        datasets.load_dataset("plasticc_ddf")
    assert not list(home.glob("plasticc_ddf-*"))


def test_a_cache_that_fails_to_write_leaves_no_file_behind(zenodo, monkeypatch):
    def full_disk(file, **arrays):
        file.write(b"partly")
        raise OSError("No space left on device")

    monkeypatch.setattr(_plasticc.np, "savez", full_disk)
    with pytest.raises(OSError, match="No space left"):
        datasets.load_dataset("plasticc_ddf")
    home = datasets.data_home()
    assert not [p for p in home.iterdir() if p.name.startswith(".")]
    assert not list(home.glob("plasticc_ddf-*"))
