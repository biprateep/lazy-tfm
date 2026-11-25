# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""PLAsTiCC's Deep Drilling Field objects: light curves and their redshifts.

The Photometric LSST Astronomical Time-Series Classification Challenge
(PLAsTiCC; The PLAsTiCC team et al. 2018, arXiv:1810.00001) simulated three
years of LSST light curves. Its unblinded data, Zenodo record 2539456,
publish the true redshift of every object. The first file of test light
curves holds exactly the objects of the Deep Drilling Fields (DDF), the
fields LSST revisits most often. :func:`load` keeps the extragalactic ones,
with the host galaxy's photo-z and one series of fluxes per band.
:func:`lazy.datasets.load_dataset` downloads the files; this module holds
what is PLAsTiCC's own.
"""

from __future__ import annotations

from collections.abc import Callable
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from lazy import _home
from lazy import _typing

__all__ = ["RECORD", "SHA256", "SIZES", "TARGET", "URL", "load"]

RECORD = 2539456
"""The Zenodo record of PLAsTiCC's unblinded data."""
URL = f"https://zenodo.org/records/{RECORD}/files/"
"""Where the record's files are published, each under its name."""

_LIGHT_CURVES = "plasticc_test_lightcurves_01.csv.gz"
_METADATA = "plasticc_test_metadata.csv.gz"
SIZES = {_LIGHT_CURVES: 148_537_739, _METADATA: 157_245_185}
"""The size in bytes of each file read."""
SHA256 = {
    _LIGHT_CURVES: (
        "388c76d92b3a9572f9beff646ec240e852c4be164c2a63d735247d0b810899f3"
    ),
    _METADATA: (
        "8048d59cf84ab30fa539ad69aeeebdac6c7347a149b8724e4a32adc9bbf6897e"
    ),
}
"""The SHA-256 checksum of each file read, as hex."""

TARGET = "true_z"
"""The target: the simulated redshift."""
_FEATURES = ("hostgal_photoz", "hostgal_photoz_err", "mwebv", "ra", "decl")
_BANDS = "ugrizy"  # Passbands 0 to 5.
_FLUX = tuple(f"flux_{band}" for band in _BANDS)
_POINTS = {
    "object_id": np.int64,
    "mjd": np.float64,
    "passband": np.int64,
    "flux": np.float64,
}
# Rows of the metadata, which covers all 3.5 million test objects, read at
# a time.
_CHUNK_ROWS = 500_000
# The parsed table; its version changes with what _parse writes.
_CACHE = "plasticc_ddf-v1.npz"


def load(
    folder: pathlib.Path, fetch: Callable[[str], pathlib.Path]
) -> tuple[pd.DataFrame, _typing.FloatArray, dict[str, str]]:
    """Loads the extragalactic DDF objects, parsing the files once.

    The parsed table is cached in ``folder``, and later loads read it alone.

    Args:
        folder: The directory of the cache.
        fetch: Returns the local path of a published file, given its name,
            downloading it if needed.

    Returns:
        The features, one row per object in order of ``object_id``: the
        host galaxy's photo-z and its error, the Milky Way's E(B-V), the
        position in degrees, and ``flux_<band>`` for each of ``ugrizy``,
        each cell a 1-D float64 array of that band's fluxes in order of
        time. Then the target, ``true_z``, shape (n_rows,), as float64,
        and the modalities, the flux columns mapped to ``"time"``.
    """
    path = folder / _CACHE
    if not path.exists():
        _write(path, _parse(fetch(_LIGHT_CURVES), fetch(_METADATA)))
    with np.load(path) as saved:
        X = pd.DataFrame({column: saved[column] for column in _FEATURES})
        y = saved[TARGET]
        counts = saved["counts"]
        series = np.split(saved["flux"], np.cumsum(counts)[:-1])
    for band, column in enumerate(_FLUX):
        X[column] = pd.Series(series[band :: len(_BANDS)], dtype=object)
    return X, y, dict.fromkeys(_FLUX, "time")


def _parse(
    light_curves: pathlib.Path, metadata: pathlib.Path
) -> dict[str, Any]:
    """Parses the two CSV files into the arrays of the cache.

    Args:
        light_curves: The test light curves of the Deep Drilling Fields.
        metadata: The metadata of every test object.

    Returns:
        Each feature and the target, float64, shape (n_rows,); ``flux``,
        float64, shape (n_points,), every object's fluxes in order of
        object, band and time; and ``counts``, shape (n_rows, 6), the number
        of fluxes of each object in each band.
    """
    points = pd.read_csv(light_curves, usecols=list(_POINTS), dtype=_POINTS)
    ids = np.unique(points["object_id"])
    meta = pd.concat(
        # Galactic objects have no redshift: true_z == 0.
        chunk[chunk["object_id"].isin(ids) & (chunk[TARGET] > 0)]
        for chunk in pd.read_csv(
            metadata,
            usecols=["object_id", TARGET, *_FEATURES],
            skipinitialspace=True,  # The columns are padded with spaces.
            chunksize=_CHUNK_ROWS,
        )
    ).sort_values("object_id")
    kept = meta["object_id"].to_numpy()
    points = points[points["object_id"].isin(kept)]
    row = np.searchsorted(kept, points["object_id"].to_numpy())
    band = points["passband"].to_numpy()
    order = np.lexsort((points["mjd"].to_numpy(), band, row))
    counts = np.bincount(
        row * len(_BANDS) + band, minlength=kept.size * len(_BANDS)
    )
    arrays: dict[str, Any] = {
        column: meta[column].to_numpy(np.float64)
        for column in (*_FEATURES, TARGET)
    }
    arrays["flux"] = points["flux"].to_numpy()[order]
    arrays["counts"] = counts.reshape(kept.size, len(_BANDS))
    return arrays


def _write(path: pathlib.Path, arrays: dict[str, Any]) -> None:
    """Writes the cache whole, then moves it into place."""
    with _home.atomic_file(path) as out:
        np.savez(out, **arrays)
