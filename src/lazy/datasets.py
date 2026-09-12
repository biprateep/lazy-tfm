"""Built-in photo-z catalogues: fetching them, caching them, loading them.

The library is general purpose -- every estimator takes whatever tabular
features you hand it -- but a photo-z method is hard to judge without a shared
benchmark, so the LSST DESC PZ Data Challenge catalogue (DC1, Schmidt et al.
2020) is available as a one-liner::

    from lazy.datasets import fetch_dc1
    train = fetch_dc1("train")
    train.raw.head()

The two files total about 1 GB. They are downloaded once from Zenodo record
10975874, checksummed, and cached under :func:`data_home` -- by default
``$XDG_CACHE_HOME/lazy-photoz`` or ``~/.cache/lazy-photoz``, overridable with
the ``LAZY_DATA_HOME`` environment variable or the ``data_home`` argument.
Nothing is re-downloaded if the cached file is already there and intact.

DC1 is Buzzard v1.0 photometry: ``ugrizy`` magnitudes at LSST ten-year depth,
``i < 25.3``, ``0 < z < 2``, 43,486 training and 390,990 test galaxies. Each
file also carries the challenge's ``trainZ`` baseline PDFs, in which every
galaxy is assigned the same training-set N(z); :func:`load_trainz` returns them,
and they are the reference a real estimator has to beat.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["BANDS", "Catalog", "data_home", "fetch_dc1", "load_trainz"]

BANDS = ("U", "G", "R", "I", "Z", "Y")
MAG_COLUMNS = BANDS
ERR_COLUMNS = tuple(f"{band}ERR" for band in BANDS)
RAW_COLUMNS = MAG_COLUMNS + ERR_COLUMNS

Split = Literal["train", "test"]

_ZENODO = "https://zenodo.org/records/10975874/files"
_FILES = {"train": "trainz_train.npz", "test": "trainz_test.npz"}
_CAT_KEYS = {"train": "train_cat", "test": "test_cat"}
_CDE_KEYS = {"train": "cde_train", "test": "cde_test"}
_SHA256 = {
    "trainz_train.npz": "fe54d2034c554ec1f675e7d5789fd5ec4564a72d22bc5008fc218003bc6fb6dc",
    "trainz_test.npz": "a1b1a2b191d0edbcb83b3a5e87cc8f7c2cd0dc5c564a1bb37258f6e1588c98ce",
}


@dataclass(frozen=True)
class Catalog:
    """Photometry, truth and identifiers for one catalogue split."""

    split: str
    raw: pd.DataFrame
    redshift: np.ndarray
    object_id: np.ndarray
    #: For a combined catalogue: 0 for rows from the train file, 1 from test.
    source: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.redshift)

    def __repr__(self) -> str:
        return f"Catalog(split={self.split!r}, n={len(self)}, columns={list(self.raw.columns)})"


def data_home(data_home: str | Path | None = None) -> Path:
    """The directory cached catalogues live in, created if it does not exist.

    Resolution order: the ``data_home`` argument, then ``$LAZY_DATA_HOME``,
    then ``$XDG_CACHE_HOME/lazy-photoz``, then ``~/.cache/lazy-photoz``.

    Nothing about the location is baked in, so a shared scratch filesystem or a
    node-local disk is one environment variable away and the same code runs
    unchanged on a laptop and on a cluster.
    """
    explicit = data_home or os.environ.get("LAZY_DATA_HOME")
    if explicit:
        root = Path(explicit)
    else:
        cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        root = Path(cache) / "lazy-photoz"
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def fetch_dc1(
    split: Split = "train",
    *,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Catalog:
    """Load one DC1 split, downloading and caching it on first use.

    Parameters
    ----------
    split
        ``"train"`` (43,486 rows) or ``"test"`` (390,990 rows).
    data_home
        Cache directory; see :func:`data_home`.
    download_if_missing
        When ``False``, raise rather than reach for the network. Use it on a
        compute node with no outbound connection, having warmed the cache
        elsewhere.

    Returns
    -------
    Catalog
        ``raw`` holds the six magnitudes and their six errors as float32;
        ``redshift`` is the true redshift as float64.
    """
    path = _cached_path(split, root=data_home, download_if_missing=download_if_missing)
    with np.load(path, allow_pickle=True) as saved:
        cat = saved[_CAT_KEYS[split]]
        raw = pd.DataFrame({column: np.asarray(cat[column], dtype=np.float32) for column in RAW_COLUMNS})
        redshift = np.asarray(cat["SPECZ"], dtype=np.float64)
        object_id = np.asarray(cat["ID"], dtype=np.int64)
    return Catalog(split=split, raw=raw, redshift=redshift, object_id=object_id)


def load_trainz(
    split: Split = "test",
    *,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """The DC1 ``trainZ`` baseline PDFs for one split, as ``(z_grid, pdfs)``.

    Every row is the same training-set N(z): the challenge's deliberately
    trivial estimator, and the floor any real method must clear. The stored
    array is a per-bin probability on a 0.01-wide grid, so normalise it as a
    density (:meth:`lazy.grid.RedshiftGrid.normalize`) before scoring.
    """
    path = _cached_path(split, root=data_home, download_if_missing=download_if_missing)
    with np.load(path, allow_pickle=True) as saved:
        return (
            np.asarray(saved["z_grid"], dtype=float),
            np.asarray(saved[_CDE_KEYS[split]], dtype=float),
        )


def _cached_path(split: Split, *, root: str | Path | None, download_if_missing: bool) -> Path:
    if split not in _FILES:
        raise ValueError(f"split must be one of {sorted(_FILES)}, got {split!r}")
    name = _FILES[split]
    dest = data_home(root) / name
    if dest.exists():
        return dest
    if not download_if_missing:
        raise FileNotFoundError(
            f"{dest} is not cached and download_if_missing=False. "
            f"Run fetch_dc1({split!r}) on a machine with network access first."
        )
    _download(f"{_ZENODO}/{name}", dest, _SHA256[name])
    return dest


def _download(url: str, dest: Path, sha256: str) -> None:
    """Fetch ``url`` to ``dest`` atomically, refusing a corrupted download."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"lazy.datasets: downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - the URL is a fixed https literal
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != sha256:
            raise OSError(f"checksum mismatch for {url}\n  got  {digest}\n  want {sha256}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"lazy.datasets: cached {dest} ({dest.stat().st_size / 1e6:.0f} MB)", flush=True)
