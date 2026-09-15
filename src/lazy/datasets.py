"""Built-in photo-z catalogues: fetching them, caching them, turning them into features.

The library is general purpose -- every estimator takes whatever tabular
features you hand it -- but a photo-z method is hard to judge without a shared
benchmark, so the LSST DESC PZ Data Challenge catalogue (DC1, Schmidt et al.
2020) is available as a one-liner::

    from lazy.datasets import fetch_dc1

    catalog = fetch_dc1()                     # train and test, concatenated
    X = catalog.features("mag-color")         # the tabular view a model sees

One call loads the data. The challenge's own split is one flag away, for
reproducing its numbers::

    train, test = fetch_dc1(split=True)

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

Feature construction lives on :class:`Catalog` because these recipes are
photometry-specific, not general tabular engineering -- but they act on a plain
:class:`~pandas.DataFrame`, so a catalogue assembled outside this module goes
through the same code, either via :meth:`Catalog.from_frame` or by calling
:meth:`Catalog.build_features` on the frame directly.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["BANDS", "FEATURE_MODES", "Catalog", "data_home", "fetch_dc1", "load_trainz"]

BANDS = ("U", "G", "R", "I", "Z", "Y")
MAG_COLUMNS = BANDS
ERR_COLUMNS = tuple(f"{band}ERR" for band in BANDS)
RAW_COLUMNS = MAG_COLUMNS + ERR_COLUMNS

FeatureMode = Literal["mag", "mag-color", "all"]

#: Feature mode -> one-line description, for help text and documentation.
FEATURE_MODES: dict[str, str] = {
    "mag": "the magnitudes and their errors",
    "mag-color": "the reference magnitude and the adjacent colours, with errors in quadrature",
    "all": "every magnitude, every colour, and all their errors",
}

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
    """Photometry, truth and identifiers for a catalogue, and the feature views of it.

    ``raw`` is the photometry as loaded: one column per band, one ``<BAND>ERR``
    column per band. :meth:`features` turns it into the columns an estimator
    actually sees.
    """

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

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        redshift: str | np.ndarray,
        object_id: str | np.ndarray | None = None,
        split: str = "custom",
    ) -> Catalog:
        """Wrap a catalogue you assembled yourself, so it gets the same treatment as DC1.

        Parameters
        ----------
        frame
            Photometry, one column per band and one ``<BAND>ERR`` column per
            band. Columns named by ``redshift`` or ``object_id`` are moved out
            of it rather than left among the features.
        redshift
            The true redshifts, or the name of the column holding them.
        object_id
            Identifiers, or the name of the column holding them. Defaults to a
            row counter.
        split
            A label, carried through to :func:`repr` and nothing else.

        Examples
        --------
        >>> frame = pd.DataFrame({b: [20.0] for b in BANDS} | {f"{b}ERR": [0.1] for b in BANDS})
        >>> catalog = Catalog.from_frame(frame.assign(z=[0.5]), redshift="z")
        >>> len(catalog), list(catalog.raw.columns) == list(RAW_COLUMNS)
        (1, True)
        """
        raw = frame.reset_index(drop=True)
        dropped = [name for name in (redshift, object_id) if isinstance(name, str)]
        if isinstance(redshift, str):
            z = np.asarray(raw[redshift], dtype=np.float64)
        else:
            z = np.asarray(redshift, dtype=np.float64).ravel()
        if isinstance(object_id, str):
            ids = np.asarray(raw[object_id])
        elif object_id is None:
            ids = np.arange(len(raw), dtype=np.int64)
        else:
            ids = np.asarray(object_id).ravel()
        raw = raw.drop(columns=dropped)
        if len(z) != len(raw) or len(ids) != len(raw):
            raise ValueError(
                f"frame has {len(raw)} rows but redshift has {len(z)} and object_id has {len(ids)}"
            )
        return cls(split=split, raw=raw, redshift=z, object_id=ids)

    def features(
        self,
        mode: FeatureMode = "mag-color",
        *,
        bands: tuple[str, ...] = BANDS,
        reference_band: str = "I",
    ) -> pd.DataFrame:
        """The feature view named by ``mode``, built from this catalogue's photometry.

        A thin wrapper over :meth:`build_features`; see it for what each mode
        contains.
        """
        return self.build_features(self.raw, mode, bands=bands, reference_band=reference_band)

    @staticmethod
    def build_features(
        raw: pd.DataFrame,
        mode: FeatureMode = "mag-color",
        *,
        bands: tuple[str, ...] = BANDS,
        reference_band: str = "I",
    ) -> pd.DataFrame:
        """Turn raw photometry into the tabular view an estimator sees.

        Which columns you feed a tabular foundation model matters more than most
        hyper-parameters. Colours beat magnitudes, and propagating the magnitude
        errors into colour errors beats leaving them as magnitude errors, so the
        default is the reference magnitude plus adjacent colours with matching
        errors.

        Parameters
        ----------
        raw
            Photometry with one column per band and one ``<BAND>ERR`` column per
            band. Any DataFrame will do -- it need not have come from
            :func:`fetch_dc1`.
        mode
            One of :data:`FEATURE_MODES`.

            ``mag``
                The magnitudes and their errors, untouched. The honest
                baseline: whatever the model achieves here, it achieved without
                help.
            ``mag-color``
                The reference-band magnitude and its error, the five unique
                adjacent colours (``u-g``, ``g-r``, ``r-i``, ``i-z``, ``z-y``),
                and their errors, each the quadrature sum of the two magnitude
                errors forming the colour. The default: same twelve numbers as
                ``mag``, in the representation that carries the spectral shape
                setting the redshift.
            ``all``
                Every magnitude, every one of the fifteen pairwise colours, and
                the errors for all of them -- magnitude errors as given, colour
                errors in quadrature. Redundant by construction, and a test of
                whether the model would rather be handed the differences than
                infer them.
        bands
            Band names in wavelength order.
        reference_band
            The single magnitude ``mag-color`` keeps.

        Returns
        -------
        pandas.DataFrame
            One row per input row, float32 columns, names chosen to be readable
            in feature-importance output (``"G-R"``, ``"G-RERR"``, ...).

        Examples
        --------
        >>> raw = pd.DataFrame(
        ...     {b: np.full(3, 20.0) for b in BANDS} | {f"{b}ERR": np.full(3, 0.1) for b in BANDS}
        ... )
        >>> list(Catalog.build_features(raw).columns)
        ['I', 'U-G', 'G-R', 'R-I', 'I-Z', 'Z-Y', 'IERR', 'U-GERR', 'G-RERR', 'R-IERR', 'I-ZERR', 'Z-YERR']
        """
        if mode not in FEATURE_MODES:
            raise ValueError(f"unknown feature mode {mode!r}; known: {sorted(FEATURE_MODES)}")
        err_columns = tuple(f"{band}ERR" for band in bands)
        missing = [c for c in (*bands, *err_columns) if c not in raw.columns]
        if missing:
            raise KeyError(f"raw photometry is missing columns: {missing}")
        if reference_band not in bands:
            raise ValueError(f"reference_band {reference_band!r} is not one of {bands}")

        mags = raw[list(bands)].to_numpy(dtype=np.float32)
        errs = raw[list(err_columns)].to_numpy(dtype=np.float32)
        index = {band: i for i, band in enumerate(bands)}

        # Every mode is a choice of which magnitudes to keep and which colours
        # to form; the errors follow from that choice, in quadrature for colours.
        if mode == "mag":
            keep, pairs = bands, ()
        elif mode == "mag-color":
            keep, pairs = (reference_band,), tuple(itertools.pairwise(bands))
        else:
            keep, pairs = bands, tuple(itertools.combinations(bands, 2))

        out: dict[str, np.ndarray] = {band: mags[:, index[band]] for band in keep}
        for first, second in pairs:
            out[f"{first}-{second}"] = mags[:, index[first]] - mags[:, index[second]]
        for band in keep:
            out[f"{band}ERR"] = errs[:, index[band]]
        for first, second in pairs:
            out[f"{first}-{second}ERR"] = np.hypot(errs[:, index[first]], errs[:, index[second]])
        return pd.DataFrame(out)


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
    split: bool = False,
    *,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> Catalog | tuple[Catalog, Catalog]:
    """Load DC1, downloading and caching it on first use.

    Parameters
    ----------
    split
        ``False`` (the default) returns one catalogue holding both files,
        434,476 rows, train first, with ``source`` marking which file each row
        came from. ``True`` returns ``(train, test)`` -- the challenge's own
        43,486/390,990 split, for reproducing its numbers.
    data_home
        Cache directory; see :func:`data_home`.
    download_if_missing
        When ``False``, raise rather than reach for the network. Use it on a
        compute node with no outbound connection, having warmed the cache
        elsewhere.

    Returns
    -------
    Catalog or tuple of Catalog
        ``raw`` holds the six magnitudes and their six errors as float32;
        ``redshift`` is the true redshift as float64.

    Notes
    -----
    Both files are read either way -- the flag chooses how they are handed
    back, not how much is downloaded.
    """
    train = _load_split("train", root=data_home, download_if_missing=download_if_missing)
    test = _load_split("test", root=data_home, download_if_missing=download_if_missing)
    if split:
        return train, test
    return Catalog(
        split="train+test",
        raw=pd.concat([train.raw, test.raw], ignore_index=True),
        redshift=np.concatenate([train.redshift, test.redshift]),
        object_id=np.concatenate([train.object_id, test.object_id]),
        source=np.concatenate([np.zeros(len(train), dtype=np.int8), np.ones(len(test), dtype=np.int8)]),
    )


def load_trainz(
    split: bool = False,
    *,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> tuple[np.ndarray, ...]:
    """The DC1 ``trainZ`` baseline PDFs, as ``(z_grid, pdfs)``.

    Every row is the same training-set N(z): the challenge's deliberately
    trivial estimator, and the floor any real method must clear. The stored
    array is a per-bin probability on a 0.01-wide grid, so normalise it as a
    density (:meth:`lazy.grid.RedshiftGrid.normalize`) before scoring.

    ``split`` follows :func:`fetch_dc1`: ``False`` stacks the two files in the
    same order, so the rows line up with ``fetch_dc1()``; ``True`` returns
    ``(z_grid, train_pdfs, test_pdfs)``.
    """
    z_grid, train = _load_trainz_split("train", root=data_home, download_if_missing=download_if_missing)
    _, test = _load_trainz_split("test", root=data_home, download_if_missing=download_if_missing)
    if split:
        return z_grid, train, test
    return z_grid, np.concatenate([train, test])


def _load_split(name: str, *, root: str | Path | None, download_if_missing: bool) -> Catalog:
    """One DC1 file as a :class:`Catalog`."""
    path = _cached_path(name, root=root, download_if_missing=download_if_missing)
    with np.load(path, allow_pickle=True) as saved:
        cat = saved[_CAT_KEYS[name]]
        raw = pd.DataFrame({column: np.asarray(cat[column], dtype=np.float32) for column in RAW_COLUMNS})
        redshift = np.asarray(cat["SPECZ"], dtype=np.float64)
        object_id = np.asarray(cat["ID"], dtype=np.int64)
    return Catalog(split=name, raw=raw, redshift=redshift, object_id=object_id)


def _load_trainz_split(
    name: str, *, root: str | Path | None, download_if_missing: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The ``z_grid`` and ``trainZ`` PDFs stored in one DC1 file."""
    path = _cached_path(name, root=root, download_if_missing=download_if_missing)
    with np.load(path, allow_pickle=True) as saved:
        return (
            np.asarray(saved["z_grid"], dtype=float),
            np.asarray(saved[_CDE_KEYS[name]], dtype=float),
        )


def _cached_path(name: str, *, root: str | Path | None, download_if_missing: bool) -> Path:
    dest = data_home(root) / _FILES[name]
    if dest.exists():
        return dest
    if not download_if_missing:
        raise FileNotFoundError(
            f"{dest} is not cached and download_if_missing=False. "
            "Run fetch_dc1() on a machine with network access first."
        )
    _download(f"{_ZENODO}/{_FILES[name]}", dest, _SHA256[_FILES[name]])
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
