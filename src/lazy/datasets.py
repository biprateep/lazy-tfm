# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Built-in photo-z catalogues: fetching, caching, turning them into features.

The library is general purpose -- every estimator takes whatever tabular
features you hand it -- but a photo-z method is hard to judge without a shared
benchmark, so the LSST DESC PZ Data Challenge catalogue (DC1, Schmidt et al.
2020) is available as a one-liner::

    from lazy import datasets

    catalog = datasets.fetch_dc1()      # train and test, concatenated
    X = catalog.features("mag-color")   # the tabular view a model sees

One call loads the data. The challenge's own split is one flag away, for
reproducing its numbers::

    train, test = datasets.fetch_dc1(split=True)

Neither split is a realistic training set: DC1 hands every training galaxy a
redshift, and real spectroscopic samples are bright, incomplete, and cut in
redshift by which features fall in the observed window.
:func:`fetch_dc1_biased` builds that case. It merges both files, reshuffles
them, and cuts the merged catalogue into a training set a HSC-like campaign
would plausibly have produced and a hold-out that is distributed like the
catalogue::

    split = datasets.fetch_dc1_biased()
    split.biased        # 35,011 galaxies the selection function kept
    split.calibration   # 10,000 representative galaxies, to repair the model
    split.test          # 19,383 representative galaxies, to score on

The selection function itself lives in :mod:`lazy.selection`; it acts on any
photometry, so a catalogue of your own can be biased the same way.

The two files total about 1 GB. They are downloaded once from Zenodo record
10975874, checksummed, and cached under :func:`data_home` -- by default
``$XDG_CACHE_HOME/lazy-tfm`` or ``~/.cache/lazy-tfm``, overridable with
the ``LAZY_DATA_HOME`` environment variable or the ``data_home`` argument.
Nothing is re-downloaded if the cached file is already there and intact.

DC1 is Buzzard v1.0 photometry: ``ugrizy`` magnitudes at LSST ten-year depth,
``i < 25.3``, ``0 < z < 2``, 43,486 training and 390,990 test galaxies. Each
file also carries the challenge's ``trainZ`` baseline PDFs, in which every
galaxy is assigned the same training-set N(z); :func:`load_trainz` returns
them, and they are the reference a real estimator has to beat.

Feature construction lives here because these recipes are photometry-specific,
not general tabular engineering -- but they act on a plain
:class:`~pandas.DataFrame`, so a catalogue assembled outside this module goes
through the same code, either via :meth:`Catalog.from_frame` or by calling
:func:`build_features` on the frame directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import hashlib
import itertools
import os
import pathlib
import typing
from typing import Any, Literal, TypeAlias
import urllib.request

import numpy as np
import numpy.typing as npt
import pandas as pd

from lazy import _typing
from lazy import selection

__all__ = [
    "BANDS",
    "FEATURE_MODES",
    "Catalog",
    "SelectionSplit",
    "build_features",
    "data_home",
    "fetch_dc1",
    "fetch_dc1_biased",
    "fetch_hsc_grid",
    "load_trainz",
    "make_selection_split",
]

BANDS = ("U", "G", "R", "I", "Z", "Y")
MAG_COLUMNS = BANDS
ERR_COLUMNS = tuple(f"{band}ERR" for band in BANDS)
RAW_COLUMNS = MAG_COLUMNS + ERR_COLUMNS

FeatureMode: TypeAlias = Literal["mag", "mag-color", "all"]

#: Feature mode -> one-line description, for help text and documentation.
FEATURE_MODES: dict[str, str] = {
    "mag": "the magnitudes and their errors",
    "mag-color": (
        "the reference magnitude and the adjacent colours, with errors in"
        " quadrature"
    ),
    "all": "every magnitude, every colour, and all their errors",
}

_ZENODO = "https://zenodo.org/records/10975874/files"
_FILES = {"train": "trainz_train.npz", "test": "trainz_test.npz"}
_CAT_KEYS = {"train": "train_cat", "test": "test_cat"}
_CDE_KEYS = {"train": "cde_train", "test": "cde_test"}
# The HSC spectroscopic success grid that drives the selection, taken from the
# DESC repository the port in `lazy.selection` was written against, pinned to
# the commit it was read from.
_RAIL = (
    "https://raw.githubusercontent.com/LSSTDESC/rail_astro_tools/9c176272"
    "/src/rail/examples_data/creation_data/data"
)
_HSC_FILE = "hsc_ratios_and_specz.hdf5"
_SHA256 = {
    "trainz_train.npz": (
        "fe54d2034c554ec1f675e7d5789fd5ec4564a72d22bc5008fc218003bc6fb6dc"
    ),
    "trainz_test.npz": (
        "a1b1a2b191d0edbcb83b3a5e87cc8f7c2cd0dc5c564a1bb37258f6e1588c98ce"
    ),
    _HSC_FILE: (
        "9621f9e30baeb87c53a2a8e803043f7c598479f93785a3707954e133074328de"
    ),
}

#: Defaults reproducing the paired split the paper reports: the biased training
#: set is sized to the DC1 training file, and the calibration sample is stolen
#: from the hold-out rather than added to it, so nothing is scored twice.
N_TRAIN = 35_000
N_CALIBRATION = 10_000


@dataclasses.dataclass(frozen=True)
class Catalog:
    """Photometry, truth and identifiers for a catalogue of galaxies.

    ``raw`` is the photometry as loaded: one column per band, one
    ``<BAND>ERR`` column per band. :meth:`features` turns it into the columns
    an estimator actually sees.

    Attributes:
        split: A label naming the catalogue, carried through to :func:`repr`.
        raw: The photometry, magnitudes and their errors in mag, one row per
            galaxy.
        redshift: The true redshifts, shape (n_galaxies,).
        object_id: Identifiers, shape (n_galaxies,).
        source: For a combined catalogue: 0 for rows from the train file, 1
            from test, shape (n_galaxies,); otherwise None.
    """

    split: str
    raw: pd.DataFrame
    redshift: _typing.FloatArray
    object_id: npt.NDArray[Any]
    source: npt.NDArray[np.int8] | None = None

    def __len__(self) -> int:
        return len(self.redshift)

    def __repr__(self) -> str:
        return (
            f"Catalog(split={self.split!r}, n={len(self)}, "
            f"columns={list(self.raw.columns)})"
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        redshift: str | npt.ArrayLike,
        object_id: str | npt.ArrayLike | None = None,
        split: str = "custom",
    ) -> Catalog:
        """Wraps a catalogue you assembled yourself, to treat it like DC1.

        Args:
            frame: Photometry, one column per band and one ``<BAND>ERR``
                column per band, in mag. Columns named by ``redshift`` or
                ``object_id`` are moved out of it rather than left among the
                features.
            redshift: The true redshifts, shape (n_galaxies,), or the name of
                the column holding them.
            object_id: Identifiers, shape (n_galaxies,), or the name of the
                column holding them. Defaults to a row counter.
            split: A label, carried through to :func:`repr` and nothing else.

        Returns:
            The catalogue.

        Examples:
            >>> frame = pd.DataFrame(
            ...     {b: [20.0] for b in BANDS}
            ...     | {f"{b}ERR": [0.1] for b in BANDS}
            ... )
            >>> catalog = Catalog.from_frame(
            ...     frame.assign(z=[0.5]), redshift="z"
            ... )
            >>> len(catalog), list(catalog.raw.columns) == list(RAW_COLUMNS)
            (1, True)
        """
        raw = frame.reset_index(drop=True)
        dropped = [
            name for name in (redshift, object_id) if isinstance(name, str)
        ]
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
                f"frame has {len(raw)} rows but redshift has {len(z)} and "
                f"object_id has {len(ids)}"
            )
        return cls(split=split, raw=raw, redshift=z, object_id=ids)

    def take(self, rows: npt.ArrayLike, *, split: str | None = None) -> Catalog:
        """Returns the rows named by ``rows``, as a catalogue of their own.

        Args:
            rows: Integer positions, in the order you want them back, or a
                boolean mask of shape (n_galaxies,).
            split: A label for the result; defaults to one naming the parent.

        Returns:
            The subset, with every per-galaxy field taken alike.

        Examples:
            >>> frame = pd.DataFrame(
            ...     {b: [20.0, 21.0] for b in BANDS}
            ...     | {f"{b}ERR": [0.1, 0.2] for b in BANDS}
            ... )
            >>> catalog = Catalog.from_frame(
            ...     frame, redshift=np.array([0.5, 1.5])
            ... )
            >>> subset = catalog.take([1], split="bright")
            >>> len(subset), subset.redshift, subset.split
            (1, array([1.5]), 'bright')
        """
        positions = np.asarray(rows)
        if positions.dtype == bool:
            positions = np.flatnonzero(positions)
        positions = positions.astype(np.int64, copy=False)
        return Catalog(
            split=(
                split
                if split is not None
                else f"{self.split}[{len(positions)}]"
            ),
            raw=self.raw.iloc[positions].reset_index(drop=True),
            redshift=self.redshift[positions],
            object_id=self.object_id[positions],
            source=None if self.source is None else self.source[positions],
        )

    def features(
        self,
        mode: FeatureMode = "mag-color",
        *,
        bands: Sequence[str] = BANDS,
        reference_band: str = "I",
    ) -> pd.DataFrame:
        """Returns the feature view named by ``mode`` of this photometry.

        A thin wrapper over :func:`build_features`; see it for what each mode
        contains and for the arguments.

        Args:
            mode: One of :data:`FEATURE_MODES`.
            bands: Band names in wavelength order.
            reference_band: The single magnitude ``mag-color`` keeps.

        Returns:
            One row per galaxy, float32 columns.
        """
        return build_features(
            self.raw, mode, bands=bands, reference_band=reference_band
        )


def build_features(
    raw: pd.DataFrame,
    mode: FeatureMode = "mag-color",
    *,
    bands: Sequence[str] = BANDS,
    reference_band: str = "I",
) -> pd.DataFrame:
    """Turns raw photometry into the tabular view an estimator sees.

    Which columns you feed a tabular foundation model matters more than most
    hyper-parameters. Colours beat magnitudes, and propagating the magnitude
    errors into colour errors beats leaving them as magnitude errors, so the
    default is the reference magnitude plus adjacent colours with matching
    errors.

    Args:
        raw: Photometry with one column per band and one ``<BAND>ERR`` column
            per band, in mag. Any DataFrame will do -- it need not have come
            from :func:`fetch_dc1`.
        mode: One of :data:`FEATURE_MODES`.

            ``mag``
                The magnitudes and their errors, untouched. The honest
                baseline: whatever the model achieves here, it achieved
                without help.
            ``mag-color``
                The reference-band magnitude and its error, the five unique
                adjacent colours (``u-g``, ``g-r``, ``r-i``, ``i-z``,
                ``z-y``), and their errors, each the quadrature sum of the two
                magnitude errors forming the colour. The default: same twelve
                numbers as ``mag``, in the representation that carries the
                spectral shape setting the redshift.
            ``all``
                Every magnitude, every one of the fifteen pairwise colours,
                and the errors for all of them -- magnitude errors as given,
                colour errors in quadrature. Redundant by construction, and a
                test of whether the model would rather be handed the
                differences than infer them.
        bands: Band names in wavelength order.
        reference_band: The single magnitude ``mag-color`` keeps; one of
            ``bands``.

    Returns:
        One row per input row, float32 columns in mag, names chosen to be
        readable in feature-importance output (``"G-R"``, ``"G-RERR"``, ...).

    Raises:
        KeyError: If ``raw`` lacks a magnitude or error column for one of
            ``bands``.

    Examples:
        >>> raw = pd.DataFrame(
        ...     {b: np.full(3, 20.0) for b in BANDS}
        ...     | {f"{b}ERR": np.full(3, 0.1) for b in BANDS}
        ... )
        >>> columns = list(build_features(raw).columns)
        >>> columns[:6]
        ['I', 'U-G', 'G-R', 'R-I', 'I-Z', 'Z-Y']
        >>> columns[6:]
        ['IERR', 'U-GERR', 'G-RERR', 'R-IERR', 'I-ZERR', 'Z-YERR']
    """
    if mode not in FEATURE_MODES:
        raise ValueError(
            f"unknown feature mode {mode!r}; known: {sorted(FEATURE_MODES)}"
        )
    bands = tuple(bands)
    err_columns = tuple(f"{band}ERR" for band in bands)
    missing = [c for c in (*bands, *err_columns) if c not in raw.columns]
    if missing:
        raise KeyError(f"raw photometry is missing columns: {missing}")
    if reference_band not in bands:
        raise ValueError(
            f"reference_band {reference_band!r} is not one of {bands}"
        )

    mags = raw[list(bands)].to_numpy(dtype=np.float32)
    errs = raw[list(err_columns)].to_numpy(dtype=np.float32)
    index = {band: i for i, band in enumerate(bands)}

    # Every mode is a choice of which magnitudes to keep and which colours to
    # form; the errors follow from that choice, in quadrature for colours.
    keep: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    if mode == "mag":
        keep, pairs = bands, ()
    elif mode == "mag-color":
        keep, pairs = (reference_band,), tuple(itertools.pairwise(bands))
    else:
        keep, pairs = bands, tuple(itertools.combinations(bands, 2))

    out: dict[str, npt.NDArray[np.float32]] = {
        band: mags[:, index[band]] for band in keep
    }
    for first, second in pairs:
        out[f"{first}-{second}"] = (
            mags[:, index[first]] - mags[:, index[second]]
        )
    for band in keep:
        out[f"{band}ERR"] = errs[:, index[band]]
    for first, second in pairs:
        out[f"{first}-{second}ERR"] = np.hypot(
            errs[:, index[first]], errs[:, index[second]]
        )
    return pd.DataFrame(out)


def data_home(data_home: str | pathlib.Path | None = None) -> pathlib.Path:
    """Returns the directory cached catalogues live in, creating it if needed.

    Resolution order: the ``data_home`` argument, then ``$LAZY_DATA_HOME``,
    then ``$XDG_CACHE_HOME/lazy-tfm``, then ``~/.cache/lazy-tfm``.

    Nothing about the location is baked in, so a shared scratch filesystem or
    a node-local disk is one environment variable away and the same code runs
    unchanged on a laptop and on a cluster.

    Args:
        data_home: An explicit cache directory, overriding the environment.

    Returns:
        The cache directory, with ``~`` expanded.
    """
    explicit = data_home or os.environ.get("LAZY_DATA_HOME")
    if explicit:
        root = pathlib.Path(explicit)
    else:
        cache = (
            os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache"
        )
        root = pathlib.Path(cache) / "lazy-tfm"
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


@typing.overload
def fetch_dc1(
    split: Literal[False] = ...,
    *,
    data_home: str | pathlib.Path | None = ...,
    download_if_missing: bool = ...,
) -> Catalog: ...


@typing.overload
def fetch_dc1(
    split: Literal[True],
    *,
    data_home: str | pathlib.Path | None = ...,
    download_if_missing: bool = ...,
) -> tuple[Catalog, Catalog]: ...


@typing.overload
def fetch_dc1(
    split: bool,
    *,
    data_home: str | pathlib.Path | None = ...,
    download_if_missing: bool = ...,
) -> Catalog | tuple[Catalog, Catalog]: ...


def fetch_dc1(
    split: bool = False,
    *,
    data_home: str | pathlib.Path | None = None,
    download_if_missing: bool = True,
) -> Catalog | tuple[Catalog, Catalog]:
    """Loads DC1, downloading and caching it on first use.

    Both files are read either way -- the flag chooses how they are handed
    back, not how much is downloaded.

    Args:
        split: ``False`` (the default) returns one catalogue holding both
            files, 434,476 rows, train first, with ``source`` marking which
            file each row came from. ``True`` returns ``(train, test)`` -- the
            challenge's own 43,486/390,990 split, for reproducing its numbers.
        data_home: Cache directory; see :func:`data_home`.
        download_if_missing: When ``False``, raise rather than reach for the
            network. Use it on a compute node with no outbound connection,
            having warmed the cache elsewhere.

    Returns:
        A :class:`Catalog`, or a tuple (train, test) of them. ``raw`` holds
        the six magnitudes and their six errors, in mag, as float32;
        ``redshift`` is the true redshift as float64.

    Raises:
        FileNotFoundError: If a file is not cached and
            ``download_if_missing`` is ``False``.
        OSError: If a download fails or does not match its checksum.
    """
    train = _load_split(
        "train", root=data_home, download_if_missing=download_if_missing
    )
    test = _load_split(
        "test", root=data_home, download_if_missing=download_if_missing
    )
    if split:
        return train, test
    return Catalog(
        split="train+test",
        raw=pd.concat([train.raw, test.raw], ignore_index=True),
        redshift=np.concatenate([train.redshift, test.redshift]),
        object_id=np.concatenate([train.object_id, test.object_id]),
        source=np.concatenate(
            [
                np.zeros(len(train), dtype=np.int8),
                np.ones(len(test), dtype=np.int8),
            ]
        ),
    )


def load_trainz(
    split: bool = False,
    *,
    data_home: str | pathlib.Path | None = None,
    download_if_missing: bool = True,
) -> tuple[_typing.FloatArray, ...]:
    """Loads the DC1 ``trainZ`` baseline PDFs.

    Every row is the same training-set N(z): the challenge's deliberately
    trivial estimator, and the floor any real method must clear. The stored
    array is a per-bin probability on a 0.01-wide grid, so normalise it as a
    density (:meth:`lazy.grid.RedshiftGrid.normalize`) before scoring.

    Args:
        split: Follows :func:`fetch_dc1`: ``False`` stacks the two files in
            the same order, so the rows line up with ``fetch_dc1()``; ``True``
            keeps them apart.
        data_home: Cache directory; see :func:`data_home`.
        download_if_missing: When ``False``, raise rather than reach for the
            network.

    Returns:
        A tuple (z_grid, pdfs), or (z_grid, train_pdfs, test_pdfs) when
        ``split`` is true. ``z_grid`` has shape (n_z,); each PDF array has
        shape (n_galaxies, n_z).

    Raises:
        FileNotFoundError: If a file is not cached and
            ``download_if_missing`` is ``False``.
        OSError: If a download fails or does not match its checksum.
    """
    z_grid, train = _load_trainz_split(
        "train", root=data_home, download_if_missing=download_if_missing
    )
    _, test = _load_trainz_split(
        "test", root=data_home, download_if_missing=download_if_missing
    )
    if split:
        return z_grid, train, test
    return z_grid, np.concatenate([train, test])


def _load_split(
    name: str, *, root: str | pathlib.Path | None, download_if_missing: bool
) -> Catalog:
    """Loads one DC1 file as a :class:`Catalog`."""
    path = _cached_path(
        name, root=root, download_if_missing=download_if_missing
    )
    with np.load(path, allow_pickle=True) as saved:
        cat = saved[_CAT_KEYS[name]]
        raw = pd.DataFrame(
            {
                column: np.asarray(cat[column], dtype=np.float32)
                for column in RAW_COLUMNS
            }
        )
        redshift = np.asarray(cat["SPECZ"], dtype=np.float64)
        object_id = np.asarray(cat["ID"], dtype=np.int64)
    return Catalog(split=name, raw=raw, redshift=redshift, object_id=object_id)


def _load_trainz_split(
    name: str, *, root: str | pathlib.Path | None, download_if_missing: bool
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Loads the ``z_grid`` and ``trainZ`` PDFs stored in one DC1 file."""
    path = _cached_path(
        name, root=root, download_if_missing=download_if_missing
    )
    with np.load(path, allow_pickle=True) as saved:
        return (
            np.asarray(saved["z_grid"], dtype=float),
            np.asarray(saved[_CDE_KEYS[name]], dtype=float),
        )


def _cached_path(
    name: str, *, root: str | pathlib.Path | None, download_if_missing: bool
) -> pathlib.Path:
    """Returns the cached copy of the DC1 file ``name``, fetched if needed."""
    return _cached_file(
        _FILES[name],
        f"{_ZENODO}/{_FILES[name]}",
        root=root,
        download_if_missing=download_if_missing,
        hint="fetch_dc1()",
    )


def _cached_file(
    filename: str,
    url: str,
    *,
    root: str | pathlib.Path | None,
    download_if_missing: bool,
    hint: str,
) -> pathlib.Path:
    """Returns the cached copy of one remote file, fetched and checksummed."""
    dest = data_home(root) / filename
    if dest.exists():
        return dest
    if not download_if_missing:
        raise FileNotFoundError(
            f"{dest} is not cached and download_if_missing=False. "
            f"Run {hint} on a machine with network access first."
        )
    _download(url, dest, _SHA256[filename])
    return dest


def _download(url: str, dest: pathlib.Path, sha256: str) -> None:
    """Fetches ``url`` to ``dest`` atomically, refusing a corrupted download."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"lazy.datasets: downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != sha256:
            raise OSError(
                f"checksum mismatch for {url}\n  got  {digest}\n  want {sha256}"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(
        f"lazy.datasets: cached {dest} ({dest.stat().st_size / 1e6:.0f} MB)",
        flush=True,
    )


# -- the spectroscopically selected split ----------------------------------


@dataclasses.dataclass(frozen=True)
class SelectionSplit:
    """A biased training set, and representative galaxies to calibrate with.

    What :func:`make_selection_split` returns:

    * :attr:`biased` -- the galaxies the selection function kept. Bright,
      incomplete, and truncated in redshift: the training set.
    * :attr:`calibration` -- a small sample distributed like the catalogue.
      Not scored on: it is there to be *used*, as extra context, as importance
      weights, or as a recalibration set.
    * :attr:`test` -- the rest of the hold-out, distributed like the catalogue
      and disjoint from everything above.
    * :attr:`unbiased` -- ``None`` unless ``control=True`` was asked for, in
      which case it holds exactly as many galaxies as :attr:`biased`, drawn at
      random from the same pool. It is the control that separates the
      selection from the sample size, for when that distinction is the
      question.

    :attr:`rows` holds each subset's positions in the parent catalogue, so a
    split can be frozen to disk and reproduced later, and :attr:`meta` records
    how it was built.

    Attributes:
        biased: The non-representative training set.
        test: Representative galaxies, held out from everything else, to score
            on.
        calibration: A representative sample to repair the biased model with,
            or ``None``.
        unbiased: The size-matched random control, or ``None`` if it was not
            asked for.
        rows: Subset name -> sorted positions in the catalogue the split was
            cut from.
        meta: How it was built: sizes, seeds, selection arguments, solver
            history.
    """

    biased: Catalog
    test: Catalog
    calibration: Catalog | None
    unbiased: Catalog | None
    rows: dict[str, _typing.IntArray]
    meta: dict[str, Any]

    def __repr__(self) -> str:
        sizes = ", ".join(
            f"{name}={len(cat):,}" for name, cat in self.catalogs.items()
        )
        return f"SelectionSplit({sizes})"

    @property
    def catalogs(self) -> dict[str, Catalog]:
        """The subsets by name, skipping any that were not asked for."""
        named = {"biased": self.biased}
        if self.unbiased is not None:
            named["unbiased"] = self.unbiased
        if self.calibration is not None:
            named["calibration"] = self.calibration
        named["test"] = self.test
        return named

    def summary(
        self, *, magnitude: str = "I", faint: float = 24.0
    ) -> pd.DataFrame:
        """Tabulates each subset: how many, how bright, how deep in redshift.

        The row labelled ``catalog`` is the catalogue the split was cut from.
        Read it against ``test`` to see that the test set is representative,
        and against ``biased`` to see that the training set is not: on DC1 the
        selection moves the median magnitude about a magnitude brighter and
        cuts the faint fraction by roughly an order of magnitude.

        Args:
            magnitude: The band whose magnitudes are summarised.
            faint: The magnitude, in mag, beyond which a galaxy counts as
                faint.

        Returns:
            One row per subset, indexed by ``set``: the count, the median
            magnitude and redshift, the maximum redshift and the faint
            fraction.
        """
        rows = [
            _describe(cat, magnitude=magnitude, faint=faint) | {"set": name}
            for name, cat in self.catalogs.items()
        ]
        rows.append(self.meta["catalog"] | {"set": "catalog"})
        return pd.DataFrame(rows).set_index("set")


def _describe(
    catalog: Catalog, *, magnitude: str = "I", faint: float = 24.0
) -> dict[str, Any]:
    """Returns the numbers that say whether two samples are drawn alike."""
    mag = catalog.raw[magnitude].to_numpy(dtype=float)
    return {
        "n": len(catalog),
        f"median_{magnitude}": float(np.median(mag)),
        "median_z": float(np.median(catalog.redshift)),
        "max_z": float(catalog.redshift.max()),
        f"frac_{magnitude}_gt_{faint:g}": float((mag > faint).mean()),
    }


def fetch_hsc_grid(
    *,
    data_home: str | pathlib.Path | None = None,
    download_if_missing: bool = True,
) -> selection.HSCGrid:
    """Loads the HSC spectroscopic success grid, downloading it on first use.

    About 14 MB, from the DESC ``rail_astro_tools`` repository at the commit
    :mod:`lazy.selection` was ported from. The loaded grid is memoised, so the
    per-pixel redshift ceilings are computed once per process however many
    splits you cut.

    Args:
        data_home: Cache directory; see :func:`data_home`.
        download_if_missing: When ``False``, raise rather than reach for the
            network.

    Returns:
        The grid.

    Raises:
        FileNotFoundError: If the grid is not cached and
            ``download_if_missing`` is ``False``.
        OSError: If the download fails or does not match its checksum.
    """
    path = _cached_file(
        _HSC_FILE,
        f"{_RAIL}/{_HSC_FILE}",
        root=data_home,
        download_if_missing=download_if_missing,
        hint="fetch_hsc_grid()",
    )
    return _load_hsc_grid(path)


@functools.lru_cache(maxsize=4)
def _load_hsc_grid(path: pathlib.Path) -> selection.HSCGrid:
    """Reads a cached grid file once per process; the result is immutable."""
    return selection.HSCGrid.from_hdf5(path)


def make_selection_split(
    catalog: Catalog,
    *,
    n_train: int = N_TRAIN,
    n_calibration: int = N_CALIBRATION,
    control: bool = False,
    holdout_rows: npt.ArrayLike | None = None,
    grid: selection.HSCGrid | None = None,
    seed: int = 1,
    selection_seed: int = 12345,
    tolerance: int = 50,
    max_iterations: int = 8,
    initial_rate: float = 0.0864,
    **selection_options: Any,
) -> SelectionSplit:
    """Cuts a catalogue into a biased training set, a control, and a test set.

    The recipe, in order:

    1. shuffle the catalogue and cut it in two;
    2. the first part is the *pool*: apply
       :func:`~lazy.selection.grid_selection` to it, and what it keeps is
       :attr:`~SelectionSplit.biased`;
    3. the second part is the *hold-out*: split it at random into the
       calibration sample and the test set.

    Where the cut in step 1 falls is what sets the training-set size, because
    the selection keeps a near-fixed fraction of whatever it is shown. So the
    cut is solved for rather than chosen: the first pass places it using
    ``initial_rate``, measures what the selection actually kept, and re-places
    it, usually converging in two passes.

    The calibration sample comes *out* of the hold-out rather than on top of
    it, so no galaxy is both handed to a model and scored on.

    Note:
        The cut in step 1 is placed from the *measured* selection rate, so
        changing ``selection_seed`` or the selection arguments can move it by
        a few hundred galaxies and with it the hold-out, the calibration
        sample and the test set. When you are comparing selections, cut one
        split and carry :attr:`SelectionSplit.rows` between the runs rather
        than re-cutting.

        The training set cannot exceed what the selection returns on the
        whole catalogue -- on DC1 that ceiling is 37,626 galaxies, and
        reaching it would leave nothing to test on.

    Args:
        catalog: The complete catalogue to cut up, typically ``fetch_dc1()``
            -- both DC1 files merged, because the challenge's own division is
            far too small on the training side to survive a selection.
        n_train: How many galaxies the biased training set should end up
            with, to within ``tolerance``; positive. This, not the test
            fraction, is the knob worth turning: it is the axis a photo-z
            method is usually judged along.
        n_calibration: Size of the calibration sample, taken out of the
            hold-out; non-negative, and smaller than the hold-out. ``0``
            leaves the whole hold-out as the test set and
            :attr:`SelectionSplit.calibration` as ``None``.
        control: Also draw an unbiased training set of exactly the biased
            one's size from the same pool. Off by default: it answers a
            different question -- whether a result is the selection or merely
            the smaller sample -- and costs a second set of runs. It is drawn
            last, so turning it on leaves every other subset unchanged.
        holdout_rows: Restrict the hold-out to these rows. The default
            shuffles the whole catalogue, as above. Passing the rows of one
            source file keeps a model trained on the other file scorable on
            this test set without leakage.
        grid: The selection function; defaults to the cached HSC grid.
        seed: Seeds the hold-out draw, the control draw and the calibration
            steal.
        selection_seed: Seeds the selection's own per-pixel subsampling, and
            nothing else. Separate from ``seed`` so that the two questions --
            which galaxies a campaign got a redshift for, and which galaxies
            were held back from it -- can be re-rolled independently. RAIL's
            default is kept.
        tolerance: For the hold-out solver: how close to ``n_train`` is close
            enough.
        max_iterations: How many passes the hold-out solver may take.
        initial_rate: The selection rate the hold-out solver starts from
            (0.0864 is HSC's on a representative LSST-depth sample).
        **selection_options: Forwarded to
            :func:`~lazy.selection.grid_selection`: ``scaling_factor``,
            ``color_redshift_cut``, ``percentile_cut``, ``redshift_cut``,
            ``magnitude`` and ``color``.

    Returns:
        The split.

    Raises:
        RuntimeError: If the hold-out solver cannot reach ``n_train``, which
            means the catalogue is too small for a training set that size, or
            the selection keeps too little of it.
    """
    if grid is None:
        grid = fetch_hsc_grid()
    n = len(catalog)
    holdout_rows = (
        np.arange(n)
        if holdout_rows is None
        else np.asarray(holdout_rows, np.int64).ravel()
    )
    if n_calibration < 0 or n_train < 1:
        raise ValueError(
            "n_train must be positive and n_calibration non-negative"
        )
    if len(holdout_rows) <= n_calibration:
        raise ValueError(
            f"only {len(holdout_rows):,} rows may be held out, which leaves "
            f"nothing to test on after a calibration sample of "
            f"{n_calibration:,}"
        )

    rng = np.random.default_rng(seed)
    holdout, pool, biased_rows, history = _solve_holdout(
        catalog,
        rng.permutation(holdout_rows),
        n_train=n_train,
        n_calibration=n_calibration,
        grid=grid,
        selection_seed=selection_seed,
        tolerance=tolerance,
        max_iterations=max_iterations,
        initial_rate=initial_rate,
        selection_options=selection_options,
    )

    # The hold-out is already a random draw; its split into calibration and
    # test is another. The control comes last so that asking for it moves
    # nothing else.
    calibration_rows, test_rows = (
        holdout[:n_calibration],
        holdout[n_calibration:],
    )
    rows = {"biased": np.sort(biased_rows), "test": np.sort(test_rows)}
    if n_calibration:
        rows["calibration"] = np.sort(calibration_rows)
    n_overlap = None
    if control:
        unbiased_rows = rng.choice(pool, size=len(biased_rows), replace=False)
        rows["unbiased"] = np.sort(unbiased_rows)
        n_overlap = int(np.intersect1d(biased_rows, unbiased_rows).size)
    meta = {
        "n_catalog": n,
        "n_holdout": len(holdout),
        "n_pool": len(pool),
        "n_pool_unused": int(len(pool) - len(biased_rows)),
        "n_biased_also_unbiased": n_overlap,
        "seed": seed,
        "selection": {"seed": selection_seed} | dict(selection_options),
        "history": history,
        "catalog": _describe(catalog),
    }
    subsets = {name: catalog.take(rows[name], split=name) for name in rows}
    return SelectionSplit(
        biased=subsets["biased"],
        test=subsets["test"],
        calibration=subsets.get("calibration"),
        unbiased=subsets.get("unbiased"),
        rows=rows,
        meta=meta,
    )


def _solve_holdout(
    catalog: Catalog,
    candidates: _typing.IntArray,
    *,
    n_train: int,
    n_calibration: int,
    grid: selection.HSCGrid,
    selection_seed: int,
    tolerance: int,
    max_iterations: int,
    initial_rate: float,
    selection_options: Mapping[str, Any],
) -> tuple[
    _typing.IntArray, _typing.IntArray, _typing.IntArray, list[dict[str, Any]]
]:
    """Sizes the hold-out so that the selection keeps about ``n_train``.

    Args:
        catalog: The catalogue being cut up.
        candidates: Shuffled rows the hold-out may be drawn from, in order.
        n_train: The target size of the biased training set.
        n_calibration: The calibration sample size; the hold-out keeps at
            least one row more.
        grid: The selection function.
        selection_seed: Seeds the selection's per-pixel subsampling.
        tolerance: How close to ``n_train`` is close enough.
        max_iterations: How many passes may be taken.
        initial_rate: The selection rate the first pass assumes.
        selection_options: Forwarded to
            :func:`~lazy.selection.grid_selection`.

    Returns:
        A tuple (holdout, pool, biased_rows, history): the hold-out rows, in
        draw order; the rest of the catalogue, sorted; the pool rows the
        selection kept; and one record per pass.

    Raises:
        RuntimeError: If no pass lands within ``tolerance`` of ``n_train``.
    """
    n = len(catalog)
    everything = np.arange(n)
    rate = float(initial_rate)
    history: list[dict[str, Any]] = []
    for iteration in range(max_iterations):
        n_holdout = int(
            np.clip(
                round(n - n_train / rate), n_calibration + 1, len(candidates)
            )
        )
        holdout = candidates[:n_holdout]
        pool = np.setdiff1d(everything, holdout, assume_unique=True)
        keep, _ = selection.grid_selection(
            catalog.raw.iloc[pool],
            catalog.redshift[pool],
            grid=grid,
            seed=selection_seed,
            **selection_options,
        )
        biased_rows, n_selected = pool[keep], int(keep.sum())
        history.append(
            {
                "iteration": iteration,
                "n_holdout": n_holdout,
                "n_pool": len(pool),
                "n_biased": n_selected,
                "rate": float(n_selected / len(pool)),
            }
        )
        if abs(n_selected - n_train) <= tolerance:
            return holdout, pool, biased_rows, history
        if n_selected == 0:
            raise RuntimeError(
                "the selection kept nothing; check that the magnitude and "
                "colour columns are the ones the grid was built for"
            )
        rate = n_selected / len(pool)
    best = max(row["n_biased"] for row in history)
    raise RuntimeError(
        f"could not size a training set of {n_train:,} within "
        f"{max_iterations} passes; the best was {best:,}. The selection keeps "
        f"about {history[-1]['rate']:.2%} of what it is shown, so this "
        f"catalogue cannot supply more than "
        f"~{int(history[-1]['rate'] * n):,} however it is cut, and a training "
        "set that size would leave nothing to test on."
    )


def fetch_dc1_biased(
    *,
    n_train: int = N_TRAIN,
    n_calibration: int = N_CALIBRATION,
    control: bool = False,
    seed: int = 1,
    selection_seed: int = 12345,
    data_home: str | pathlib.Path | None = None,
    download_if_missing: bool = True,
    **selection_options: Any,
) -> SelectionSplit:
    """Loads DC1 and cuts the spectroscopic-selection split from it.

    Both DC1 files are merged and reshuffled first. The challenge's own
    division cannot carry this experiment: the HSC selection keeps 3,186 of
    its 43,486 training galaxies, so a biased training set would also be a
    twelve-times smaller one.

    ``n_train`` cannot reach the 43,486 of the DC1 training file. The
    selection keeps 8.66 per cent of a representative sample, so all 434,476
    galaxies yield at most 37,626 -- and that with nothing left to test on.
    The default of 35,000 is the largest round number that still leaves a
    usable hold-out, and it is the size the runs reported for this work used.

    Args:
        n_train: See :func:`make_selection_split`.
        n_calibration: See :func:`make_selection_split`.
        control: See :func:`make_selection_split`.
        seed: See :func:`make_selection_split`.
        selection_seed: See :func:`make_selection_split`.
        data_home: See :func:`fetch_dc1`. Both DC1 files and the ~14 MB HSC
            grid must be available; on a node with no network, warm the cache
            elsewhere first.
        download_if_missing: See :func:`fetch_dc1`.
        **selection_options: See :func:`make_selection_split`.

    Returns:
        The split.

    Raises:
        FileNotFoundError: If a file is not cached and
            ``download_if_missing`` is ``False``.
        OSError: If a download fails or does not match its checksum.
        RuntimeError: If the hold-out solver cannot reach ``n_train``.

    Examples:
        >>> split = fetch_dc1_biased()  # doctest: +SKIP
        >>> X = split.biased.features("mag-color")  # doctest: +SKIP
        >>> model.fit(X, split.biased.redshift)  # doctest: +SKIP
    """
    catalog = fetch_dc1(
        data_home=data_home, download_if_missing=download_if_missing
    )
    grid = fetch_hsc_grid(
        data_home=data_home, download_if_missing=download_if_missing
    )
    return make_selection_split(
        catalog,
        n_train=n_train,
        n_calibration=n_calibration,
        control=control,
        grid=grid,
        seed=seed,
        selection_seed=selection_seed,
        **selection_options,
    )
