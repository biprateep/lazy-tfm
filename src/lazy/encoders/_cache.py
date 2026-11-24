# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Embeddings on disk, one file per item.

An encoder files each item's embedding under a key of the item's content,
in a folder named after the encoder's identity
(:meth:`lazy.encoders.BaseEncoder._cache_identity`), so an item is encoded
once however many times, and in however many subsets, it is asked for::

    <root>/v1/<key of the identity>/identity.json
    <root>/v1/<key of the identity>/<2 digits>/<30 digits>.npy

The root is ``embeddings`` under :func:`lazy.datasets.data_home`, or the
directory an encoder's ``cache`` parameter names. ``v1`` names this layout.
``identity.json`` holds the identity as text, to tell whose folder it is.
Each ``.npy`` file holds one embedding, shape (width,), and its name is the
:func:`digest` of the item's content.

A file is written whole and then moved into place, so that several
processes can share a cache. A file that cannot be read, or that holds
another width, counts as missing.

Typical usage example:

  store = _cache.Store(True, identity, width)
  rows, found = store.load(keys)
  store.save(keys[~found], new_rows)
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import tempfile

import numpy as np
import numpy.typing as npt

from lazy import _home
from lazy import _typing
from lazy import _warnings

__all__ = ["LAYOUT", "Store", "clear_embedding_cache", "digest"]

LAYOUT = "v1"
"""The name of the layout the module docstring describes."""

_SUBDIRECTORY = "embeddings"


def digest(content: bytes) -> str:
    """A key for ``content``: 32 hexadecimal digits of its BLAKE2b hash."""
    return hashlib.blake2b(content, digest_size=16).hexdigest()


class Store:
    """The embeddings of one encoder, on disk.

    Attributes:
        directory: The encoder's folder, which exists once something is
            saved.
        width: The length of an embedding.
    """

    def __init__(
        self, cache: bool | str | os.PathLike[str], identity: str, width: int
    ):
        """Names the folder; nothing is read or written yet.

        Args:
            cache: ``True`` for the default cache, or the directory to use
                in its place.
            identity: What sets the encoder's output; see
                :meth:`lazy.encoders.BaseEncoder._cache_identity`.
            width: The length of an embedding.
        """
        root = _root() if isinstance(cache, bool) else pathlib.Path(cache)
        self.directory = root.expanduser() / LAYOUT / digest(identity.encode())
        self.width = width
        self._identity = identity
        self._writable = True

    def load(
        self, keys: npt.NDArray[np.object_]
    ) -> tuple[_typing.FloatArray, _typing.BoolArray]:
        """Reads the embeddings filed under ``keys``.

        Args:
            keys: One key per item, from :func:`digest`, shape (n,).

        Returns:
            A tuple (rows, found): the embeddings, shape
            (n, width), with NaN in the rows that are not in the
            cache, and which rows were found, shape (n,).
        """
        rows = np.full((len(keys), self.width), np.nan)
        found = np.zeros(len(keys), dtype=bool)
        for i, key in enumerate(keys):
            try:
                row = np.load(self._path(key), allow_pickle=False)
            except Exception:  # noqa: BLE001 - whatever is wrong with the file, it holds no embedding.
                continue
            if (
                isinstance(row, np.ndarray)
                and row.shape == (self.width,)
                and row.dtype == np.float64
            ):
                rows[i], found[i] = row, True
        return rows, found

    def save(
        self, keys: npt.NDArray[np.object_], rows: _typing.FloatArray
    ) -> None:
        """Files each row of ``rows`` under its key.

        A cache that cannot be written (a directory of another user, a
        full disk) is given up on with one warning: the embeddings are
        computed all the same.

        Args:
            keys: One key per row, shape (n,).
            rows: The embeddings, shape (n, width).

        Warns:
            RuntimeWarning: The first time a write fails.
        """
        if not self._writable or not len(keys):
            return
        try:
            note = self.directory / "identity.json"
            if not note.exists():
                self._write(note, self._identity.encode())
            for key, row in zip(keys, rows, strict=True):
                self._write(self._path(key), row)
        except OSError as error:
            self._writable = False
            _warnings.warn(
                f"The embedding cache {self.directory} cannot be written"
                f" ({error}); the embeddings are computed without it.",
                RuntimeWarning,
            )

    def _path(self, key: str) -> pathlib.Path:
        """The file of one embedding; two digits fan the files out."""
        return self.directory / key[:2] / f"{key[2:]}.npy"

    def _write(
        self, path: pathlib.Path, content: bytes | _typing.FloatArray
    ) -> None:
        """Writes one file whole, then moves it into place.

        Args:
            path: Where the file belongs.
            content: Text as bytes, or an array to write as ``.npy``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=path.parent, suffix=".partial")
        partial = pathlib.Path(name)
        try:
            with os.fdopen(handle, "wb") as file:
                if isinstance(content, bytes):
                    file.write(content)
                else:
                    np.save(file, content)
            # mkstemp makes the file private (0600); give it the mode any
            # new file would get, so that others can read a shared cache.
            os.chmod(partial, 0o666 & ~_home.umask())
            try:
                os.replace(partial, path)
            except PermissionError:
                # Windows refuses to replace a file that another process,
                # having written it first, holds open; keep that copy.
                if not path.exists():
                    raise
        finally:
            partial.unlink(missing_ok=True)


def clear_embedding_cache() -> None:
    """Deletes every embedding in the default cache.

    A cache that an encoder's ``cache`` parameter names is not touched.
    """
    shutil.rmtree(_root(), ignore_errors=True)


def _root() -> pathlib.Path:
    """The default cache directory, under :func:`lazy.datasets.data_home`."""
    return _home.data_home() / _SUBDIRECTORY
