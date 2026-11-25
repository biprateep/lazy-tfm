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
processes can share a cache. A file other than the one ``np.save`` writes
for a row of the encoder's width, such as a damaged one, counts as missing.

Typical usage example:

  store = _cache.Store(True, identity, width)
  rows, found = store.load(keys)
  store.save(keys[~found], new_rows)
"""

from __future__ import annotations

import hashlib
import io
import os
import pathlib
import shutil

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
        directory: The encoder's folder, as text, which exists once
            something is saved.
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
        self.directory = os.fspath(
            root.expanduser() / LAYOUT / digest(identity.encode())
        )
        self.width = width
        self._identity = identity
        self._writable = True
        # The header np.save writes for a row: a hit is then read without
        # np.load, and a row written in one call.
        buffer = io.BytesIO()
        np.save(buffer, np.zeros(width, dtype="<f8"))
        self._header = buffer.getvalue()[: -8 * width]

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
        size = len(self._header) + 8 * self.width
        for i, key in enumerate(keys):
            try:
                with open(self._path(key), "rb") as file:
                    data = file.read()
            except OSError:
                continue
            if len(data) == size and data.startswith(self._header):
                rows[i] = np.frombuffer(data, "<f8", offset=len(self._header))
                found[i] = True
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
            note = os.path.join(self.directory, "identity.json")
            if not os.path.exists(note):
                self._write(note, self._identity.encode())
            for key, row in zip(keys, rows, strict=True):
                content = self._header + np.asarray(row, "<f8").tobytes()
                self._write(self._path(key), content)
        except OSError as error:
            self._writable = False
            _warnings.warn(
                f"The embedding cache {self.directory} cannot be written"
                f" ({error}); the embeddings are computed without it.",
                RuntimeWarning,
            )

    def _path(self, key: str) -> str:
        """The file of one embedding; two digits fan the files out."""
        return os.path.join(self.directory, key[:2], f"{key[2:]}.npy")

    def _write(self, path: str, content: bytes) -> None:
        """Writes one file whole, then moves it into place.

        Args:
            path: Where the file belongs.
            content: The file's bytes.
        """
        try:
            with _home.atomic_file(path) as file:
                file.write(content)
        except FileNotFoundError:  # The folder's first file.
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with _home.atomic_file(path) as file:
                file.write(content)


def clear_embedding_cache() -> None:
    """Deletes every embedding in the default cache.

    A cache that an encoder's ``cache`` parameter names is not touched.
    """
    shutil.rmtree(_root(), ignore_errors=True)


def _root() -> pathlib.Path:
    """The default cache directory, under :func:`lazy.datasets.data_home`."""
    return _home.data_home() / _SUBDIRECTORY
