# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Encoders of images.

An image is an array of shape (height, width) or (height, width,
channels), the path of an image file, or a Pillow image. An array keeps
its values and may have any number of channels; a file or a Pillow image
is read as 8-bit values with 1 (grey), 3 (RGB) or 4 (RGBA) channels,
except an image of one channel of more than 8 bits, which keeps its
values. Every image of a column has ``n_channels`` channels, and every
value is finite.
The encoders here are classical: fixed computations with no weights and no
download.

:class:`Statistics`
    Summary statistics of each channel's values.
:class:`Pixels`
    The image itself, averaged down to a few pixels.

Typical usage example:

  columns = lazy.image.Pixels(size=4).fit_transform(paths)
"""

from __future__ import annotations

import abc
import itertools
import os
import pathlib
from typing import Any

import numpy as np
from PIL import Image

from lazy import _typing
from lazy.encoders import _base

__all__ = ["STATISTICS", "Pixels", "Statistics"]

STATISTICS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "q25",
    "median",
    "q75",
    "max",
    "gradient",
)
"""The columns that :class:`Statistics` gives for each channel, in order."""

_QUARTILES = (0.0, 0.25, 0.5, 0.75, 1.0)

# The Pillow mode that reads a file with each number of channels.
_MODES = {1: "L", 3: "RGB", 4: "RGBA"}


class _ImageEncoder(_base.BaseEncoder):
    """The classical image encoders' ``n_channels``, reading and cache key."""

    modality = "image"

    # Set by each encoder's __init__; declared for the type checker only.
    n_channels: int

    def _width(self) -> int:
        """The columns per channel, times ``n_channels``."""
        channels = _base.positive_integer("n_channels", self.n_channels)
        return self._per_channel() * channels

    @abc.abstractmethod
    def _per_channel(self) -> int:
        """The number of columns per channel; checks the parameters."""

    def _pixels(self, item: Any) -> _typing.FloatArray:
        """One image as an array, shape (height, width, n_channels).

        Raises:
            ValueError: If the image has another number of channels, no
                pixel, or a value that is not finite.
        """
        if isinstance(item, (str, os.PathLike)):
            with Image.open(item) as image:
                return self._pixels(image)
        if isinstance(item, Image.Image):
            item = self._converted(item)
        pixels = np.asarray(item, dtype=float)
        if pixels.ndim == 2:
            pixels = pixels[:, :, np.newaxis]
        if pixels.ndim != 3 or pixels.shape[2] != self.n_channels:
            raise ValueError(
                "an image must have the shape (height, width) or (height,"
                f" width, {self.n_channels}), as n_channels says; got"
                f" {pixels.shape}"
            )
        if not pixels.size or not np.isfinite(pixels).all():
            raise ValueError(
                "an image must have pixels, all of them finite; got the"
                f" shape {pixels.shape}"
            )
        return pixels

    def _converted(self, image: Image.Image) -> Image.Image:
        """A Pillow image in the mode of ``n_channels``.

        An image of one band of more than 8 bits (16-bit integers, 32-bit
        integers or floats) keeps its values.

        Raises:
            ValueError: If ``n_channels`` is not 1, 3 or 4, or the image has
                more than 8 bits and ``n_channels`` is not 1.
        """
        if self.n_channels not in _MODES:
            raise ValueError(
                f"an image file has 1, 3 or 4 channels: {self.n_channels=}"
            )
        if image.mode in ("I", "F") or image.mode.startswith("I;16"):
            if self.n_channels != 1:
                raise ValueError(
                    f"an image of mode {image.mode} has one channel of more"
                    f" than 8 bits: set n_channels=1; {self.n_channels=}"
                )
            return image
        return image.convert(_MODES[self.n_channels])

    def _item_bytes(self, item: Any) -> bytes:
        """The content of an image: of the file, for a path.

        A Pillow image is keyed by its pixels as this encoder reads them,
        since its own bytes may leave out a palette.
        """
        if isinstance(item, (str, os.PathLike)):
            return b"file:" + pathlib.Path(item).read_bytes()
        if isinstance(item, Image.Image):
            image = self._converted(item)
            header = f"pillow:{image.mode}:{image.size}:".encode()
            return header + image.tobytes()
        return super()._item_bytes(item)


class Statistics(_ImageEncoder):
    """Summary statistics of each channel of each image.

    Each channel gives the columns of :data:`STATISTICS`: the ``mean``,
    the ``std`` (the population one), the ``min``, the quartiles ``q25``,
    ``median`` and ``q75``, and the ``max`` of its values, and
    ``gradient``, the mean absolute difference between pixels that are
    neighbours in a row or in a column (0 for an image of one pixel). The
    columns are grouped by statistic: ``mean0``, ``mean1``, ..., then
    ``std0``, and so on.

    Attributes:
        n_channels: The number of channels of every image.
        batch_size: The most images encoded in one batch.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        n_channels: int = 3,
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = False,
    ) -> None:
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.cache = cache

    def _per_channel(self) -> int:
        """One column per statistic."""
        return len(STATISTICS)

    def _names(self) -> list[str]:
        """``mean0``, ``mean1``, ..., ``gradient<n_channels - 1>``."""
        channels = range(self.n_channels)
        names: list[str] = []
        for statistic in STATISTICS:
            names.extend(f"{statistic}{channel}" for channel in channels)
        return names

    def _encode(self, items: list[Any]) -> _typing.FloatArray:
        """The statistics, shape (len(items), 8 * n_channels)."""
        return np.array([_statistics(self._pixels(item)) for item in items])


def _statistics(pixels: _typing.FloatArray) -> _typing.FloatArray:
    """The statistics of one image, flat, grouped by statistic."""
    values = pixels.reshape(-1, pixels.shape[2])
    steps = [np.abs(np.diff(pixels, axis=axis)) for axis in (0, 1)]
    n_steps = sum(step[:, :, 0].size for step in steps)
    gradient = sum(step.sum(axis=(0, 1)) for step in steps) / max(n_steps, 1)
    return np.concatenate(
        [
            values.mean(axis=0),
            values.std(axis=0),
            *np.quantile(values, _QUARTILES, axis=0),
            gradient,
        ]
    )


class Pixels(_ImageEncoder):
    """Each image, averaged down to ``size`` by ``size`` pixels.

    The image is cut into ``size`` equal bands of rows and ``size`` equal
    bands of columns, and each output pixel is the mean of the image over
    its cell, with a pixel that straddles two cells shared by area. The
    mean of the image is therefore kept, and an image smaller than
    ``size`` is spread over the cells. The columns are ``y<i>x<j>c<k>``
    for row band ``i``, column band ``j`` and channel ``k``.

    Attributes:
        size: The number of output pixels along each side.
        n_channels: The number of channels of every image.
        batch_size: The most images encoded in one batch.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        size: int = 8,
        n_channels: int = 3,
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = False,
    ) -> None:
        self.size = size
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.cache = cache

    def _per_channel(self) -> int:
        """One column per output pixel."""
        return _base.positive_integer("size", self.size) ** 2

    def _names(self) -> list[str]:
        """``y0x0c0``, ``y0x0c1``, ..., in the order of the flat array."""
        cells = itertools.product(
            range(self.size), range(self.size), range(self.n_channels)
        )
        return [f"y{row}x{column}c{channel}" for row, column, channel in cells]

    def _encode(self, items: list[Any]) -> _typing.FloatArray:
        """The small images, flat: (len(items), size * size * n_channels)."""
        rows = []
        for item in items:
            pixels = self._pixels(item)
            height, width, _ = pixels.shape
            down = _area_weights(height, self.size)
            across = (
                down if width == height else _area_weights(width, self.size)
            )
            # (c, h, w) -> (c, size, size) -> (size, size, c).
            small = down @ pixels.transpose(2, 0, 1) @ across.T
            rows.append(small.transpose(1, 2, 0).ravel())
        return np.array(rows)


def _area_weights(n_in: int, n_out: int) -> _typing.FloatArray:
    """The weights that average ``n_in`` pixels into ``n_out`` equal cells.

    Returns:
        The share of each cell that each input pixel covers, shape
        (n_out, n_in); each row sums to 1.
    """
    edges = np.linspace(0, n_in, n_out + 1)[:, np.newaxis]
    pixel = np.arange(n_in)
    overlap = np.minimum(edges[1:], pixel + 1) - np.maximum(edges[:-1], pixel)
    return np.asarray(np.clip(overlap, 0, None) * n_out / n_in)
