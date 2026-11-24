# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Encoders of time series.

A time series is one row of a numeric array, or a sequence of numbers of
any length. Its values that are not finite are dropped, and the rest is
read as evenly sampled. The encoders here are classical: fixed
computations with no weights and no download.

:class:`Statistics`
    Summary statistics of the values: location, spread, shape, trend and
    roughness.
:class:`Spectrum`
    How the variance is shared among octaves of frequency.

Typical usage example:

  columns = lazy.time.Statistics().fit_transform(light_curves)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from lazy import _typing
from lazy.encoders import _base

__all__ = ["STATISTICS", "Spectrum", "Statistics"]

STATISTICS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "q25",
    "median",
    "q75",
    "max",
    "skewness",
    "kurtosis",
    "slope",
    "autocorrelation",
    "roughness",
    "length",
)
"""The columns of :class:`Statistics`, in order."""

_QUARTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _values(item: Any) -> _typing.FloatArray:
    """The finite values of one time series, in order.

    Raises:
        ValueError: If the item is not one-dimensional, or has no finite
            value.
    """
    series = np.asarray(item, dtype=float)
    if series.ndim != 1:
        raise ValueError(
            "a time series must be one-dimensional; got an item of shape"
            f" {series.shape}"
        )
    series = series[np.isfinite(series)]
    if not series.size:
        raise ValueError("a time series has no finite value")
    return series


class Statistics(_base.BaseEncoder):
    """Summary statistics of each time series.

    The columns are those of :data:`STATISTICS`: location, spread, shape,
    trend, lag-1 autocorrelation, roughness and length.
    :doc:`/guide/encoders` defines each. A statistic that a series does
    not define (the skewness of a constant series, the slope of one value)
    is 0.

    Attributes:
        batch_size: The most series encoded in one batch.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "time"

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = False,
    ) -> None:
        self.batch_size = batch_size
        self.cache = cache

    def _width(self) -> int:
        """The number of statistics."""
        return len(STATISTICS)

    def _names(self) -> tuple[str, ...]:
        """The names of the statistics."""
        return STATISTICS

    def _encode(self, items: list[Any]) -> _typing.FloatArray:
        """The statistics of each series, shape (len(items), 13)."""
        return np.array([_statistics(_values(item)) for item in items])


def _statistics(series: _typing.FloatArray) -> list[float]:
    """The statistics of one series of finite values, as ``STATISTICS``."""
    n = series.size
    mean = series.mean()
    if _is_constant(series):
        # No spread, shape, trend or correlation. Tested exactly, since
        # x - mean leaves rounding residue that scaling would blow up.
        return [mean, 0.0, *np.quantile(series, _QUARTILES), *[0.0] * 5, n]
    centred = series - mean
    std = series.std()
    scaled = centred / std
    time = np.arange(n) - (n - 1) / 2
    return [
        mean,
        std,
        *np.quantile(series, _QUARTILES),
        np.mean(scaled**3),
        np.mean(scaled**4) - 3,
        time @ centred / (time @ time),
        scaled[:-1] @ scaled[1:] / n,
        np.abs(np.diff(series)).mean(),
        n,
    ]


def _is_constant(series: _typing.FloatArray) -> bool:
    """Whether every value of a series is the same, one value included."""
    return bool(series.min() == series.max())


class Spectrum(_base.BaseEncoder):
    """The share of each series' variance in each octave of frequency.

    Octave ``j`` holds the frequencies above :math:`2^{-j-2}` and up to
    :math:`2^{-j-1}` cycles per sample, and the last octave everything
    below. The columns are each octave's share of the variance, from the
    highest frequencies to the lowest, and then ``peak``, the frequency
    that holds the most. :doc:`/guide/encoders` gives the equations. The
    octaves do not depend on the length, so series of different lengths
    compare. A constant series gives zeros.

    Attributes:
        n_octaves: The number of octaves.
        batch_size: The most series encoded in one batch.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "time"

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        n_octaves: int = 8,
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = False,
    ) -> None:
        self.n_octaves = n_octaves
        self.batch_size = batch_size
        self.cache = cache

    def _width(self) -> int:
        """One column per octave, and the peak frequency."""
        return _base.positive_integer("n_octaves", self.n_octaves) + 1

    def _names(self) -> list[str]:
        """``octave0`` (the highest frequencies) onwards, then ``peak``."""
        return [*(f"octave{j}" for j in range(self.n_octaves)), "peak"]

    def _encode(self, items: list[Any]) -> _typing.FloatArray:
        """The shares and the peak, shape (len(items), n_octaves + 1)."""
        rows = np.zeros((len(items), self.n_octaves + 1))
        for row, item in zip(rows, items, strict=True):
            series = _values(item)
            if _is_constant(series):
                continue
            power = np.abs(np.fft.rfft(series - series.mean())[1:]) ** 2
            if series.size % 2 == 0:
                # The Nyquist term appears once in the variance, the others
                # twice (as k and n - k).
                power[-1] /= 2
            total = power.sum()
            frequency = np.fft.rfftfreq(series.size)[1:]
            octave = np.floor(-np.log2(2 * frequency)).astype(int)
            octave = np.minimum(octave, self.n_octaves - 1)
            row[:-1] = np.bincount(octave, power, self.n_octaves) / total
            row[-1] = frequency[np.argmax(power)]
        return rows
