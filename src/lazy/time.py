# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Encoders of time series.

A time series is one row of a numeric array, or a sequence of numbers of
any length, read as evenly sampled.

:class:`Pretrained`
    A frozen Chronos network, chosen by ``model``: the default of the
    modality ``"time"``. Its weights download on first use; it needs the
    ``encoders`` extra.
:class:`Statistics`
    Summary statistics of the values: location, spread, shape, trend and
    roughness.
:class:`Spectrum`
    How the variance is shared among octaves of frequency.

Typical usage example:

  columns = lazy.time.Statistics().fit_transform(light_curves)
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import numpy as np
import numpy.typing as npt

from lazy import _typing
from lazy.encoders import _base
from lazy.encoders import _pretrained

__all__ = ["STATISTICS", "Pretrained", "Spectrum", "Statistics"]

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


class Statistics(_base.BaseEncoder):
    """Summary statistics of each time series, over its finite values.

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

    The values of a series that are not finite are dropped first.

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


@dataclasses.dataclass(frozen=True)
class _Recipe:
    """How one Chronos model turns a series into an embedding.

    Attributes:
        states: The width of the network's states.
        patch: The number of values in one token.
        context: The most values of a series, its last ones, that it reads.
        outputs: The tokens that the network puts after the summary token,
            which the embedding leaves out: Chronos-2's masked output patch.
        pinned: Whether the loader takes the precision and the attention
            that transformers is pinned to; Chronos-T5's takes no attention.
    """

    states: int
    patch: int
    context: int
    outputs: int = 0
    pinned: bool = False

    @property
    def width(self) -> int:
        """The width of the embedding: the states, the mean and the scale."""
        return self.states + 2


class Pretrained(_pretrained.PretrainedEncoder):
    """The embedding of each time series, from a frozen Chronos network.

    ``model`` picks one of the networks of ``recipes``, each pinned to a
    commit on the Hugging Face Hub: ``"chronos-t5-tiny"`` (the default),
    ``"chronos-t5-mini"``, ``"chronos-t5-small"``, ``"chronos-bolt-small"``
    and ``"chronos-2"``. Chronos-T5 reads one value per token and the last
    512 values of a series; Chronos-Bolt reads one patch of 16 values per
    token and the last 2,048, and Chronos-2 the last 8,192.
    :doc:`/guide/encoders` lists the networks and gives the evidence behind
    the choice.

    The embedding is the mean of the network's final states over the
    tokens that hold an observed value and over its summary token, then
    ``mean`` and ``scale``, the mean and the standard deviation of the
    values it read. The network scales each series by its own level, so
    these two columns keep what it removes. A value that is not finite is
    a missing value, which the network masks.

    Attributes:
        model: The network; ``"chronos-t5-tiny"`` by default.
        device: ``"auto"``, or a torch device such as ``"cpu"``.
        batch_size: The most series in one forward pass.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "time"
    package = "chronos"
    recipes = {
        "chronos-t5-tiny": _Recipe(256, 1, 512),
        "chronos-t5-mini": _Recipe(384, 1, 512),
        "chronos-t5-small": _Recipe(512, 1, 512),
        "chronos-bolt-small": _Recipe(512, 16, 2048),
        "chronos-2": _Recipe(768, 16, 8192, outputs=1, pinned=True),
    }

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        model: str = "chronos-t5-tiny",
        device: str = "auto",
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = True,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.cache = cache

    def _names(self) -> list[str]:
        """``0`` onwards for the states, then ``mean`` and ``scale``."""
        states = self._recipe().states
        return [*(str(i) for i in range(states)), "mean", "scale"]

    def _length(self, item: Any) -> int:
        """The number of values, which the tokens grow with."""
        return int(np.size(item))

    def _load(self, path: os.PathLike[str], device: str) -> Any:
        """The Chronos pipeline, on ``device``, in float32."""
        import chronos  # noqa: PLC0415 - an optional, heavy extra.
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        if self._recipe().pinned:
            settings = _pretrained.PINNED
        else:
            settings = {"torch_dtype": torch.float32}
        return chronos.BaseChronosPipeline.from_pretrained(
            path, device_map=device, **settings
        )

    def _embed(self, network: Any, items: list[Any]) -> _typing.FloatArray:
        """The mean state over each series' observed tokens, mean, scale."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        recipe = self._recipe()
        series = [_observed(item)[-recipe.context :] for item in items]
        states, _ = network.embed(
            [torch.as_tensor(values, dtype=torch.float32) for values in series]
        )
        rows = np.empty((len(series), recipe.width))
        for row, values, state in zip(rows, series, states, strict=True):
            # Chronos-2 gives each series its own tensor, of one variate.
            tokens = state.float().numpy().reshape(-1, recipe.states)
            # A batch pads on the left: a series' own tokens and its one
            # summary token come last, before the output tokens.
            observed = _tokens(values, recipe.patch)
            end = len(tokens) - recipe.outputs
            own = tokens[end - len(observed) - 1 : end]
            row[:-2] = own[np.append(observed, True)].mean(axis=0)
            row[-2:] = np.nanmean(values), np.nanstd(values)
        return rows


def _tokens(values: _typing.FloatArray, patch: int) -> npt.NDArray[np.bool_]:
    """Which tokens of a series hold an observed value.

    Args:
        values: The series, with ``NaN`` for a missing value.
        patch: The number of values in one token; a series is padded on
            the left to whole tokens.

    Returns:
        One flag per token, shape (ceil(len(values) / patch),).
    """
    padded = np.concatenate([np.full(-len(values) % patch, np.nan), values])
    return np.isfinite(padded.reshape(-1, patch)).any(axis=1)


def _observed(item: Any) -> _typing.FloatArray:
    """One time series from its first finite value to its last.

    Values that are not finite before the first finite value or after the
    last are dropped, as padding; those between them are ``NaN``.

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
    finite = np.flatnonzero(np.isfinite(series))
    if not finite.size:
        raise ValueError("a time series has no finite value")
    series = series[finite[0] : finite[-1] + 1]
    return np.where(np.isfinite(series), series, np.nan)


def _values(item: Any) -> _typing.FloatArray:
    """The finite values of one time series, in order.

    Raises:
        ValueError: If the item is not one-dimensional, or has no finite
            value.
    """
    series = _observed(item)
    return series[np.isfinite(series)]
