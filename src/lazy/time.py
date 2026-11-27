# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Encoders of time series.

A time series is one row of a numeric array, or a sequence of numbers of
any length, read as evenly sampled.

:class:`Pretrained`
    A frozen Chronos or TimesFM network, chosen by ``model``: the default
    of the modality ``"time"``. Its weights download on first use; it
    needs the ``encoders`` extra.
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
from typing import Any, ClassVar

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
    gradient_kind = "differentiable"

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

    def _stacked(self, X: Any) -> _typing.FloatArray | None:  # noqa: GS030 - scikit-learn's X.
        """The series as one array; see :func:`_stacked`."""
        return _stacked(X)

    def forward(self, items: Any) -> Any:
        """The statistics of stacked series, on torch's autograd graph.

        A quartile's gradient goes to the values it is interpolated from,
        and the length has none.

        Args:
            items: Series of one length, a float torch tensor of shape (n,
                length).

        Returns:
            What ``transform`` gives, shape (n, 13).
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        n = items.shape[1]
        mean = items.mean(dim=1, keepdim=True)
        constant = (items.min(dim=1).values == items.max(dim=1).values)[:, None]
        centred = items - mean
        variance = centred.square().mean(dim=1, keepdim=True)
        std = torch.where(constant, 1.0, variance).sqrt()
        scaled = centred / std
        time = torch.arange(n, dtype=items.dtype) - (n - 1) / 2
        shape = torch.stack(
            [
                std[:, 0],
                scaled.pow(3).mean(dim=1),
                scaled.pow(4).mean(dim=1) - 3,
                centred @ time / (time @ time).clamp(min=1),
                (scaled[:, :-1] * scaled[:, 1:]).sum(dim=1) / n,
                items.diff(dim=1).abs().sum(dim=1) / max(n - 1, 1),
            ],
            dim=1,
        )
        shape = torch.where(constant, 0.0, shape)
        quartiles = torch.quantile(
            items, torch.tensor(_QUARTILES, dtype=items.dtype), dim=1
        ).T
        return torch.cat(
            [
                mean,
                shape[:, :1],
                quartiles,
                shape[:, 1:],
                torch.full_like(mean, n),
            ],
            dim=1,
        )


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
    gradient_kind = "differentiable"

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

    def _stacked(self, X: Any) -> _typing.FloatArray | None:  # noqa: GS030 - scikit-learn's X.
        """The series as one array; see :func:`_stacked`."""
        return _stacked(X)

    def forward(self, items: Any) -> Any:
        """The spectrum of stacked series, on torch's autograd graph.

        The peak frequency is a choice among frequencies, and has no
        gradient.

        Args:
            items: Series of one length, a float torch tensor of shape (n,
                length).

        Returns:
            What ``transform`` gives, shape (n, n_octaves + 1).
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        n = items.shape[1]
        out = torch.zeros((len(items), self.n_octaves + 1), dtype=items.dtype)
        if n < 2:
            return out
        centred = items - items.mean(dim=1, keepdim=True)
        power = torch.fft.rfft(centred, dim=1)[:, 1:].abs().square()
        if n % 2 == 0:
            power = torch.cat([power[:, :-1], power[:, -1:] / 2], dim=1)
        constant = (items.min(dim=1).values == items.max(dim=1).values)[:, None]
        total = torch.where(constant, 1.0, power.sum(dim=1, keepdim=True))
        frequency = np.fft.rfftfreq(n)[1:]
        octave = np.floor(-np.log2(2 * frequency)).astype(int)
        bins = torch.as_tensor(np.minimum(octave, self.n_octaves - 1))
        shares = out[:, :-1].index_add(1, bins, power / total)
        peak = torch.as_tensor(frequency, dtype=items.dtype)[
            power.argmax(dim=1)
        ]
        return torch.where(
            constant, 0.0, torch.cat([shares, peak[:, None]], dim=1)
        )


@dataclasses.dataclass(frozen=True)
class _Recipe:
    """How one network turns a series into an embedding.

    A subclass per family of networks loads its network and pools its
    states.

    Attributes:
        states: The width of the network's states.
        patch: The number of values in one token.
        context: The most values of a series, its last ones, that it reads.
        package: The module that loads the network.
    """

    states: int
    patch: int
    context: int
    package: ClassVar[str]

    @property
    def width(self) -> int:
        """The width of the embedding: the states, the mean and the scale."""
        return self.states + 2

    def load(self, path: os.PathLike[str], device: str) -> Any:
        """The network, on ``device``, in float32."""
        raise NotImplementedError

    def pooled(
        self, network: Any, series: list[_typing.FloatArray]
    ) -> _typing.FloatArray:
        """The mean state over the observed tokens of each series' window.

        Args:
            network: What :meth:`load` returned.
            series: The whole series, with ``NaN`` for a missing value;
                none starts or ends with one.

        Returns:
            The pooled states, shape (len(series), states).
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class _Chronos(_Recipe):
    """A Chronos pipeline, which masks a missing value.

    Attributes:
        outputs: The tokens that the network puts after the summary token,
            which the embedding leaves out: Chronos-2's masked output patch.
        pinned: Whether the pipeline takes the precision and the attention
            that transformers is pinned to; Chronos-T5's takes no attention.
    """

    outputs: int = 0
    pinned: bool = False
    package: ClassVar[str] = "chronos"

    def load(self, path: os.PathLike[str], device: str) -> Any:
        """The Chronos pipeline, on ``device``, in float32."""
        import chronos  # noqa: PLC0415 - an optional, heavy extra.
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        if self.pinned:
            settings = _pretrained.PINNED
        else:
            settings = {"torch_dtype": torch.float32}
        return chronos.BaseChronosPipeline.from_pretrained(
            path, device_map=device, **settings
        )

    def pooled(
        self, network: Any, series: list[_typing.FloatArray]
    ) -> _typing.FloatArray:
        """The mean state over the observed tokens and the summary token."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        windows = [values[-self.context :] for values in series]
        states, _ = network.embed(
            [torch.as_tensor(values, dtype=torch.float32) for values in windows]
        )
        rows = np.empty((len(windows), self.states))
        for row, values, state in zip(rows, windows, states, strict=True):
            # Chronos-2 gives each series its own tensor, of one variate.
            tokens = state.float().numpy().reshape(-1, self.states)
            # A batch pads on the left: a series' own tokens and its one
            # summary token come last, before the output tokens.
            observed = np.append(_tokens(values, self.patch), True)
            end = len(tokens) - self.outputs
            row[:] = tokens[end - len(observed) : end][observed].mean(axis=0)
        return rows


@dataclasses.dataclass(frozen=True)
class _TimesFm(_Recipe):
    """A TimesFM decoder, which reads a missing value filled in.

    As TimesFM's own forecast does, each series has its missing values
    filled by linear interpolation before its last ``context`` values are
    read, and is padded on the left with values that the network masks:
    here to whole patches of the batch's longest series, where the forecast
    pads to the whole context. Each series is standardized first. The
    network normalizes the values by statistics of the series itself
    (TimesFM 2.5 each patch by those of the series up to it, TimesFM 2.0
    the series by those of its first patch of at least three values), so
    this leaves its states as they are, but keeps their precision in
    float32. TimesFM has no summary token.

    Attributes:
        freq: The frequency indicator that TimesFM 2.0 reads with each
            series; ``None`` for a network that reads none.
    """

    freq: int | None = None
    package: ClassVar[str] = "transformers"

    def load(self, path: os.PathLike[str], device: str) -> Any:
        """The decoder, without the file's forecast heads, on ``device``."""
        import transformers  # noqa: PLC0415 - an optional, heavy extra.

        network = _pretrained.load_quietly(transformers.AutoModel, path)
        return network.to(device)

    def pooled(
        self, network: Any, series: list[_typing.FloatArray]
    ) -> _typing.FloatArray:
        """The mean state over the observed patches, pooled on the device."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        filled = [_interpolated(values)[-self.context :] for values in series]
        length = self.patch * max(-(-len(v) // self.patch) for v in filled)
        values = np.zeros((len(filled), length), dtype=np.float32)
        padding = np.ones((len(filled), length), dtype=np.float32)
        observed = np.zeros((len(filled), length // self.patch))
        for i, (own, whole) in enumerate(zip(filled, series, strict=True)):
            scale = float(own.std()) or 1.0
            values[i, length - len(own) :] = (own - own.mean()) / scale
            padding[i, length - len(own) :] = 0
            flags = _tokens(whole[-self.context :], self.patch)
            observed[i, len(observed[i]) - len(flags) :] = flags
        inputs = {
            "past_values": torch.from_numpy(values),
            "past_values_padding": torch.from_numpy(padding),
        }
        if self.freq is not None:
            inputs["freq"] = torch.full((len(filled), 1), self.freq)
        inputs = {name: x.to(network.device) for name, x in inputs.items()}
        states = network(**inputs).last_hidden_state
        # Summed in float64, except on an Apple GPU, which has none.
        dtype = torch.float32 if states.device.type == "mps" else torch.float64
        states = states.to(dtype)
        mask = torch.as_tensor(observed, dtype=dtype, device=states.device)
        mask = mask[..., None]
        pooled = (states * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.cpu().numpy()


class Pretrained(_pretrained.PretrainedEncoder):
    """The embedding of each time series, from a frozen pretrained network.

    ``model`` picks one of the networks of ``recipes``, each pinned to a
    commit on the Hugging Face Hub: ``"chronos-t5-tiny"`` (the default),
    ``"chronos-t5-mini"``, ``"chronos-t5-small"``, ``"chronos-bolt-small"``,
    ``"chronos-2"``, ``"timesfm-2.0-500m"`` and ``"timesfm-2.5-200m"``.
    Chronos-T5 reads one value per token and the last 512 values of a
    series; Chronos-Bolt reads one patch of 16 values per token and the
    last 2,048, and Chronos-2 the last 8,192. TimesFM reads patches of 32
    values: TimesFM 2.0 the last 2,048, and TimesFM 2.5 the last 16,384.
    :doc:`/guide/encoders` lists the networks and gives the evidence behind
    the choice.

    The embedding is the mean of the network's final states over the
    tokens that hold an observed value and over its summary token, if it
    has one (TimesFM has none), then ``mean`` and ``scale``, the mean and
    the standard deviation of the observed values in the window it read,
    missing values left out. The network scales each
    series by its own level, so these two columns keep what it removes. A
    value that is not finite is a missing value: Chronos masks it, and
    TimesFM reads it filled in by linear interpolation, as TimesFM's own
    forecasts fill it.

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
        "chronos-t5-tiny": _Chronos(256, 1, 512),
        "chronos-t5-mini": _Chronos(384, 1, 512),
        "chronos-t5-small": _Chronos(512, 1, 512),
        "chronos-bolt-small": _Chronos(512, 16, 2048),
        "chronos-2": _Chronos(768, 16, 8192, outputs=1, pinned=True),
        # Each TimesFM's patch and context come from its config.json. The
        # states are the decoder stack's, before the forecast heads. A
        # missing value is filled as the timesfm package's forecast fills
        # it (linear_interpolation in timesfm_2p5_base.py), and as TimesFM
        # 2.0's card says it does. TimesFM 2.0 reads the frequency
        # indicator 0, "high frequency", which its card gives as the
        # default, for data up to daily.
        "timesfm-2.0-500m": _TimesFm(1280, 32, 2048, freq=0),
        "timesfm-2.5-200m": _TimesFm(1280, 32, 16384),
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

    def _package(self) -> str:
        """The module that loads ``model``, which its family names."""
        return self._recipe().package

    def _load(self, path: os.PathLike[str], device: str) -> Any:
        """The network of ``model``, on ``device``, in float32."""
        return self._recipe().load(path, device)

    def _embed(self, network: Any, items: list[Any]) -> _typing.FloatArray:
        """The mean state over each series' observed tokens, mean, scale."""
        recipe = self._recipe()
        series = [_observed(item) for item in items]
        windows = [values[-recipe.context :] for values in series]
        means = [(np.nanmean(v), np.nanstd(v)) for v in windows]
        return np.hstack([recipe.pooled(network, series), means])


def _interpolated(values: _typing.FloatArray) -> _typing.FloatArray:
    """A series with each missing value filled linearly from its neighbours.

    Args:
        values: The series, with ``NaN`` for a missing value; it starts and
            ends with a finite one.

    Returns:
        The filled series, of the same shape.
    """
    known = np.flatnonzero(np.isfinite(values))
    return np.interp(np.arange(values.size), known, values[known])


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


def _stacked(X: Any) -> _typing.FloatArray | None:  # noqa: GS030 - scikit-learn's X.
    """The series of a column as one array, shape (n, length).

    Args:
        X: One column of series.

    Returns:
        The array, or ``None`` if the series differ in length, or one is
        missing or has a value that is not finite.
    """
    items = _base._as_items(X)  # noqa: SLF001 - the encoders' reading of a column.
    if any(_base._is_missing(item) for item in items):  # noqa: SLF001 - as above.
        return None
    series = [np.asarray(item, dtype=float) for item in items]
    same = len({part.shape for part in series}) == 1 and series[0].ndim == 1
    if not same or not all(np.isfinite(part).all() for part in series):
        return None
    return np.stack(series)


def _values(item: Any) -> _typing.FloatArray:
    """The finite values of one time series, in order.

    Raises:
        ValueError: If the item is not one-dimensional, or has no finite
            value.
    """
    series = _observed(item)
    return series[np.isfinite(series)]
