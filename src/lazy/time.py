# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Encoders of time series.

A time series is one row of a numeric array, or a sequence of numbers of
any length, read as evenly sampled.

:class:`Pretrained`
    A frozen Chronos, TimesFM or TiRex network, chosen by ``model``: the
    default of the modality ``"time"``. Its weights download on first use;
    it needs the ``encoders`` extra, and TiRex the ``tirex`` extra.
:class:`Statistics`
    Summary statistics of the values: location, spread, shape, trend and
    roughness.
:class:`Spectrum`
    How the variance is shared among octaves of frequency.

Typical usage example:

  columns = lazy.time.Statistics().fit_transform(light_curves)
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
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
    """How one network turns a series into an embedding.

    A subclass per family of networks loads its network and pools its
    states.

    Attributes:
        states: The width of the pooled states.
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
    (TimesFM 2.5 and 3 each patch by those of the series up to it,
    TimesFM 2.0 the series by those of its first patch of at least three
    values), so this leaves its states as they are, but keeps their
    precision in float32. TimesFM has no summary token.

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
        for row, pad, own in zip(values, padding, filled, strict=True):
            scale = float(own.std()) or 1.0
            row[length - len(own) :] = (own - own.mean()) / scale
            pad[length - len(own) :] = 0
        windows = [values[-self.context :] for values in series]
        observed = _observed_tokens(windows, self.patch, length // self.patch)
        states = self._states(network, values, padding)
        # Summed in float64, except on an Apple GPU, which has none.
        dtype = torch.float32 if states.device.type == "mps" else torch.float64
        states = states.to(dtype)
        mask = torch.as_tensor(observed, dtype=dtype, device=states.device)
        mask = mask[..., None]
        pooled = (states * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.cpu().numpy()

    def _states(
        self,
        network: Any,
        values: npt.NDArray[np.float32],
        padding: npt.NDArray[np.float32],
    ) -> Any:
        """The decoder's final states over the patches of each series.

        Args:
            network: What :meth:`load` returned.
            values: The standardized series, padded on the left to whole
                patches of the longest, shape (n_series, length).
            padding: 1 where ``values`` is padding, else 0; the same shape.

        Returns:
            The states, a tensor of shape (n_series, length // patch,
            states) on the network's device.
        """
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        inputs = {
            "past_values": torch.from_numpy(values),
            "past_values_padding": torch.from_numpy(padding),
        }
        if self.freq is not None:
            inputs["freq"] = torch.full((len(values), 1), self.freq)
        inputs = {name: x.to(network.device) for name, x in inputs.items()}
        return network(**inputs).last_hidden_state


@dataclasses.dataclass(frozen=True)
class _TimesFm3(_TimesFm):
    """TimesFM 3, which reads a series from its first patch.

    TimesFM 3 normalizes its queries and keys after their rotary position
    embedding, so its states depend on the patch that a series starts at,
    not only on the order of its patches. Each series therefore starts at
    the first patch, as when TimesFM 3's forecast reads it alone, and is
    padded on the right, which its causal attention keeps from the
    series' own patches; their states then move back to the end. The
    network reads the series through its ``decode``, as its forecast does,
    with a horizon of one value.
    """

    package: ClassVar[str] = "timesfm3"

    def load(self, path: os.PathLike[str], device: str) -> Any:
        """The network, on ``device``, in float32."""
        import timesfm3.torch  # noqa: PLC0415 - an optional, heavy extra.
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        # huggingface_hub's loader prints that it reads a local directory.
        with contextlib.redirect_stdout(None):
            network = timesfm3.torch.TimesFM3Torch.from_pretrained(
                os.fspath(path)
            )
        return network.to(device, torch.float32).eval()

    def _states(
        self,
        network: Any,
        values: npt.NDArray[np.float32],
        padding: npt.NDArray[np.float32],
    ) -> Any:
        """The final states over each series' patches, read from the start."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        length = values.shape[1]
        tokens = length // self.patch
        # Each series moves left by its whole patches of padding.
        shifts = padding.argmin(axis=1) // self.patch
        order = (np.arange(length) + self.patch * shifts[:, None]) % length
        values = np.take_along_axis(values, order, axis=1)
        padded = np.take_along_axis(padding.astype(bool), order, axis=1)
        device = next(network.parameters()).device
        _, outputs = network.decode(
            torch.from_numpy(values)[:, None].to(device),
            horizon=1,
            mask=torch.from_numpy(padded).to(device),
            return_aux_outputs=True,
        )
        states = outputs["__call__:transformer_output"][:, 0, :tokens]
        # Each series' states move back to the end, where pooled reads them.
        back = (np.arange(tokens) - shifts[:, None]) % tokens
        back = torch.as_tensor(back, device=states.device)[..., None]
        return torch.take_along_dim(states, back, dim=1)


@dataclasses.dataclass(frozen=True)
class _TiRex(_Recipe):
    """TiRex, an xLSTM, which masks a missing value.

    As TiRex's own embedding does (``TiRexEmbedding`` in tirex-ts), each
    series' window is padded on the left with missing values to the whole
    context, and every layer's states are kept: each layer's mean over the
    observed patches is scaled to unit length, and the layers' means are
    joined and standardized together, by a layer norm without weights.
    The network's matrix products run in float64, except on an Apple GPU,
    which has none; its RMS norms round to float32, as tirex-ts writes
    them. A row then depends on its batch only by float rounding.
    """

    package: ClassVar[str] = "tirex"

    def load(self, path: os.PathLike[str], device: str) -> Any:
        """The network, on ``device``, in float64 where it can be."""
        import tirex  # noqa: PLC0415 - an optional, heavy extra.
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        # tirex-ts prints that it reads a local file; its CUDA kernels,
        # which run in bfloat16, are not used.
        with contextlib.redirect_stdout(None):
            network = tirex.TiRexZero.from_pretrained(
                os.fspath(pathlib.Path(path) / "model.ckpt"),
                backend="torch",
                device=device,
            )
        apple = torch.device(device).type == "mps"
        return network.to(torch.float32 if apple else torch.float64).eval()

    def pooled(
        self, network: Any, series: list[_typing.FloatArray]
    ) -> _typing.FloatArray:
        """Each layer's mean state over the observed patches, normalized."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        windows = [values[-self.context :] for values in series]
        batch = np.full((len(windows), self.context), np.nan)
        for row, values in zip(batch, windows, strict=True):
            row[len(row) - len(values) :] = values
        observed = _observed_tokens(
            windows, self.patch, self.context // self.patch
        )
        weight = next(network.parameters())
        context = torch.as_tensor(
            batch, dtype=weight.dtype, device=weight.device
        )
        # As TiRex's _prepare_context_for_embedding: the tokenizer
        # standardizes the window, and the network reads each patch's mask.
        tokens, _ = network.tokenizer.input_transform(context)
        mask = tokens.isfinite().to(tokens.dtype)
        tokens = torch.nan_to_num(tokens, nan=network.config.nan_mask_value)
        _, states = network._forward_model(  # noqa: SLF001 - TiRexEmbedding's.
            torch.cat([tokens, mask], dim=2), return_all_hidden=True
        )
        # The states are (series, patches, layers, width).
        weights = torch.as_tensor(observed, dtype=states.dtype)
        weights = weights.to(states.device)[..., None, None]
        means = (states * weights).sum(dim=1) / weights.sum(dim=1)
        means = torch.nn.functional.normalize(means, dim=-1).flatten(1)
        rows = torch.nn.functional.layer_norm(means, means.shape[-1:])
        return rows.cpu().double().numpy()


class Pretrained(_pretrained.PretrainedEncoder):
    """The embedding of each time series, from a frozen pretrained network.

    ``model`` picks one of the networks of ``recipes``, each pinned to a
    commit on the Hugging Face Hub: ``"chronos-t5-tiny"`` (the default),
    ``"chronos-t5-mini"``, ``"chronos-t5-small"``, ``"chronos-bolt-small"``,
    ``"chronos-2"``, ``"timesfm-2.0-500m"``, ``"timesfm-2.5-200m"``,
    ``"timesfm-3.0"`` and ``"tirex"``. Chronos-T5 reads one value per token
    and the last 512 values of a series; Chronos-Bolt reads one patch of 16
    values per token and the last 2,048, and Chronos-2 the last 8,192.
    TimesFM reads patches of 32 values: TimesFM 2.0 the last 2,048,
    TimesFM 2.5 the last 16,384 and TimesFM 3 the last 15,360. TiRex reads
    patches of 32 values and the last 2,048. TimesFM 3 and TiRex restrict
    commercial use, as their licences state.
    :doc:`/guide/encoders` lists the networks and gives the evidence behind
    the choice.

    The embedding is the mean of the network's final states over the
    tokens that hold an observed value and over its summary token, if it
    has one (TimesFM has none), or, for TiRex, each layer's such mean,
    normalized as its authors do; then ``mean`` and ``scale``, the mean and
    the standard deviation of the observed values in the window it read,
    missing values left out. The network scales each
    series by its own level, so these two columns keep what it removes. A
    value that is not finite is a missing value: Chronos and TiRex mask
    it, and TimesFM reads it filled in by linear interpolation, as
    TimesFM's own forecasts fill it.

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
        # The context is the longest that TimesFM 3's forecast reads
        # (_MAX_CONTEXT_LENGTH in timesfm3_forecaster.py), which fills a
        # missing value as TimesFM 2.5's does.
        "timesfm-3.0": _TimesFm3(1280, 32, 15360),
        # 12 layers of 512 states, patches of 32 values and the context of
        # 2,048 (train_ctx_len), from the checkpoint's hyper-parameters.
        "tirex": _TiRex(12 * 512, 32, 2048),
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


def _observed_tokens(
    windows: list[_typing.FloatArray], patch: int, tokens: int
) -> npt.NDArray[np.float64]:
    """Which tokens of a batch hold an observed value, as weights.

    Args:
        windows: The values each series' network reads, with ``NaN`` for
            a missing value.
        patch: The number of values in one token.
        tokens: The tokens of the batch; each series' come last.

    Returns:
        1 for a token with an observed value, else 0, shape
        (len(windows), tokens).
    """
    observed = np.zeros((len(windows), tokens))
    for row, values in zip(observed, windows, strict=True):
        flags = _tokens(values, patch)
        row[tokens - len(flags) :] = flags
    return observed


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
