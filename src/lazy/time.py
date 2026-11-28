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

from lazy import _arrays
from lazy import _steps
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


class _SeriesEncoder(_base.BaseEncoder):
    """What the encoders of series share: their forward's input.

    Their embeddings are differentiable in the values of a series.
    """

    modality = "time"
    gradient_kind = "differentiable"

    def _array(self, item: Any) -> _typing.FloatArray | None:
        """One series as an array, shape (length,); ``None`` with a gap.

        ``transform`` drops or masks the values that are not finite, so a
        series with one has no array whose elements line up with its values.
        """
        series = np.asarray(item, dtype=float)
        if series.ndim != 1 or not series.size or not np.isfinite(series).all():
            return None
        return series

    def _stack(self, items: Any) -> Any:
        """Stacked series, as a forward takes them.

        Args:
            items: A float torch tensor of shape (n, length).

        Returns:
            ``items`` itself.

        Raises:
            lazy._steps.BarrierError: If the tensor is not of that shape,
                or, unless the encoder pads its items, a value is not
                finite: ``transform`` drops such a value, which no function
                of the tensor's elements does.
        """
        if items.ndim != 2 or not (
            self.pads_items or bool(items.isfinite().all())
        ):
            raise _steps.BarrierError(
                f"{type(self).__name__} has a forward only for series of"
                " one length whose values are all finite"
            )
        return items


class Statistics(_SeriesEncoder):
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

        n = self._stack(items).shape[1]
        mean = items.mean(dim=1, keepdim=True)
        constant = (items.min(dim=1).values == items.max(dim=1).values)[:, None]
        centred = items - mean
        variance = centred.square().mean(dim=1, keepdim=True)
        std = torch.where(constant, 1.0, variance).sqrt()
        scaled = centred / std
        time = (
            torch.arange(n, dtype=items.dtype, device=items.device)
            - (n - 1) / 2
        )
        shape = torch.stack(
            [
                std[:, 0],
                scaled.pow(3).mean(dim=1),
                scaled.pow(4).mean(dim=1) - 3,
                # 0.5 is its least value above zero, at two values.
                centred @ time / (time @ time).clamp(min=0.5),
                (scaled[:, :-1] * scaled[:, 1:]).sum(dim=1) / n,
                items.diff(dim=1).abs().sum(dim=1) / max(n - 1, 1),
            ],
            dim=1,
        )
        shape = torch.where(constant, 0.0, shape)
        quartiles = torch.quantile(items, items.new_tensor(_QUARTILES), dim=1).T
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


class Spectrum(_SeriesEncoder):
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

    def forward(self, items: Any) -> Any:
        """The spectrum of stacked series, on torch's autograd graph.

        The peak frequency is a choice among frequencies, so it has no
        gradient: it is that of ``transform``, computed off the graph.

        Args:
            items: Series of one length, a float torch tensor of shape (n,
                length).

        Returns:
            What ``transform`` gives, shape (n, n_octaves + 1).
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        n = self._stack(items).shape[1]
        encoded = items.new_tensor(self._encode(list(_arrays.detached(items))))
        if n < 2:
            # On the graph, with no gradient: one value has no spectrum.
            return encoded + items[:, :1] * 0
        centred = items - items.mean(dim=1, keepdim=True)
        power = torch.fft.rfft(centred, dim=1)[:, 1:].abs().square()
        if n % 2 == 0:
            power = torch.cat([power[:, :-1], power[:, -1:] / 2], dim=1)
        constant = (items.min(dim=1).values == items.max(dim=1).values)[:, None]
        total = torch.where(constant, 1.0, power.sum(dim=1, keepdim=True))
        frequency = np.fft.rfftfreq(n)[1:]
        octave = np.floor(-np.log2(2 * frequency)).astype(int)
        bins = torch.as_tensor(
            np.minimum(octave, self.n_octaves - 1), device=items.device
        )
        shares = items.new_zeros((len(items), self.n_octaves)).index_add(
            1, bins, power / total
        )
        shares = torch.where(constant, 0.0, shares)
        return torch.cat([shares, encoded[:, -1:]], dim=1)


@dataclasses.dataclass(frozen=True)
class _Recipe:
    """How one network turns a series into an embedding.

    A subclass per family of networks loads its network and pools its
    states.

    Attributes:
        states: The width of the pooled states.
        patch: The number of values in one token.
        context: The most values of a series, its last ones, that it reads.
        quantized: Whether the network reads each value as one of a set of
            tokens, which have no gradient: Chronos-T5's.
        package: The module that loads the network.
    """

    states: int
    patch: int
    context: int
    quantized: ClassVar[bool] = False
    package: ClassVar[str]

    @property
    def width(self) -> int:
        """The width of the embedding: the states, the mean and the scale."""
        return self.states + 2

    def load(self, path: os.PathLike[str], device: str) -> Any:
        """The network, on ``device``, in float32."""
        raise NotImplementedError

    def pooled(self, network: Any, series: list[Any]) -> Any:
        """The mean state over the observed tokens of each series' window.

        It keeps torch's autograd graph, unless the network is quantized.

        Args:
            network: What :meth:`load` returned.
            series: The whole series, float torch tensors of shape
                (length,), with ``NaN`` for a missing value; none starts or
                ends with one.

        Returns:
            The pooled states, a float torch tensor of shape (len(series),
            states).
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class _Chronos(_Recipe):
    """A Chronos pipeline, which masks a missing value.

    Chronos-Bolt and Chronos-2 read the series through their model's
    ``encode``, as the pipeline's ``embed`` does, but on torch's autograd
    graph.

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
        pipeline = chronos.BaseChronosPipeline.from_pretrained(
            path, device_map=device, **settings
        )
        norm = getattr(getattr(pipeline, "model", None), "instance_norm", None)
        if isinstance(norm, chronos.chronos_bolt.InstanceNorm):
            pipeline.model.instance_norm = _masked_norm(norm)
        return pipeline

    def pooled(self, network: Any, series: list[Any]) -> Any:
        """The mean state over the observed tokens and the summary token."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        # The model's instance norm takes the square root of a constant
        # window's variance, 0, whose gradient is not finite: such a window
        # reaches the network without one, as in the classical encoders.
        windows = [
            _steady(values[-self.context :].float()) for values in series
        ]
        states = self._states(network, windows)
        rows = []
        for values, state in zip(windows, states, strict=True):
            # Chronos-2 gives each series' states as one variate.
            tokens = state.float().reshape(-1, self.states)
            # A batch pads on the left: a series' own tokens and its one
            # summary token come last, before the output tokens.
            flags = _tokens(_arrays.detached(values), self.patch)
            observed = torch.as_tensor(np.append(flags, True))
            end = len(tokens) - self.outputs
            own = tokens[end - len(observed) : end]
            rows.append(own[observed.to(own.device)].mean(dim=0))
        return torch.stack(rows)

    def _states(self, network: Any, windows: list[Any]) -> Any:
        """The token states of each window, on torch's autograd graph.

        Args:
            network: What :meth:`load` returned.
            windows: The windows, float32 torch tensors of shape (length,),
                with ``NaN`` for a missing value.

        Returns:
            Each window's states, padded on the left to the batch's longest
            window: a tensor of shape (len(windows), tokens, states), or a
            sequence of len(windows) tensors with ``tokens * states``
            values each.
        """
        model = network.model
        context = _left_padded(windows, max(map(len, windows)))
        states = model.encode(context=context.to(model.device))[0]
        # Chronos-2 gives the states in an output of the encoder.
        return getattr(states, "last_hidden_state", states)


@dataclasses.dataclass(frozen=True)
class _ChronosT5(_Chronos):
    """A Chronos-T5 pipeline, which reads each value as one of its tokens.

    The tokens have no gradient, so it reads the series through the
    pipeline's ``embed``.
    """

    quantized: ClassVar[bool] = True

    def _states(self, network: Any, windows: list[Any]) -> Any:
        """The token states of each window, from the pipeline's ``embed``."""
        states, _ = network.embed([values.cpu() for values in windows])
        return states


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

    def pooled(self, network: Any, series: list[Any]) -> Any:
        """The mean state over the observed patches, pooled on the device."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        # Filled before the window is cut, so that a missing value at its
        # start is filled from the value before it.
        filled = [_interpolated(values)[-self.context :] for values in series]
        length = self.patch * max(-(-len(v) // self.patch) for v in filled)
        padding = np.ones((len(filled), length), dtype=np.float32)
        rows = []
        for pad, own in zip(padding, filled, strict=True):
            mean, scale = _spread(own, constant=1.0)
            rows.append(((own - mean) / scale).float())
            pad[length - len(own) :] = 0
        windows = [
            _arrays.detached(values[-self.context :]) for values in series
        ]
        observed = _observed_tokens(windows, self.patch, length // self.patch)
        values = _left_padded(rows, length, fill=0.0)
        states = self._states(network, values, padding)
        # Summed in float64, except on an Apple GPU, which has none.
        dtype = torch.float32 if states.device.type == "mps" else torch.float64
        states = states.to(dtype)
        mask = torch.as_tensor(observed, dtype=dtype, device=states.device)
        mask = mask[..., None]
        return (states * mask).sum(dim=1) / mask.sum(dim=1)

    def _states(
        self, network: Any, values: Any, padding: npt.NDArray[np.float32]
    ) -> Any:
        """The decoder's final states over the patches of each series.

        Args:
            network: What :meth:`load` returned.
            values: The standardized series, padded on the left to whole
                patches of the longest, a float32 torch tensor of shape
                (n_series, length).
            padding: 1 where ``values`` is padding, else 0; the same shape.

        Returns:
            The states, a tensor of shape (n_series, length // patch,
            states) on the network's device.
        """
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        inputs = {
            "past_values": values,
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
        self, network: Any, values: Any, padding: npt.NDArray[np.float32]
    ) -> Any:
        """The final states over each series' patches, read from the start."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        length = values.shape[1]
        tokens = length // self.patch
        # Each series moves left by its whole patches of padding.
        shifts = padding.argmin(axis=1) // self.patch
        order = (np.arange(length) + self.patch * shifts[:, None]) % length
        values = torch.take_along_dim(
            values, torch.as_tensor(order, device=values.device), dim=1
        )
        padded = np.take_along_axis(padding.astype(bool), order, axis=1)
        device = next(network.parameters()).device
        _, outputs = network.decode(
            values[:, None].to(device),
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

    def pooled(self, network: Any, series: list[Any]) -> Any:
        """Each layer's mean state over the observed patches, normalized."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        weight = next(network.parameters())
        windows = [
            values[-self.context :].to(weight.device, weight.dtype)
            for values in series
        ]
        context = _left_padded(windows, self.context)
        observed = _observed_tokens(
            [_arrays.detached(values) for values in windows],
            self.patch,
            self.context // self.patch,
        )
        # As TiRex's _prepare_context_for_embedding: the tokenizer
        # standardizes the window, and the network reads each patch's mask.
        tokens = self._scaled(context)
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
        return torch.nn.functional.layer_norm(means, means.shape[-1:])

    def _scaled(self, context: Any) -> Any:
        """The patches of standardized values that TiRex's tokenizer gives.

        It computes what ``PatchedTokenizer.input_transform`` of tirex-ts
        does, with no ``NaN`` on the autograd graph, whose gradient would
        then be ``NaN`` everywhere; the gradient of a constant window is
        finite too.

        Args:
            context: The windows, padded on the left with ``NaN``, a float
                torch tensor of shape (n_series, context).

        Returns:
            The standardized values, ``NaN`` where missing, shape
            (n_series, context // patch, patch).
        """
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        observed = context.isfinite()
        values = torch.where(observed, context, 0.0)
        count = observed.sum(dim=1, keepdim=True)
        loc = values.sum(dim=1, keepdim=True) / count
        centred = torch.where(observed, values - loc, 0.0)
        variance = centred.square().sum(dim=1, keepdim=True) / count
        # tirex-ts's StandardScaler: a constant window is scaled by its
        # level, plus its epsilon.
        scale = _arrays.deviation(variance, loc.abs() + 1e-5)
        scaled = torch.where(observed, centred / scale, np.nan)
        return scaled.unfold(-1, self.patch, self.patch)


class Pretrained(_pretrained.PretrainedEncoder, _SeriesEncoder):
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

    The encoder's ``forward`` gives the embeddings of series on torch's
    autograd graph, without the cache, so that an attribution reaches each
    value. Series of other lengths are stacked padded on the left with
    ``NaN``, which, as a missing value, has no gradient. It does so for
    every network but Chronos-T5, which reads each value as one of a set of
    tokens and is a barrier to gradients.

    Attributes:
        model: The network; ``"chronos-t5-tiny"`` by default.
        device: ``"auto"``, or a torch device such as ``"cpu"``.
        batch_size: The most series in one forward pass.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "time"
    package = "chronos"
    pads_items = "left"
    barrier_reason = (
        "Chronos-T5 reads each value as one of a set of tokens, which have"
        " no gradient"
    )
    recipes = {
        "chronos-t5-tiny": _ChronosT5(256, 1, 512),
        "chronos-t5-mini": _ChronosT5(384, 1, 512),
        "chronos-t5-small": _ChronosT5(512, 1, 512),
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

    @property
    def gradient_kind(self) -> str:  # type: ignore[override]  # By model.
        """``"barrier"`` for Chronos-T5, else ``"differentiable"``."""
        quantized = self._recipe().quantized
        return "barrier" if quantized else "differentiable"

    def _array(self, item: Any) -> _typing.FloatArray | None:
        """One series, from its first finite value to its last.

        Returns:
            The series, ``NaN`` for a missing value, shape (length,);
            ``None`` if it has no finite value.
        """
        try:
            return _observed(item)
        except ValueError:
            return None

    def _embed(self, network: Any, items: list[Any]) -> _typing.FloatArray:
        """The mean state over each series' observed tokens, mean, scale."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        series = [torch.from_numpy(_observed(item)) for item in items]
        return self._embedded(network, series).cpu().numpy()

    def forward(self, items: Any) -> Any:
        """The embeddings of stacked series, on torch's autograd graph.

        It runs the network on its ``device``, without the cache, in
        passes of at most ``batch_size`` series.

        Args:
            items: Series padded on the left with ``NaN``, which is also a
                missing value, a float torch tensor of shape (n, length).

        Returns:
            What ``transform`` gives for the series without their padding,
            shape (n, width), of the type and on the device of ``items``.

        Raises:
            lazy._steps.BarrierError: If ``model`` is Chronos-T5, ``items``
                has another shape, or a series has no finite value.
        """
        if self.gradient_kind == "barrier":
            raise _steps.BarrierError(self.barrier_reason)
        items = self._stack(items)
        network = self._network()
        series = [_trimmed(row) for row in items]
        embeddings = self._in_passes(
            series, lambda part: self._embedded(network, part)
        )
        return embeddings.to(items)

    def _embedded(self, network: Any, series: list[Any]) -> Any:
        """The embeddings of series, on torch's autograd graph.

        Args:
            network: What :meth:`_load` returned.
            series: Float torch tensors of shape (length,), with ``NaN``
                for a missing value; none starts or ends with one.

        Returns:
            The embeddings, a float64 tensor of shape (len(series), width)
            on the device of ``series``.
        """
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        recipe = self._recipe()
        spread = torch.stack(
            [
                torch.stack(_spread(values[-recipe.context :], constant=0.0))
                for values in series
            ]
        ).double()
        pooled = recipe.pooled(network, series)
        return torch.cat([pooled.to(spread), spread], dim=1)


def _spread(values: Any, constant: float) -> tuple[Any, Any]:
    """The mean and the standard deviation of a series' finite values.

    Args:
        values: A float torch tensor of shape (length,), with at least one
            finite value.
        constant: The standard deviation to give a constant series.

    Returns:
        Two tensors of shape (): the mean and the population standard
        deviation, whose gradient is finite for a constant series too.
    """
    observed = values[values.isfinite()]
    variance = observed.var(correction=0)
    return observed.mean(), _arrays.deviation(variance, constant)


def _steady(values: Any) -> Any:
    """A window, detached from the autograd graph if its values are equal.

    Args:
        values: The window a network normalizes, a float torch tensor of
            shape (length,), with ``NaN`` for a missing value.

    Returns:
        ``values``, without a gradient if its finite values are all equal.
    """
    observed = values[values.isfinite()]
    return values.detach() if bool((observed == observed[0]).all()) else values


def _left_padded(windows: list[Any], length: int, fill: float = np.nan) -> Any:
    """Series padded on the left with ``fill`` to ``length`` and stacked.

    Args:
        windows: Float torch tensors of shape (at most ``length``,).
        length: The length of each padded series.
        fill: The value of the padding.

    Returns:
        A tensor of shape (len(windows), length).
    """
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    return torch.stack(
        [
            torch.cat([values.new_full((length - len(values),), fill), values])
            for values in windows
        ]
    )


def _interpolated(values: Any) -> Any:
    """A series with each missing value filled linearly from its neighbours.

    It computes what ``np.interp`` does, on torch's autograd graph: a
    filled value's gradient goes to its two neighbours.

    Args:
        values: The series, a float torch tensor of shape (length,), with
            ``NaN`` for a missing value; it starts and ends with a finite
            one.

    Returns:
        The filled series, of the same shape.
    """
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    finite = values.isfinite()
    if bool(finite.all()):
        return values
    known = np.flatnonzero(_arrays.detached(finite))
    position = np.arange(len(values))
    # The known values on either side of each position, as np.interp's.
    right = np.clip(np.searchsorted(known, position, side="right"), 1, None)
    left = known[right - 1]
    right = known[np.minimum(right, len(known) - 1)]
    start = values[torch.as_tensor(left, device=values.device)]
    end = values[torch.as_tensor(right, device=values.device)]
    span = values.new_tensor(np.maximum(right - left, 1))
    offset = values.new_tensor(position - left)
    filled = (end - start) / span * offset + start
    return torch.where(finite, values, filled)


def _trimmed(row: Any) -> Any:
    """A padded series from its first finite value to its last.

    Args:
        row: A float torch tensor of shape (length,), with ``NaN`` for
            padding or a missing value.

    Returns:
        The series, a view of ``row`` that keeps its gradient.

    Raises:
        lazy._steps.BarrierError: If it has no finite value.
    """
    finite = np.flatnonzero(np.isfinite(_arrays.detached(row)))
    if not finite.size:
        raise _steps.BarrierError("a series has no finite value")
    return row[finite[0] : finite[-1] + 1]


def _masked_norm(norm: Any) -> Any:
    """Chronos's instance norm, with no ``NaN`` on the autograd graph.

    Args:
        norm: A ``chronos.chronos_bolt.InstanceNorm``.

    Returns:
        An instance norm that gives what ``norm`` gives, but whose gradient
        is finite where the context has a missing value, and for a
        constant window.
    """
    import chronos  # noqa: PLC0415 - an optional, heavy extra.
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    class MaskedNorm(chronos.chronos_bolt.InstanceNorm):
        """``InstanceNorm`` whose statistics skip a missing value."""

        def forward(
            self, x: Any, loc_scale: tuple[Any, Any] | None = None
        ) -> tuple[Any, tuple[Any, Any]]:
            """Standardizes ``x``, as ``InstanceNorm.forward`` does.

            Its ``(x - loc).square().nanmean()`` and division multiply a
            ``NaN`` by a zero gradient, which gives ``NaN``; here a
            missing value takes no part in the arithmetic.
            """
            if loc_scale is not None:
                return super().forward(x, loc_scale)
            dtype = x.dtype
            x = x.to(torch.float32)
            observed = x.isfinite()
            values = torch.where(observed, x, 0.0)
            count = observed.sum(dim=-1, keepdim=True)
            # A window with no observed value: loc 0 and scale 1, as
            # upstream's nan_to_num gives.
            some = count > 0
            count = torch.where(some, count, 1)
            loc = values.sum(dim=-1, keepdim=True) / count
            centred = torch.where(observed, values - loc, 0.0)
            variance = centred.square().sum(dim=-1, keepdim=True) / count
            scale = _arrays.deviation(variance, self.eps)
            scale = torch.where(some, scale, 1.0)
            scaled = centred / scale
            if self.use_arcsinh:
                scaled = torch.arcsinh(scaled)
            scaled = torch.where(observed, scaled, np.nan)
            return scaled.to(dtype), (loc, scale)

    return MaskedNorm(eps=norm.eps, use_arcsinh=norm.use_arcsinh)


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
