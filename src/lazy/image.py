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

:class:`Pretrained`
    A frozen vision network, chosen by ``model``: the default of the
    modality ``"image"``. Its weights download on first use; it needs the
    ``encoders`` extra.
:class:`Statistics`
    Summary statistics of each channel's values.
:class:`Pixels`
    The image itself, averaged down to a few pixels.

Typical usage example:

  columns = lazy.image.Pixels(size=4).fit_transform(paths)
"""

from __future__ import annotations

import abc
from collections.abc import Callable
import dataclasses
import functools
import itertools
import math
import os
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from lazy import _arrays
from lazy import _steps
from lazy import _typing
from lazy.encoders import _base
from lazy.encoders import _pretrained

__all__ = ["STATISTICS", "Pixels", "Pretrained", "Statistics"]

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
    """The classical image encoders' ``n_channels``, reading and cache key.

    Their embeddings are differentiable in the pixels.
    """

    modality = "image"
    gradient_kind = "differentiable"

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
        pixels = np.asarray(self._read_item(item), dtype=float)
        if pixels.ndim == 2:
            pixels = pixels[:, :, np.newaxis]
        if pixels.ndim != 3 or pixels.shape[2] != self.n_channels:
            raise ValueError(
                "an image must have the shape (height, width) or (height,"
                f" width, {self.n_channels}), as n_channels says; got"
                f" {pixels.shape}"
            )
        return _finite(pixels)

    def _read_item(self, item: Any) -> Any:
        """A file or a Pillow image as an array, in ``n_channels``' mode."""
        return _read(item, self._converted)

    def _array(self, item: Any) -> _typing.FloatArray | None:
        """One image as an array, shape (height, width, n_channels)."""
        return self._pixels(item)

    def _stack(self, items: Any) -> Any:
        """Stacked images, shape (n, height, width, n_channels).

        Args:
            items: A float torch tensor of that shape, or of shape (n,
                height, width) for images of one channel.

        Returns:
            The images with their axis of channels.

        Raises:
            lazy._steps.BarrierError: If the tensor has another shape.
        """
        if items.ndim == 3:
            items = items[..., None]
        if items.ndim != 4 or items.shape[3] != self.n_channels:
            raise _steps.BarrierError(
                f"{type(self).__name__} has a forward only for images of"
                f" one size with {self.n_channels} channels; got a tensor"
                f" of shape {tuple(items.shape)}"
            )
        return items

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
        if _is_deep(image):
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
        if isinstance(item, Image.Image):
            item = self._converted(item)
        return _image_bytes(item) or super()._item_bytes(item)


def _is_deep(image: Image.Image) -> bool:
    """Whether a Pillow image is one band of more than 8 bits."""
    return image.mode in ("I", "F") or image.mode.startswith("I;16")


def _read(item: Any, convert: Callable[[Image.Image], Image.Image]) -> Any:
    """A file or a Pillow image as an array; any other item as it is.

    Args:
        item: One image.
        convert: What makes of a Pillow image the one that is read.

    Returns:
        The array of the converted image, or ``item`` itself.
    """
    if isinstance(item, (str, os.PathLike)):
        with Image.open(item) as image:
            return _read(image, convert)
    if isinstance(item, Image.Image):
        return np.asarray(convert(item))
    return item


def _rgb(image: Image.Image) -> Image.Image:
    """A Pillow image as RGB, unless it is one band of more than 8 bits."""
    return image if _is_deep(image) else image.convert("RGB")


def _finite(pixels: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """``pixels``, checked to hold at least one value, all finite.

    Raises:
        ValueError: If it is empty or a value is not finite.
    """
    if not pixels.size or not np.isfinite(pixels).all():
        raise ValueError(
            "an image must have pixels, all of them finite; got the shape"
            f" {pixels.shape}"
        )
    return pixels


def _image_bytes(item: Any) -> bytes | None:
    """The content of a file or of a Pillow image; ``None`` for others."""
    if isinstance(item, (str, os.PathLike)):
        return b"file:" + pathlib.Path(item).read_bytes()
    if isinstance(item, Image.Image):
        return f"pillow:{item.mode}:{item.size}:".encode() + item.tobytes()
    return None


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

    def forward(self, items: Any) -> Any:
        """The statistics of stacked images, on torch's autograd graph.

        A quartile's gradient goes to the pixels it is interpolated from,
        and the standard deviation of a channel of one value has none.

        Args:
            items: Images, a float torch tensor of shape (n, height, width,
                n_channels).

        Returns:
            What ``transform`` gives, shape (n, 8 * n_channels).
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        items = self._stack(items)
        values = items.reshape(len(items), -1, items.shape[-1])
        variance = values.var(dim=1, correction=0)
        std = torch.where(
            variance > 0, torch.where(variance > 0, variance, 1.0).sqrt(), 0.0
        )
        quartiles = torch.quantile(values, values.new_tensor(_QUARTILES), dim=1)
        steps = [items.diff(dim=axis).abs() for axis in (1, 2)]
        n_steps = sum(step[0, :, :, 0].numel() for step in steps)
        gradient = sum(step.sum(dim=(1, 2)) for step in steps) / max(n_steps, 1)
        return torch.cat([values.mean(dim=1), std, *quartiles, gradient], dim=1)


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

    def forward(self, items: Any) -> Any:
        """The small images of stacked images, on torch's autograd graph.

        Args:
            items: Images, a float torch tensor of shape (n, height, width,
                n_channels).

        Returns:
            What ``transform`` gives, shape (n, size * size *
            n_channels).
        """
        items = self._stack(items)
        down, across = (
            items.new_tensor(_area_weights(n, self.size))
            for n in items.shape[1:3]
        )
        # (n, c, h, w) -> (n, c, size, size) -> (n, size, size, c).
        small = down @ items.permute(0, 3, 1, 2) @ across.T
        return small.permute(0, 2, 3, 1).reshape(len(items), -1)


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


def _bilinear(x: _typing.FloatArray) -> _typing.FloatArray:
    """Pillow's bilinear (triangle) kernel, at distances ``x`` in pixels."""
    return np.clip(1.0 - np.abs(x), 0.0, None)


def _bicubic(x: _typing.FloatArray) -> _typing.FloatArray:
    """Pillow's bicubic kernel (Keys, a = -0.5), at distances ``x``."""
    a = -0.5
    x = np.abs(x)
    near = ((a + 2) * x - (a + 3)) * x * x + 1
    far = (((x - 5) * x + 8) * x - 4) * a
    return np.where(x < 1, near, np.where(x < 2, far, 0.0))


# Each resampling filter of the processors, as Pillow defines it
# (Resample.c): its support at unit scale, in pixels, and its kernel.
_FILTERS = {"bilinear": (1.0, _bilinear), "bicubic": (2.0, _bicubic)}


def _resampling(
    n_in: int, start: float, stop: float, n_out: int, resample: str
) -> tuple[npt.NDArray[np.int64], _typing.FloatArray]:
    """The weights that resample a span of one axis as Pillow does.

    They are those of Pillow's ``Image.resize`` with a ``box``, in float:
    the span [start, stop) of ``n_in`` pixels becomes ``n_out`` pixels,
    and a kernel widened by the reduction smooths as it shrinks. Each
    output pixel reads only the input pixels in Pillow's window around
    it, so the weights form a sparse matrix.

    Args:
        n_in: The number of pixels along the axis.
        start: Where the span starts, in pixels.
        stop: Where it stops, in pixels.
        n_out: The number of output pixels.
        resample: A key of :data:`_FILTERS`.

    Returns:
        A tuple (indices, weights): the (output, input) index of each
        weight that is not 0, shape (2, n_weights), in row-major order,
        and the weights, shape (n_weights,). The weights of each output
        pixel sum to 1.
    """
    support, kernel = _FILTERS[resample]
    scale = (stop - start) / n_out
    stretch = max(scale, 1.0)
    centre = start + (np.arange(n_out) + 0.5) * scale
    # Pillow's window: from the first input pixel whose centre is within
    # the widened support, as wide as Pillow allocates; the kernel is 0
    # beyond its support.
    reach = support * stretch
    first = np.maximum(np.trunc(centre - reach + 0.5), 0).astype(np.int64)
    pixel = first[:, None] + np.arange(2 * math.ceil(reach) + 1)
    distance = (pixel + 0.5 - centre[:, None]) / stretch
    weights = np.where(pixel < n_in, kernel(distance), 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    kept = weights != 0
    return np.stack([np.nonzero(kept)[0], pixel[kept]]), weights[kept]


@functools.lru_cache(maxsize=64)
def _sparse_resampling(recipe: _Recipe, n: int, short: int, dtype: Any) -> Any:
    """The resampling of an axis by a recipe, a sparse torch tensor.

    Args:
        recipe: The recipe, which sets the span and the filter.
        n: The number of pixels along the axis.
        short: The number along the image's shorter axis.
        dtype: The torch type of the weights.

    Returns:
        The weights of :func:`_resampling`, a COO tensor of shape
        (recipe.side, n) on the CPU, which no caller changes.
    """
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    indices, values = _resampling(
        n, *recipe.span(n, short), recipe.side, recipe.resample
    )
    # Not an inference tensor, even when first built by a transform: a
    # forward on the autograd graph reuses it.
    with torch.inference_mode(False):
        return torch.sparse_coo_tensor(
            torch.from_numpy(indices),
            torch.from_numpy(values).to(dtype),
            size=(recipe.side, n),
            # Built in row-major order, with no repeated index.
            check_invariants=False,
            is_coalesced=True,
        )


@dataclasses.dataclass(frozen=True)
class _Recipe:
    """How one model turns an image into an embedding.

    The preprocessing is the processor's saved with the weights
    (``preprocessor_config.json``), pinned here: the image is resampled
    to ``side`` by ``side`` pixels, its 8-bit values scaled to [0, 1],
    then normalized by ``mean`` and ``std``.

    Attributes:
        width: The width of the embedding.
        network: The class of transformers that loads the network.
        output: The output of the network that is the embedding.
        resample: The filter, a key of :data:`_FILTERS`.
        side: The side of the square the network reads, in pixels.
        shortest_edge: If set, the image is resized for its shorter side
            to have this many pixels, and the central square of ``side``
            is cut, as transformers' processors do.
        zoom: If set, the central square of ``1 / zoom`` of the shorter
            side is resized to ``side``; with neither, the whole image is.
        mean: The mean of each channel, subtracted after scaling.
        std: The standard deviation of each channel, divided by last.
        requires: The packages the network runs on besides transformers.
    """

    width: int
    network: str = "AutoModel"
    output: str = "pooler_output"
    resample: str = "bicubic"
    side: int = 224
    shortest_edge: int | None = None
    zoom: float | None = None
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    requires: tuple[str, ...] = ()

    def span(self, n: int, short: int) -> tuple[float, float]:
        """The span of an axis that becomes the ``side`` pixels read.

        Args:
            n: The number of pixels along the axis.
            short: The number along the image's shorter axis.

        Returns:
            A tuple (start, stop), in pixels.
        """
        if self.shortest_edge is not None:
            # transformers' get_resize_output_image_size and center_crop.
            resized = (
                self.shortest_edge
                if n == short
                else int(self.shortest_edge * n / short)
            )
            top = (resized - self.side) // 2
            return top * n / resized, (top + self.side) * n / resized
        if self.zoom is not None:
            half = short / self.zoom / 2
            return n / 2 - half, n / 2 + half
        return 0.0, float(n)

    def pixels(self, images: Any) -> Any:
        """What the network reads of images, on torch's autograd graph.

        ``torch.sparse.mm`` may not run on every device, such as Apple's
        MPS, so the images are resampled on the CPU; the gradient follows
        them there and back.

        Args:
            images: Images of one size, a float torch tensor of shape (n,
                height, width) or (n, height, width, 3), with 8-bit values
                (0 to 255).

        Returns:
            The pixels, a tensor of shape (n, 3, side, side), of the type
            of ``images``, on the CPU.
        """
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        grey = images.ndim == 3
        images = images.cpu()
        if grey:
            images = images[..., None]
        n, height, width, channels = images.shape
        rows, columns = (
            _sparse_resampling(self, axis, min(height, width), images.dtype)
            for axis in (height, width)
        )
        # Pillow resamples each row, then each column, and clips each pass
        # to the 8-bit range, which the overshoot of the bicubic kernel
        # leaves. Each pass is (side, axis) @ (axis, the rest).
        across = torch.sparse.mm(columns, images.permute(2, 0, 1, 3).flatten(1))
        across = across.clamp(0, 255).unflatten(1, (n, height, channels))
        square = torch.sparse.mm(rows, across.permute(2, 1, 0, 3).flatten(1))
        square = square.clamp(0, 255).unflatten(1, (n, self.side, channels))
        # (side, n, side, channels) -> (n, channels, side, side).
        square = square.permute(1, 3, 0, 2)
        if grey:
            square = square.expand(-1, 3, -1, -1)
        mean, std = (
            _arrays.constant(np.array(values)[:, None, None], square)
            for values in (self.mean, self.std)
        )
        return (square / 255 - mean) / std


# The normalization of DINOv2's and of CLIP's processors.
_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_OPENAI = (
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)

# The version of Pretrained's preprocessing, in its cache identity, so
# that a cache filed before a change to it is not read: bump it with
# every change to what the networks read of an image.
_PREPROCESSING = 2

# The stretch of an image that is not 8-bit: the percentiles of each band
# that map to 0 and 1, and the softening of the asinh that follows.
_LOW, _HIGH, _SOFTENING = 1.0, 99.0, 20.0


class Pretrained(_pretrained.PretrainedEncoder):
    r"""The embedding of each image, from a frozen vision network.

    ``model`` picks one of the networks of ``recipes``, each pinned to a
    commit on the Hugging Face Hub with its preprocessing (resizing,
    cropping, normalization): ``"dinov2-with-registers-small"`` (the
    default), ``"dinov2-large"``, ``"siglip2-base-patch16-224"``,
    ``"clip-vit-base-patch16"`` and ``"zoobot-encoder-convnext_nano"``.
    The embedding is the class token for DINOv2, the attention-pooled
    tokens for SigLIP 2, the projected class token for CLIP and the pooled
    features of the last stage for Zoobot, a network trained on galaxy
    images. Zoobot reads the central square of an image, of 1/1.2 of its
    shorter side, at 224 by 224 pixels with values in [0, 1].
    :doc:`/guide/encoders` lists the networks and gives the evidence behind
    the choice.

    The preprocessing runs in torch, in float64: the image is resampled
    with the filter of the network's processor, with Pillow's weights,
    without rounding to 8 bits in between.

    The networks read 8-bit RGB. A file, a Pillow image or a ``uint8``
    array of one, three or four (RGBA, whose alpha is dropped) channels is
    read as such. An array of another type, or
    a file of one band of more than 8 bits, such as an image of fluxes, is
    stretched first, band by band: its 1st and 99th percentiles map to 0
    and 1, then :math:`\operatorname{asinh}(20 x) /
    \operatorname{asinh}(20)`, clipped to [0, 1], maps to 0 to 255. The
    stretch keeps the contrast of an image but not its brightness, which a
    table column can carry.

    A gradient passes every network: ``forward`` gives the embeddings of
    images of one size on torch's autograd graph, without the cache, so
    that an attribution reaches each pixel. It reaches the values of an
    8-bit image, and the stretched values of an image that is stretched.

    Attributes:
        model: The network; ``"dinov2-with-registers-small"`` by default.
        device: ``"auto"``, or a torch device such as ``"cpu"``.
        batch_size: The most images in one forward pass.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "image"
    package = "transformers"
    gradient_kind = "differentiable"
    # Each processor's settings, from its preprocessor_config.json at the
    # pinned commit. CLIP's omits the scale, whose default is 1 / 255.
    recipes = {
        "dinov2-with-registers-small": _Recipe(
            384, shortest_edge=256, mean=_IMAGENET[0], std=_IMAGENET[1]
        ),
        "dinov2-large": _Recipe(
            1024, shortest_edge=256, mean=_IMAGENET[0], std=_IMAGENET[1]
        ),
        "siglip2-base-patch16-224": _Recipe(
            768,
            "SiglipVisionModel",
            resample="bilinear",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        "clip-vit-base-patch16": _Recipe(
            512,
            "CLIPVisionModelWithProjection",
            "image_embeds",
            shortest_edge=224,
            mean=_OPENAI[0],
            std=_OPENAI[1],
        ),
        # Zoobot's preprocessing at test time, from galaxy-datasets'
        # minimal_view_config: no normalization, which its config.json,
        # copied from ImageNet's ConvNeXt, would wrongly apply.
        "zoobot-encoder-convnext_nano": _Recipe(
            640, resample="bilinear", zoom=1.2, requires=("timm",)
        ),
    }

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        model: str = "dinov2-with-registers-small",
        device: str = "auto",
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = True,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.cache = cache

    def _load(self, path: os.PathLike[str], device: str) -> Any:
        """The network, on ``device``."""
        import transformers  # noqa: PLC0415 - an optional, heavy extra.

        network = getattr(transformers, self._recipe().network)
        # A checkpoint of two towers holds the text tower's weights too.
        return _pretrained.load_quietly(network, path).to(device)

    def _identity_head(self) -> list[str]:
        """The base's, and the version of the preprocessing."""
        return [*super()._identity_head(), f"preprocessing {_PREPROCESSING}"]

    def _array(self, item: Any) -> _typing.FloatArray | None:
        """One image as the network reads it; see :func:`_image`."""
        return _image(item)

    def _keeps_values(self, item: Any) -> bool:
        """Whether the image is 8-bit, which the network reads unstretched."""
        return bool(np.asarray(self._read_item(item)).dtype == np.uint8)

    def _embed(self, network: Any, items: list[Any]) -> _typing.FloatArray:
        """The embedding of each image, as the recipe names it.

        Each image is resampled as it is read, so that one image at a time
        is held at its full size in float64.
        """
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        recipe = self._recipe()
        pixels = torch.cat(
            [
                recipe.pixels(torch.from_numpy(_image(item))[None])
                for item in items
            ]
        )
        return self._embedded(network, pixels).float().cpu().numpy()

    def forward(self, items: Any) -> Any:
        """The embeddings of stacked images, on torch's autograd graph.

        It runs the network on its ``device``, without the cache, in
        passes of at most ``batch_size`` images.

        Args:
            items: Images of one size, grey or RGB, with values from 0 to
                255 (stretched, for an image that is not 8-bit): a float
                torch tensor of shape (n, height, width) or (n, height,
                width, 3).

        Returns:
            What ``transform`` gives, shape (n, width), of the type and on
            the device of ``items``.

        Raises:
            lazy._steps.BarrierError: If ``items`` has another shape.
        """
        if items.ndim not in (3, 4) or items.ndim == 4 and items.shape[3] != 3:
            raise _steps.BarrierError(
                "Pretrained has a forward only for images of one size, grey"
                f" or RGB; got a tensor of shape {tuple(items.shape)}"
            )
        network, recipe = self._network(), self._recipe()
        embeddings = self._in_passes(
            items, lambda part: self._embedded(network, recipe.pixels(part))
        )
        return embeddings.to(items)

    def _embedded(self, network: Any, pixels: Any) -> Any:
        """The embeddings of what the network reads, on the autograd graph.

        Args:
            network: What :meth:`_load` returned.
            pixels: What :meth:`_Recipe.pixels` gives, shape (n, 3, side,
                side).

        Returns:
            The embeddings, a tensor of shape (n, width) on the device of
            the network.
        """
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        result = network(pixel_values=pixels.to(network.device, torch.float32))
        return getattr(result, self._recipe().output)

    def _read_item(self, item: Any) -> Any:
        """A file or a Pillow image as an array: RGB, or a deep band."""
        return _read(item, _rgb)

    def _item_bytes(self, item: Any) -> bytes:
        """The content of an image: of the file for a path, else its RGB."""
        if isinstance(item, Image.Image):
            item = _rgb(item)
        return _image_bytes(item) or super()._item_bytes(item)


def _image(item: Any) -> _typing.FloatArray:
    """One image as a network reads it, before it is resampled.

    An image of 8 bits keeps its values; any other is stretched to them
    (:func:`_stretch`).

    Returns:
        Values from 0 to 255, shape (height, width) or (height, width, 3).

    Raises:
        ValueError: If an array has neither 1 nor 3 channels (nor 4, for
            8-bit RGBA), no pixel, or a value that is not finite.
    """
    array = np.asarray(_read(item, _rgb))
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 3 and array.shape[2] == 4 and array.dtype == np.uint8:
        array = array[:, :, :3]  # RGBA: the alpha goes, as Pillow drops it.
    if array.ndim not in (2, 3) or array.ndim == 3 and array.shape[2] != 3:
        raise ValueError(
            "an image for a pretrained network has the shape (height,"
            f" width) or (height, width, 3); got {array.shape}"
        )
    pixels = _finite(array).astype(float)
    return pixels if array.dtype == np.uint8 else _stretch(pixels)


def _stretch(pixels: _typing.FloatArray) -> _typing.FloatArray:
    """An image of any range on the 8-bit scale, band by band.

    See :class:`Pretrained`; the values are not rounded.

    Args:
        pixels: Finite values, shape (height, width) or (height, width, 3).

    Returns:
        The stretched image, from 0 to 255, of the same shape.
    """
    bands = pixels.reshape(*pixels.shape[:2], -1)
    low, high = np.percentile(bands, (_LOW, _HIGH), axis=(0, 1))
    # A band of one value has no contrast to stretch: it maps to 0.
    span = np.where(high > low, high - low, 1.0)
    scaled = np.arcsinh(_SOFTENING * (bands - low) / span)
    unit = np.clip(scaled / np.arcsinh(_SOFTENING), 0.0, 1.0)
    return (unit * 255).reshape(pixels.shape)
