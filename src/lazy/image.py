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
from collections.abc import Sequence
import dataclasses
import functools
import itertools
import os
import pathlib
import types
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from lazy import _steps
from lazy import _typing
from lazy.encoders import _base
from lazy.encoders import _pretrained
from lazy.models import _hub

__all__ = ["STATISTICS", "VIEWS", "Pixels", "Pretrained", "Statistics"]

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

VIEWS: types.MappingProxyType[str, Image.Transpose | None] = (
    types.MappingProxyType(
        {
            "identity": None,
            "rotate_90": Image.Transpose.ROTATE_90,
            "rotate_180": Image.Transpose.ROTATE_180,
            "rotate_270": Image.Transpose.ROTATE_270,
            "flip_left_right": Image.Transpose.FLIP_LEFT_RIGHT,
            "flip_top_bottom": Image.Transpose.FLIP_TOP_BOTTOM,
            "transpose": Image.Transpose.TRANSPOSE,
            "transverse": Image.Transpose.TRANSVERSE,
        }
    )
)
"""The views of an image that :class:`Pretrained` embeds, by Pillow's names:
the eight rotations and flips of a square (the dihedral group D4), the
rotations counterclockwise. ``"d4"`` names all of them. Read-only."""

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
        return _finite(pixels)

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


@dataclasses.dataclass(frozen=True)
class _Recipe:
    """How one model turns an image into an embedding.

    Attributes:
        width: The width of the embedding.
        network: The class of transformers that loads the network.
        output: The output of the network that is the embedding.
        size: ``None`` to preprocess as the processor saved with the
            weights says; else the side, in pixels, of the square that
            :func:`_centred` cuts and scales.
        zoom: How much :func:`_centred` magnifies the centre of an image.
        requires: The packages the network runs on besides transformers.
    """

    width: int
    network: str = "AutoModel"
    output: str = "pooler_output"
    size: int | None = None
    zoom: float = 1.0
    requires: tuple[str, ...] = ()


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

    The networks read 8-bit RGB. A file, a Pillow image or a ``uint8``
    array of one, three or four (RGBA, whose alpha is dropped) channels is
    read as such. An array of another type, or
    a file of one band of more than 8 bits, such as an image of fluxes, is
    stretched first, band by band: its 1st and 99th percentiles map to 0
    and 1, then :math:`\operatorname{asinh}(20 x) /
    \operatorname{asinh}(20)`, clipped to [0, 1], maps to 0 to 255. The
    stretch keeps the contrast of an image but not its brightness, which a
    table column can carry.

    Attributes:
        model: The network; ``"dinov2-with-registers-small"`` by default.
        views: The rotations and flips of each image whose embeddings are
            averaged, at one forward pass each: ``None`` (the default) for
            the image as it is, ``"d4"`` for all eight of :data:`VIEWS`, or
            a list of their names. :doc:`/guide/encoders` shows their use.
        device: ``"auto"``, or a torch device such as ``"cpu"``.
        batch_size: The most images in one forward pass.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "image"
    package = "transformers"
    recipes = {
        "dinov2-with-registers-small": _Recipe(384),
        "dinov2-large": _Recipe(1024),
        "siglip2-base-patch16-224": _Recipe(768, "SiglipVisionModel"),
        "clip-vit-base-patch16": _Recipe(
            512, "CLIPVisionModelWithProjection", "image_embeds"
        ),
        # Zoobot's preprocessing at test time, from galaxy-datasets'
        # minimal_view_config: no normalization, which its config.json,
        # copied from ImageNet's ConvNeXt, would wrongly apply.
        "zoobot-encoder-convnext_nano": _Recipe(
            640, size=224, zoom=1.2, requires=("timm",)
        ),
    }

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        model: str = "dinov2-with-registers-small",
        views: str | Sequence[str] | None = None,
        device: str = "auto",
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = True,
    ) -> None:
        self.model = model
        self.views = views
        self.device = device
        self.batch_size = batch_size
        self.cache = cache

    def fit(self, X: Any, y: Any = None) -> Pretrained:  # noqa: GS030 - scikit-learn's X, y.
        """Checks ``views``, then the rest, as every pretrained encoder does.

        Args:
            X: One column of images.
            y: Ignored; scikit-learn passes it.

        Returns:
            The encoder itself.

        Raises:
            UnknownNameError: If ``model`` is not one of ``recipes``, or a
                view is not one of :data:`VIEWS`.
            ValueError: If ``views`` is not ``None``, ``"d4"`` or a list or
                tuple of distinct names, or ``device`` or another parameter
                is not valid.
            ImportError: If the ``encoders`` extra is not installed.
            RuntimeError: If ``device`` names an accelerator that is not
                there.
        """
        _transposes(self.views)
        super().fit(X, y)
        return self

    def _load(self, path: os.PathLike[str], device: str) -> Any:
        """The preprocessing of images to pixels, and the network."""
        import transformers  # noqa: PLC0415 - an optional, heavy extra.

        recipe = self._recipe()
        network = getattr(transformers, recipe.network)
        if recipe.size is None:
            # Pillow's resampling, pinned: with torchvision installed the
            # default would be torchvision's, whose pixels differ.
            processor = transformers.AutoImageProcessor.from_pretrained(
                path, backend="pil"
            )
            preprocess = functools.partial(_processed, processor)
        else:
            preprocess = functools.partial(
                _centred, size=recipe.size, zoom=recipe.zoom
            )
        # A checkpoint of two towers holds the text tower's weights too.
        model = _pretrained.load_quietly(network, path)
        return preprocess, model.to(device)

    def _embed(self, network: Any, items: list[Any]) -> _typing.FloatArray:
        """The mean embedding of each image's views, as the recipe names it."""
        preprocess, model = network
        images = [_rgb(item) for item in items]
        transposes = _transposes(self.views)
        recipe = self._recipe()
        # Summed on the device, so that the next view's preprocessing
        # overlaps this view's forward pass.
        total: Any = 0  # A torch tensor after the first view.
        for transpose in transposes:
            # Not `if transpose`: FLIP_LEFT_RIGHT is 0.
            shown = [
                image if transpose is None else image.transpose(transpose)
                for image in images
            ]
            pixels = preprocess(shown).to(model.device)
            result = model(pixel_values=pixels)
            total = total + getattr(result, recipe.output).float()
        return (total / len(transposes)).cpu().numpy()

    def _item_bytes(self, item: Any) -> bytes:
        """The content of an image: of the file for a path, else its RGB."""
        if isinstance(item, Image.Image):
            item = _rgb(item)
        return _image_bytes(item) or super()._item_bytes(item)


def _processed(processor: Any, images: list[Image.Image]) -> Any:
    """The pixels that a processor saved with the weights makes of images.

    Args:
        processor: A transformers image processor.
        images: RGB images.

    Returns:
        A torch tensor, shape (len(images), 3, height, width).
    """
    return processor(images=images, return_tensors="pt")["pixel_values"]


def _centred(images: list[Image.Image], size: int, zoom: float) -> Any:
    """The centre of each image, magnified, as Zoobot reads a galaxy.

    The square at the centre of an image, of ``1 / zoom`` of its shorter
    side, is resized to ``size`` by ``size`` pixels (bilinear, which
    smooths as it shrinks), and its 8-bit values are scaled to [0, 1].

    Args:
        images: RGB images.
        size: The side of the square, in pixels.
        zoom: How much the centre is magnified.

    Returns:
        A float32 torch tensor, shape (len(images), 3, size, size).
    """
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    squares = []
    for image in images:
        width, height = image.size
        side = min(width, height) / zoom
        left, top = (width - side) / 2, (height - side) / 2
        box = (left, top, left + side, top + side)
        square = image.resize((size, size), Image.Resampling.BILINEAR, box=box)
        squares.append(np.asarray(square))
    pixels = np.stack(squares).transpose(0, 3, 1, 2) / 255
    return torch.as_tensor(pixels, dtype=torch.float32)


def _transposes(
    views: str | Sequence[str] | None,
) -> list[Image.Transpose | None]:
    """The Pillow transposes that ``views`` names, in order.

    Raises:
        ValueError: If ``views`` is not ``None``, ``"d4"`` or a non-empty
            list of distinct names of :data:`VIEWS`.
    """
    if views is None:
        return [None]
    if isinstance(views, str) and views == "d4":
        return list(VIEWS.values())
    if (
        isinstance(views, str)
        or not isinstance(views, Sequence)
        or not views
        or not all(isinstance(view, str) for view in views)
        or len(set(views)) != len(views)
    ):
        raise ValueError(
            "views must be None, 'd4' or a list or tuple of distinct names of"
            f" {list(VIEWS)}: {views=}"
        )
    for view in views:
        if view not in VIEWS:
            raise _hub.unknown_name("view", view, list(VIEWS), argument="views")
    return [VIEWS[view] for view in views]


def _rgb(item: Any) -> Image.Image:
    """One image as 8-bit RGB, for a network.

    Raises:
        ValueError: If an array has neither 1 nor 3 channels (nor 4, for
            8-bit RGBA), no pixel, or a value that is not finite.
    """
    if isinstance(item, (str, os.PathLike)):
        with Image.open(item) as image:
            return _rgb(image)
    if isinstance(item, Image.Image) and not _is_deep(item):
        return item.convert("RGB")
    array = np.asarray(item)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 3 and array.shape[2] == 4 and array.dtype == np.uint8:
        array = array[:, :, :3]  # RGBA: the alpha goes, as Pillow drops it.
    if array.ndim not in (2, 3) or array.ndim == 3 and array.shape[2] != 3:
        raise ValueError(
            "an image for a pretrained network has the shape (height,"
            f" width) or (height, width, 3); got {array.shape}"
        )
    if array.dtype != np.uint8:
        array = _stretch(_finite(array.astype(float)))
    return Image.fromarray(_finite(array)).convert("RGB")


def _stretch(pixels: _typing.FloatArray) -> npt.NDArray[np.uint8]:
    """An image of any range as 8 bits, band by band; see :class:`Pretrained`.

    Args:
        pixels: Finite values, shape (height, width) or (height, width, 3).

    Returns:
        The stretched image, of the same shape.
    """
    bands = pixels.reshape(*pixels.shape[:2], -1)
    low, high = np.percentile(bands, (_LOW, _HIGH), axis=(0, 1))
    # A band of one value has no contrast to stretch: it maps to 0.
    span = np.where(high > low, high - low, 1.0)
    scaled = np.arcsinh(_SOFTENING * (bands - low) / span)
    unit = np.clip(scaled / np.arcsinh(_SOFTENING), 0.0, 1.0)
    return np.rint(unit * 255).astype(np.uint8).reshape(pixels.shape)
