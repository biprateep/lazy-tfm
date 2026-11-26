# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The contract of the encoders that run a pretrained network.

A pretrained encoder runs one of several frozen networks, which its
``model`` parameter names. Each model's weights are pinned to a commit on
the Hugging Face Hub in :data:`lazy.encoders.CHECKPOINTS`, keyed by the
encoder's modality and the model, and its recipe (pooling, prompt, context)
is pinned in the encoder's class. The weights download on first use, with
a :class:`~lazy.LicenseWarning`, or ahead of time through
:func:`lazy.download_checkpoint`, and load once per process and device, in
the cache that the models share (:func:`lazy.clear_model_cache` empties
it). The packages come from the ``encoders`` extra.

A subclass declares ``recipes`` (model name -> recipe, with the width of
an embedding as ``width``) and the import name of its ``package``, which
:meth:`PretrainedEncoder._package` may vary by model, and implements
:meth:`PretrainedEncoder._load` and :meth:`PretrainedEncoder._embed`.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from importlib import metadata
import os
from typing import Any, ClassVar

import numpy as np

from lazy import _typing
from lazy.encoders import _base
from lazy.models import _device
from lazy.models import _ensemble
from lazy.models import _hub
from lazy.models import _weights

__all__ = ["EXTRA", "PINNED", "PretrainedEncoder", "load_quietly"]

EXTRA = "encoders"
"""The extra of lazy-tfm that installs the packages of the encoders."""

PINNED: dict[str, str] = {"dtype": "float32", "attn_implementation": "sdpa"}
"""What ``transformers`` loads a network with, never left to its defaults:
the precision, which a model's config may set lower, and the attention."""


class PretrainedEncoder(_base.BaseEncoder):
    """An encoder that runs a frozen network with pinned weights.

    Beyond :class:`~lazy.encoders.BaseEncoder`, its constructor takes
    ``model`` and ``device``. An embedding depends on its own item alone,
    up to float rounding: a batch pads its items to one shape.

    The cache keys an embedding by the versions of lazy, of the package
    that loads the network and of torch, the repository and the revision
    of the weights, the class and its parameters but ``device``. The cache
    is on by default.

    Attributes:
        model: The network, one of ``recipes``; see
            :func:`lazy.list_versions` with the modality.
        device: ``"auto"`` (CUDA, then Apple's MPS, then the CPU), or a
            torch device such as ``"cpu"`` or ``"cuda:1"``.
    """

    recipes: ClassVar[Mapping[str, Any]]
    package: ClassVar[str]

    _UNKEYED = (*_base.BASE_PARAMS, "device")

    # Set by each encoder's __init__; declared for the type checker only.
    model: str
    device: Any

    @abc.abstractmethod
    def _load(self, path: os.PathLike[str], device: str) -> Any:
        """Builds the network from the downloaded weights.

        Args:
            path: The snapshot directory of the model's checkpoint.
            device: The torch device to put it on.

        Returns:
            Whatever :meth:`_embed` needs: the network, its preprocessor.
        """

    @abc.abstractmethod
    def _embed(self, network: Any, items: list[Any]) -> _typing.FloatArray:
        """The embeddings of a batch, without gradients.

        Args:
            network: What :meth:`_load` returned.
            items: At most ``batch_size`` items, none of them missing.

        Returns:
            The embeddings, shape (len(items), :meth:`_width`).
        """

    def _recipe(self) -> Any:
        """The recipe of ``model``.

        Raises:
            UnknownNameError: If ``model`` is not one of ``recipes``.
        """
        try:
            return self.recipes[self.model]
        except (KeyError, TypeError):
            raise _hub.unknown_name(
                "model", self.model, sorted(self.recipes), argument="model"
            ) from None

    def _checkpoint(self) -> _hub.Checkpoint:
        """The pinned weights of ``model``."""
        return _hub.get_checkpoint(self.modality, self.model)

    def _width(self) -> int:
        """The width of the model's embedding; checks ``model``."""
        return self._recipe().width

    def fit(self, X: Any, y: Any = None) -> PretrainedEncoder:  # noqa: GS030 - scikit-learn's X, y.
        """Checks the model, the package, the device and the column.

        Args:
            X: One column of items; see :class:`~lazy.encoders.BaseEncoder`.
            y: Ignored; scikit-learn passes it.

        Returns:
            The encoder itself.

        Raises:
            UnknownNameError: If ``model`` is not one of ``recipes``.
            ImportError: If the ``encoders`` extra is not installed.
            ValueError: If ``device`` is not a device, or a parameter of
                :class:`~lazy.encoders.BaseEncoder` is not valid.
            RuntimeError: If ``device`` names an accelerator that is not
                there.
        """
        self._recipe()
        self._import()
        _device.resolve_device(self.device)
        super().fit(X, y)
        return self

    def _encode(self, items: list[Any]) -> _typing.FloatArray:
        """The embeddings of a batch, from the network of this process."""
        torch = self._import()
        device = _device.resolve_device(self.device)
        checkpoint = self._checkpoint()
        path = checkpoint.download()
        # The class is part of the key: a subclass may load differently.
        key = _weights.key(
            checkpoint.backend,
            checkpoint.version,
            path,
            device,
            "float32",
            type(self),
        )
        network = _weights.network(key, lambda: self._load(path, device))
        with torch.inference_mode():
            return np.asarray(self._embed(network, items), dtype=float)

    def _import(self) -> Any:
        """Imports the packages; returns torch.

        Raises:
            ImportError: If the ``encoders`` extra is not installed.
        """
        name = f"lazy.{self.modality}.{type(self).__name__}"
        for package in (self._package(), *self._requires()):
            _ensemble.import_extra(package, needed_by=name, extra=EXTRA)
        return _ensemble.import_extra("torch", needed_by=name, extra=EXTRA)

    def _package(self) -> str:
        """The module that loads ``model``; a subclass may vary it by model."""
        return self.package

    def _requires(self) -> tuple[str, ...]:
        """The packages ``model`` runs on besides the loader, by name.

        A recipe lists them as ``requires``, a name that is both the
        distribution's and the module's; most need none.
        """
        return getattr(self._recipe(), "requires", ())

    def _identity_head(self) -> list[str]:
        """The versions of lazy, its packages and torch; the weights' commit."""
        checkpoint = self._checkpoint()
        return [
            *super()._identity_head(),
            metadata.version(checkpoint.package),
            *(metadata.version(package) for package in self._requires()),
            metadata.version("torch"),
            checkpoint.repo_id,
            checkpoint.revision or "",
        ]


def load_quietly(network: Any, path: os.PathLike[str]) -> Any:
    """Loads a network with :data:`PINNED`, without transformers' report.

    A checkpoint can hold weights that the network does not use, such as
    the text tower of a model of two towers or a forecast head, which
    transformers would list as unexpected.

    Args:
        network: A ``transformers`` class of network, such as ``AutoModel``.
        path: The snapshot directory of the checkpoint.

    Returns:
        The network, in evaluation mode.
    """
    import transformers  # noqa: PLC0415 - an optional, heavy extra.

    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        return network.from_pretrained(path, **PINNED).eval()
    finally:
        transformers.logging.set_verbosity(verbosity)
