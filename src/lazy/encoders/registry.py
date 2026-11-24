# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The encoder registry: which encoders exist and what they are called.

An encoder joins it through :func:`register`, which refuses a class that
falls short of the contract on :class:`~lazy.encoders.BaseEncoder`
(:mod:`lazy.encoders._conformance`). Registering is what lets
:func:`get_encoder` build it by name and the conformance tests find it.
"""

from __future__ import annotations

from typing import Any

from lazy import image
from lazy import text
from lazy import time
from lazy.encoders import _base
from lazy.encoders import _conformance
from lazy.models import _hub

__all__ = [
    "DEFAULT_ENCODERS",
    "ENCODERS",
    "get_encoder",
    "list_encoders",
    "register",
]

ENCODERS: dict[str, type[_base.BaseEncoder]] = {}
"""Short name -> encoder class. Filled only through :func:`register`."""

DEFAULT_ENCODERS: dict[str, str] = {
    "image": "image_pixels",
    "text": "text_hashing",
    "time": "time_statistics",
}
"""Modality -> the name of its default encoder, which :func:`get_encoder`
builds for the modality's name."""


def register(name: str, cls: type[_base.BaseEncoder]) -> None:
    """Adds an encoder to the registry, once it meets the encoder contract.

    Args:
        name: The short name :func:`get_encoder` will know it by, which is
            not a modality. A class already registered under it is
            replaced.
        cls: The encoder class.

    Raises:
        TypeError: If the class falls short of the contract; the message
            lists every problem.
        ValueError: If ``name`` is a modality, which names its default.
    """
    if name in _base.MODALITIES:
        raise ValueError(
            f"{name!r} is a modality, which names its default encoder:"
            " register the encoder under another name"
        )
    _conformance.check_static(cls)
    ENCODERS[name] = cls


register("image_pixels", image.Pixels)
register("image_statistics", image.Statistics)
register("text_hashing", text.Hashing)
register("text_statistics", text.Statistics)
register("time_spectrum", time.Spectrum)
register("time_statistics", time.Statistics)


def list_encoders(modality: str | None = None) -> list[str]:
    """Returns the encoder names, sorted.

    Args:
        modality: Only the encoders of this modality, one of
            :data:`~lazy.encoders.MODALITIES`; ``None`` for all of them.

    Raises:
        UnknownNameError: If ``modality`` is not one of them.
    """
    if modality is not None and modality not in _base.MODALITIES:
        raise _hub.unknown_name("modality", modality, _base.MODALITIES)
    return sorted(
        name
        for name, cls in ENCODERS.items()
        if modality is None or cls.modality == modality
    )


def get_encoder(name: str, **params: Any) -> _base.BaseEncoder:
    """Constructs the encoder registered under ``name``.

    Args:
        name: An encoder name, one of :func:`list_encoders`, or a modality,
            for its default encoder (:data:`DEFAULT_ENCODERS`).
        **params: Passed to the encoder's constructor.

    Returns:
        The unfitted encoder.

    Raises:
        UnknownNameError: If ``name`` is neither a modality nor the name of
            a registered encoder.
    """
    try:
        cls = ENCODERS[DEFAULT_ENCODERS.get(name, name)]
    except KeyError:
        known = [*DEFAULT_ENCODERS, *list_encoders()]
        raise _hub.unknown_name("encoder", name, known) from None
    return cls(**params)
