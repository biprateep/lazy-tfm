# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The static contract an encoder must meet before it can be registered.

:func:`lazy.encoders.register` runs :func:`check_static` on every encoder,
so none joins the registry without the shared parameters and a modality,
and no pretrained one without the package that loads it, ``model`` and
``device``, and a recipe for exactly the models pinned for its modality.
That an encoder also behaves as the contract asks (rows that do not depend
on their batch, say) is for its own tests to show.

Typical usage example:

  _conformance.check_static(MyEncoder, "mine")  # raises TypeError
"""

from __future__ import annotations

import inspect

from lazy.encoders import _base
from lazy.encoders import _pretrained
from lazy.models import _conformance
from lazy.models import _hub

__all__ = ["check_static", "problems"]


def problems(cls: type) -> list[str]:
    """Lists every way ``cls`` falls short of the encoder contract.

    Args:
        cls: The candidate encoder class.

    Returns:
        One message per problem; empty when the class conforms.
    """
    if not (isinstance(cls, type) and issubclass(cls, _base.BaseEncoder)):
        return [f"{cls!r} is not a BaseEncoder subclass"]
    found = []
    if cls.modality not in _base.MODALITIES:
        found.append(
            f"modality = {cls.modality!r}, not one of {_base.MODALITIES}"
        )
    if inspect.isabstract(cls):
        missing = sorted(cls.__abstractmethods__)
        found.append(f"abstract methods are not implemented: {missing}")
    names, defaults = _base.BASE_PARAMS, _base.BASE_DEFAULTS
    if issubclass(cls, _pretrained.PretrainedEncoder):
        default = _hub.DEFAULT_ENCODER_MODELS.get(cls.modality)
        names = (*names, "model", "device")
        defaults = {**defaults, "model": default, "device": "auto"}
        found.extend(_pretrained_problems(cls))
    found.extend(_conformance.parameter_problems(cls, names, defaults))
    return found


def _pretrained_problems(cls: type[_pretrained.PretrainedEncoder]) -> list[str]:
    """What a pretrained encoder lacks: a package, pinned weights per model.

    Args:
        cls: The candidate class.

    Returns:
        One message per problem.
    """
    found = []
    if not getattr(cls, "package", ""):
        found.append("package, which loads the weights, is not declared")
    recipes = set(getattr(cls, "recipes", {}))
    pinned = set(_hub.list_versions(cls.modality))
    if recipes != pinned:
        found.append(
            f"recipes {sorted(recipes)} are not the models pinned for"
            f" {cls.modality!r} in ENCODER_CHECKPOINTS, {sorted(pinned)}"
        )
    return found


def check_static(cls: type) -> None:
    """Raises unless ``cls`` meets the encoder contract.

    Args:
        cls: The candidate encoder class.

    Raises:
        TypeError: If it falls short, listing every problem.
    """
    _conformance.refuse(cls, problems(cls))
