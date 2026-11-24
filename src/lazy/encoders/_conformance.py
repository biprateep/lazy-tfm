# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The static contract an encoder must meet before it can be registered.

:func:`lazy.encoders.register` runs :func:`check_static` on every encoder,
so none joins the registry without the shared parameters and a modality.
That an encoder also behaves as the contract asks (rows that do not depend
on their batch, say) is for its own tests to show.

Typical usage example:

  _conformance.check_static(MyEncoder, "mine")  # raises TypeError
"""

from __future__ import annotations

import inspect

from lazy.encoders import _base
from lazy.models import _conformance

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
    found.extend(
        _conformance.parameter_problems(
            cls, _base.BASE_PARAMS, _base.BASE_DEFAULTS
        )
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
