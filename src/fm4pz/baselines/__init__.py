# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""The classical baselines, each a :class:`fm4pz.cv.Method`.

:func:`get` returns a baseline by name; the modules are imported only when
asked for, so that one baseline's dependencies never slow another's start.
"""

from __future__ import annotations

import importlib

from fm4pz import cv

__all__ = ["NAMES", "get"]

#: Each baseline's name and the module that defines it.
NAMES: dict[str, str] = {
    "flexzboost": "fm4pz.baselines.flexzboost",
    "cmnn": "fm4pz.baselines.cmnn",
    "gpz": "fm4pz.baselines.gpz",
}


def get(name: str) -> cv.Method:
    """Returns the baseline called ``name``.

    Args:
        name: One of :data:`NAMES`.

    Returns:
        A fresh instance of the baseline.

    Raises:
        ValueError: If ``name`` is not a known baseline.
    """
    if name not in NAMES:
        raise ValueError(f"unknown baseline: {name=}; known: {list(NAMES)}")
    return importlib.import_module(NAMES[name]).METHOD
