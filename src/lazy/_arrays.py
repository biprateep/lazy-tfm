# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Array arithmetic that the distributions and the ensemble share."""

from collections.abc import Iterable
from typing import Any

from lazy import _typing


def weighted_sum(
    weights: Iterable[float], terms: Iterable[_typing.FloatArray]
) -> Any:
    """``sum(weight * term)``, added in order into one running total.

    The terms may come from a generator, one at a time: only the total and
    the current term are held.

    Args:
        weights: One weight per term.
        terms: Arrays of one shape, as many as ``weights``.

    Returns:
        The weighted sum, an array of the terms' shape, or ``None`` for no
        terms.
    """
    total = None
    for weight, term in zip(weights, terms, strict=True):
        if total is None:
            total = weight * term
        else:
            total += weight * term
        # Released before the next term is computed, not after.
        del term
    return total
