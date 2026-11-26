# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Array arithmetic that the distributions, the metrics and the ensemble share.

The functions that take ``Any`` arrays work on any array of the Python array
API standard, numpy's or torch's, through ``array_api_compat``; on a torch
tensor they keep its autograd graph. :func:`update`, :func:`put` and
:func:`clip` write into a writable numpy array, which saves the copy of a
large array, so a caller passes them only arrays it made itself.
"""

from collections.abc import Callable, Iterable
import operator
from typing import Any

import array_api_compat
import numpy as np

from lazy import _typing


def weighted_sum(
    weights: Iterable[float], terms: Iterable[_typing.FloatArray]
) -> Any:
    """``sum(weight * term)``, added in order into one running total.

    The terms may come from a generator, one at a time: only the total and
    the current term are held. The total starts as the first weighted term,
    so it keeps that term's memory order (see :func:`weighted_total`); the
    ensemble's pools and the mixtures' pdf, cdf and variance round as that
    order makes them, and a C-ordered start changes their last bits.

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


def weighted_total(weights: Iterable[float], terms: Iterable[Any]) -> Any:
    """``sum(weight * term)``, out of place on torch's arrays.

    Out of place keeps torch's graph. The terms are added in order into
    zeros, so that the total is C-ordered whatever the terms' order: numpy
    sums a row of a C-ordered array pairwise, and a row of an F-ordered one
    in sequence, which rounds differently. A mixture's densities on a grid
    are normalized by such row sums, and they round as a C-ordered total
    makes them; :func:`weighted_sum` keeps the first term's order instead,
    and the two cannot be one function without changing one of them.

    Args:
        weights: One weight per term.
        terms: Arrays of one shape, as many as ``weights``, numpy arrays or
            torch tensors.

    Returns:
        The weighted sum, an array of the terms' kind and shape.
    """
    total = None
    for weight, term in zip(weights, terms, strict=True):
        if total is None:
            xp = array_api_compat.array_namespace(term)
            total = xp.zeros(
                term.shape,
                dtype=term.dtype,
                device=array_api_compat.device(term),
            )
        total = update(total, operator.add, weight * term)
    return total


def detached(array: Any) -> _typing.FloatArray:
    """An array's values as a numpy array, off any autograd graph.

    A search or a lookup of an index carries no gradient, so numpy does it
    on these values, and the arithmetic gathers with the indices it finds.

    Args:
        array: A numpy array or a torch tensor.

    Returns:
        Its values; a numpy array is returned as it is.
    """
    if array_api_compat.is_torch_array(array):
        array = array.detach().cpu()
    return np.asarray(array)


_IN_PLACE = {
    operator.add: operator.iadd,
    operator.sub: operator.isub,
    operator.mul: operator.imul,
    operator.truediv: operator.itruediv,
}


def _writable(array: Any) -> bool:
    """Whether ``array`` is a numpy array that may be written into."""
    return isinstance(array, np.ndarray) and array.flags.writeable


def update(array: Any, operation: Callable[[Any, Any], Any], other: Any) -> Any:
    """``operation(array, other)``, written into ``array`` when it can be.

    Args:
        array: A numpy array the caller made, or a torch tensor.
        operation: ``operator.add``, ``sub``, ``mul`` or ``truediv``.
        other: The other operand, broadcastable to ``array``'s shape.

    Returns:
        ``array`` itself, updated, if it is a writable numpy array; else a
        new array, which keeps torch's graph.
    """
    if _writable(array):
        return _IN_PLACE[operation](array, other)
    return operation(array, other)


def put(array: Any, mask: Any, value: Any) -> Any:
    """``array`` with ``value`` where ``mask`` holds, written in if it can be.

    Args:
        array: A numpy array the caller made, or a torch tensor.
        mask: Booleans broadcastable to ``array``'s shape, a numpy array
            or a torch tensor.
        value: A number, or an array broadcastable to ``array``'s shape.

    Returns:
        ``array`` itself, updated, if it is a writable numpy array; else a
        new array, which keeps torch's graph.
    """
    if _writable(array):
        if np.ndim(mask) == 1 and np.ndim(value) == 0:
            # A mask of the last axis indexes its columns, faster than a
            # copyto that scans the whole array.
            array[..., mask] = value
        else:
            np.copyto(array, value, where=mask)
        return array
    xp = array_api_compat.array_namespace(array)
    if not isinstance(value, float):
        value = xp.asarray(value)
    return xp.where(xp.asarray(mask), value, array)


def clip(array: Any, low: float, high: float | None = None) -> Any:
    """``np.clip(array, low, high)``, written in if it can be.

    Args:
        array: A numpy array the caller made, or a torch tensor.
        low: The lowest value kept.
        high: The highest value kept, or ``None`` for no upper bound.

    Returns:
        ``array`` itself, clipped, if it is a writable numpy array; else a
        new array, which keeps torch's graph. NaN stays NaN, as in
        ``np.clip``.
    """
    if _writable(array):
        return np.clip(array, low, high, out=array)
    return array_api_compat.array_namespace(array).clip(array, low, high)
