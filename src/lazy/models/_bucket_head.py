# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""What every backend with a bucket head shares.

A model whose regressor is a classifier over fixed buckets of the
standardised target, such as TabPFN or LimiX-2, answers with logits over
those buckets. Its density is a
:class:`~lazy.distributions.HistogramDistribution` over the buckets' borders
in the target's units, and its native grid is every member group's buckets.
:func:`bucket_edges` and :func:`native_grid` build those, for any backend
with bins (TabFM's grid too). :func:`softmax` turns logits into
probabilities for any classifier head, TabFM's included. TabPFN's public
:func:`~lazy.models.tabpfn.bucket_masses` applies ``torch.softmax`` to
upstream's logits instead: its rounding differs from numpy's by a few units
in the last place, and TabPFN's densities are pinned to it.

Typical usage example:

  edges = _bucket_head.bucket_edges(borders, mean, std)
  masses = _bucket_head.softmax(logits, temperature)
  dist = distributions.HistogramDistribution(edges, masses)
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lazy import _typing
from lazy import grid as grid_lib

__all__ = ["bucket_edges", "native_grid", "softmax"]


def softmax(
    logits: _typing.FloatArray, temperature: float
) -> _typing.FloatArray:
    """Temperature-scaled softmax over the last axis.

    The result has the dtype of ``logits / temperature``: float32 logits stay
    float32, and integer logits become float64.

    Args:
        logits: Logits, any shape; classes on the last axis.
        temperature: Divides the logits before the softmax.

    Returns:
        Probabilities of the same shape, summing to one over the last axis.

    Examples:
        >>> softmax(np.zeros((1, 4)), temperature=0.9).round(3).tolist()
        [[0.25, 0.25, 0.25, 0.25]]
    """
    scaled = logits / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    weights = np.exp(scaled, out=scaled)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights


def bucket_edges(
    borders: _typing.FloatArray, mean: float, std: float
) -> _typing.FloatArray:
    """The buckets' borders in the target's units.

    Args:
        borders: The borders in standardised units, shape
            ``(n_buckets + 1,)``.
        mean: The context targets' mean.
        std: Their standard deviation, positive.

    Returns:
        ``borders * std + mean``, made non-decreasing, the same shape.
    """
    return np.maximum.accumulate(borders * std + mean)


def native_grid(edges: Iterable[_typing.FloatArray]) -> grid_lib.Grid:
    """The histogram grid over every member group's buckets.

    Under bagging each member standardises its target on its own bag, and so
    places its buckets there; the grid holds all of them.

    Args:
        edges: Each member group's bucket borders in the target's units.

    Returns:
        A ``"histogram"`` grid whose edges are the union of ``edges``.
    """
    union = np.unique(np.concatenate(list(edges)))
    if union.size < 3:
        # A constant target has one bucket; a grid needs two, and halving
        # it changes no density.
        union = np.r_[union[0], union.mean(), union[-1]]
    return grid_lib.Grid.from_edges(union, normalization="histogram")
