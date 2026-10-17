# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""What every backend with a quantile regression head shares.

A model whose regressor answers with quantiles of the predictive CDF (TabICL
and Kumo Tabular, each with 999) needs the same pieces around it: the levels
the quantiles sit at, its targets standardized in float64 before the model
sees them in float32, a native grid wide enough for the quantiles to
extrapolate into, and a warning when a row puts noticeable probability
outside that grid. :class:`QuantileHead` holds them, so that such a backend
declares ``native_output = "quantiles"``, mixes the class in before
:class:`~lazy.models._ensemble.ContextEnsembleEstimator` and implements only
what is its own.

Typical usage example:

  class MyQuantiles(_quantile_head.QuantileHead,
                    _ensemble.ContextEnsembleEstimator):
      native_output = "quantiles"
      ...
"""

from __future__ import annotations

import os
from typing import Any
import warnings

import numpy as np
import pandas as pd

from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib

__all__ = [
    "NATIVE_PAD",
    "NATIVE_QUANTILE_BINS",
    "OUTSIDE_TOLERANCE",
    "QuantileHead",
    "mass_outside",
    "quantile_levels",
    "target_scaling",
]

# Warnings skip every frame inside this package, so that they point at the
# user's call however deep in the package (or LazyModel) they are raised.
_PACKAGE_PREFIX = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
)

#: Bins of the native grid across the training targets' range: as fine as
#: 999 quantiles resolve. The padding adds bins of the same width.
NATIVE_QUANTILE_BINS = 1000

#: How far the native grid extends beyond the training targets on each
#: side, as a fraction of their range: the quantiles extrapolate.
NATIVE_PAD = 0.25

#: The probability a row may put outside the native grid before
#: tabulating it there warns.
OUTSIDE_TOLERANCE = 0.01


def quantile_levels(n_quantiles: int) -> _typing.FloatArray:
    """Returns the cumulative probabilities a quantile head's outputs sit at.

    The levels are interior points of ``[0, 1]``: the model never claims to
    know where the 0th or 100th percentile is.

    Args:
        n_quantiles: How many quantiles the model returns per row.

    Returns:
        The evenly spaced levels, shape ``(n_quantiles,)``.

    Examples:
        >>> quantile_levels(3).tolist()
        [0.25, 0.5, 0.75]
    """
    return np.linspace(0.0, 1.0, int(n_quantiles) + 2)[1:-1]


def target_scaling(targets: _typing.FloatArray) -> tuple[float, float]:
    """The (offset, scale) that standardize the targets: mean and std.

    A target whose values are all equal is constant, with the constant as
    its offset and a scale of 1, as in LimiX and upstream TabICL. Its std
    would decide otherwise by float rounding: ``np.std(np.full(100, 0.1))``
    is 2.8e-17, which would shrink the answer to a point, while that of
    ``np.full(100, 0.3)`` is exactly 0.

    Args:
        targets: The context targets, shape ``(n,)``.

    Returns:
        A tuple ``(offset, scale)``.
    """
    low, high = float(np.min(targets)), float(np.max(targets))
    if low == high:
        return low, 1.0
    scale = float(np.std(targets))
    return float(np.mean(targets)), scale if scale > 0 else 1.0


def mass_outside(
    dist: distributions.QuantileDistribution, grid: grid_lib.Grid
) -> _typing.FloatArray:
    """Each row's probability outside the grid, exactly.

    Args:
        dist: The predicted quantiles.
        grid: The grid they are tabulated on.

    Returns:
        The probability below ``grid.y_min`` plus that above ``grid.y_max``,
        tails included (they reach one quantile spacing beyond the outermost
        quantiles), shape ``(n_rows,)``.
    """
    cdf = dist.cdf([grid.y_min, grid.y_max])
    return cdf[:, 0] + (1.0 - cdf[:, 1])


class QuantileHead:
    """The native grid and density tabulation of a quantile-head backend.

    Mixed in before :class:`~lazy.models._ensemble.ContextEnsembleEstimator`,
    whose ``_chunks`` and ``native_grid_`` it uses. The native grid has
    :data:`NATIVE_QUANTILE_BINS` equal-width bins across the range of every
    context target (bagged or not), padded by :data:`NATIVE_PAD` of that
    range on each side; tabulating a row that puts more than
    :data:`OUTSIDE_TOLERANCE` of its probability outside it warns.
    """

    # The whole context's target range, recorded at fit for the grid.
    _support: tuple[float, float]

    # Provided by ContextEnsembleEstimator; declared for the type checker.
    native_grid_: grid_lib.Grid
    _chunks: Any

    def _before_fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        del X  # Unused: the native grid depends on the targets alone.
        # The native grid spans every context target, not only those some
        # bag happened to draw.
        self._support = (float(np.min(y)), float(np.max(y)))

    def _native_grid(self) -> grid_lib.Grid:
        low, high = self._support
        if high > low:
            pad = NATIVE_PAD * (high - low)
            n_bins = round(NATIVE_QUANTILE_BINS * (1.0 + 2.0 * NATIVE_PAD))
        else:
            pad = 0.01 * max(abs(high), 1.0)
            n_bins = NATIVE_QUANTILE_BINS
        return grid_lib.Grid.linear(
            low - pad, high + pad, n_bins, normalization="histogram"
        )

    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        if X.shape[0] == 0:
            return np.empty((0, grid.n_bins))
        blocks = []
        outside = 0
        for dist in self._chunks(X):
            blocks.append(dist.on_grid(grid))
            if grid == self.native_grid_ and isinstance(
                dist, distributions.QuantileDistribution
            ):
                outside += int(
                    np.count_nonzero(
                        mass_outside(dist, grid) > OUTSIDE_TOLERANCE
                    )
                )
        if outside:
            warnings.warn(
                f"{outside} of {len(X)} rows put more than "
                f"{OUTSIDE_TOLERANCE:.0%} of their probability outside the "
                f"native grid [{grid.y_min:g}, {grid.y_max:g}], and their "
                "densities there are renormalised. Pass a wider y_grid, or "
                "use predict_distribution, which has full support.",
                UserWarning,
                skip_file_prefixes=(_PACKAGE_PREFIX,),
            )
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _progress_postfix(
        self, dist: distributions.Distribution
    ) -> dict[str, Any]:
        del dist  # Unused: the count is recorded when predicting.
        return {"quantiles": getattr(self, "n_quantiles_", None)}
