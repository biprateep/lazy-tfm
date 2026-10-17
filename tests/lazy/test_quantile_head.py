# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""What the quantile-head backends share: their tails and target scaling."""

import numpy as np
import pytest

import lazy
from lazy.models import _quantile_head


def test_mass_outside_counts_a_tail_beyond_the_grid_exactly():
    """The tails reach one spacing past the outermost quantiles."""
    levels = _quantile_head.quantile_levels(99)
    dist = lazy.distributions.QuantileDistribution(
        levels, np.linspace(0.0, 1.0, 99)[None]
    )
    spacing = 1.0 / 98
    grid = lazy.Grid.linear(-0.5 * spacing, 1.0 + 0.25 * spacing, 10)
    outside = _quantile_head.mass_outside(dist, grid)
    np.testing.assert_allclose(outside, [0.005 + 0.0075], rtol=1e-12)
    np.testing.assert_allclose(
        outside, 1.0 - dist.histogramize(grid.edges).masses.sum(), rtol=1e-12
    )


@pytest.mark.parametrize("pad", [0.0, 1e-9])
def test_no_mass_is_outside_a_grid_that_holds_the_tails(pad):
    levels = _quantile_head.quantile_levels(9)
    dist = lazy.distributions.QuantileDistribution(
        levels, np.linspace(0.0, 1.0, 9)[None]
    )
    low, high = dist.ppf([0.0, 1.0])[0]
    grid = lazy.Grid.linear(low - pad, high + pad, 10)
    np.testing.assert_array_equal(_quantile_head.mass_outside(dist, grid), 0)
