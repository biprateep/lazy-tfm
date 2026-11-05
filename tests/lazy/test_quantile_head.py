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
    outside = dist._bin_densities_and_outside(grid)[1]
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
    np.testing.assert_array_equal(dist._bin_densities_and_outside(grid)[1], 0)


def test_the_mass_outside_from_the_tabulation_is_the_cdf_at_the_ends():
    """To the bit, on rows outside either end, both, or neither."""
    rng = np.random.default_rng(0)
    levels = _quantile_head.quantile_levels(999)
    centres = np.array([-0.9, 0.9, 0.0, 0.0, 0.0])
    spreads = np.array([0.3, 0.3, 2.0, 0.1, 0.0])
    locs = np.sort(rng.normal(size=(5, 999)), axis=1) * spreads[:, None]
    dist = lazy.distributions.QuantileDistribution(
        levels, centres[:, None] + locs
    )
    grid = lazy.Grid.linear(-1.0, 1.0, 1500, normalization="histogram")
    density, outside = dist._bin_densities_and_outside(grid)
    cdf = dist.cdf([grid.y_min, grid.y_max])
    assert outside.tobytes() == (cdf[:, 0] + (1.0 - cdf[:, 1])).tobytes()
    assert grid.normalize(density).tobytes() == dist.on_grid(grid).tobytes()
    assert (outside[:3] > 0.01).all() and (outside[3:] == 0).all()


@pytest.mark.parametrize("value", [0.1, 0.3, -1e4])
def test_a_constant_target_is_scaled_by_one(value):
    """Whatever its std rounds to: 2.8e-17 for 0.1 and 0 for 0.3 here."""
    assert _quantile_head.target_scaling(np.full(100, value)) == (value, 1.0)


def test_a_varying_target_is_standardized_by_its_mean_and_std():
    targets = np.array([0.1, 0.1, 0.1, np.nextafter(0.1, 1.0)])
    offset, scale = _quantile_head.target_scaling(targets)
    assert offset == np.mean(targets)
    assert scale == np.std(targets) > 0
