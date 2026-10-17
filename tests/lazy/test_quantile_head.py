# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""What the quantile-head backends share: their tails and target scaling."""

import fakes
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


@pytest.mark.parametrize("value", [0.1, 0.3, -1e4])
def test_a_constant_target_is_scaled_by_one(value):
    """Whatever its std rounds to: 2.8e-17 for 0.1 and 0 for 0.3 here."""
    assert _quantile_head.target_scaling(np.full(100, value)) == (value, 1.0)


def test_a_varying_target_is_standardized_by_its_mean_and_std():
    targets = np.array([0.1, 0.1, 0.1, np.nextafter(0.1, 1.0)])
    offset, scale = _quantile_head.target_scaling(targets)
    assert offset == np.mean(targets)
    assert scale == np.std(targets) > 0


@pytest.mark.parametrize("bag_size", [None, 50])
@pytest.mark.parametrize("name", ["tabicl", "kumo"])
def test_a_constant_context_gives_the_same_answer_at_any_value(
    monkeypatch, name, bag_size
):
    """The answer about the constant does not hang on float rounding."""
    X = np.random.default_rng(0).normal(size=(100, 3))
    backend = fakes.Backend(name, monkeypatch)
    answers = []
    for value in (0.1, 0.3):
        model = backend(bag_size=bag_size).fit(X, np.full(100, value))
        answers.append(model.predict_distribution(X[:5]).locs - value)
    np.testing.assert_allclose(answers[0], answers[1], rtol=0, atol=1e-12)
    # On the scale of one standardized unit, not of float rounding.
    assert (np.ptp(answers[0], axis=1) > 1.0).all()
