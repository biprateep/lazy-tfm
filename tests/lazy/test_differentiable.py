# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The differentiable forward pass that every backend may offer.

Each backend's own pass is tested beside it (``test_tabpfn.py``); here are
the base class's checks and the torch densities.
"""

import numpy as np
import pytest
import standins

from lazy import distributions
from lazy.models import _differentiable

torch = pytest.importorskip("torch")


def test_a_torch_histogram_has_the_log_density_of_its_numpy_twin():
    # Zero-width buckets inside and at the end: the last edge closes the
    # last bucket of non-zero width.
    edges = np.array([0.0, 1.0, 1.0, 3.0, 4.0, 4.0])
    masses = np.array([[0.2, 0.0, 0.5, 0.3, 0.0], [0.1, 0.0, 0.1, 0.8, 0.0]])
    twin = distributions.HistogramDistribution(edges, masses)
    histogram = _differentiable.TorchHistogram(
        edges, torch.log(torch.tensor(masses))
    )
    for y in ([0.5, 4.0], [1.0, 3.5], [-1.0, 4.5], [3.0, 0.0]):
        with np.errstate(divide="ignore"):  # A density of 0 logs to -inf.
            expected = np.log(np.diag(twin.pdf(y)))
        np.testing.assert_allclose(histogram.log_prob(y).numpy(), expected)


def test_a_torch_mixture_weights_its_parts():
    edges = np.array([0.0, 1.0, 2.0])
    parts = tuple(
        _differentiable.TorchHistogram(
            edges, torch.log(torch.tensor(m, dtype=torch.float64))
        )
        for m in ([[0.5, 0.5]], [[0.9, 0.1]])
    )
    mixture = _differentiable.TorchMixture(parts, np.array([0.25, 0.75]))
    expected = np.log(0.25 * 0.5 + 0.75 * 0.9)
    np.testing.assert_allclose(mixture.log_prob([0.5]).numpy(), [expected])


def _fit(**params):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    model = standins.HistogramStandIn(n_estimators=2, progress=False, **params)
    return model.fit(X, X[:, 0]), X


def test_a_backend_without_the_pass_says_so():
    model, X = _fit(transforms=("none",))
    tensor = torch.tensor(X)
    with pytest.raises(NotImplementedError, match="no differentiable"):
        model._differentiable_density(tensor, tensor, X[:, 0])


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"transforms": ("quantile",)}, "no feature transform or clip"),
        ({"transforms": ("none",), "outlier_threshold": 3.0}, "or clip"),
    ],
)
def test_a_fit_that_is_not_differentiable_is_refused(params, message):
    model, X = _fit(**params)
    tensor = torch.tensor(X)
    with pytest.raises(ValueError, match=message):
        model._differentiable_density(tensor, tensor, X[:, 0])


def test_a_context_of_another_size_is_refused():
    model, X = _fit(transforms=("none",))
    tensor = torch.tensor(X)
    with pytest.raises(ValueError, match="n_context_"):
        model._differentiable_density(tensor, tensor[:10], X[:10, 0])
