# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""TabFM's differentiable forward pass.

The hermetic test runs upstream's classifier on the fake backbone
(tests/lazy/fakes/tabfm.py); the last loads TabFM on the CPU, and runs only
with ``LAZY_RUN_CHECKPOINT_TESTS=1``.
"""

import fakes
import numpy as np
import pytest

import lazy


def _density_and_reference(model, X, y, X_test, y_test):
    """The differentiable log-density at ``y_test``, and predict's density.

    Returns:
        A tuple (log_density, reference, query, context), the last two the
        tensors that collect the gradients.
    """
    torch = pytest.importorskip("torch")
    query = torch.tensor(X_test, requires_grad=True)
    context = torch.tensor(X, requires_grad=True)
    log_density = model._differentiable_density(query, context, y).log_prob(
        y_test
    )
    reference = np.diag(model.predict_distribution(X_test).pdf(y_test))
    return log_density, reference, query, context


def _has_gradients(*tensors):
    """Whether each tensor collected a finite, non-zero gradient."""
    return all(
        bool(t.grad.isfinite().all()) and bool(t.grad.abs().sum() > 0)
        for t in tensors
    )


@pytest.mark.parametrize("n_dither", [1, 2])
def test_the_differentiable_pass_matches_predict_and_has_gradients(
    monkeypatch, n_dither
):
    # Three classes per level, so that a class shift undone the wrong way
    # shows, and three members, each with its own column order.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(70, 3))
    y = X[:, 0] + 0.3 * rng.normal(size=70)
    model = fakes.install("tabfm", monkeypatch)(
        n_estimators=3,
        n_coarse_bins=3,
        n_fine_bins=3,
        n_dither=n_dither,
        device="cpu",
        progress=False,
        transforms=("none",),
        outlier_threshold=None,
    ).fit(X[:60], y[:60])
    log_density, reference, query, context = _density_and_reference(
        model, X[:60], y[:60], X[60:], y[60:]
    )
    # The fake's logits are float32 on both paths.
    np.testing.assert_allclose(
        log_density.exp().detach().numpy(), reference, rtol=1e-5
    )
    log_density.sum().backward()
    assert _has_gradients(query, context)
    assert model._network().scale.grad is None


@pytest.mark.checkpoint
@pytest.mark.parametrize("n_dither", [1, 2])
def test_the_differentiable_pass_matches_predict_on_the_checkpoint(n_dither):
    # Small, since loading the checkpoint alone takes about 13 GB of a CPU
    # machine's memory.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = 2.0 + X[:, 0] + 0.1 * rng.normal(size=50)
    model = lazy.TabFMHistogram(
        n_estimators=2,
        n_dither=n_dither,
        device="cpu",
        progress=False,
        transforms=("none",),
        outlier_threshold=None,
    ).fit(X[:40], y[:40])
    log_density, reference, query, context = _density_and_reference(
        model, X[:40], y[:40], X[40:], y[40:]
    )
    # Predict decodes against a prefilled cache and this pass runs the
    # whole forward, so the two differ by float32 rounding, a few parts in
    # a million of the density.
    np.testing.assert_allclose(
        log_density.exp().detach().numpy(), reference, rtol=1e-4
    )
    log_density.sum().backward()
    assert _has_gradients(query, context)
