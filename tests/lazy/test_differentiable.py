# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The differentiable forward pass that every backend may offer.

Each backend's own pass is tested beside it (``test_tabpfn.py``); here are
the base class's checks and the torch densities.
"""

import fakes
import numpy as np
import pytest
import standins

from lazy import distributions
from lazy import grid as grid_lib
from lazy import metrics
from lazy.models import _differentiable
from lazy.models import _transforms

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


def test_a_value_in_an_empty_bucket_leaves_finite_gradients():
    log_masses = torch.log(torch.tensor([[0.5, 0.0, 0.5]])).requires_grad_()
    histogram = _differentiable.TorchHistogram(
        np.array([0.0, 1.0, 2.0, 3.0]), log_masses
    )
    mixture = _differentiable.TorchMixture((histogram,), np.array([1.0]))
    log_density = mixture.log_prob([1.5])
    log_density.sum().backward()
    assert log_density.item() == -np.inf
    assert log_masses.grad.isfinite().all()


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


def _fake(name, monkeypatch, **params):
    """A backend's fake fitted on 40 rows, with 20 more as queries."""
    rng = np.random.default_rng(1)
    X = rng.lognormal(size=(60, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=60)
    model = fakes.install(name, monkeypatch)(
        n_estimators=4, device="cpu", progress=False, **params
    ).fit(X[:40], y[:40])
    return model, X[:40], y[:40], X[40:]


@pytest.mark.parametrize(
    "params",
    [
        {"transforms": "auto"},
        {"transforms": ("none",), "outlier_threshold": 3.0},
    ],
)
def test_what_the_models_own_package_applies_is_refused(monkeypatch, params):
    model, X, y, queries = _fake("tabpfn", monkeypatch, **params)
    with pytest.raises(ValueError, match="TabPFN.s own package applies"):
        model._differentiable_density(torch.tensor(queries), torch.tensor(X), y)


@pytest.mark.parametrize(
    "transforms",
    [
        ("power",),
        ("quantile", "quantile_uniform+original"),
        ("robust", "quantile_rtdl", "none"),
    ],
)
def test_the_pass_runs_through_the_transforms_that_lazy_applies(
    monkeypatch, transforms
):
    model, X, y, queries = _fake(
        "tabpfn", monkeypatch, transforms=transforms, outlier_threshold=None
    )
    assert any(t is not None for t in model.transformers_)
    query = torch.tensor(queries, requires_grad=True)
    density = model._differentiable_density(query, torch.tensor(X), y)
    np.testing.assert_allclose(
        density.mean().detach().numpy(),
        model.predict_distribution(queries).mean(),
        rtol=1e-4,
    )
    density.mean().sum().backward()
    assert query.grad.isfinite().all() and query.grad.abs().sum() > 0


def test_the_soft_clip_gives_a_tensor_what_it_gives_an_array():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    X[:, 2] = np.nan  # No standard deviation: the column passes through.
    queries = np.array([[0.0, 40.0, 1.0], [-30.0, 0.5, -7.0]])
    clip = _transforms.SoftClip(2.0).fit(X)
    point = torch.tensor(queries, requires_grad=True)
    clipped = clip.transform(point)
    np.testing.assert_allclose(
        clipped.detach().numpy(), clip.transform(queries)
    )
    clipped.sum().backward()
    # Beyond a bound, the slope is that of the logarithm of the magnitude.
    assert point.grad[0, 1].item() == pytest.approx(1 / 41)
    assert point.grad[1, 0].item() == pytest.approx(1 / 31)
    assert point.grad[:, 2].tolist() == [1.0, 1.0]


def test_a_context_of_another_size_is_refused():
    model, X = _fit(transforms=("none",))
    tensor = torch.tensor(X)
    with pytest.raises(ValueError, match="n_context_"):
        model._differentiable_density(tensor, tensor[:10], X[:10, 0])


def test_torch_quantiles_have_the_log_density_of_their_numpy_twin():
    levels = np.array([0.1, 0.3, 0.5, 0.9])
    # Rows: plain; a tie inside; a tie at each end, so a tail takes the
    # nearest non-zero spacing; values out of order, made monotone.
    locs = np.array(
        [
            [0.0, 1.0, 2.0, 4.0],
            [0.0, 1.0, 1.0, 4.0],
            [1.0, 1.0, 2.0, 2.0],
            [0.0, 2.0, 1.0, 4.0],
        ]
    )
    twin = distributions.QuantileDistribution(levels, locs)
    quantiles = _differentiable.TorchQuantiles(
        levels, torch.tensor(locs, dtype=torch.float64)
    )
    for y in (
        [-0.5, -0.9, 0.5, 2.5],
        [1.0, 1.0, 2.0, 6.5],
        [4.5, 5.0, 2.9, -2.1],
    ):
        with np.errstate(divide="ignore"):  # A density of 0 logs to -inf.
            expected = np.log(np.diag(twin.pdf(y)))
        np.testing.assert_allclose(quantiles.log_prob(y).numpy(), expected)


def test_torch_quantiles_average_as_their_numpy_twins():
    levels = np.array([0.25, 0.5, 0.75])
    parts = [np.array([[0.0, 1.0, 2.0]]), np.array([[1.0, 3.0, 4.0]])]
    expected = distributions.QuantileDistribution.average(
        [distributions.QuantileDistribution(levels, p) for p in parts],
        [0.25, 0.75],
    )
    average = _differentiable.TorchQuantiles.average(
        [
            _differentiable.TorchQuantiles(levels, torch.tensor(p))
            for p in parts
        ],
        np.array([0.25, 0.75]),
    )
    np.testing.assert_allclose(average.locs.numpy(), expected.locs)


def test_torch_quantiles_at_other_levels_are_not_averaged():
    parts = [
        _differentiable.TorchQuantiles(levels, torch.zeros(1, 2))
        for levels in (np.array([0.25, 0.75]), np.array([0.2, 0.8]))
    ]
    with pytest.raises(ValueError, match="same levels"):
        _differentiable.TorchQuantiles.average(parts, np.array([0.5, 0.5]))


def test_torch_quantiles_trim_as_their_numpy_twins():
    levels = np.array([0.25, 0.5, 0.75])
    parts = [np.array([[v, v + 1.0, v + 3.0]]) for v in (0.0, 1.0, 9.0, -4.0)]
    expected = distributions.QuantileDistribution.trimmed_average(
        [distributions.QuantileDistribution(levels, p) for p in parts], 0.3
    )
    trimmed = _differentiable.TorchQuantiles.trimmed_average(
        [
            _differentiable.TorchQuantiles(levels, torch.tensor(p))
            for p in parts
        ],
        0.3,
    )
    np.testing.assert_allclose(trimmed.locs.numpy(), expected.locs)


def _twins():
    """Torch densities and their numpy twins, from leaves with gradients.

    A histogram with zero-width buckets, inside and at the end; quantiles
    with ties, a row of equal values and values out of order; a mixture of
    a histogram with buckets of its own and one on the grid's own edges.
    """
    edges = np.array([0.0, 1.0, 1.0, 3.0, 4.0, 4.0])
    masses = np.array([[0.2, 0.1, 0.4, 0.3, 0.0], [0.1, 0.0, 0.1, 0.7, 0.1]])
    log_masses = torch.log(torch.tensor(masses)).requires_grad_()
    probabilities = torch.exp(log_masses).detach().numpy()
    yield (
        _differentiable.TorchHistogram(edges, log_masses),
        distributions.HistogramDistribution(edges, probabilities),
        [log_masses],
    )
    levels = np.array([0.1, 0.3, 0.5, 0.9])
    locs = np.array(
        [[0.0, 1.0, 1.0, 4.0], [2.0, 2.0, 2.0, 2.0], [0.5, 2.0, 1.0, 3.5]]
    )
    leaf = torch.tensor(locs, requires_grad=True)
    yield (
        _differentiable.TorchQuantiles(levels, leaf),
        distributions.QuantileDistribution(levels, locs),
        [leaf],
    )
    own = np.linspace(-1.0, 5.0, 7)
    parts = [(edges, masses), (own, np.full((2, 6), 1.0 / 6.0))]
    leaves = [torch.log(torch.tensor(m)).requires_grad_() for _, m in parts]
    weights = np.array([0.25, 0.75])
    yield (
        _differentiable.TorchMixture(
            tuple(
                _differentiable.TorchHistogram(e, leaf)
                for (e, _), leaf in zip(parts, leaves, strict=True)
            ),
            weights,
        ),
        distributions.MixtureDistribution(
            tuple(
                distributions.HistogramDistribution(
                    e, torch.exp(leaf).detach().numpy()
                )
                for (e, _), leaf in zip(parts, leaves, strict=True)
            ),
            weights,
        ),
        leaves,
    )


_GRIDS = [
    grid_lib.Grid.linear(-1.0, 5.0, 6),
    grid_lib.Grid.linear(-1.0, 5.0, 6, normalization="histogram"),
    grid_lib.Grid.linear(0.5, 3.0, 9),
]


@pytest.mark.parametrize("grid", _GRIDS, ids=["trapezoid", "own", "inner"])
def test_torch_densities_have_the_grid_statistics_of_their_numpy_twins(grid):
    for density, twin, _ in _twins():
        y = np.linspace(-2.0, 6.0, density.mean().shape[0])
        on_grid = density.on_grid(grid)
        pit, terms = metrics._row_terms(
            y, grid, on_grid, metrics.DENSITY_SCORES
        )
        expected = twin.on_grid(grid)
        cde, twin_pit, crps, nll = metrics.per_object_metrics(y, grid, expected)
        np.testing.assert_allclose(on_grid.detach().numpy(), expected)
        np.testing.assert_allclose(density.mean().detach().numpy(), twin.mean())
        for ours, twin_terms in [
            (terms["cde_loss"], cde),
            (pit, twin_pit),
            (terms["crps"], crps),
            (terms["nll"], nll),
        ]:
            np.testing.assert_allclose(
                ours.detach().numpy(), twin_terms, atol=1e-12
            )


@pytest.mark.parametrize("grid", _GRIDS, ids=["trapezoid", "own", "inner"])
def test_the_grid_statistics_of_torch_densities_have_gradients(grid):
    for density, _, leaves in _twins():
        y = np.linspace(0.7, 2.8, density.mean().shape[0])
        _, terms = metrics._row_terms(
            y, grid, density.on_grid(grid), metrics.DENSITY_SCORES
        )
        for statistic in (
            terms["cde_loss"],
            terms["crps"],
            terms["nll"],
            density.mean(),
        ):
            gradients = torch.autograd.grad(
                statistic.sum(), leaves, retain_graph=True, allow_unused=True
            )
            flat = torch.cat([g.ravel() for g in gradients if g is not None])
            assert torch.isfinite(flat).all()
            assert flat.abs().sum() > 0


def test_a_context_that_moves_refuses_a_transform_fitted_on_the_context(
    monkeypatch,
):
    model, X, y, queries = _fake(
        "tabpfn", monkeypatch, transforms=("quantile",), outlier_threshold=None
    )
    tensors = torch.tensor(queries), torch.tensor(X)
    model._differentiable_density(*tensors, y)
    with pytest.raises(ValueError, match="would not follow a context that"):
        model._differentiable_density(*tensors, y, context_moves=True)
