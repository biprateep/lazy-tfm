# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LazyEnsembleModel: the pools, the members, the scikit-learn contract.

The pools are checked against analytic Gaussians, and everything else on the
k-nearest-neighbour stand-ins registered under backend names, so the suite
runs on a CPU with nothing downloaded. One test pools two real checkpoints,
and runs only with ``LAZY_RUN_CHECKPOINT_TESTS=1``.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn import base as sklearn_base
from sklearn import model_selection
from sklearn.utils import estimator_checks
import standins

import lazy
from lazy import base
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import ensemble
from lazy.models import registry

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)

# Every test but the warning's own fits on fewer rows than the warning's
# threshold.
pytestmark = pytest.mark.filterwarnings("ignore::lazy.EnsembleSizeWarning")


class Gaussian(base.BaseDensityRegressor):
    """N(offset + x0, scale**2) per row, binned finely and exactly."""

    def __init__(
        self,
        offset=0.0,
        scale=1.0,
        *,
        y_grid=None,
        random_state=0,
        device="auto",
    ):
        self.offset = offset
        self.scale = scale
        self.y_grid = y_grid
        self.random_state = random_state
        self.device = device

    def _fit(self, X, y):
        self.native_grid_ = grid_lib.Grid.linear(
            -12.0, 12.0, 20_000, normalization="histogram"
        )

    def means(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.offset + frame.iloc[:, 0].to_numpy(dtype=float)

    def _predict_distribution(self, X):
        edges = self.native_grid_.edges
        cdf = stats.norm.cdf(edges[None, :], self.means(X)[:, None], self.scale)
        return distributions.HistogramDistribution(edges, np.diff(cdf, axis=1))

    def _predict_pdf(self, X, grid):
        return self._predict_distribution(X).on_grid(grid)

    def _default_grid(self):
        if self.y_grid is None:
            return self.native_grid_
        return super()._default_grid()


@pytest.fixture
def gaussian_data():
    """One feature, the member means' offset; targets within [-3, 3]."""
    X = np.linspace(-0.5, 0.5, 50)[:, None]
    y = np.linspace(-3.0, 3.0, 50)
    query = np.array([[-0.3], [0.0], [0.2]])
    return X, y, query


@pytest.fixture
def standin_backends(monkeypatch):
    """The stand-ins, registered under the names of three real backends."""
    monkeypatch.setitem(
        registry.ESTIMATORS, "tabpfn", standins.HistogramStandIn
    )
    monkeypatch.setitem(registry.ESTIMATORS, "tabicl", standins.QuantileStandIn)
    monkeypatch.setitem(
        registry.ESTIMATORS, "limix", standins.ScaffoldedHistogramStandIn
    )


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0] + 0.3 * X[:, 1] ** 2 + rng.normal(0.0, 0.2, 120)
    return X, y


def _fit_gaussians(pooling, gaussian_data, **params):
    X, y, _ = gaussian_data
    members = [Gaussian(0.0, 0.5), Gaussian(0.4, 0.8)]
    return lazy.LazyEnsembleModel(members, pooling=pooling, **params).fit(X, y)


# -- the pools ---------------------------------------------------------------


def test_the_geometric_pool_of_two_gaussians_is_their_product(gaussian_data):
    model = _fit_gaussians("geometric", gaussian_data)
    query = gaussian_data[2]
    dist = model.predict_distribution(query)
    mean_a, mean_b = query[:, 0], query[:, 0] + 0.4
    precision_a, precision_b = 1 / 0.5**2, 1 / 0.8**2
    mean = (precision_a * mean_a + precision_b * mean_b) / (
        precision_a + precision_b
    )
    variance = 2.0 / (precision_a + precision_b)
    np.testing.assert_allclose(dist.mean(), mean, atol=1e-6)
    np.testing.assert_allclose(dist.var(), variance, rtol=1e-5)


def test_the_linear_pool_of_two_gaussians_is_their_mixture(gaussian_data):
    model = _fit_gaussians("linear", gaussian_data)
    query = gaussian_data[2]
    dist = model.predict_distribution(query)
    means = np.stack([query[:, 0], query[:, 0] + 0.4])
    variances = np.array([[0.5**2], [0.8**2]])
    mean = means.mean(axis=0)
    variance = (variances + means**2).mean(axis=0) - mean**2
    # The mixture is cut at the grid's ends, 4.9 and 6.4 sd from the wider
    # member's mean, which moves it by about 1e-6.
    np.testing.assert_allclose(dist.mean(), mean, atol=1e-5)
    np.testing.assert_allclose(dist.var(), variance, rtol=1e-5)


def test_the_quantile_pool_of_two_gaussians_averages_mean_and_sd(
    gaussian_data,
):
    model = _fit_gaussians("quantile", gaussian_data)
    query = gaussian_data[2]
    dist = model.predict_distribution(query)
    assert isinstance(dist, distributions.QuantileDistribution)
    levels = np.array([0.01, 0.16, 0.5, 0.84, 0.99])
    expected = stats.norm.ppf(
        levels[None, :], (query + 0.2), np.mean([0.5, 0.8])
    )
    np.testing.assert_allclose(dist.ppf(levels), expected, atol=1e-5)
    # The ends are in the grid, so the pool has no point masses.
    grid = model.native_grid_
    ends = dist.ppf([0.0, 1.0])
    assert np.all((ends >= grid.y_min) & (ends <= grid.y_max))


def test_the_geometric_pool_follows_the_product_bin_by_bin(gaussian_data):
    """The product of the densities, on the grid, not only its moments."""
    model = _fit_gaussians("geometric", gaussian_data)
    query = gaussian_data[2]
    grid = model.native_grid_
    pooled = model.predict_proba(query)
    mean_a, mean_b = query[:, 0], query[:, 0] + 0.4
    product = np.sqrt(
        stats.norm.pdf(grid.centers, mean_a[:, None], 0.5)
        * stats.norm.pdf(grid.centers, mean_b[:, None], 0.8)
    )
    product /= (product * grid.widths).sum(axis=1, keepdims=True)
    # Far in the tails, the log floor (1e-10 of the uniform density) props
    # the pool up, by at most 1e-7.
    np.testing.assert_allclose(pooled, product, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("pooling", ["geometric", "linear"])
def test_one_member_is_that_member(standin_backends, data, pooling):
    X, y = data
    model = lazy.LazyEnsembleModel(["tabpfn"], pooling=pooling).fit(X, y)
    member = lazy.LazyModel("tabpfn").fit(X, y)
    np.testing.assert_allclose(
        model.predict_proba(X[:10], "native"),
        member.predict_proba(X[:10], model.native_grid_),
        rtol=1e-8,
        atol=1e-9,
    )


def test_one_quantile_member_is_that_member(standin_backends, data):
    X, y = data
    model = lazy.LazyEnsembleModel(["tabicl"], pooling="quantile").fit(X, y)
    member = lazy.LazyModel("tabicl").fit(X, y)
    values = np.linspace(y.min(), y.max(), 50)
    np.testing.assert_allclose(
        model.predict_distribution(X[:10]).cdf(values),
        member.predict_distribution(X[:10]).cdf(values),
        atol=1e-12,
    )


@pytest.mark.parametrize("pooling", ensemble.POOLINGS)
def test_predictions_have_the_documented_shapes(
    standin_backends, data, pooling
):
    X, y = data
    model = lazy.LazyEnsembleModel(["tabpfn", "tabicl"], pooling=pooling)
    model.fit(X, y)
    grid = model.native_grid_
    pdfs = model.predict_proba(X[:7])
    assert pdfs.shape == (7, grid.n_bins)
    np.testing.assert_allclose((pdfs * grid.widths).sum(axis=1), 1.0)
    trapezoid = lazy.Grid.linear(-4.0, 4.0, 300)
    on_grid = model.predict_proba(X[:7], trapezoid)
    np.testing.assert_allclose(
        np.trapezoid(on_grid, trapezoid.centers, axis=1), 1.0
    )
    assert model.predict(X[:7]).shape == (7,)
    assert model.predict_quantiles(X[:7]).shape == (7, 3)
    assert model.predict_interval(X[:7], 0.68).shape == (7, 2)
    pit = model.predict_pit(X[:7], y[:7])
    assert pit.shape == (7,)
    assert np.all((pit >= 0) & (pit <= 1))
    assert np.isfinite(model.score(X, y))
    table = model.evaluate(X, y)
    assert table["model"].iloc[0] == model.name_
    assert model.name_ == f"{pooling}(tabpfn:v0, tabicl:v0)"


def test_chunking_changes_nothing(standin_backends, data):
    X, y = data
    whole = lazy.LazyEnsembleModel(["tabpfn", "limix"], chunk_size=0)
    chunked = sklearn_base.clone(whole).set_params(chunk_size=7)
    np.testing.assert_allclose(
        whole.fit(X, y).predict_proba(X),
        chunked.fit(X, y).predict_proba(X),
    )


# -- the pooling grid --------------------------------------------------------


def test_the_pooling_grid_covers_the_targets_with_a_margin(gaussian_data):
    model = _fit_gaussians("geometric", gaussian_data)
    grid = model.native_grid_
    assert grid.normalization == "histogram"
    assert grid.y_min == pytest.approx(-3.0 - 0.25 * 6.0)
    assert grid.y_max == pytest.approx(3.0 + 0.25 * 6.0)
    # As fine as the members' 0.0012-wide buckets.
    assert grid.widths[0] <= 24.0 / 20_000 + 1e-12
    assert model.provenance_["pool_grid"]["n_bins"] == grid.n_bins


def test_the_pooling_grid_stops_where_every_member_does(gaussian_data):
    X, y, _ = gaussian_data
    member = Gaussian(0.0, 0.5, y_grid=lazy.Grid.linear(-3.2, 3.2, 100))
    member.fit(X, y)
    # A member without a native grid answers on its default grid.
    del member.native_grid_
    grid = ensemble._pool_grid((-3.0, 3.0), [member.grid_], None)
    assert (grid.y_min, grid.y_max) == pytest.approx((-3.2, 3.2))
    assert grid.n_bins == ensemble.MIN_POOL_BINS


def test_pool_bins_sets_the_resolution(gaussian_data):
    model = _fit_gaussians("linear", gaussian_data, pool_bins=300)
    assert model.native_grid_.n_bins == 300
    assert model.predict_proba(gaussian_data[2]).shape == (3, 300)


def test_y_grid_is_the_default_output_grid(standin_backends, data):
    X, y = data
    grid = lazy.Grid.linear(-5.0, 5.0, 80)
    model = lazy.LazyEnsembleModel(["tabpfn", "tabicl"], y_grid=grid)
    model.fit(X, y)
    assert model.grid_ == grid
    assert model.predict_proba(X[:3]).shape == (3, 80)
    assert model.predict_proba(X[:3], "native").shape[1] == (
        model.native_grid_.n_bins
    )
    other = lazy.Grid.linear(-5.0, 5.0, 40)
    assert model.predict_proba(X[:3], other).shape == (3, 40)


# -- the members -------------------------------------------------------------


def test_members_are_named_after_their_backend(standin_backends, data):
    X, y = data
    model = lazy.LazyEnsembleModel(
        [
            "tabpfn",
            lazy.LazyModel("tabpfn", n_estimators=2),
            standins.QuantileStandIn(),
            "tabpfn",
        ]
    )
    model.fit(X, y)
    assert list(model.named_estimators_) == [
        "tabpfn",
        "tabpfn_2",
        "quantilestandin",
        "tabpfn_3",
    ]
    assert model.named_estimators_.tabpfn_2.n_estimators == 2
    assert model.estimators_[2] is model.named_estimators_["quantilestandin"]
    np.testing.assert_allclose(model.weights_, 0.25)
    assert set(model.provenance_["members"]) == set(model.named_estimators_)
    assert model.provenance_["pooling"] == "geometric"


def test_members_are_clones(standin_backends, data):
    X, y = data
    given = lazy.LazyModel("tabpfn", n_estimators=2)
    model = lazy.LazyEnsembleModel([given, "tabicl"]).fit(X, y)
    assert model.estimators_[0] is not given
    assert not hasattr(given, "estimator_")


def test_random_state_and_device_reach_members_at_their_default(
    standin_backends, data
):
    X, y = data
    model = lazy.LazyEnsembleModel(
        ["tabpfn", lazy.LazyModel("tabicl", random_state=7)],
        random_state=3,
        device="cpu",
    )
    model.fit(X, y)
    tabpfn, tabicl = model.estimators_
    assert tabpfn.random_state == 3
    assert tabicl.random_state == 7
    assert tabpfn.device == tabicl.device == "cpu"
    assert tabpfn.estimator_.random_state_ == 3


def test_a_member_explicit_value_wins():
    member = lazy.LazyModel("tabicl", device="cuda:1", random_state=5)
    configured = ensemble._configure(member, 3, "cpu")
    assert configured.device == "cuda:1"
    assert configured.random_state == 5
    assert member is not configured
    plain = ensemble._configure(Gaussian(), 3, "cpu")
    assert (plain.random_state, plain.device) == (3, "cpu")


def test_get_params_lists_the_members_parameters(standin_backends):
    model = lazy.LazyEnsembleModel(["tabpfn", "tabicl"])
    params = model.get_params()
    assert params["tabpfn__n_estimators"] == 8
    assert isinstance(params["tabicl"], lazy.LazyModel)
    assert "tabpfn__n_estimators" not in model.get_params(deep=False)


def test_set_params_reaches_members_without_touching_the_list(
    standin_backends,
):
    given = ["tabpfn", "tabicl"]
    model = lazy.LazyEnsembleModel(given)
    model.set_params(tabpfn__n_estimators=2, pooling="linear")
    assert given == ["tabpfn", "tabicl"]
    assert model.models[0].get_params()["n_estimators"] == 2
    assert model.pooling == "linear"
    model.set_params(tabicl=lazy.LazyModel("tabicl", n_estimators=3))
    assert model.models[1].n_estimators == 3
    with pytest.raises(ValueError, match="invalid parameter"):
        model.set_params(tabfm__n_estimators=2)
    with pytest.raises(ValueError, match="does_not_exist"):
        model.set_params(tabpfn__does_not_exist=2)


def test_clone_keeps_every_parameter(standin_backends):
    model = lazy.LazyEnsembleModel(
        ["tabpfn", lazy.LazyModel("tabicl", n_estimators=3)],
        pooling="quantile",
        pool_bins=400,
    )
    copy = sklearn_base.clone(model)
    assert copy.get_params()["tabicl__n_estimators"] == 3
    assert copy.pooling == "quantile"
    assert copy.models[1] is not model.models[1]


def test_grid_search_tunes_a_member(standin_backends, data):
    X, y = data
    search = model_selection.GridSearchCV(
        lazy.LazyEnsembleModel(["tabpfn", "tabicl"]),
        {"tabpfn__n_estimators": [1, 2]},
        cv=2,
    )
    search.fit(X, y)
    assert search.best_params_["tabpfn__n_estimators"] in (1, 2)
    assert len(search.cv_results_["mean_test_score"]) == 2
    best = search.best_estimator_
    assert (
        best.estimators_[0].n_estimators
        == (search.best_params_["tabpfn__n_estimators"])
    )


@pytest.mark.parametrize(
    "check",
    [
        estimator_checks.check_no_attributes_set_in_init,
        estimator_checks.check_get_params_invariance,
        estimator_checks.check_parameters_default_constructible,
        estimator_checks.check_set_params,
    ],
    ids=lambda c: c.__name__,
)
def test_sklearn_contract(check):
    model = lazy.LazyEnsembleModel(["tabpfn", "tabicl"])
    check(type(model).__name__, model)


# -- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"models": ["tabpfn"], "pooling": "max"}, ValueError, "pooling"),
        ({"models": "tabpfn"}, ValueError, "non-empty list"),
        ({"models": []}, ValueError, "non-empty list"),
        ({"models": ["tabdpt"]}, ValueError, "unknown model"),
        ({"models": [3]}, TypeError, "int"),
        ({"models": ["tabpfn"], "pool_bins": 1}, ValueError, "pool_bins"),
        ({"models": ["tabpfn"], "chunk_size": -1}, ValueError, "chunk_size"),
    ],
)
def test_invalid_parameters_fail_at_fit(
    standin_backends, data, params, error, match
):
    X, y = data
    model = lazy.LazyEnsembleModel(**params)  # Construction checks nothing.
    with pytest.raises(error, match=match):
        model.fit(X, y)


def test_a_small_context_warns_with_several_members(standin_backends, data):
    X, y = data
    with pytest.warns(lazy.EnsembleSizeWarning, match="fewer members"):
        lazy.LazyEnsembleModel(["tabpfn", "tabicl"]).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", lazy.EnsembleSizeWarning)
        lazy.LazyEnsembleModel(["tabpfn"]).fit(X, y)


def test_a_large_context_does_not_warn(gaussian_data):
    X = np.linspace(-0.5, 0.5, ensemble.SMALL_CONTEXT_ROWS)[:, None]
    y = np.linspace(-3.0, 3.0, ensemble.SMALL_CONTEXT_ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", lazy.EnsembleSizeWarning)
        lazy.LazyEnsembleModel([Gaussian(), Gaussian(0.1)]).fit(X, y)


# -- features and persistence ------------------------------------------------


def test_feature_names_reach_the_members(standin_backends, data):
    X, y = data
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    model = lazy.LazyEnsembleModel(["tabpfn", "tabicl"]).fit(frame, y)
    for member in model.estimators_:
        assert list(member.feature_names_in_) == ["a", "b", "c"]
    np.testing.assert_allclose(
        model.predict_proba(frame[["c", "a", "b"]].iloc[:5]),
        model.predict_proba(frame.iloc[:5]),
    )
    with pytest.raises(ValueError, match="feature names"):
        model.predict_proba(frame.rename(columns={"a": "z"}))


def test_unnamed_features_stay_unnamed(standin_backends, data):
    X, y = data
    model = lazy.LazyEnsembleModel(["tabpfn", "tabicl"]).fit(X, y)
    assert not hasattr(model, "feature_names_in_")
    assert not hasattr(model.estimators_[0].estimator_, "feature_names_in_")
    with pytest.warns(UserWarning, match="has feature names"):
        model.predict_proba(pd.DataFrame(X[:3], columns=["a", "b", "c"]))


def test_a_fitted_ensemble_pickles(standin_backends, data):
    X, y = data
    model = lazy.LazyEnsembleModel(["tabpfn", "tabicl"]).fit(X, y)
    again = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        again.predict_proba(X[:5]), model.predict_proba(X[:5])
    )


# -- real checkpoints ----------------------------------------------------------


@needs_checkpoint
@pytest.mark.gpu
def test_tabpfn_fast_and_tabicl_pool_on_a_cpu():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0.0, 0.1, 300)
    members = [lazy.LazyModel("tabpfn", version="v3.5-fast"), "tabicl"]
    scores = {}
    for pooling in ensemble.POOLINGS:
        model = lazy.LazyEnsembleModel(
            members, pooling=pooling, device="cpu"
        ).fit(X[:200], y[:200])
        pdfs = model.predict_proba(X[200:])
        grid = model.native_grid_
        np.testing.assert_allclose((pdfs * grid.widths).sum(axis=1), 1.0)
        assert model.estimators_[0].device == "cpu"
        scores[pooling] = model.score(X[200:], y[200:])
    singles = [
        lazy.LazyModel(name, device="cpu", **params)
        .fit(X[:200], y[:200])
        .score(X[200:], y[200:])
        for name, params in [
            ("tabpfn", {"version": "v3.5-fast"}),
            ("tabicl", {}),
        ]
    ]
    # Pooling two good models should not fall below the worse of them.
    assert scores["geometric"] > min(singles)
