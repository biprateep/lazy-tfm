# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The estimator protocol, exercised through a backend-free stand-in.

Nothing here needs a GPU or a checkpoint: a Gaussian whose mean tracks the first
feature satisfies the same contract the real backends do, so it can carry the
whole scikit-learn conformance suite.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn import base as sklearn_base
from sklearn import exceptions

import lazy


class GaussianDummy(lazy.BaseDensityRegressor):
    """p(z | x) = N(x's first column + offset, sigma). Not a photo-z method."""

    def __init__(self, *, sigma=0.1, z_grid=None):
        self.sigma = sigma
        self.z_grid = z_grid

    def _fit(self, X, y):
        self.offset_ = float(np.mean(y) - np.mean(X.iloc[:, 0]))

    def _predict_pdf(self, X, grid):
        centers = np.asarray(X.iloc[:, 0], dtype=float) + self.offset_
        return np.exp(
            -0.5
            * ((grid.centers[None, :] - centers[:, None]) / self.sigma) ** 2
        )


@pytest.fixture
def data():
    generator = np.random.default_rng(0)
    X = pd.DataFrame(
        {"a": generator.uniform(0.2, 1.8, 64), "b": generator.normal(size=64)}
    )
    return X, X["a"].to_numpy()


def test_fit_returns_self_and_sets_fitted_attributes(data):
    X, y = data
    est = GaussianDummy()
    assert est.fit(X, y) is est
    assert est.n_features_in_ == 2
    assert list(est.feature_names_in_) == ["a", "b"]
    assert est.grid_ is lazy.DC1_GRID


def test_predict_proba_is_normalized_on_the_grid(data):
    X, y = data
    pdfs = GaussianDummy().fit(X, y).predict_proba(X)
    assert pdfs.shape == (len(X), lazy.DC1_GRID.n_bins)
    assert np.allclose(np.trapezoid(pdfs, lazy.DC1_GRID.centers, axis=1), 1.0)


def test_predict_pdf_is_an_alias_of_predict_proba(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    assert np.array_equal(est.predict_pdf(X), est.predict_proba(X))


def test_predict_recovers_the_truth_for_a_well_specified_model(data):
    X, y = data
    z = GaussianDummy(sigma=0.05).fit(X, y).predict(X)
    # One grid bin plus the peak's discreteness.
    assert np.abs(z - y).max() < 0.02


@pytest.mark.parametrize("method", lazy.POINT_ESTIMATORS)
def test_every_point_estimate_method_is_usable(data, method):
    X, y = data
    z = GaussianDummy().fit(X, y).predict(X, method=method)
    assert z.shape == (len(X),)
    assert np.isfinite(z).all()


def test_z_peak_is_the_default_method(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    assert np.array_equal(est.predict(X), est.predict(X, method="z_peak"))


def test_unknown_method_names_the_alternatives(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    with pytest.raises(ValueError, match="method must be one of"):
        # The un-prefixed spelling is not accepted.
        est.predict(X, method="peak")


def test_score_is_negated_cde_loss_so_higher_is_better(data):
    X, y = data
    sharp = GaussianDummy(sigma=0.05).fit(X, y).score(X, y)
    blunt = GaussianDummy(sigma=0.5).fit(X, y).score(X, y)
    assert sharp > blunt


# -- the grid is an argument to the prediction, not to the fit ---------------


def test_a_grid_passed_at_call_time_is_used(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    assert est.predict_proba(X, lazy.Grid.linear(0.0, 3.0, 37)).shape == (
        len(X),
        37,
    )
    assert est.predict_proba(X, np.linspace(0.05, 2.95, 30)).shape == (
        len(X),
        30,
    )
    assert est.predict(X, z_grid=np.linspace(0.05, 2.95, 30)).shape == (len(X),)


def test_one_fitted_model_answers_on_many_grids_without_refitting(data):
    X, y = data
    est = GaussianDummy(sigma=0.05).fit(X, y)
    coarse = est.predict(X, z_grid=lazy.Grid.linear(0.0, 2.0, 50))
    fine = est.predict(X, z_grid=lazy.Grid.linear(0.0, 2.0, 400))
    assert est.offset_ is not None  # nothing was refitted
    # Same estimator, coarser resolution.
    assert np.abs(coarse - fine).max() < 0.05


def test_the_constructor_grid_is_the_default(data):
    X, y = data
    est = GaussianDummy(z_grid=lazy.Grid.linear(0.0, 3.0, 37)).fit(X, y)
    assert est.predict_proba(X).shape == (len(X), 37)
    # ... and a call-time grid still wins
    assert est.predict_proba(X, lazy.Grid.linear(0.0, 2.0, 11)).shape == (
        len(X),
        11,
    )


def test_predict_cdf_is_monotone_and_reaches_one(data):
    X, y = data
    cdf = GaussianDummy().fit(X, y).predict_cdf(X)
    assert (np.diff(cdf, axis=1) >= -1e-12).all()
    assert np.allclose(cdf[:, -1], 1.0, atol=1e-9)


# -- input validation --------------------------------------------------------


def test_columns_may_be_reordered_but_not_renamed(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    assert np.allclose(est.predict_proba(X[["b", "a"]]), est.predict_proba(X))
    with pytest.raises(ValueError, match="feature names differ"):
        est.predict_proba(X.rename(columns={"a": "c"}))


def test_feature_count_mismatch_is_caught(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    with pytest.raises(ValueError, match="fitted with 2"):
        est.predict_proba(X[["a"]])


def test_numpy_input_is_accepted(data):
    X, y = data
    est = GaussianDummy().fit(X.to_numpy(), y)
    assert est.n_features_in_ == 2
    assert not hasattr(est, "feature_names_in_")  # as in scikit-learn
    assert est.predict_proba(X.to_numpy()).shape == (
        len(X),
        lazy.DC1_GRID.n_bins,
    )


def test_mismatched_lengths_are_rejected(data):
    X, y = data
    with pytest.raises(ValueError, match="rows but y has"):
        GaussianDummy().fit(X, y[:-1])


def test_non_finite_targets_are_rejected(data):
    X, y = data
    y = y.copy()
    y[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        GaussianDummy().fit(X, y)


def test_predict_before_fit_raises(data):
    X, _ = data
    with pytest.raises(exceptions.NotFittedError):
        GaussianDummy().predict_proba(X)


# -- scikit-learn conformance ------------------------------------------------


def test_sklearn_get_set_params_and_clone_round_trip():
    est = GaussianDummy(sigma=0.3)
    assert est.get_params()["sigma"] == 0.3
    copy = sklearn_base.clone(est)
    assert copy.get_params() == est.get_params()
    assert not hasattr(copy, "offset_")
    est.set_params(sigma=0.7)
    assert est.sigma == 0.7


def test_point_estimates_returns_every_definition_at_once(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    estimates = est.point_estimates(est.predict_proba(X))
    assert set(estimates) == set(lazy.POINT_ESTIMATORS)


def test_evaluate_returns_a_one_row_metric_table(data):
    X, y = data
    table = GaussianDummy().fit(X, y).evaluate(X, y)
    assert len(table) == 1
    assert table["model"].iloc[0] == "GaussianDummy"
    assert table["point_estimate"].iloc[0] == "z_peak"
    assert {"bias", "sigma_iqr", "cde_loss", "pit_ks"} <= set(table.columns)


# -- native distributions, quantiles and native grids ------------------------


class _WithNativeGrid(GaussianDummy):
    """The stand-in, plus a native grid fixed at fit."""

    def _fit(self, X, y):
        super()._fit(X, y)
        self.native_grid_ = lazy.Grid.linear(
            0.0, 3.0, 30, normalization="histogram"
        )


def test_the_default_distribution_is_the_density_on_the_default_grid(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    dist = est.predict_distribution(X)
    assert isinstance(dist, lazy.distributions.HistogramDistribution)
    np.testing.assert_array_equal(dist.bins, lazy.DC1_GRID.edges)
    masses = est.predict_proba(X) * lazy.DC1_GRID.widths
    np.testing.assert_allclose(
        dist.probabilities,
        masses / masses.sum(axis=1, keepdims=True),
        atol=1e-12,
    )


def test_quantiles_agree_with_the_grid_median(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    median = est.predict_quantiles(X, [0.5])[:, 0]
    np.testing.assert_allclose(
        median,
        est.predict(X, method="z_median"),
        atol=lazy.DC1_GRID.widths[0],
    )
    low, mid, high = est.predict_quantiles(X).T
    assert (low <= mid).all() and (mid <= high).all()


def test_the_native_grid_is_requested_by_name(data):
    X, y = data
    est = _WithNativeGrid().fit(X, y)
    assert est.predict_proba(X, "native").shape == (len(X), 30)
    assert est.grid_ == lazy.DC1_GRID  # still the default
    native = _WithNativeGrid(z_grid="native").fit(X, y)
    assert native.grid_ == native.native_grid
    assert native.predict_proba(X).shape == (len(X), 30)


def test_an_estimator_without_a_native_grid_says_so(data):
    X, y = data
    est = GaussianDummy().fit(X, y)
    with pytest.raises(AttributeError, match="no native grid"):
        est.predict_proba(X, "native")


def test_as_grid_refuses_the_native_sentinel():
    with pytest.raises(ValueError, match="native grid"):
        lazy.as_grid("native")


@pytest.mark.parametrize(
    "call",
    [
        lambda est, X, y: est.predict_proba(X),
        lambda est, X, y: est.predict_pdf(X),
        lambda est, X, y: est.predict_cdf(X),
        lambda est, X, y: est.predict(X),
        lambda est, X, y: est.predict_distribution(X),
        lambda est, X, y: est.predict_quantiles(X),
        lambda est, X, y: est.score(X, y),
        lambda est, X, y: est.evaluate(X, y),
        lambda est, X, y: est.point_estimates(np.ones((len(X), 200))),
        lambda est, X, y: est.predict_proba(X, "native"),
        lambda est, X, y: est.native_grid,
    ],
)
@pytest.mark.parametrize("z_grid", [None, "native"])
def test_every_method_of_an_unfitted_model_raises_not_fitted(
    data, call, z_grid
):
    X, y = data
    with pytest.raises(exceptions.NotFittedError):
        call(_WithNativeGrid(z_grid=z_grid), X, y)


def test_the_native_grid_error_tells_unfitted_from_absent(data):
    X, y = data
    with pytest.raises(exceptions.NotFittedError, match="not fitted yet"):
        GaussianDummy().native_grid  # noqa: B018 - the access is the test.
    with pytest.raises(AttributeError, match="has no native grid") as info:
        GaussianDummy().fit(X, y).native_grid  # noqa: B018 - as above.
    assert not isinstance(info.value, exceptions.NotFittedError)


class _CountingDummy(GaussianDummy):
    """The stand-in, counting how often inference runs."""

    def _predict_pdf(self, X, grid):
        self.calls_ = getattr(self, "calls_", 0) + 1
        return super()._predict_pdf(X, grid)


@pytest.mark.parametrize(
    "call",
    [
        lambda est, X, y: est.predict(X, method="z_mode"),
        lambda est, X, y: est.evaluate(X, y, method="z_mode"),
    ],
)
def test_an_unknown_method_is_refused_before_inference(data, call):
    X, y = data
    est = _CountingDummy().fit(X, y)
    with pytest.raises(ValueError, match="method must be one of"):
        call(est, X, y)
    assert getattr(est, "calls_", 0) == 0
