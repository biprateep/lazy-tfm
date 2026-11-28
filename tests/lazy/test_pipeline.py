# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LazyPipeline, on a stand-in model."""

import inspect
import warnings

import numpy as np
import pytest
import sklearn
from sklearn import base as sklearn_base
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing
import standins

import lazy

GRID = lazy.Grid.linear(-6.0, 6.0, 40, normalization="trapezoid")

# The methods that return arrays or a table, with the arguments after X, by
# position.
FORWARDED = {
    "predict": lambda y: ("median", GRID),
    "predict_proba": lambda y: (GRID,),
    "predict_pdf": lambda y: (GRID,),
    "predict_cdf": lambda y: (GRID,),
    "predict_quantiles": lambda y: ([0.1, 0.5],),
    "predict_interval": lambda y: (0.8,),
    "predict_pit": lambda y: (y,),
    "evaluate": lambda y: (y, "median", GRID),
}


@pytest.fixture(name="regression")
def _regression():
    """Features, shape (60, 3), and a target that follows the first."""
    rng = np.random.default_rng(3)
    X = 5.0 + 3.0 * rng.normal(size=(60, 3))
    return X, 2.0 * X[:, 0] + rng.normal(0.0, 0.1, 60)


@pytest.fixture(name="fitted")
def _fitted(regression):
    """A fitted pipeline, and its model fitted alone on the scaled features."""
    X, y = regression
    model = standins.HistogramStandIn(n_estimators=2, progress=False)
    piped = lazy.make_pipeline(
        "passthrough", preprocessing.StandardScaler(), model
    )
    piped = sklearn_base.clone(piped).fit(X, y)
    scaled = preprocessing.StandardScaler().fit_transform(X)
    return piped, model.fit(scaled, y), scaled


class _Unreachable(sklearn_base.TransformerMixin, sklearn_base.BaseEstimator):
    """A step that fits, and fails the test if anything transforms with it."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise AssertionError("a step ran before the arguments were checked")


@pytest.mark.parametrize(
    ("method", "argument"),
    [("predict_quantiles", [0.5, 1.5]), ("predict_interval", 1.5)],
)
def test_a_bad_level_is_refused_before_any_step_runs(
    regression, method, argument
):
    X, y = regression
    piped = lazy.make_pipeline(
        _Unreachable(), standins.HistogramStandIn(n_estimators=2)
    )
    piped.steps[-1][1].fit(X, y)
    piped.steps[0][1].fit(X)
    with pytest.raises(ValueError, match="must"):
        getattr(piped, method)(X, argument)


@pytest.mark.parametrize("method", [*FORWARDED, "score"])
def test_an_empty_x_is_refused_before_any_step_runs(regression, method):
    X, y = regression
    piped = lazy.make_pipeline(
        _Unreachable(), standins.HistogramStandIn(n_estimators=2)
    )
    piped.steps[-1][1].fit(X, y)
    piped.steps[0][1].fit(X)
    arguments = FORWARDED.get(method, lambda y: (y,))(y[:0])
    with pytest.raises(ValueError, match="0 sample"):
        getattr(piped, method)(X[:0], *arguments)


@pytest.mark.parametrize("method", FORWARDED)
def test_a_density_method_answers_as_the_model_on_transformed_features(
    regression, fitted, method
):
    X, y = regression
    piped, model, scaled = fitted
    args = FORWARDED[method](y)
    np.testing.assert_array_equal(
        np.asarray(getattr(piped, method)(X, *args)),
        np.asarray(getattr(model, method)(scaled, *args)),
    )


def test_the_native_distribution_is_the_models_on_transformed_features(
    regression, fitted
):
    X, _ = regression
    piped, model, scaled = fitted
    np.testing.assert_array_equal(
        piped.predict_distribution(X).ppf([0.3]),
        model.predict_distribution(scaled).ppf([0.3]),
    )


def test_score_uses_the_grid_it_is_given(regression, fitted):
    """scikit-learn's Pipeline.score drops keywords without routing."""
    X, y = regression
    piped, model, scaled = fitted
    assert piped.score(X, y, GRID) == model.score(scaled, y, GRID)
    assert piped.score(X, y) == model.score(scaled, y)
    assert piped.score(X, y, GRID) != piped.score(X, y)


def test_a_method_the_last_step_lacks_is_absent(regression, fitted):
    X, y = regression
    piped, _, _ = fitted
    plain = lazy.make_pipeline(linear_model.LinearRegression()).fit(X, y)
    assert not hasattr(plain, "predict_quantiles")
    assert not hasattr(piped[:-1], "predict")


def test_a_pipeline_can_end_in_a_pipeline(regression, fitted):
    X, y = regression
    piped, _, _ = fitted
    nested = lazy.make_pipeline("passthrough", sklearn_base.clone(piped))
    np.testing.assert_array_equal(
        nested.fit(X, y).predict_quantiles(X, [0.5]),
        piped.predict_quantiles(X, [0.5]),
    )


@pytest.mark.parametrize("method", [*FORWARDED, "score"])
def test_a_method_has_the_signature_of_the_models(method):
    """A change to the model's API must reach the pipeline."""
    ours = inspect.signature(getattr(lazy.LazyPipeline, method))
    theirs = inspect.signature(getattr(lazy.BaseDensityRegressor, method))
    assert [(p.name, p.kind, p.default) for p in ours.parameters.values()] == [
        (p.name, p.kind, p.default) for p in theirs.parameters.values()
    ]


def test_every_prediction_method_of_a_model_is_forwarded():
    predictions = {
        name
        for name in vars(lazy.BaseDensityRegressor)
        if name.startswith("predict") or name in ("score", "evaluate")
    }
    assert predictions == {*FORWARDED, "predict_distribution", "score"}


class _Weighted(sklearn_base.TransformerMixin, sklearn_base.BaseEstimator):
    """A transformer that takes sample weights, and records those of fit."""

    def fit(self, X, y=None, sample_weight=None):
        self.weights_ = sample_weight
        return self

    def transform(self, X, sample_weight=None):
        return X


# _Weighted's inherited fit_transform passes no metadata to transform.
@pytest.mark.filterwarnings("ignore:.*fit_transform. does not forward")
def test_a_search_routes_metadata_to_fit_and_scores_without_it(regression):
    """The methods that predict take no metadata, and the routing says so."""
    X, y = regression
    weights = np.arange(1.0, 61.0)
    with sklearn.config_context(enable_metadata_routing=True):
        step = _Weighted().set_fit_request(sample_weight=True)
        step = step.set_transform_request(sample_weight=True)
        piped = lazy.make_pipeline(
            step, standins.HistogramStandIn(n_estimators=2, progress=False)
        )
        result = model_selection.cross_validate(
            piped,
            X,
            y,
            cv=2,
            params={"sample_weight": weights},
            return_estimator=True,
            error_score="raise",
        )
    assert np.isfinite(result["test_score"]).all()
    fold = result["estimator"][0][0].weights_
    assert len(fold) == 30 and set(fold) <= set(weights)


@pytest.mark.parametrize("name", ["predict", "evaluate"])
def test_a_pipeline_warns_once_of_a_point_estimate_left_out(
    fitted, regression, name
):
    piped, _, _ = fitted
    X, y = regression
    args = (X, y) if name == "evaluate" else (X,)
    with pytest.warns(FutureWarning, match="lazy 0.8.0") as record:
        getattr(piped, name)(*args)
    assert len(record) == 1
    assert record[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(piped, name)(*args, method="mode")
