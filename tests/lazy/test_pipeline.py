# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LazyPipeline, on a stand-in model."""

import inspect

import numpy as np
import pytest
from sklearn import base as sklearn_base
from sklearn import linear_model
from sklearn import preprocessing
import standins

import lazy

GRID = lazy.Grid.linear(-6.0, 6.0, 40)

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
