# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The Tabularizer's adaptive reducer, trained through a pipeline's model."""

import pickle

import fakes
import numpy as np
import pandas as pd
import pytest
from sklearn import base as sklearn_base
from sklearn import exceptions
from sklearn import preprocessing
import standins

import lazy
from lazy import encoders
from lazy import interp

torch = pytest.importorskip("torch")

# The settings of a model with a differentiable forward pass.
DIFFERENTIABLE = {"transforms": ("none",), "outlier_threshold": None}


@pytest.fixture(name="model")
def _model(monkeypatch):
    """TabPFN's fake, unfitted, as a model with a differentiable pass."""
    fakes.install("tabpfn", monkeypatch)
    return lazy.LazyModel(
        "tabpfn", n_estimators=2, device="cpu", progress=False, **DIFFERENTIABLE
    )


@pytest.fixture(name="data")
def _data():
    """A context of 120 rows and 40 more, each with a series and a number.

    The fake predicts from the mean of a row's columns, and the target is
    a nonlinear function of two values of the series, which no principal
    component carries alone.
    """
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(160, 6))
    X = pd.DataFrame({"curve": list(curves), "number": rng.normal(size=160)})
    y = np.sin(2.0 * curves[:, 0]) + curves[:, 1] ** 2
    y = y + 0.05 * rng.normal(size=160)
    return X[:120], y[:120], X[120:], y[120:]


def _pipeline(model, **reducer):
    """The model behind the series' values, reduced to 3 columns."""
    tab = lazy.Tabularizer(
        {"curve": lazy.image.Pixels(size=1, n_channels=6)},
        n_columns=3,
        reducer=encoders.AdaptiveReducer(**reducer) if reducer else "pca",
    )
    return lazy.make_pipeline(tab, sklearn_base.clone(model))


def _as_images(X):
    """Each series as an image of one pixel and six channels."""
    return X.assign(curve=[curve.reshape(1, 1, 6) for curve in X["curve"]])


def test_step_zero_is_the_pca_reduction_then_the_model(model, data):
    X, y, X_test = _as_images(data[0]), data[1], _as_images(data[2])
    adaptive = _pipeline(model, n_steps=0).fit(X, y)
    fixed = _pipeline(model).fit(X, y)
    assert len(adaptive[0].history_) == 1 and adaptive[0].best_step_ == 0
    np.testing.assert_allclose(
        adaptive[0].transform(X_test), fixed[0].transform(X_test)
    )
    np.testing.assert_allclose(
        adaptive.predict_proba(X_test), fixed.predict_proba(X_test)
    )
    assert adaptive[0].get_feature_names_out()[0] == "embedding__z0"


def test_training_keeps_the_step_with_the_lowest_validation_nll(model, data):
    X, y = _as_images(data[0]), data[1]
    fitted = _pipeline(model, n_steps=30, learning_rate=1e-2).fit(X, y)
    nll = fitted[0].history_["validation_nll"]
    assert len(nll) == 31
    assert fitted[0].best_step_ == nll.idxmin()
    # The fake reads the mean of the new columns, so training finds the
    # target's nonlinear function of the values.
    assert nll[fitted[0].best_step_] < nll[0] - 0.3
    # The model was fitted again on what the trained reducer gives.
    np.testing.assert_allclose(
        fitted.predict(X[:5]),
        sklearn_base.clone(model)
        .fit(fitted[0].transform(X), y)
        .predict(fitted[0].transform(X[:5])),
    )


def test_a_training_that_only_worsens_keeps_the_start(model, data):
    X, y, X_test = _as_images(data[0]), data[1], _as_images(data[2])
    worse = _pipeline(model, n_steps=1, learning_rate=10.0).fit(X, y)
    nll = worse[0].history_["validation_nll"]
    assert nll[1] > nll[0]
    assert worse[0].best_step_ == 0
    np.testing.assert_allclose(
        worse[0].transform(X_test),
        _pipeline(model).fit(X, y)[0].transform(X_test),
    )


def test_the_same_seed_gives_the_same_fit_and_another_seed_another(model, data):
    X, y, X_test = _as_images(data[0]), data[1], _as_images(data[2])
    fits = [
        _pipeline(model, n_steps=3, random_state=seed).fit(X, y)[0]
        for seed in (0, 0, 1)
    ]
    np.testing.assert_array_equal(
        fits[0].transform(X_test), fits[1].transform(X_test)
    )
    assert not np.allclose(fits[0].transform(X_test), fits[2].transform(X_test))


def test_a_module_is_trained_as_the_reducer_and_the_given_one_is_not(
    model, data
):
    X, y, X_test = _as_images(data[0]), data[1], _as_images(data[2])
    torch.manual_seed(0)
    module = torch.nn.Linear(6, 2).double()
    before = module.weight.detach().clone()
    tab = lazy.Tabularizer(
        {"curve": lazy.image.Pixels(size=1, n_channels=6)},
        n_columns=None,
        reducer=encoders.AdaptiveReducer(
            module, n_steps=10, learning_rate=1e-2
        ),
    )
    fitted = lazy.make_pipeline(tab, model).fit(X, y)
    # The module given is left as it is; a copy of it is trained.
    assert torch.equal(module.weight, before)
    assert fitted[0].network_ is not module
    out = fitted[0].transform(X_test)
    assert list(out.columns[:2]) == ["embedding__z0", "embedding__z1"]
    assert (
        fitted[0].history_["train_nll"].iloc[-1]
        < (fitted[0].history_["train_nll"].iloc[0])
    )
    with pytest.raises(ValueError, match="gives 2 columns, more than n_col"):
        lazy.Tabularizer(
            {"curve": lazy.image.Pixels(size=1, n_channels=6)},
            n_columns=1,
            reducer=module,
        ).fit(X)


def test_a_gradient_method_passes_the_trained_reducer(model, data):
    X, y, X_test, y_test = data
    X, X_test = _as_images(X), _as_images(X_test)
    fitted = _pipeline(model, n_steps=2).fit(X, y)
    explanation = interp.explain(
        fitted, X_test[:4], y_test[:4], X, y, method="gradient"
    )
    assert explanation.feature_names == ("curve", "number")
    assert explanation.item_values["curve"].shape == (3, 4, 1, 1, 6)


def test_later_steps_are_trained_through_and_fitted_again(model, data):
    X, y = _as_images(data[0]), data[1]
    tab = _pipeline(model, n_steps=5, learning_rate=1e-2)[0]
    fitted = lazy.make_pipeline(tab, preprocessing.StandardScaler(), model).fit(
        X, y
    )
    # The scaler is fitted on the trained reducer's columns.
    np.testing.assert_allclose(
        fitted[1].mean_, fitted[0].transform(X).mean().to_numpy()
    )
    blocked = lazy.make_pipeline(
        sklearn_base.clone(tab),
        preprocessing.KBinsDiscretizer(3, encode="ordinal"),
        model,
    )
    with pytest.raises(ValueError, match="KBinsDiscretizer has no differenti"):
        blocked.fit(X, y)


def test_a_failed_training_leaves_the_pipeline_unfitted(monkeypatch, data):
    X, y = _as_images(data[0]), data[1]
    fakes.install("tabpfn", monkeypatch)
    # The default recipe has no differentiable pass.
    model = lazy.LazyModel(
        "tabpfn", n_estimators=2, device="cpu", progress=False
    )
    pipe = _pipeline(model, n_steps=1)
    with pytest.raises(ValueError, match="an adaptive reducer"):
        pipe.fit(X, y)
    with pytest.raises(exceptions.NotFittedError):
        pipe.predict(X)
    with pytest.raises(exceptions.NotFittedError):
        pipe[0].transform(X)


def test_a_module_of_the_wrong_width_leaves_the_tabularizer_unfitted(data):
    X = _as_images(data[0])
    tab = lazy.Tabularizer(
        {"curve": lazy.image.Pixels(size=1, n_channels=6)},
        n_columns=None,
        reducer=torch.nn.Linear(5, 3).double(),
    )
    with pytest.raises(RuntimeError):
        tab.fit(X)
    with pytest.raises(exceptions.NotFittedError):
        tab.transform(X)


def test_a_refit_at_a_stage_trains_the_reducer_through_the_model(model, data):
    X, y, X_test = _as_images(data[0]), data[1], _as_images(data[2])
    pipe = _pipeline(model, n_steps=30, learning_rate=1e-2)
    fitted = sklearn_base.clone(pipe).fit(X, y)
    assert fitted[0].best_step_ > 0
    _, stage, context = interp._staged(pipe, "tabularizer", X_test, X, y, True)
    np.testing.assert_allclose(context, fitted[0].transform(X))
    np.testing.assert_allclose(stage, fitted[0].transform(X_test))


def test_a_model_without_a_differentiable_pass_is_refused(data):
    X, y = _as_images(data[0]), data[1]
    standin = standins.HistogramStandIn(
        n_estimators=2, progress=False, transforms=("none",)
    )
    with pytest.raises(ValueError, match="no differentiable forward pass"):
        _pipeline(standin, n_steps=1).fit(X, y)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"hidden": 0}, "hidden == 0, must be >= 1"),
        ({"n_steps": -1}, "n_steps == -1, must be >= 0"),
        ({"learning_rate": 0.0}, "learning_rate == 0.0, must be > 0"),
        ({"validation_fraction": 1.0}, "validation_fraction == 1.0, must"),
        (
            {"validation_fraction": 0.6, "query_fraction": 0.5},
            "must be below 1, to leave support rows",
        ),
        ({"random_state": -1}, "random_state == -1, must be >= 0"),
        ({"network": "cnn"}, "network must be 'mlp' or a torch.nn.Module"),
    ],
)
def test_a_bad_setting_is_refused_by_name(model, data, params, message):
    X, y = _as_images(data[0]), data[1]
    with pytest.raises(ValueError, match=message):
        _pipeline(model, **params).fit(X, y)


def test_a_bad_reducer_and_a_missing_number_are_refused(model, data):
    X, y = _as_images(data[0]), data[1]
    columns = {"curve": lazy.image.Pixels(size=1, n_channels=6)}
    with pytest.raises(ValueError, match="reducer must be 'pca', 'mlp'"):
        lazy.Tabularizer(columns, reducer="svd").fit(X)
    gappy = X.copy()
    gappy.loc[3, "number"] = np.nan
    with pytest.raises(ValueError, match="without missing or infinite"):
        _pipeline(model, n_steps=1).fit(gappy, y)


def test_outside_a_pipeline_the_mlp_stays_pca_and_pickles(model, data):
    X = _as_images(data[0])
    tab = _pipeline(model, n_steps=3)[0].fit(X)
    assert tab.trains_through_model and not hasattr(tab, "history_")
    np.testing.assert_allclose(
        tab.transform(X), _pipeline(model)[0].fit(X).transform(X)
    )
    again = pickle.loads(pickle.dumps(tab))
    pd.testing.assert_frame_equal(again.transform(X), tab.transform(X))


@pytest.mark.checkpoint
def test_tabpfn_behind_a_trained_reducer_scores_better_than_behind_pca():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(500, 40))
    direction = rng.normal(size=40)
    projection = values @ (direction / np.linalg.norm(direction))
    y = projection + np.sin(2.0 * projection) + 0.1 * rng.normal(size=500)
    X = pd.DataFrame({"item": [row.reshape(1, 1, 40) for row in values]})
    model = lazy.LazyModel(
        "tabpfn",
        version="v3.5-fast",
        n_estimators=1,
        device="cpu",
        progress=False,
        **DIFFERENTIABLE,
    )

    def held_out_nll(n_steps):
        reducer = encoders.AdaptiveReducer(
            hidden=16, n_steps=n_steps, learning_rate=3e-2, query_fraction=0.3
        )
        tab = lazy.Tabularizer(
            {"item": lazy.image.Pixels(size=1, n_channels=40)},
            n_columns=4,
            reducer=reducer,
        )
        fitted = lazy.make_pipeline(tab, model).fit(X[:300], y[:300])
        density = fitted.predict_distribution(X[300:]).pdf(y[300:])
        return -np.log(np.diag(density)).mean()

    pca, trained = held_out_nll(0), held_out_nll(30)
    # The target follows one direction of 40 that no principal component of
    # isotropic columns follows, so the trained network finds what PCA
    # misses.
    assert trained < pca - 0.5
