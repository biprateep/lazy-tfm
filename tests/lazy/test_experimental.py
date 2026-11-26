# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""lazy.experimental.ContextAdapter, on TabPFN's fake and its checkpoint."""

import fakes
import numpy as np
import pandas as pd
import pytest
from sklearn import base as sklearn_base
from sklearn import decomposition
from sklearn import preprocessing
import standins

import lazy
from lazy import experimental

# The settings of a model with a differentiable forward pass.
DIFFERENTIABLE = {"transforms": ("none",), "outlier_threshold": None}


@pytest.fixture(name="model")
def _model(monkeypatch):
    """TabPFN's fake, unfitted, as a model with a differentiable pass."""
    pytest.importorskip("torch")
    fakes.install("tabpfn", monkeypatch)
    return lazy.LazyModel(
        "tabpfn", n_estimators=2, device="cpu", progress=False, **DIFFERENTIABLE
    )


@pytest.fixture(name="data")
def _data():
    """A context of 120 rows and 6 columns, and 40 rows to predict.

    The fake predicts from the mean of a row's columns, and the target is
    a nonlinear function of two columns, which no principal component
    carries alone.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 6)), columns=list("abcdef"))
    y = np.sin(2.0 * X["a"]) + X["b"] ** 2 + 0.05 * rng.normal(size=160)
    return X[:120], y[:120], X[120:], y[120:]


def test_step_zero_is_pca_of_the_standardized_columns_then_the_model(
    model, data
):
    X, y, X_test, _ = data
    adapter = experimental.ContextAdapter(model, n_components=3, n_steps=0)
    adapter.fit(X, y)
    scaler = preprocessing.StandardScaler().fit(X)
    pca = decomposition.PCA(3, svd_solver="full").fit(scaler.transform(X))
    reference = sklearn_base.clone(model).fit(
        pca.transform(scaler.transform(X)), y
    )
    expected = reference.predict_proba(pca.transform(scaler.transform(X_test)))
    assert len(adapter.history_) == 1
    assert adapter.best_step_ == 0
    # The kept network is the float64 PCA; scikit-learn's PCA also
    # subtracts the standardized columns' mean, zero to rounding.
    np.testing.assert_allclose(
        adapter.predict_proba(X_test), expected, rtol=1e-9, atol=1e-12
    )


def test_training_keeps_the_step_with_the_lowest_validation_nll(model, data):
    X, y, _, _ = data
    adapter = experimental.ContextAdapter(
        model, n_components=3, n_steps=30, learning_rate=1e-2
    ).fit(X, y)
    nll = adapter.history_["validation_nll"]
    assert len(nll) == 31
    assert adapter.best_step_ == nll.idxmin()
    # The target follows sin(2a) + b**2, which the fake reads only through
    # the mean of the new columns, so training finds it (measured: from
    # 2.75 nats at step 0 to 1.68 at step 30).
    assert nll[adapter.best_step_] < nll[0] - 0.3


def test_a_training_that_only_worsens_keeps_pca(model, data):
    X, y, X_test, _ = data
    pca = experimental.ContextAdapter(model, n_components=3, n_steps=0)
    worse = sklearn_base.clone(pca).set_params(n_steps=1, learning_rate=10.0)
    worse.fit(X, y)
    # A step of 10 throws the network off (measured: validation NLL from
    # 2.75 to 3.39 nats; for seeds 1 and 2, from 3.39 to 4.57 and from
    # 2.99 to 3.69).
    nll = worse.history_["validation_nll"]
    assert nll[1] > nll[0]
    assert worse.best_step_ == 0
    np.testing.assert_array_equal(
        worse.transform(X_test), pca.fit(X, y).transform(X_test)
    )


def test_the_network_alone_receives_finite_gradients_at_step_one(
    model, data, monkeypatch
):
    torch = pytest.importorskip("torch")
    gradients = []
    step = torch.optim.Adam.step

    def recording_step(self, *args, **kwargs):
        if not gradients:
            (group,) = self.param_groups
            gradients.extend(p.grad.clone() for p in group["params"])
        return step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", recording_step)
    X, y, _, _ = data
    experimental.ContextAdapter(model, n_components=3, hidden=5, n_steps=1).fit(
        X, y
    )
    components, hidden_weight, hidden_bias, output_weight, output_bias = (
        gradients
    )
    assert [tuple(g.shape) for g in gradients] == [
        (3, 6),
        (5, 3),
        (5,),
        (3, 5),
        (3,),
    ]
    assert all(bool(g.isfinite().all()) for g in gradients)
    for gradient in (components, output_weight, output_bias):
        assert bool(gradient.abs().sum() > 0)
    # The output layer starts at zero, so nothing reaches the hidden layer
    # through it until it moves.
    assert not hidden_weight.any() and not hidden_bias.any()


def test_the_same_seed_gives_the_same_fit_and_another_seed_another(model, data):
    X, y, X_test, _ = data
    adapter = experimental.ContextAdapter(model, n_components=3, n_steps=3)
    first = sklearn_base.clone(adapter).fit(X, y)
    second = sklearn_base.clone(adapter).fit(X, y)
    other = sklearn_base.clone(adapter).set_params(random_state=1).fit(X, y)
    pd.testing.assert_frame_equal(first.history_, second.history_)
    np.testing.assert_array_equal(
        first.predict_proba(X_test), second.predict_proba(X_test)
    )
    assert not first.history_.equals(other.history_)


def test_unmapped_columns_pass_through_unchanged(model, data):
    X, y, X_test, _ = data
    adapter = experimental.ContextAdapter(
        model, columns=["b", "d", "e"], n_components=2, n_steps=2
    ).fit(X, y)
    np.testing.assert_array_equal(adapter.is_mapped_, [0, 1, 0, 1, 1, 0])
    features = adapter.transform(X_test)
    assert features.shape == (40, 5)
    np.testing.assert_array_equal(features[:, 2:], X_test[["a", "c", "f"]])


def test_a_pipeline_encodes_items_and_adapts_them(model):
    rng = np.random.default_rng(1)
    table = pd.DataFrame(
        {
            "number": rng.normal(size=60),
            "curve": [rng.normal(size=rng.integers(5, 40)) for _ in range(60)],
        }
    )
    y = table["number"].to_numpy() + 0.1 * rng.normal(size=60)
    pipe = lazy.make_pipeline(
        lazy.Tabularizer({"curve": "time_statistics"}, n_components=None),
        experimental.ContextAdapter(model, n_components=4, n_steps=2),
    ).fit(table[:40], y[:40])
    densities = pipe.predict_proba(table[40:])
    assert densities.shape == (20, pipe[-1].grid_.n_bins)
    assert np.isfinite(densities).all()
    assert pipe.predict_quantiles(table[40:]).shape == (20, 3)


def test_a_model_without_a_differentiable_pass_is_refused(monkeypatch, data):
    pytest.importorskip("torch")
    fakes.install("tabpfn", monkeypatch)
    X, y, _, _ = data
    model = lazy.LazyModel("tabpfn", n_estimators=2, progress=False)
    with pytest.raises(
        ValueError, match=r"ContextAdapter: .*transforms=\('none',\)"
    ):
        experimental.ContextAdapter(model, n_components=3, n_steps=0).fit(X, y)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"n_components": 0}, "n_components == 0"),
        ({"n_components": 7}, "n_components must be at most 6"),
        ({"hidden": 0}, "hidden == 0"),
        ({"n_steps": -1}, "n_steps == -1"),
        ({"learning_rate": 0.0}, "learning_rate == 0.0"),
        ({"validation_fraction": 1.0}, "validation_fraction == 1.0"),
        ({"query_fraction": 0.0}, "query_fraction == 0.0"),
        (
            {"validation_fraction": 0.5, "query_fraction": 0.5},
            "validation_fraction \\+ query_fraction must be below 1",
        ),
        ({"query_fraction": 0.001, "n_components": 3}, "0 training queries"),
        ({"columns": ["z"]}, "columns must be names .* 'z' is neither"),
        ({"columns": [6]}, "columns must be names .* 6 is neither"),
        ({"columns": []}, "columns must name at least one column"),
        ({"columns": "abc"}, "columns must be a list"),
    ],
)
def test_a_bad_parameter_is_refused_by_name(data, params, message):
    X, y, _, _ = data
    model = standins.HistogramStandIn()
    adapter = experimental.ContextAdapter(model, **params)
    with pytest.raises(ValueError, match=message):
        adapter.fit(X, y)


def test_an_infinite_validation_nll_at_every_step_warns_and_keeps_pca(
    model, data, monkeypatch
):
    torch = pytest.importorskip("torch")

    def infinite(*args):
        del args  # Unused: every NLL is infinite.
        return torch.tensor(np.inf, requires_grad=True) * 1.0

    monkeypatch.setattr(experimental, "_nll", infinite)
    X, y, _, _ = data
    adapter = experimental.ContextAdapter(model, n_components=3, n_steps=2)
    with pytest.warns(lazy.LazyWarning, match="no step had a finite"):
        adapter.fit(X, y)
    assert adapter.best_step_ == 0


def test_a_generator_as_random_state_is_refused(data):
    """A shared generator would make refits differ, and change it."""
    X, y, _, _ = data
    adapter = experimental.ContextAdapter(
        standins.HistogramStandIn(), random_state=np.random.default_rng(0)
    )
    with pytest.raises(TypeError, match="random_state"):
        adapter.fit(X, y)


def test_columns_by_name_are_refused_among_repeated_names(data):
    X, y, _, _ = data
    X = X.set_axis(list("aabcde"), axis=1)
    adapter = experimental.ContextAdapter(
        standins.HistogramStandIn(), columns=["a"]
    )
    with pytest.raises(ValueError, match="column names to be unique"):
        adapter.fit(X, y)


def test_a_context_with_a_missing_value_is_refused(data):
    X, y, _, _ = data
    X = X.copy()
    X.iloc[0, 0] = np.nan
    adapter = experimental.ContextAdapter(standins.HistogramStandIn())
    with pytest.raises(ValueError, match="without missing or infinite"):
        adapter.fit(X, y)


@pytest.mark.checkpoint
def test_tabpfn_behind_a_trained_adapter_scores_better_than_behind_pca():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 40))
    direction = rng.normal(size=40)
    projection = X @ (direction / np.linalg.norm(direction))
    y = projection + np.sin(2.0 * projection) + 0.1 * rng.normal(size=500)
    model = lazy.LazyModel(
        "tabpfn",
        version="v3.5-fast",
        n_estimators=1,
        device="cpu",
        progress=False,
        **DIFFERENTIABLE,
    )

    def held_out_nll(n_steps):
        adapter = experimental.ContextAdapter(
            model,
            n_components=4,
            hidden=16,
            n_steps=n_steps,
            learning_rate=3e-2,
            query_fraction=0.3,
        ).fit(X[:300], y[:300])
        density = adapter.predict_distribution(X[300:]).pdf(y[300:])
        return -np.log(np.diag(density)).mean()

    pca, trained = held_out_nll(0), held_out_nll(30)
    # The target follows one direction of 40 that no principal component of
    # isotropic columns follows, so the trained network finds what PCA
    # misses (measured on 200 held-out rows: 1.38 nats behind PCA, -0.18
    # behind the network; -0.19 against 1.36 for the data of seed 2).
    assert trained < pca - 0.5
