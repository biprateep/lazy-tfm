# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""lazy.interp: the Shapley arithmetic on known games, and explain itself."""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn import exceptions
import standins

import lazy
from lazy import interp


def _game(n_features, seed, pairwise_only):
    """A value function with known values, and its table by bit mask.

    The values have shape (2**p, 2, 3): two statistics of three rows.
    """
    rng = np.random.default_rng(seed)
    masks = np.arange(1 << n_features)
    members = (masks[:, None] >> np.arange(n_features)) & 1 == 1
    if pairwise_only:
        single = rng.normal(size=(n_features, 2, 3))
        pair = np.triu(rng.normal(size=(n_features, n_features)), 1)
        table = np.einsum("mj,jsn->msn", members, single)
        table += np.einsum("mj,mk,jk->m", members, members, pair)[:, None, None]
    else:
        table = rng.normal(size=(1 << n_features, 2, 3))

    def value_function(coalitions):
        return table[coalitions @ (1 << np.arange(n_features))]

    return value_function, table


def _shapley_by_definition(table, n_features):
    """The mean gain of each feature over every order of the features."""
    orders = list(itertools.permutations(range(n_features)))
    values = np.zeros((*table.shape[1:], n_features))
    for order in orders:
        mask = 0
        for j in order:
            values[..., j] += table[mask | (1 << j)] - table[mask]
            mask |= 1 << j
    return values / len(orders)


def test_exact_shapley_values_are_the_mean_gain_over_every_order():
    value_function, table = _game(4, seed=0, pairwise_only=False)
    values, full, empty = interp._exact_shapley(value_function, 4)
    np.testing.assert_allclose(
        values, _shapley_by_definition(table, 4), atol=1e-12
    )
    np.testing.assert_array_equal(full, table[-1])
    np.testing.assert_array_equal(empty, table[0])


def test_antithetic_pairs_are_exact_when_features_interact_in_pairs():
    value_function, table = _game(5, seed=1, pairwise_only=True)
    values, full, empty = interp._sampled_shapley(
        value_function, 5, 4, np.random.default_rng(0)
    )
    np.testing.assert_allclose(
        values, _shapley_by_definition(table, 5), atol=1e-12
    )
    np.testing.assert_array_equal(full, table[-1])
    np.testing.assert_array_equal(empty, table[0])


@pytest.fixture(name="data")
def _data():
    """A target that follows ``signal``; ``noise`` and ``zero`` say nothing.

    Returns the context features and targets, then those of the 40 rows to
    explain.
    """
    rng = np.random.default_rng(5)
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=240),
            "noise": rng.normal(size=240),
            "zero": np.zeros(240),
        }
    )
    y = 3.0 * X["signal"].to_numpy() + rng.normal(0.0, 0.1, 240)
    return X[:200], y[:200], X[200:], y[200:]


@pytest.fixture(name="model")
def _model():
    return standins.HistogramStandIn(n_estimators=2, progress=False)


@pytest.mark.parametrize(
    "options",
    [
        {"method": "shapley"},
        {"method": "shapley", "n_permutations": 4},
        {"method": "loco"},
        {"method": "permutation"},
    ],
    ids=["exact", "sampled", "loco", "permutation"],
)
def test_the_feature_that_sets_the_target_is_the_most_important(
    data, model, options
):
    X_context, y_context, X, y = data
    model.fit(X_context, y_context)
    explanation = interp.explain(model, X, y, X_context, y_context, **options)
    importance = explanation.importance()
    assert list(importance.index) == ["signal", "noise", "zero"]
    assert list(importance.columns) == list(interp.STATISTICS)
    assert (importance.idxmax() == "signal").all()
    # The stand-in's CRPS without the signal is the spread of the target,
    # about 3, so the signal gains more than 1 by any method.
    assert importance.loc["signal", "crps"] > 1.0
    assert explanation.table("nll").shape == (40, 3)


def test_shapley_values_sum_to_what_the_features_add(data, model):
    X_context, y_context, X, y = data
    explanation = interp.explain(model, X, y, X_context, y_context)
    assert explanation.method == "shapley"
    np.testing.assert_allclose(
        explanation.values.sum(axis=-1),
        explanation.full_values - explanation.base_values,
        atol=1e-10,
    )


def test_loco_is_the_loss_of_a_refit_without_the_feature(data, model):
    X_context, y_context, X, y = data
    grid = lazy.Grid.linear(-12.0, 12.0, 300)
    explanation = interp.explain(
        model, X, y, X_context, y_context, method="loco", y_grid=grid
    )
    kept = ["noise", "zero"]
    pdfs = model.fit(X_context[kept], y_context).predict_proba(X[kept], grid)
    without = -lazy.metrics.per_object_crps(y, grid, pdfs)
    np.testing.assert_allclose(
        explanation.table("crps")["signal"],
        explanation.full_values[0] - without,
        atol=1e-12,
    )


def test_permuting_a_constant_feature_changes_nothing(data, model):
    X_context, y_context, X, y = data
    model.fit(X_context, y_context)
    before = X.copy()
    explanation = interp.explain(model, X, y, n_repeats=2)
    assert explanation.method == "permutation"
    assert explanation.base_values is None
    np.testing.assert_array_equal(explanation.values[..., 2], 0.0)
    pd.testing.assert_frame_equal(X, before)


def test_a_group_of_columns_is_one_feature(data, model):
    X_context, y_context, X, y = data
    grouped = interp.explain(
        model,
        X,
        y,
        X_context,
        y_context,
        groups={"rest": ["zero", "noise"]},
    )
    assert grouped.feature_names == ("signal", "rest")
    assert grouped.values.shape == (3, 40, 2)
    singles = interp.explain(
        model, X, y, X_context, y_context, groups={"s": ["signal"]}
    )
    plain = interp.explain(model, X, y, X_context, y_context)
    np.testing.assert_array_equal(singles.values, plain.values)


def test_the_same_seed_gives_the_same_values(data, model):
    X_context, y_context, X, y = data
    first, second = (
        interp.explain(
            model, X, y, X_context, y_context, n_permutations=2, random_state=7
        )
        for _ in range(2)
    )
    np.testing.assert_array_equal(first.values, second.values)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"method": "shap"}, "method must be"),
        ({"method": "loco", "X_context": None}, "given together"),
        ({"method": "loco", "X_context": None, "y_context": None}, "needs X_"),
        ({"n_permutations": 3}, "must be even"),
        ({"groups": {"g": ["missing"]}}, "not a column of X"),
        ({"groups": {"a": ["zero"], "b": ["zero"]}}, "in two groups"),
        ({"groups": {"noise": ["zero"]}}, "name of a column"),
        ({"n_repeats": 3}, "n_repeats does not apply"),
        ({"method": "loco", "n_permutations": 2}, "does not apply"),
    ],
)
def test_a_bad_argument_is_refused_by_name(data, model, options, message):
    X_context, y_context, X, y = data
    arguments = {"X_context": X_context, "y_context": y_context} | options
    with pytest.raises(ValueError, match=message):
        interp.explain(model, X, y, **arguments)


def test_a_context_with_other_columns_is_refused(data, model):
    X_context, y_context, X, y = data
    with pytest.raises(ValueError, match="must have the columns of X"):
        interp.explain(model, X, y, X_context[["signal", "zero"]], y_context)


def test_permutation_needs_a_fitted_model(data, model):
    _, _, X, y = data
    with pytest.raises(exceptions.NotFittedError):
        interp.explain(model, X, y)


def test_an_unknown_statistic_is_refused():
    explanation = interp.Explanation(
        "loco", ("crps",), ("a",), np.zeros((1, 2, 1)), np.zeros((1, 2)), None
    )
    with pytest.raises(ValueError, match="statistic must be one of"):
        explanation.table("nll")


def test_unnamed_rows_reach_a_model_fitted_on_unnamed_rows(data, model):
    X_context, y_context, X, y = data
    named = interp.explain(model, X, y, X_context, y_context, method="loco")
    arrays = interp.explain(
        model,
        X.to_numpy(),
        y,
        X_context.to_numpy(),
        y_context,
        method="loco",
    )
    assert arrays.feature_names == ("x0", "x1", "x2")
    np.testing.assert_array_equal(arrays.values, named.values)
    model.fit(X_context.to_numpy(), y_context)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        interp.explain(model, X.to_numpy(), y, n_repeats=1)


def test_permutation_scores_a_row_the_same_whatever_rows_come_with_it(
    data, model
):
    X_context, y_context, X, y = data
    model.fit(X_context, y_context)
    few = interp.explain(model, X[:3], y[:3])
    many = interp.explain(model, X, y)
    np.testing.assert_allclose(
        few.full_values, many.full_values[:, :3], atol=1e-12
    )


def test_half_a_context_and_no_rows_are_refused(data, model):
    X_context, y_context, X, y = data
    with pytest.raises(ValueError, match="given together"):
        interp.explain(model, X, y, y_context=y_context)
    with pytest.raises(ValueError, match="X has no rows"):
        interp.explain(model, X[:0], y[:0], X_context, y_context)
    with pytest.raises(ValueError, match="X_context has 200 rows but y_con"):
        interp.explain(model, X, y, X_context, y_context[:-1])
