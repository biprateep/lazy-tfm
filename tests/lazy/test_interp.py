# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""lazy.interp: the Shapley arithmetic on known games, and explain itself."""

import itertools
import warnings

import fakes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn import base as sklearn_base
from sklearn import dummy
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
        ({"method": "gradient", "n_steps": 4}, "n_steps does not apply"),
        ({"baseline": np.zeros(3)}, "baseline does not apply"),
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
    for show in (explanation.table, explanation.plot):
        with pytest.raises(ValueError, match="statistic must be one of"):
            show("nll")


def _bars(ax):
    """The tick labels and bar lengths of a bar chart, from bottom to top."""
    labels = [label.get_text() for label in ax.get_yticklabels()]
    return labels, [bar.get_width() for bar in ax.patches]


def test_plot_draws_the_importance_with_the_most_important_on_top():
    # Rows (1, -3) and (-2, 0) and (0.5, 0.5): mean |values| 2, 1, 0.5.
    values = np.array([[[1.0, -2.0, 0.5], [-3.0, 0.0, 0.5]]])
    explanation = interp.Explanation(
        "loco", ("mean",), ("a", "b", "c"), values, np.zeros((1, 2)), None
    )
    ax = explanation.plot("mean")
    plt.close(ax.figure)
    assert _bars(ax) == (["c", "b", "a"], [0.5, 1.0, 2.0])
    assert ax.get_xlabel() == "|mean| (target units)"


def test_plot_draws_into_a_given_axes():
    explanation = interp.Explanation(
        "loco", ("crps",), ("a",), np.ones((1, 2, 1)), np.zeros((1, 2)), None
    )
    figure, ax = plt.subplots()
    plt.close(figure)
    assert explanation.plot(ax=ax) is ax
    assert _bars(ax) == (["a"], [1.0])


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


# -- the gradient methods ----------------------------------------------------


@pytest.fixture(name="differentiable")
def _differentiable(monkeypatch):
    """TabPFN's fake, fitted with a differentiable pass, and 20 rows.

    A batch holds 7 query points, so that "gradient" spans three batches.
    Returns the model, the context's features and targets, then those of
    the rows.
    """
    pytest.importorskip("torch")
    monkeypatch.setattr(interp, "_QUERY_BUDGET", 7)
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + 0.1 * rng.normal(size=60)
    model = fakes.install("tabpfn", monkeypatch)(
        n_estimators=2,
        device="cpu",
        progress=False,
        transforms=("none",),
        outlier_threshold=None,
    ).fit(X[:40], y[:40])
    return model, X[:40], y[:40], X[40:], y[40:]


def test_integrated_gradients_sum_to_what_the_features_add(
    differentiable, monkeypatch
):
    model, X_context, y_context, X, y = differentiable
    # A batch of 24 rows at 32 steps, and of 3 rows at 256, the last of 2: a
    # batch's points mixed up between its rows would break the sum.
    monkeypatch.setattr(interp, "_QUERY_BUDGET", 768)
    errors = []
    for n_steps in (32, 256):
        explanation = interp.explain(
            model,
            X,
            y,
            X_context,
            y_context,
            method="integrated_gradients",
            n_steps=n_steps,
        )
        gap = explanation.full_values - explanation.base_values
        errors.append(np.abs(explanation.values.sum(axis=-1) - gap).max())
    # A right Riemann sum's error falls as 1 / n_steps: by 8 here, while
    # the fake's float32 logits add about 1e-6.
    assert errors[1] < errors[0] / 4
    assert errors[1] < 0.02 * np.abs(gap).max()


def test_gradient_times_input_is_the_derivative_times_the_offset(
    differentiable,
):
    model, X_context, y_context, X, y = differentiable
    grid = lazy.Grid.linear(-4.0, 4.0, 200)
    baseline = np.array([0.5, -1.0, 2.0])
    explanation = interp.explain(
        model,
        X,
        y,
        X_context,
        y_context,
        method="gradient",
        baseline=baseline,
        y_grid=grid,
    )
    # Central differences of the statistics as predict gives them.
    step = 1e-3
    expected = np.empty_like(explanation.values)
    for j, column in enumerate(X.columns):
        moved = [
            X.assign(**{column: X[column] + sign * step}) for sign in (1, -1)
        ]
        up, down = (interp._statistics(model, m, y, True, grid) for m in moved)
        expected[..., j] = (
            (up - down) / (2 * step) * (X[column] - baseline[j]).to_numpy()
        )
    # The fake's float32 logits make the differences good to 1e-3.
    np.testing.assert_allclose(explanation.values, expected, atol=2e-3)
    np.testing.assert_allclose(
        explanation.full_values,
        interp._statistics(model, X, y, True, grid),
        rtol=1e-5,
        atol=1e-6,
    )
    assert explanation.base_values is None


def test_a_context_other_than_the_fits_is_refused(differentiable):
    model, X_context, y_context, X, y = differentiable
    with pytest.raises(ValueError, match="the context the model was fitted"):
        interp.explain(
            model, X, y, X_context, y_context + 1.0, method="gradient"
        )


def test_gradients_are_taken_inside_no_grad(differentiable):
    torch = pytest.importorskip("torch")
    model, X_context, y_context, X, y = differentiable
    options = {"method": "gradient", "y_grid": lazy.Grid.linear(-4, 4, 50)}
    plain = interp.explain(model, X, y, X_context, y_context, **options)
    with torch.no_grad():
        inside = interp.explain(model, X, y, X_context, y_context, **options)
    np.testing.assert_array_equal(plain.values, inside.values)


def test_a_lazy_model_explains_as_its_backend_does(differentiable):
    model, X_context, y_context, X, y = differentiable
    wrapped = lazy.LazyModel("tabpfn", **model.get_params())
    wrapped.fit(X_context, y_context)
    first, second = (
        interp.explain(m, X, y, X_context, y_context, method="gradient")
        for m in (model, wrapped)
    )
    np.testing.assert_array_equal(first.values, second.values)


def test_reordered_columns_keep_their_gradients(differentiable):
    model, X_context, y_context, X, y = differentiable
    order = ["c", "a", "b"]
    options = {"method": "integrated_gradients", "n_steps": 4}
    plain = interp.explain(model, X, y, X_context, y_context, **options)
    moved = interp.explain(
        model, X[order], y, X_context[order], y_context, **options
    )
    assert moved.feature_names == tuple(order)
    np.testing.assert_allclose(moved.values, plain.values[..., [2, 0, 1]])
    np.testing.assert_allclose(moved.base_values, plain.base_values)


def test_a_group_takes_the_sum_of_its_columns_gradients(differentiable):
    model, X_context, y_context, X, y = differentiable
    options = {"method": "gradient", "y_grid": lazy.Grid.linear(-4, 4, 50)}
    plain = interp.explain(model, X, y, X_context, y_context, **options)
    grouped = interp.explain(
        model, X, y, X_context, y_context, groups={"ac": ["a", "c"]}, **options
    )
    assert grouped.feature_names == ("ac", "b")
    np.testing.assert_allclose(
        grouped.values,
        np.stack([plain.values[..., [0, 2]].sum(-1), plain.values[..., 1]], -1),
    )


@pytest.mark.parametrize(
    ("model", "message", "needs_torch"),
    [
        (dummy.DummyRegressor(), "needs a lazy backend", False),
        (
            standins.HistogramStandIn(n_estimators=2, progress=False),
            r"'gradient': .*fit with transforms=\('none',\)",
            True,
        ),
        (
            standins.HistogramStandIn(
                n_estimators=2, progress=False, transforms=("none",)
            ),
            "'gradient': .*no differentiable forward pass",
            True,
        ),
    ],
    ids=["no-pass", "transformed", "no-forward"],
)
def test_a_model_without_a_differentiable_pass_is_refused(
    data, model, message, needs_torch
):
    X_context, y_context, X, y = data
    if needs_torch:  # The pass itself refuses.
        pytest.importorskip("torch")
    model.fit(X_context, y_context)
    with pytest.raises(ValueError, match=message):
        interp.explain(model, X, y, X_context, y_context, method="gradient")


@pytest.mark.checkpoint
def test_integrated_gradients_find_tabpfns_signal():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = 2.0 * X[:, 0] + 0.1 * rng.normal(size=120)
    model = lazy.TabPFNBarDistribution(
        n_estimators=2,
        device="cpu",
        progress=False,
        transforms=("none",),
        outlier_threshold=None,
    ).fit(X[:100], y[:100])
    explanation = interp.explain(
        model,
        X[100:],
        y[100:],
        X[:100],
        y[:100],
        method="integrated_gradients",
        n_steps=8,
    )
    importance = explanation.importance()
    assert np.isfinite(explanation.values).all()
    assert importance["mean"].idxmax() == "x0"
    # y = 2 x0 + noise of 0.1, so the mean moves with x0 alone: x0's
    # importance is near 2 and the others' near 0 (measured: 1.92 against
    # below 0.002), and a factor of 5 leaves room for a different seed.
    assert importance.loc["x0", "mean"] > 5 * importance["mean"][1:].max()


# -- the context rows ---------------------------------------------------------


def test_leaving_a_context_row_out_is_what_a_refit_without_it_loses(
    data, model
):
    X_context, y_context, X, y = data
    X_context, y_context = X_context[:30], y_context[:30]
    grid = lazy.Grid.linear(-12.0, 12.0, 300)
    explanation = interp.explain_context(
        model,
        X,
        y,
        X_context,
        y_context,
        method="loo",
        rows=[17, 4],
        y_grid=grid,
    )
    np.testing.assert_array_equal(explanation.context_rows, [17, 4])

    def crps_and_mean(keep):
        fitted = model.fit(X_context[keep], y_context[keep])
        pdfs = fitted.predict_proba(X, grid)
        crps = -lazy.metrics.per_object_crps(y, grid, pdfs)
        return np.stack([crps, fitted.predict_distribution(X).mean()])

    full = crps_and_mean(np.ones(30, dtype=bool))
    np.testing.assert_allclose(
        explanation.full_values[[0, 2]], full, atol=1e-12
    )
    for k, row in enumerate([17, 4]):
        expected = full - crps_and_mean(np.arange(30) != row)
        assert np.abs(expected).max() > 0  # The stand-in's neighbours move.
        np.testing.assert_allclose(
            explanation.values[[0, 2], :, k], expected, atol=1e-12
        )


def test_a_number_of_rows_draws_the_same_rows_for_the_same_seed(data, model):
    X_context, y_context, X, y = data
    X_context, y_context = X_context[:30], y_context[:30]
    first, second, other = (
        interp.explain_context(
            model,
            X[:5],
            y[:5],
            X_context,
            y_context,
            method="loo",
            rows=4,
            random_state=seed,
        )
        for seed in (3, 3, 4)
    )
    assert len(np.unique(first.context_rows)) == 4
    np.testing.assert_array_equal(first.context_rows, second.context_rows)
    assert not np.array_equal(first.context_rows, other.context_rows)
    given = interp.explain_context(
        model,
        X[:5],
        y[:5],
        X_context,
        y_context,
        method="loo",
        rows=first.context_rows,
    )
    np.testing.assert_array_equal(given.values, first.values)


def test_context_gradients_are_the_derivative_along_the_rows_offset(
    differentiable,
):
    model, X_context, y_context, X, y = differentiable
    grid = lazy.Grid.linear(-4.0, 4.0, 200)
    baseline = np.array([0.5, -1.0, 2.0])
    explanation = interp.explain_context(
        model, X, y, X_context, y_context, baseline=baseline, y_grid=grid
    )
    assert explanation.values.shape == (3, 20, 40)
    np.testing.assert_array_equal(explanation.context_rows, np.arange(40))
    np.testing.assert_allclose(
        explanation.full_values,
        interp._statistics(model, X, y, True, grid),
        rtol=1e-5,
        atol=1e-6,
    )
    # The sum over a row's features of the derivative times the offset is
    # the derivative along the offset: central differences of predict's
    # statistics, refitting with the row moved along it. The fake's float32
    # logits err by about 1e-7 / step (measured: 1e-5 at this step, values
    # up to 0.8); a smaller step only raises that.
    step = 1e-2
    for row in (0, 13, 39):
        offset = X_context.iloc[row].to_numpy() - baseline
        moved = []
        for sign in (1, -1):
            context = X_context.copy()
            context.iloc[row] += sign * step * offset
            refit = sklearn_base.clone(model).fit(context, y_context)
            moved.append(interp._statistics(refit, X, y, True, grid))
        expected = (moved[0] - moved[1]) / (2 * step)
        np.testing.assert_allclose(
            explanation.values[..., row], expected, atol=1e-4
        )


def test_top_ranks_context_rows_by_absolute_value():
    values = np.array([[[0.1, -0.5, 0.3], [0.0, 0.2, -0.1]]])
    explanation = interp.ContextExplanation(
        "loo", ("nll",), values, np.array([10, 20, 30]), np.zeros((1, 2))
    )
    top = explanation.top(2, "nll")
    assert list(top.columns) == ["row", "rank", "context_row", "value"]
    assert top["row"].tolist() == [0, 0, 1, 1]
    assert top["rank"].tolist() == [1, 2, 1, 2]
    assert top["context_row"].tolist() == [20, 30, 20, 30]
    assert top["value"].tolist() == [-0.5, 0.3, 0.2, -0.1]
    assert len(explanation.top(5, "nll")) == 6
    with pytest.raises(ValueError, match="statistic must be one of"):
        explanation.top(1)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"method": "shapley"}, "method must be one of"),
        ({"rows": 3}, "rows does not apply"),
        ({"method": "loo", "baseline": np.zeros(3)}, "baseline does not"),
        ({"baseline": np.zeros((2, 2))}, "baseline must be finite"),
        ({"method": "loo", "rows": 0}, "rows == 0, must be >= 1"),
        ({"method": "loo", "rows": [1, 1]}, "rows repeats rows"),
        ({"method": "loo", "rows": [200]}, r"rows must be row positions in"),
        ({"method": "loo", "rows": [0.5]}, "rows must be integer row"),
        ({"method": "loo", "rows": [[1, 2]]}, "a 1-D array of positions"),
        ({"method": "loo", "rows": []}, "a 1-D array of positions"),
        ({"random_state": 5}, "random_state does not apply"),
    ],
)
def test_a_bad_context_argument_is_refused_by_name(
    data, model, options, message
):
    X_context, y_context, X, y = data
    with pytest.raises(ValueError, match=message):
        interp.explain_context(model, X, y, X_context, y_context, **options)


@pytest.mark.parametrize(
    ("model", "message", "needs_torch"),
    [
        (dummy.DummyRegressor(), "needs a lazy backend", False),
        (
            standins.HistogramStandIn(n_estimators=2, progress=False),
            r"'gradient': .*fit with transforms=\('none',\)",
            True,
        ),
    ],
    ids=["no-pass", "transformed"],
)
def test_context_gradients_refuse_a_model_without_a_differentiable_pass(
    data, model, message, needs_torch
):
    X_context, y_context, X, y = data
    if needs_torch:  # The pass itself refuses.
        pytest.importorskip("torch")
    model.fit(X_context, y_context)
    with pytest.raises(ValueError, match=message):
        interp.explain_context(model, X, y, X_context, y_context)


def test_context_gradients_refuse_a_context_other_than_the_fits(
    differentiable,
):
    model, X_context, y_context, X, y = differentiable
    with pytest.raises(ValueError, match="the context the model was fitted"):
        interp.explain_context(model, X, y, X_context, y_context + 1.0)


@pytest.mark.checkpoint
def test_a_context_outlier_moves_tabpfns_prediction_beside_it_most():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(65, 3))
    X[0] = [1.5, -1.0, 0.5]  # The outlier, off the mean of the context.
    X[60] = X[0] + 0.05  # The explained row beside it.
    y = 2.0 * X[:, 0] + 0.1 * rng.normal(size=65)
    y[0] += 10.0
    model = lazy.TabPFNBarDistribution(
        n_estimators=2,
        device="cpu",
        progress=False,
        transforms=("none",),
        outlier_threshold=None,
    ).fit(X[:60], y[:60])
    explanation = interp.explain_context(model, X[60:], y[60:], X[:60], y[:60])
    assert np.isfinite(explanation.values).all()
    # The outlier pulls the mean of the row beside it most: 20 times more
    # than that of any other explained row here (measured: 3.0 against
    # 0.15). It did so for 18 of 18 data sets measured, by as little as 1.1
    # times. Its rank among that row's context rows by NLL varies more: in
    # the top 5 for 5 of seeds 0 to 9, and as low as 13th.
    outlier = np.abs(explanation.values[2, :, 0])
    assert outlier[0] > 2 * outlier[1:].max()
