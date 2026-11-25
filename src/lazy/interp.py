# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Which features a model's predicted distributions rely on.

:func:`explain` attributes three statistics of each explained row's predicted
distribution to the features: minus its CRPS and minus its negative log
likelihood at the true target, and its mean. It needs only ``fit`` and
``predict_distribution``, so it works on every model::

    explanation = interp.explain(model, X_test, y_test, X_train, y_train)
    explanation.importance()  # One row per feature, one column per statistic.
    explanation.table("nll")  # One row per explained row.

The methods are Shapley values and leave-one-out, which remove a feature by
refitting the model without its column, and permutation, which keeps the
fitted model. The "Interpretability" page of the documentation
(``docs/guide/interpretability.md``) defines each one.

The values of the CRPS and of the NLL are signed so that a positive value
means the feature improves the prediction. Explain rows that are not in the
context: a model recognizes its context rows and predicts their targets
whatever the features.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
import numbers
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base as sklearn_base
from sklearn.utils import validation

from lazy import _inputs
from lazy import _typing
from lazy import base
from lazy import grid as grid_lib
from lazy import metrics

__all__ = [
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_REPEATS",
    "EXACT_MAX_FEATURES",
    "METHODS",
    "STATISTICS",
    "Explanation",
    "explain",
]

STATISTICS: tuple[str, ...] = ("crps", "nll", "mean")
"""The statistics that are attributed. They are minus the CRPS and minus the
negative log likelihood of each row's predicted distribution at its true
target, and the mean of that distribution.
"""

METHODS: tuple[str, ...] = ("shapley", "loco", "permutation")
"""The accepted values of ``method``, besides ``"auto"``."""

EXACT_MAX_FEATURES = 10
"""The most features for which ``"shapley"`` is exact by default: 1,024 fits."""

DEFAULT_PERMUTATIONS = 32
"""The permutations ``"shapley"`` samples by default above that."""

DEFAULT_REPEATS = 5
"""The shuffles of each feature that ``"permutation"`` averages by default."""

# From coalitions, shape (m, p) of bool, to every statistic of every explained
# row, shape (m, n_statistics, n_rows).
_ValueFunction = Callable[[npt.NDArray[np.bool_]], _typing.FloatArray]


@dataclasses.dataclass(frozen=True)
class Explanation:
    """The attributions of :func:`explain`.

    Attributes:
        method: The method that ran, one of :data:`METHODS`.
        statistics: The statistics attributed, in the order of the first
            axis of ``values``.
        feature_names: The features, in the order of the last axis of
            ``values``: column names, or group names.
        values: Each feature's attribution, shape
            (n_statistics, n_rows, n_features). The CRPS is in the target's
            units, the NLL in nats, and the mean in the target's units.
        full_values: Each statistic with every feature present, shape
            (n_statistics, n_rows).
        base_values: Each statistic with no feature, shape
            (n_statistics, n_rows), for ``"shapley"``, whose values sum to
            ``full_values - base_values`` over the features; ``None`` for
            the other methods.
    """

    method: str
    statistics: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: _typing.FloatArray
    full_values: _typing.FloatArray
    base_values: _typing.FloatArray | None

    def table(self, statistic: str = "crps") -> pd.DataFrame:
        """One statistic's attributions, per explained row.

        Args:
            statistic: One of ``statistics``.

        Returns:
            A table of shape (n_rows, n_features), with the features as
            columns.

        Raises:
            ValueError: If ``statistic`` is not one of ``statistics``.
        """
        if statistic not in self.statistics:
            raise ValueError(
                f"statistic must be one of {self.statistics}: {statistic=}"
            )
        return pd.DataFrame(
            self.values[self.statistics.index(statistic)],
            columns=list(self.feature_names),
        )

    def importance(self) -> pd.DataFrame:
        """Each feature's importance over the explained rows.

        For the CRPS and the NLL it is the mean attribution: how much the
        feature lowers the average loss (for ``"shapley"``, the SAGE value
        of Covert et al. 2020). For the mean it is the mean absolute
        attribution, since contributions to a prediction cancel over rows.

        Returns:
            A table of shape (n_features, n_statistics), with the features
            as rows and ``statistics`` as columns.
        """
        columns = {
            name: (np.abs(values) if name == "mean" else values).mean(axis=0)
            for name, values in zip(self.statistics, self.values, strict=True)
        }
        return pd.DataFrame(columns, index=list(self.feature_names))


def explain(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    X: Any,
    y: npt.ArrayLike,
    X_context: Any = None,
    y_context: npt.ArrayLike | None = None,
    *,
    method: str = "auto",
    groups: Mapping[str, Sequence[str]] | None = None,
    y_grid: grid_lib.Grid | npt.ArrayLike | None = None,
    n_permutations: int | None = None,
    n_repeats: int | None = None,
    random_state: int = 0,
) -> Explanation:
    """Attributes each row's predicted distribution to the features.

    ``"shapley"`` fits the model once per coalition, ``2**p`` times when it
    is exact and at most ``n_permutations * (p - 1) + 2`` times when it is
    sampled; ``"loco"`` fits it ``p + 1`` times; ``"permutation"`` predicts
    ``1 + p * n_repeats`` times. Every fit serves all the explained rows.
    The distributions of all the explained rows are held at once, and so
    is every coalition's statistics, ``24 * n_rows`` bytes each: about
    60 kB per explained row at the peak for exact values of 10 features.

    Args:
        model: An estimator with ``fit`` and ``predict_distribution``.
            ``"permutation"`` needs it fitted; the other methods fit clones
            of it and leave it as it is. A clone fitted on some of the
            columns gets, as its ``categorical_features``, those columns'
            mask; inside a pipeline, whose steps change the columns, the
            model should mark its categorical columns by their pandas type
            (``"from_dtype"``).
        X: The rows to explain, shape (n_rows, n_columns), in any form the
            models take. They should not be rows of the context.
        y: Their finite true targets, shape (n_rows,).
        X_context: The context to refit on, shape (n_context, n_columns),
            with the columns of ``X``; needed by ``"shapley"`` and
            ``"loco"``.
        y_context: Its finite targets, shape (n_context,).
        method: One of :data:`METHODS`, or ``"auto"``: ``"shapley"`` when a
            context is given, and ``"permutation"`` otherwise.
            ``"permutation"`` does not use a context.
        groups: Columns to treat as one feature each, as
            ``{feature name: column names}``: the columns of an embedding,
            say. A column in no group is a feature of its own. ``None`` for
            no groups.
        y_grid: The grid on which the CRPS and the NLL are computed, the
            same for every coalition. ``None`` means the fitted model's
            default grid for ``"permutation"``, and
            :func:`lazy.base.default_grid` over the targets of the rows and
            of the context for the other methods.
        n_permutations: For ``"shapley"``, the even number of permutations
            to sample, in antithetic pairs; ``None`` for exact values up to
            :data:`EXACT_MAX_FEATURES` features and
            :data:`DEFAULT_PERMUTATIONS` permutations above.
        n_repeats: For ``"permutation"``, the shuffles of each feature;
            ``None`` for :data:`DEFAULT_REPEATS`.
        random_state: The seed of the permutations, the shuffles and the
            column of noise that stands in for no feature at all.

    Returns:
        The attributions.

    Raises:
        ValueError: If an argument is not as described above, or
            ``n_permutations`` or ``n_repeats`` is given to a method that
            does not use it.
        TypeError: If ``n_permutations`` or ``n_repeats`` is not an integer.
        sklearn.exceptions.NotFittedError: If ``method`` is
            ``"permutation"`` and ``model`` is not fitted.
    """
    method = _check_method(method, X_context, y_context)
    _check_options(method, n_permutations=n_permutations, n_repeats=n_repeats)
    X, named = _frame(X)
    if X.empty:
        raise ValueError("X has no rows to explain")
    y = _inputs.check_target(y, len(X))
    names, feature_of = _features(X.columns, groups)
    rng = np.random.default_rng(random_state)

    if method == "permutation":
        validation.check_is_fitted(model)
        if y_grid is None and not hasattr(model, "grid_"):
            raise ValueError(
                "y_grid is needed: the model has no default grid (grid_)"
            )
        grid = model.grid_ if y_grid is None else grid_lib.as_grid(y_grid)
        values, full = _permutation(
            model, X, y, named, feature_of, grid, _count(n_repeats), rng
        )
        return Explanation(method, STATISTICS, names, values, full, None)

    context, _ = _frame(X_context)
    context_targets = _inputs.check_target(
        y_context, len(context), names=("X_context", "y_context")
    )
    if not context.columns.equals(X.columns):
        raise ValueError(
            "X_context must have the columns of X:"
            f" {list(context.columns)} != {list(X.columns)}"
        )
    if y_grid is None:
        targets = np.concatenate([y, context_targets])
        grid = base.default_grid(float(targets.min()), float(targets.max()))
    else:
        grid = grid_lib.as_grid(y_grid)
    value_function = _refit_values(
        model, context, context_targets, X, y, named, feature_of, grid, rng
    )
    n_features = len(names)
    if method == "loco":
        coalitions = np.vstack(
            [np.ones(n_features, dtype=bool), ~np.eye(n_features, dtype=bool)]
        )
        table = value_function(coalitions)
        values = np.moveaxis(table[0] - table[1:], 0, -1)
        return Explanation(method, STATISTICS, names, values, table[0], None)
    if n_permutations is None and n_features <= EXACT_MAX_FEATURES:
        values, full, empty = _exact_shapley(value_function, n_features)
    else:
        values, full, empty = _sampled_shapley(
            value_function, n_features, _count(n_permutations, even=True), rng
        )
    return Explanation(method, STATISTICS, names, values, full, empty)


def _check_method(method: str, X_context: Any, y_context: Any) -> str:  # noqa: GS030 - scikit-learn's X, y.
    """The method to run, with ``"auto"`` resolved.

    Args:
        method: One of :data:`METHODS`, or ``"auto"``.
        X_context: The context features, or ``None``.
        y_context: The context targets, or ``None``.

    Returns:
        One of :data:`METHODS`.

    Raises:
        ValueError: If ``method`` is unknown, only one half of the context
            is given, or the method refits and there is no context.
    """
    if (X_context is None) != (y_context is None):
        raise ValueError("X_context and y_context must be given together")
    if method == "auto":
        method = "permutation" if X_context is None else "shapley"
    if method not in METHODS:
        raise ValueError(
            f"method must be 'auto' or one of {METHODS}: {method=}"
        )
    if method != "permutation" and X_context is None:
        raise ValueError(
            f"method={method!r} refits the model, so it needs X_context and"
            " y_context"
        )
    return method


def _check_options(method: str, **options: int | None) -> None:
    """Refuses an option that ``method`` does not use.

    Args:
        method: One of :data:`METHODS`.
        **options: ``n_permutations`` and ``n_repeats``, as given.

    Raises:
        ValueError: If an option of another method is not ``None``.
    """
    used = {"shapley": "n_permutations", "permutation": "n_repeats"}
    for name, value in options.items():
        if value is not None and used.get(method) != name:
            raise ValueError(f"{name} does not apply to method={method!r}")


def _count(value: int | None, even: bool = False) -> int:
    """``n_permutations`` (``even``) or ``n_repeats``, checked or defaulted.

    Args:
        value: The option as given; ``None`` for its default.
        even: Whether the option is ``n_permutations``, which must be even.

    Returns:
        The count.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If it is too small, or odd where it must be even.
    """
    name = "n_permutations" if even else "n_repeats"
    if value is None:
        return DEFAULT_PERMUTATIONS if even else DEFAULT_REPEATS
    validation.check_scalar(value, name, numbers.Integral, min_val=1 + even)
    if even and value % 2:
        raise ValueError(
            "n_permutations must be even, since permutations are drawn in"
            f" antithetic pairs: {value=}"
        )
    return int(value)


def _features(
    columns: pd.Index, groups: Mapping[str, Sequence[str]] | None
) -> tuple[tuple[str, ...], _typing.IntArray]:
    """The features to attribute to, and the feature of each column.

    Args:
        columns: The column names of ``X``.
        groups: ``{feature name: column names}``, or ``None``.

    Returns:
        A tuple (names, feature_of): each feature's name, in the order its
        first column appears, and the index of each column's feature, shape
        (n_columns,).

    Raises:
        ValueError: If a group names an unknown column, two groups share a
            column, or a group has the name of a column outside it.
    """
    owner: dict[Any, str] = {}
    for name, members in (groups or {}).items():
        if name in columns and name not in members:
            raise ValueError(
                f"groups[{name!r}] has the name of a column of X that is not"
                " in it"
            )
        for column in members:
            if column not in columns:
                raise ValueError(
                    f"groups[{name!r}] names {column!r}, which is not a"
                    f" column of X: {list(columns)}"
                )
            if owner.setdefault(column, name) != name:
                raise ValueError(
                    f"Column {column!r} is in two groups: {owner[column]!r}"
                    f" and {name!r}"
                )
    index: dict[str, int] = {}
    feature_of = [
        index.setdefault(owner.get(column, str(column)), len(index))
        for column in columns
    ]
    return tuple(index), np.array(feature_of)


def _frame(features: Any) -> tuple[pd.DataFrame, bool]:
    """Features as a DataFrame of their own types, named ``x0, ...`` if not.

    Args:
        features: Features, shape (n, p), in any form a model accepts.

    Returns:
        A tuple (frame, named), as :func:`lazy._inputs.to_frame` returns it,
        with placeholder names for a table without names.
    """
    frame, named = _inputs.to_frame(features)
    if not named:
        frame.columns = [f"x{i}" for i in range(frame.shape[1])]
    return frame, named


def _table(
    frame: pd.DataFrame, named: bool
) -> pd.DataFrame | _typing.FloatArray:
    """``frame`` as a model should get it.

    A model compares the column names of its queries with those of its fit,
    so a table that the caller gave without names goes to it without names.

    Args:
        frame: Features, shape (n, p).
        named: Whether the caller named the columns.

    Returns:
        The frame, or its values when the names are placeholders.
    """
    return frame if named else frame.to_numpy()


def _statistics(
    model: Any,
    X: pd.DataFrame,  # noqa: GS030 - scikit-learn's X, y.
    y: _typing.FloatArray,
    named: bool,
    grid: grid_lib.Grid,
) -> _typing.FloatArray:
    """Every statistic of each row's predicted distribution.

    Args:
        model: A fitted model.
        X: Query rows, shape (n, p).
        y: Their true targets, shape (n,).
        named: Whether the caller named the columns; see :func:`_table`.
        grid: The grid on which the CRPS and the NLL are computed.

    Returns:
        The statistics, in the order of :data:`STATISTICS`, shape (3, n).
    """
    distribution = model.predict_distribution(_table(X, named))
    _, _, crps, nll = metrics.per_object_metrics(
        y, grid, distribution.on_grid(grid)
    )
    return np.stack([-crps, -nll, distribution.mean()])


def _refit_values(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    X_context: pd.DataFrame,
    y_context: _typing.FloatArray,
    X: pd.DataFrame,
    y: _typing.FloatArray,
    named: bool,
    feature_of: _typing.IntArray,
    grid: grid_lib.Grid,
    rng: np.random.Generator,
) -> _ValueFunction:
    """A value function that removes absent features and refits.

    For a coalition, a clone of ``model`` is fitted on the context's columns
    of the features present, and predicts the same columns of ``X``. No
    model fits on no column at all, so the empty coalition gets one column
    of standard normal noise: its prediction is then the model's estimate of
    the target's marginal distribution. A clone that takes
    ``categorical_features`` gets the mask of the columns present.

    Args:
        model: The model to clone.
        X_context: The context, shape (n_context, n_columns).
        y_context: Its targets, shape (n_context,).
        X: The rows to explain, shape (n, n_columns).
        y: Their true targets, shape (n,).
        named: Whether the caller named the columns; see :func:`_table`.
        feature_of: The index of each column's feature, shape (n_columns,).
        grid: The grid on which the CRPS and the NLL are computed.
        rng: The source of the column of noise.

    Returns:
        The value function.
    """
    noise_context = pd.DataFrame({"noise": rng.standard_normal(len(X_context))})
    noise_queries = pd.DataFrame({"noise": rng.standard_normal(len(X))})
    params = model.get_params()
    is_categorical = (
        _inputs.categorical_mask(
            params["categorical_features"], X_context, named
        )
        if "categorical_features" in params
        else None
    )

    def values(coalitions: npt.NDArray[np.bool_]) -> _typing.FloatArray:
        """Every statistic of every row, for each coalition."""
        table = []
        for coalition in coalitions:
            clone = sklearn_base.clone(model)
            context, queries, marked = noise_context, noise_queries, None
            if coalition.any():
                present = coalition[feature_of]
                context, queries = X_context.loc[:, present], X.loc[:, present]
                if is_categorical is not None:
                    marked = is_categorical[present]
            if is_categorical is not None:
                clone.set_params(categorical_features=marked)
            fitted = clone.fit(context, y_context)
            table.append(_statistics(fitted, queries, y, True, grid))
        return np.stack(table)

    return values


def _exact_shapley(
    value_function: _ValueFunction, n_features: int
) -> tuple[_typing.FloatArray, _typing.FloatArray, _typing.FloatArray]:
    """Shapley values over all ``2**n_features`` coalitions.

    Feature ``j`` gets the sum over the coalitions ``S`` without it of
    ``|S|! (p - |S| - 1)! / p! * (v(S + j) - v(S))``; the weight equals
    ``1 / (p * comb(p - 1, |S|))``.

    Args:
        value_function: From coalitions, shape (m, p), to values, shape
            (m, n_statistics, n_rows).
        n_features: The number of features, p.

    Returns:
        A tuple (values, full, empty): the Shapley values, shape
        (n_statistics, n_rows, p), and the values of the full and of the
        empty coalition, each shape (n_statistics, n_rows).
    """
    p = n_features
    index = np.arange(1 << p)  # A coalition's index is its bit mask.
    coalitions = (index[:, None] >> np.arange(p)) & 1 == 1
    table = value_function(coalitions)
    weights = np.array([1 / (p * math.comb(p - 1, s)) for s in range(p)])
    sizes = coalitions.sum(axis=1)
    values = []
    for j in range(p):
        without = index[~coalitions[:, j]]
        gains = table[without | (1 << j)] - table[without]
        values.append(np.tensordot(weights[sizes[without]], gains, axes=1))
    return np.stack(values, axis=-1), table[-1], table[0]


def _sampled_shapley(
    value_function: _ValueFunction,
    n_features: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[_typing.FloatArray, _typing.FloatArray, _typing.FloatArray]:
    """Shapley values from antithetic pairs of permutations.

    Each permutation adds the features one at a time, and a feature's value
    is its mean gain over the permutations. Each permutation is paired with
    its reverse, which makes the values exact when no more than two
    features interact.

    Args:
        value_function: From coalitions, shape (m, p), to values, shape
            (m, n_statistics, n_rows).
        n_features: The number of features, p.
        n_permutations: The even number of permutations.
        rng: The source of the permutations.

    Returns:
        A tuple (values, full, empty), as :func:`_exact_shapley` returns.
    """
    p = n_features
    half = rng.permuted(np.tile(np.arange(p), (n_permutations // 2, 1)), axis=1)
    orders = np.concatenate([half, half[:, ::-1]])
    # rank[t, j] is the step at which permutation t adds feature j, and its
    # prefix k holds the features added in the first k steps.
    rank = np.argsort(orders, axis=1)
    prefixes = rank[:, None, :] < np.arange(p + 1)[None, :, None]
    coalitions, inverse = np.unique(
        prefixes.reshape(-1, p), axis=0, return_inverse=True
    )
    table = value_function(coalitions)
    step = inverse.reshape(len(orders), p + 1)
    gains = table[step[:, 1:]] - table[step[:, :-1]]  # (t, step, s, n)
    by_feature = np.take_along_axis(gains, rank[:, :, None, None], axis=1)
    values = np.moveaxis(by_feature.mean(axis=0), 0, -1)
    return values, table[step[0, -1]], table[step[0, 0]]


def _permutation(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    X: pd.DataFrame,
    y: _typing.FloatArray,
    named: bool,
    feature_of: _typing.IntArray,
    grid: grid_lib.Grid,
    n_repeats: int,
    rng: np.random.Generator,
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """What each statistic loses when one feature is shuffled among rows.

    Args:
        model: A fitted model.
        X: The rows to explain, shape (n, n_columns).
        y: Their true targets, shape (n,).
        named: Whether the caller named the columns; see :func:`_table`.
        feature_of: The index of each column's feature, shape (n_columns,);
            a feature's columns are shuffled together.
        grid: The grid on which the CRPS and the NLL are computed.
        n_repeats: The shuffles of each feature, which are averaged.
        rng: The source of the shuffles.

    Returns:
        A tuple (values, full): the mean loss of each statistic, shape
        (n_statistics, n, n_features), and the statistics of the rows as
        they are, shape (n_statistics, n).
    """
    full = _statistics(model, X, y, named, grid)
    n_features = feature_of.max() + 1
    values = np.zeros((*full.shape, n_features))
    for j in range(n_features):
        columns = feature_of == j
        for _ in range(n_repeats):
            shuffled = X.copy()
            moved = X.loc[:, columns].iloc[rng.permutation(len(X))]
            # A frame keeps each column's type, a category's included.
            shuffled.loc[:, columns] = moved.set_axis(X.index)
            values[..., j] += full - _statistics(
                model, shuffled, y, named, grid
            )
    return values / n_repeats, full
