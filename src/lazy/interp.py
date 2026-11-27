# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Which features and context rows a model's predicted distributions rely on.

:func:`explain` attributes three statistics of each explained row's predicted
distribution to the features: minus its CRPS and minus its negative log
likelihood at the true target, and its mean. Its removal methods need only
``fit`` and ``predict_distribution``, so they work on every model::

    explanation = interp.explain(model, X_test, y_test, X_train, y_train)
    explanation.importance()  # One row per feature, one column per statistic.
    explanation.table("nll")  # One row per explained row.

The methods are Shapley values and leave-one-out, which remove a feature by
refitting the model without its column; permutation, which keeps the fitted
model; and gradient times input and integrated gradients, which take
gradients of the statistics through a fitted model's differentiable forward
pass. The "Interpretability" page of the documentation
(``docs/guide/interpretability.md``) defines each one.

:func:`explain_context` attributes the same statistics to the rows of the
context, by gradient times input in each context row's features or by
leaving each context row out::

    explanation = interp.explain_context(
        model, X_test, y_test, X_train, y_train
    )
    explanation.top(5, "nll")  # Each row's 5 most influential context rows.

The values of the CRPS and of the NLL are signed so that a positive value
means the feature, or the context row, improves the prediction. Explain rows
that are not in the context: a model recognizes its context rows and
predicts their targets whatever the features.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
import numbers
import typing
from typing import Any

import array_api_compat
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
from lazy.models import _differentiable as differentiable

if typing.TYPE_CHECKING:
    from matplotlib import axes as mpl_axes

__all__ = [
    "CONTEXT_METHODS",
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_REPEATS",
    "DEFAULT_STEPS",
    "EXACT_MAX_FEATURES",
    "METHODS",
    "STATISTICS",
    "ContextExplanation",
    "Explanation",
    "explain",
    "explain_context",
]

STATISTICS: tuple[str, ...] = ("crps", "nll", "mean")
"""The statistics that are attributed. They are minus the CRPS and minus the
negative log likelihood of each row's predicted distribution at its true
target, and the mean of that distribution.
"""

METHODS: tuple[str, ...] = (
    "shapley",
    "loco",
    "permutation",
    "gradient",
    "integrated_gradients",
)
"""The accepted values of ``method``, besides ``"auto"``."""

CONTEXT_METHODS: tuple[str, ...] = ("gradient", "loo")
"""The accepted values of :func:`explain_context`'s ``method``."""

EXACT_MAX_FEATURES = 10
"""The most features for which ``"shapley"`` is exact by default: 1,024 fits."""

DEFAULT_PERMUTATIONS = 32
"""The permutations ``"shapley"`` samples by default above that."""

DEFAULT_REPEATS = 5
"""The shuffles of each feature that ``"permutation"`` averages by default."""

DEFAULT_STEPS = 32
"""The steps of ``"integrated_gradients"``'s Riemann sum by default."""

_GRADIENT_METHODS = ("gradient", "integrated_gradients")

# The x-axis label of Explanation.plot, for each statistic's importance.
_IMPORTANCE_LABELS = {
    "crps": "minus CRPS (target units)",
    "nll": "minus NLL (nats)",
    "mean": "|mean| (target units)",
}

# The query points on one autograd graph at a time, about.
_QUERY_BUDGET = 64

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
            (n_statistics, n_rows), for ``"shapley"``, and at the baseline
            for ``"integrated_gradients"``; the values of either sum to
            ``full_values - base_values`` over the features, for the
            latter to the error of its Riemann sum. ``None`` for the other
            methods.
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
        return pd.DataFrame(
            self.values[_statistic_index(self.statistics, statistic)],
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

    def plot(
        self, statistic: str = "crps", *, ax: mpl_axes.Axes | None = None
    ) -> mpl_axes.Axes:
        """Draws one statistic's :meth:`importance` as horizontal bars.

        The most important feature is at the top. The plot changes no
        global state: call :func:`lazy.plotting.use_style` first for the
        house rcParams.

        Args:
            statistic: One of ``statistics``.
            ax: The axes to draw into; a new column-width figure, taller
                with more features, if None.

        Returns:
            The axes drawn into.

        Raises:
            ValueError: If ``statistic`` is not one of ``statistics``.
        """
        _statistic_index(self.statistics, statistic)
        # Matplotlib is slow to import, and ``import lazy`` leaves it out.
        from lazy import plotting  # noqa: PLC0415 - see above.

        return plotting.plot_importance(
            self.importance()[statistic],
            _IMPORTANCE_LABELS[statistic],
            ax=ax,
        )


@dataclasses.dataclass(frozen=True)
class ContextExplanation:
    """The attributions of :func:`explain_context`.

    Attributes:
        method: The method that ran, one of :data:`CONTEXT_METHODS`.
        statistics: The statistics attributed, in the order of the first
            axis of ``values``.
        values: Each explained context row's attribution, shape
            (n_statistics, n_rows, n_context_rows_explained). The CRPS is in
            the target's units, the NLL in nats, and the mean in the
            target's units.
        context_rows: The positions in ``X_context`` of the explained
            context rows, in the order of the last axis of ``values``, shape
            (n_context_rows_explained,).
        full_values: Each statistic with the whole context, shape
            (n_statistics, n_rows).
    """

    method: str
    statistics: tuple[str, ...]
    values: _typing.FloatArray
    context_rows: _typing.IntArray
    full_values: _typing.FloatArray

    def top(self, k: int, statistic: str = "crps") -> pd.DataFrame:
        """Each explained row's most influential context rows.

        The most influential have the largest absolute attribution.

        Args:
            k: How many context rows to give per explained row; all of them
                if there are fewer.
            statistic: One of ``statistics``.

        Returns:
            A table with one row per explained row and rank, shape
            (n_rows * k, 4), sorted by both. Its columns are ``row``, the
            explained row's position in ``X``; ``rank``, from 1 for the most
            influential; ``context_row``, its position in ``X_context``; and
            ``value``, its attribution.

        Raises:
            ValueError: If ``statistic`` is not one of ``statistics``, or
                ``k`` is below 1.
            TypeError: If ``k`` is not an integer.
        """
        values = self.values[_statistic_index(self.statistics, statistic)]
        validation.check_scalar(k, "k", numbers.Integral, min_val=1)
        k = min(int(k), values.shape[1])
        order = np.argsort(-np.abs(values), axis=1, kind="stable")[:, :k]
        n_rows = len(values)
        return pd.DataFrame(
            {
                "row": np.repeat(np.arange(n_rows), k),
                "rank": np.tile(np.arange(1, k + 1), n_rows),
                "context_row": self.context_rows[order].ravel(),
                "value": np.take_along_axis(values, order, axis=1).ravel(),
            }
        )


def _statistic_index(statistics: tuple[str, ...], statistic: str) -> int:
    """The position of ``statistic`` in ``statistics``.

    Args:
        statistics: The statistics of an explanation.
        statistic: The one asked for.

    Returns:
        Its position.

    Raises:
        ValueError: If it is not one of ``statistics``.
    """
    if statistic not in statistics:
        raise ValueError(f"statistic must be one of {statistics}: {statistic=}")
    return statistics.index(statistic)


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
    n_steps: int | None = None,
    baseline: npt.ArrayLike | None = None,
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

    ``"gradient"`` runs the model's differentiable forward pass once, and
    three backward passes through it, one per statistic, for each batch of
    64 rows; ``"integrated_gradients"`` takes ``n_steps`` points on the
    path to each row, so it runs ``n_steps`` times as many batches, and
    predicts the baseline once. Both also predict the rows once, a batch
    at a time, to check that ``X_context`` is the model's context, so that
    a wrong one fails after the first batch. Neither refits, and each
    pass's autograd graph holds the context and one batch of about 64
    points, or of one row's ``n_steps`` points if there are more.

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
        X_context: The context, shape (n_context, n_columns), with the
            columns of ``X``: the one to refit on for ``"shapley"`` and
            ``"loco"``, and the one ``model`` was fitted on for the
            gradient methods.
        y_context: Its finite targets, shape (n_context,).
        method: One of :data:`METHODS`, or ``"auto"``: ``"shapley"`` when a
            context is given, and ``"permutation"`` otherwise.
            ``"permutation"`` does not use a context. ``"gradient"``
            (gradient times input) and ``"integrated_gradients"`` need
            ``model`` fitted with a differentiable forward pass: a
            backend, on numeric columns, with no transform or clip that
            the backend's own package applies (``transforms=("none",)``
            and ``outlier_threshold=None`` always qualify).
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
        n_steps: For ``"integrated_gradients"``, the steps of the Riemann
            sum along the path from the baseline to each row; ``None`` for
            :data:`DEFAULT_STEPS`.
        baseline: For the gradient methods, the finite row that stands in
            for no feature at all, shape (n_columns,); ``None`` for the mean
            of ``X_context``.
        random_state: The seed of the permutations, the shuffles and the
            column of noise that stands in for no feature at all.

    Returns:
        The attributions.

    Raises:
        ValueError: If an argument is not as described above, an option
            is given to a method that does not use it, or a gradient
            method's model has no differentiable forward pass.
        TypeError: If ``n_permutations``, ``n_repeats`` or ``n_steps`` is
            not an integer.
        sklearn.exceptions.NotFittedError: If ``method`` is
            ``"permutation"`` or a gradient method and ``model`` is not
            fitted.
    """
    method = _check_method(method, X_context, y_context)
    _check_options(
        method,
        n_permutations=n_permutations,
        n_repeats=n_repeats,
        n_steps=n_steps,
        baseline=baseline,
    )
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
        n_repeats = _count(n_repeats, "n_repeats", DEFAULT_REPEATS)
        values, full = _permutation(
            model, X, y, named, feature_of, grid, n_repeats, rng
        )
        return Explanation(method, STATISTICS, names, values, full, None)

    context, context_targets, grid = _context(
        X_context, y_context, X, y, y_grid
    )
    if method in _GRADIENT_METHODS:
        baseline = _check_baseline(baseline, X.shape[1])
        n_steps = (
            _count(n_steps, "n_steps", DEFAULT_STEPS)
            if method == "integrated_gradients"
            else 1
        )
        values, full, empty = _gradient_attributions(
            model,
            method,
            context,
            context_targets,
            X,
            y,
            named,
            grid,
            baseline,
            n_steps,
        )
        # A feature's attribution is the sum of its columns'.
        one_hot = feature_of[:, None] == np.arange(len(names))
        return Explanation(
            method, STATISTICS, names, values @ one_hot, full, empty
        )
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
        n_permutations = _count(
            n_permutations, "n_permutations", DEFAULT_PERMUTATIONS, even=True
        )
        values, full, empty = _sampled_shapley(
            value_function, n_features, n_permutations, rng
        )
    return Explanation(method, STATISTICS, names, values, full, empty)


def explain_context(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    X: Any,
    y: npt.ArrayLike,
    X_context: Any,
    y_context: npt.ArrayLike,
    *,
    method: str = "gradient",
    y_grid: grid_lib.Grid | npt.ArrayLike | None = None,
    baseline: npt.ArrayLike | None = None,
    rows: int | npt.ArrayLike | None = None,
    random_state: int | None = None,
) -> ContextExplanation:
    """Attributes each row's predicted distribution to the context rows.

    ``"gradient"`` gives context row ``r`` gradient times input in its
    features, summed over them: the derivative of an explained row's
    statistic in each feature of ``r``, times that feature's offset from
    the baseline. It runs the model's differentiable forward pass once for
    each batch of 64 rows, and one backward pass through it for each
    statistic of each row, ``3 * n_rows`` in all, each about as costly as
    the forward pass; it also predicts the rows once, a batch at a time,
    to check that ``X_context`` is the model's context. The autograd graph
    holds the
    context and one batch of rows, and ``values`` holds
    ``24 * n_rows * n_context`` bytes.

    ``"loo"`` (leave one out) gives context row ``r`` the statistic of a
    clone of ``model`` fitted on the whole context minus that of a clone
    fitted without ``r``: ``n_context_rows_explained + 1`` fits, each of
    which serves all the explained rows.
    A model that bags its context rows (``bag_size``) draws other bags
    without ``r``, so its values hold that redraw as well as ``r``'s own
    effect; fit it without bagging to measure ``r`` alone.

    Args:
        model: An estimator with ``fit`` and ``predict_distribution``.
            ``"gradient"`` needs it fitted on ``X_context`` and
            ``y_context`` with a differentiable forward pass, as
            :func:`explain`'s gradient methods do. A transform or a clip
            that lazy fitted on the context is held fixed, so the values
            leave out how it depends on each context row. ``"loo"`` fits
            clones of it and leaves it as it is.
        X: The rows to explain, shape (n_rows, n_columns), in any form the
            models take. They should not be rows of the context.
        y: Their finite true targets, shape (n_rows,).
        X_context: The context, shape (n_context, n_columns), with the
            columns of ``X``.
        y_context: Its finite targets, shape (n_context,).
        method: One of :data:`CONTEXT_METHODS`.
        y_grid: The grid on which the CRPS and the NLL are computed;
            ``None`` for :func:`lazy.base.default_grid` over the targets of
            the rows and of the context.
        baseline: For ``"gradient"``, the finite row from which each context
            row's offset is measured, shape (n_columns,); ``None`` for the
            mean of ``X_context``.
        rows: For ``"loo"``, the context rows to explain: a number of them,
            drawn without replacement, or their distinct positions in
            ``X_context``, shape (n_context_rows_explained,). ``None`` for
            every row.
        random_state: For ``"loo"``, the seed of the rows that an integer
            ``rows`` draws; ``None`` for 0.

    Returns:
        The attributions.

    Raises:
        ValueError: If an argument is not as described above, an option is
            given to the method that does not use it, or the model of
            ``"gradient"`` has no differentiable forward pass.
        sklearn.exceptions.NotFittedError: If ``method`` is ``"gradient"``
            and ``model`` is not fitted.
    """
    if method not in CONTEXT_METHODS:
        raise ValueError(f"method must be one of {CONTEXT_METHODS}: {method=}")
    _check_options(
        method, baseline=baseline, rows=rows, random_state=random_state
    )
    X, named = _frame(X)
    if X.empty:
        raise ValueError("X has no rows to explain")
    y = _inputs.check_target(y, len(X))
    context, context_targets, grid = _context(
        X_context, y_context, X, y, y_grid
    )
    if method == "gradient":
        baseline = _check_baseline(baseline, X.shape[1])
        chosen = np.arange(len(context))
        values, full = _context_gradients(
            model, context, context_targets, X, y, named, grid, baseline
        )
    else:
        chosen = _context_rows(
            rows, len(context), np.random.default_rng(random_state or 0)
        )
        values, full = _leave_one_out(
            model, context, context_targets, X, y, named, grid, chosen
        )
    return ContextExplanation(method, STATISTICS, values, chosen, full)


def _context(  # noqa: GS030 - scikit-learn's X, y.
    X_context: Any,
    y_context: Any,
    X: pd.DataFrame,
    y: _typing.FloatArray,
    y_grid: grid_lib.Grid | npt.ArrayLike | None,
) -> tuple[pd.DataFrame, _typing.FloatArray, grid_lib.Grid]:
    """The context, checked against the rows, and the grid of the scores.

    Args:
        X_context: The context features, as given.
        y_context: Their targets, as given.
        X: The rows to explain, shape (n, n_columns).
        y: Their true targets, shape (n,).
        y_grid: The grid as given, or ``None`` for
            :func:`lazy.base.default_grid` over ``y`` and ``y_context``.

    Returns:
        A tuple (X_context, y_context, grid): the context as a frame, shape
        (n_context, n_columns), its targets, shape (n_context,), and the
        grid.

    Raises:
        ValueError: If the targets are not finite, their number is not the
            context's, or the context has other columns than ``X``.
    """
    context, _ = _frame(X_context)
    targets = _inputs.check_target(
        y_context, len(context), names=("X_context", "y_context")
    )
    if not context.columns.equals(X.columns):
        raise ValueError(
            "X_context must have the columns of X:"
            f" {list(context.columns)} != {list(X.columns)}"
        )
    if y_grid is not None:
        return context, targets, grid_lib.as_grid(y_grid)
    both = np.concatenate([y, targets])
    grid = base.default_grid(float(both.min()), float(both.max()))
    return context, targets, grid


def _context_rows(
    rows: int | npt.ArrayLike | None, n_context: int, rng: np.random.Generator
) -> _typing.IntArray:
    """The positions of the context rows that ``"loo"`` explains.

    Args:
        rows: ``rows`` as given.
        n_context: The number of context rows.
        rng: The source of the rows that an integer draws.

    Returns:
        The positions, shape (n_context_rows_explained,): sorted when drawn,
        in the given order when given.

    Raises:
        ValueError: If a number of rows is not in [1, n_context], or the
            positions are not a non-empty 1-D array of distinct integers in
            [0, n_context).
    """
    if rows is None:
        return np.arange(n_context)
    if isinstance(rows, numbers.Integral):
        validation.check_scalar(
            rows, "rows", numbers.Integral, min_val=1, max_val=n_context
        )
        return np.sort(rng.choice(n_context, int(rows), replace=False))
    shape = np.shape(rows)
    if len(shape) != 1 or not shape[0]:
        raise ValueError(
            "rows must be a number of rows or a 1-D array of positions in"
            f" X_context, of at least one: got shape {shape}"
        )
    return _inputs.check_rows(rows, n_context, name="rows")


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
            is given, or the method needs a context and there is none.
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
        raise ValueError(f"method={method!r} needs X_context and y_context")
    return method


def _check_options(method: str, **options: Any) -> None:
    """Refuses an option that ``method`` does not use.

    Args:
        method: One of :data:`METHODS` or :data:`CONTEXT_METHODS`.
        **options: The options of :func:`explain` or of
            :func:`explain_context`, as given.

    Raises:
        ValueError: If an option of another method is not ``None``.
    """
    used = {
        "shapley": ("n_permutations",),
        "permutation": ("n_repeats",),
        "gradient": ("baseline",),
        "integrated_gradients": ("n_steps", "baseline"),
        "loo": ("rows", "random_state"),
    }
    for name, value in options.items():
        if value is not None and name not in used.get(method, ()):
            raise ValueError(f"{name} does not apply to method={method!r}")


def _count(
    value: int | None, name: str, default: int, even: bool = False
) -> int:
    """A count option, such as ``n_repeats``, checked or defaulted.

    Args:
        value: The option as given; ``None`` for its default.
        name: The option's name, for the errors.
        default: Its default.
        even: Whether it must be even, as ``n_permutations`` must.

    Returns:
        The count.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If it is too small, or odd where it must be even.
    """
    if value is None:
        return default
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
    return np.stack(_scores(distribution, y, grid))


def _scores(
    distribution: Any, y: _typing.FloatArray, grid: grid_lib.Grid
) -> tuple[Any, Any, Any]:
    """The statistics of numpy's or torch's distributions, in order.

    Args:
        distribution: One distribution per row, with ``on_grid`` and
            ``mean``: a :mod:`lazy.distributions` one, or a torch density,
            whose statistics keep its autograd graph.
        y: Their true targets, shape (n,).
        grid: The grid on which the CRPS and the NLL are computed.

    Returns:
        The statistics, in the order of :data:`STATISTICS`, each of the
        distribution's array kind and shape (n,).
    """
    densities = distribution.on_grid(grid)
    if not array_api_compat.is_torch_array(densities):
        # A copy, as per_object_metrics takes: _grid_scores normalizes it.
        densities = np.array(densities, dtype=np.float64)
    crps, nll = metrics._grid_scores(y, grid, densities)  # noqa: SLF001 - per_object_metrics's code, for torch too.
    return -crps, -nll, distribution.mean()


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
            table.append(
                _fit_statistics(
                    clone, context, y_context, queries, y, True, grid
                )
            )
        return np.stack(table)

    return values


def _fit_statistics(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    X_context: pd.DataFrame,
    y_context: _typing.FloatArray,
    X: pd.DataFrame,
    y: _typing.FloatArray,
    named: bool,
    grid: grid_lib.Grid,
) -> _typing.FloatArray:
    """Fits ``model`` on a context, and scores every row.

    Args:
        model: An unfitted model, which is fitted in place.
        X_context: The context, shape (n_context, p).
        y_context: Its targets, shape (n_context,).
        X: Query rows, shape (n, p).
        y: Their true targets, shape (n,).
        named: Whether the caller named the columns; see :func:`_table`.
        grid: The grid on which the CRPS and the NLL are computed.

    Returns:
        The statistics, in the order of :data:`STATISTICS`, shape (3, n).
    """
    model.fit(_table(X_context, named), y_context)
    return _statistics(model, X, y, named, grid)


def _leave_one_out(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    X_context: pd.DataFrame,
    y_context: _typing.FloatArray,
    X: pd.DataFrame,
    y: _typing.FloatArray,
    named: bool,
    grid: grid_lib.Grid,
    chosen: _typing.IntArray,
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """What each statistic loses when a clone is fitted without a row.

    Args:
        model: The model to clone.
        X_context: The context, shape (n_context, n_columns).
        y_context: Its targets, shape (n_context,).
        X: The rows to explain, shape (n, n_columns).
        y: Their true targets, shape (n,).
        named: Whether the caller named the columns; see :func:`_table`.
        grid: The grid on which the CRPS and the NLL are computed.
        chosen: The positions of the context rows to leave out, shape (m,).

    Returns:
        A tuple (values, full): each statistic with the whole context minus
        that without each row, shape (n_statistics, n, m), and the former,
        shape (n_statistics, n).
    """
    full = _fit_statistics(
        sklearn_base.clone(model), X_context, y_context, X, y, named, grid
    )
    values = np.empty((*full.shape, len(chosen)))
    for index, row in enumerate(chosen):
        keep = np.delete(np.arange(len(X_context)), row)
        values[..., index] = full - _fit_statistics(
            sklearn_base.clone(model),
            X_context.iloc[keep],
            y_context[keep],
            X,
            y,
            named,
            grid,
        )
    return values, full


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


def _gradient_attributions(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    method: str,
    X_context: pd.DataFrame,
    y_context: _typing.FloatArray,
    X: pd.DataFrame,
    y: _typing.FloatArray,
    named: bool,
    grid: grid_lib.Grid,
    baseline: _typing.FloatArray | None,
    n_steps: int,
) -> tuple[_typing.FloatArray, _typing.FloatArray, _typing.FloatArray | None]:
    """Gradient times input, or integrated gradients, of every statistic.

    Column ``j`` of row ``x`` gets ``x_j - b_j`` times the mean of the
    statistic's derivative in ``x_j`` at ``b + (k / m) (x - b)`` for
    ``k = 1, ..., m``: at ``x`` alone for ``"gradient"``, whose ``m`` is 1.
    Each point's statistics depend only on its own features, so the
    gradient of their sum over a batch of points is each point's gradient.
    A batch holds the ``m`` points of each of about
    ``_QUERY_BUDGET / m`` rows.

    Args:
        model: A model fitted on ``X_context`` and ``y_context``.
        method: ``"gradient"`` or ``"integrated_gradients"``.
        X_context: Its context, shape (n_context, n_columns).
        y_context: Its targets, shape (n_context,).
        X: The rows to explain, shape (n, n_columns).
        y: Their true targets, shape (n,).
        named: Whether the caller named the columns; see :func:`_table`.
        grid: The grid on which the CRPS and the NLL are computed.
        baseline: The baseline ``b``, checked, shape (n_columns,), or
            ``None`` for the mean of ``X_context``.
        n_steps: The steps ``m``; 1 for ``"gradient"``.

    Returns:
        A tuple (values, full, base): each column's attribution, shape
        (n_statistics, n, n_columns); the statistics of the rows, shape
        (n_statistics, n); and, for ``"integrated_gradients"``, those at
        the baseline, of that shape, or ``None`` for ``"gradient"``.

    Raises:
        sklearn.exceptions.NotFittedError: If the model is not fitted.
        ValueError: As :func:`_differentiable` and :func:`_density` raise,
            if ``X_context`` and ``y_context`` are not the model's
            context, or if torch's inference mode is on.
    """
    model, order, context, queries = _differentiable(
        model, method, X_context, X, named
    )
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    differentiable.refuse_inference_mode(f"method={method!r}")
    origin = context.mean(axis=0) if baseline is None else baseline[order]
    context_tensor = torch.tensor(context)
    delta = queries - origin
    fractions = np.arange(1, n_steps + 1) / n_steps
    chunk = max(1, _QUERY_BUDGET // n_steps)
    gradients = np.empty((len(STATISTICS), *queries.shape))
    full = np.empty((len(STATISTICS), len(queries)))
    # Inside torch.no_grad() too, which would build no graph.
    with torch.enable_grad():
        for start in range(0, len(queries), chunk):
            rows = slice(start, start + chunk)
            path = origin + fractions[:, None, None] * delta[rows]  # (m, c, p)
            query = torch.tensor(
                path.reshape(-1, path.shape[-1]), requires_grad=True
            )
            density = differentiable.density(
                model, f"method={method!r}", query, context_tensor, y_context
            )
            statistics = _scores(density, np.tile(y[rows], n_steps), grid)
            # The last step, k = m, is the row itself.
            at_points = torch.stack(statistics).detach().numpy()
            full[:, rows] = at_points.reshape(len(STATISTICS), n_steps, -1)[
                :, -1
            ]
            # The three backward passes share one forward pass's graph,
            # which the last of them frees.
            for index, statistic in enumerate(statistics):
                (gradient,) = torch.autograd.grad(
                    statistic.sum(),
                    query,
                    retain_graph=index < len(statistics) - 1,
                )
                gradients[index, rows] = (
                    gradient.numpy().reshape(path.shape).sum(axis=0)
                )
            # After the backward passes have freed this batch's graph.
            _check_context(
                model, method, X.iloc[rows], named, y_context, full[-1, rows]
            )
    values = (gradients / n_steps * delta)[..., np.argsort(order)]
    if method == "gradient":
        return values, full, None
    # The baseline is one row; each explained row scores it at its target.
    at_baseline = pd.DataFrame([origin[np.argsort(order)]], columns=X.columns)
    distribution = model.predict_distribution(_table(at_baseline, named))
    base = _scores(distribution[np.zeros(len(y), dtype=int)], y, grid)
    return values, full, np.stack(base)


def _context_gradients(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    X_context: pd.DataFrame,
    y_context: _typing.FloatArray,
    X: pd.DataFrame,
    y: _typing.FloatArray,
    named: bool,
    grid: grid_lib.Grid,
    baseline: _typing.FloatArray | None,
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Gradient times input of every statistic in the context's features.

    Context row ``r`` gets the sum over its columns ``j`` of the
    statistic's derivative in ``x_rj`` times ``x_rj - b_j``. Every row's
    statistics depend on the whole context, so each statistic of each row
    takes a backward pass of its own, all through one forward pass per
    batch of ``_QUERY_BUDGET`` rows, which keeps one graph in memory.

    Args:
        model: A model fitted on ``X_context`` and ``y_context``.
        X_context: Its context, shape (n_context, n_columns).
        y_context: Its targets, shape (n_context,).
        X: The rows to explain, shape (n, n_columns).
        y: Their true targets, shape (n,).
        named: Whether the caller named the columns; see :func:`_table`.
        grid: The grid on which the CRPS and the NLL are computed.
        baseline: The baseline ``b``, checked, shape (n_columns,), or
            ``None`` for the mean of ``X_context``.

    Returns:
        A tuple (values, full): each context row's attribution, shape
        (n_statistics, n, n_context), and the statistics of the rows, shape
        (n_statistics, n).

    Raises:
        sklearn.exceptions.NotFittedError: If the model is not fitted.
        ValueError: As :func:`_differentiable` and :func:`_density` raise,
            if ``X_context`` and ``y_context`` are not the model's
            context, or if torch's inference mode is on.
    """
    model, order, context, queries = _differentiable(
        model, "gradient", X_context, X, named
    )
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    differentiable.refuse_inference_mode("method='gradient'")
    origin = context.mean(axis=0) if baseline is None else baseline[order]
    offsets = torch.tensor(context - origin)
    context_tensor = torch.tensor(context, requires_grad=True)
    values = np.empty((len(STATISTICS), len(queries), len(context)))
    full = np.empty((len(STATISTICS), len(queries)))
    # Inside torch.no_grad() too, which would build no graph.
    with torch.enable_grad():
        for start in range(0, len(queries), _QUERY_BUDGET):
            rows = slice(start, start + _QUERY_BUDGET)
            query = torch.tensor(queries[rows])
            density = differentiable.density(
                model, "method='gradient'", query, context_tensor, y_context
            )
            statistics = torch.stack(_scores(density, y[rows], grid))
            full[:, rows] = statistics.detach().numpy()
            # One backward pass per statistic of a row; the last frees the
            # graph.
            flat = statistics.reshape(-1)
            for index, statistic in enumerate(flat):
                (gradient,) = torch.autograd.grad(
                    statistic,
                    context_tensor,
                    retain_graph=index < len(flat) - 1,
                )
                which, row = divmod(index, len(query))
                values[which, start + row] = (
                    (gradient * offsets).sum(dim=1).numpy()
                )
            # After the backward passes have freed this batch's graph.
            _check_context(
                model,
                "gradient",
                X.iloc[rows],
                named,
                y_context,
                full[-1, rows],
            )
    return values, full


def _differentiable(  # noqa: GS030 - scikit-learn's X.
    model: Any,
    method: str,
    X_context: pd.DataFrame,
    X: pd.DataFrame,
    named: bool,
) -> tuple[Any, _typing.IntArray, _typing.FloatArray, _typing.FloatArray]:
    """The backend whose forward pass a gradient method runs, and its inputs.

    Args:
        model: A model fitted on ``X_context``.
        method: The gradient method, for the errors.
        X_context: Its context, shape (n_context, n_columns).
        X: The rows to explain, shape (n, n_columns).
        named: Whether the caller named the columns; see :func:`_table`.

    Returns:
        A tuple (backend, order, context, queries): the backend; the
        position in ``X`` of each of the fit's columns, shape (n_columns,);
        and the context and the rows as floats in the fit's column order,
        shapes (n_context, n_columns) and (n, n_columns).

    Raises:
        sklearn.exceptions.NotFittedError: If the model is not fitted.
        ValueError: If the model has no differentiable forward pass, a
            column is not numeric, a feature is missing, or the columns do
            not match the fit's.
    """
    validation.check_is_fitted(model)
    model = differentiable.backend(model, f"method={method!r}")
    # Where each of the fit's columns is in X, by predict's rule for names.
    positions = pd.DataFrame([np.arange(X.shape[1])], columns=X.columns)
    order = model._align_to_fit(positions, named).to_numpy()[0]  # noqa: SLF001 - predict's rule.
    context = _float_features(X_context, "X_context", method)[:, order]
    queries = _float_features(X, "X", method)[:, order]
    return model, order, context, queries


def _check_context(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    method: str,
    X: pd.DataFrame,
    named: bool,
    y_context: _typing.FloatArray,
    means: _typing.FloatArray,
) -> None:
    """Refuses a context other than the one the model was fitted on.

    The forward pass's context is the caller's; predict's is the fit's.
    Their means agree to float32 rounding, far inside this tolerance, when
    they are the same context.

    Args:
        model: The backend.
        method: The gradient method, for the errors.
        X: The rows explained, shape (n, n_columns).
        named: Whether the caller named the columns; see :func:`_table`.
        y_context: The caller's context targets, shape (n_context,).
        means: The rows' means from the forward pass on the caller's
            context, shape (n,).

    Raises:
        ValueError: If predict's means differ from ``means``.
    """
    predicted = model.predict_distribution(_table(X, named)).mean()
    tolerance = 1e-3 * (np.std(y_context) + np.abs(np.mean(y_context)))
    if not np.allclose(means, predicted, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"method={method!r} needs X_context and y_context to be the"
            " context the model was fitted on: its forward pass on them"
            " predicts means that differ from predict's by up to"
            f" {np.abs(means - predicted).max():.3g}"
        )


def _check_baseline(
    baseline: npt.ArrayLike | None, n_columns: int
) -> _typing.FloatArray | None:
    """``baseline`` as a float array, checked; ``None`` stays ``None``.

    Args:
        baseline: The baseline as given.
        n_columns: The number of columns of ``X``.

    Returns:
        The baseline, float64, shape (n_columns,), or ``None``.

    Raises:
        ValueError: If it is not a finite row of ``n_columns`` numbers.
    """
    if baseline is None:
        return None
    try:
        row = np.asarray(baseline, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"baseline must be numbers: {error}") from error
    if row.shape != (n_columns,) or not np.isfinite(row).all():
        raise ValueError(
            f"baseline must be finite, of shape ({n_columns},): got {row}"
        )
    return row


def _float_features(
    frame: pd.DataFrame, name: str, method: str
) -> _typing.FloatArray:
    """A table's values, which gradients need numeric and present.

    Args:
        frame: Features, shape (n, p).
        name: The argument's name, for the errors.
        method: The gradient method, for the errors.

    Returns:
        The values, float64, shape (n, p).

    Raises:
        ValueError: If a column is not numeric, or a value is missing or
            infinite.
    """
    other = frame.select_dtypes(exclude="number").columns
    if len(other):
        raise ValueError(
            f"method={method!r} needs numeric columns; {name} has others:"
            f" {list(other)}"
        )
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            f"method={method!r} needs {name} without missing or infinite values"
        )
    return values
