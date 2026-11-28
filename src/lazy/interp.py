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

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base as sklearn_base
from sklearn.utils import validation

from lazy import _inputs
from lazy import _paths
from lazy import _steps
from lazy import _typing
from lazy import _warnings
from lazy import base
from lazy import grid as grid_lib
from lazy import metrics
from lazy import pipeline
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
    "Barrier",
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

Barrier = _steps.Barrier
"""A step that a gradient cannot cross, with the reason; see
:attr:`Explanation.barriers`."""

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
        stage: The step of a pipeline whose output columns are the
            features; ``None`` when they are the model's raw input.
        barriers: For a gradient method without ``at``, the steps that the
            gradient could not cross, each with the reason: the features
            are the outputs of the last of them, and of each encoder among
            them.
        item_values: For a gradient method, each attribution to a column
            of items spread over the items' elements, such as the pixels
            of an image: the column's name -> an array of shape
            (n_statistics, n_rows, ...) that sums to the column's
            ``values`` over its trailing axes. ``None`` if no feature is a
            column of items.
    """

    method: str
    statistics: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: _typing.FloatArray
    full_values: _typing.FloatArray
    base_values: _typing.FloatArray | None
    stage: str | None = None
    barriers: tuple[Barrier, ...] = ()
    item_values: Mapping[str, _typing.FloatArray] | None = None

    @property
    def sum_error(self) -> _typing.FloatArray | None:
        """How far each row's values are from summing to what they should.

        The values of ``"shapley"`` and of ``"integrated_gradients"`` sum
        over the features to ``full_values - base_values``: exactly for
        Shapley values, and to the error of the Riemann sum for integrated
        gradients, which more ``n_steps`` lower.

        Returns:
            ``values.sum(axis=-1) - (full_values - base_values)``, shape
            (n_statistics, n_rows), or ``None`` for a method without
            ``base_values``.
        """
        if self.base_values is None:
            return None
        gained = self.full_values - self.base_values
        return np.asarray(self.values.sum(axis=-1) - gained)

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

    def top(  # noqa: GS030 - scikit-learn's X.
        self, k: int, statistic: str = "crps", X_context: Any = None
    ) -> pd.DataFrame:
        """Each explained row's most influential context rows.

        The most influential have the largest absolute attribution.

        Args:
            k: How many context rows to give per explained row; all of them
                if there are fewer.
            statistic: One of ``statistics``.
            X_context: The context that was explained, to show each context
                row's features beside its attribution; ``None`` for none.

        Returns:
            A table with one row per explained row and rank, shape
            (n_rows * k, 4), sorted by both. Its columns are ``row``, the
            explained row's position in ``X``; ``rank``, from 1 for the most
            influential; ``context_row``, its position in ``X_context``; and
            ``value``, its attribution. With ``X_context``, the columns of
            that context row follow, under the context's column names, or
            ``x0`` onwards for a table without names.

        Raises:
            ValueError: If ``statistic`` is not one of ``statistics``,
                ``k`` is below 1, or ``X_context`` has fewer rows than the
                context rows explained need.
            TypeError: If ``k`` is not an integer.
        """
        values = self.values[_statistic_index(self.statistics, statistic)]
        validation.check_scalar(k, "k", numbers.Integral, min_val=1)
        k = min(int(k), values.shape[1])
        order = np.argsort(-np.abs(values), axis=1, kind="stable")[:, :k]
        n_rows = len(values)
        table = pd.DataFrame(
            {
                "row": np.repeat(np.arange(n_rows), k),
                "rank": np.tile(np.arange(1, k + 1), n_rows),
                "context_row": self.context_rows[order].ravel(),
                "value": np.take_along_axis(values, order, axis=1).ravel(),
            }
        )
        if X_context is None:
            return table
        context, _ = _frame(X_context)
        if len(context) <= self.context_rows.max():
            raise ValueError(
                f"X_context has {len(context)} rows, but context row"
                f" {self.context_rows.max()} was explained: give the context"
                " that explain_context was given"
            )
        features = context.iloc[table["context_row"]].reset_index(drop=True)
        return pd.concat([table, features], axis=1)


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
    at: str | None = None,
    random_state: int = 0,
) -> Explanation:
    """Attributes each row's predicted distribution to the features.

    The features are the columns of ``X``, unless ``at`` names a step of a
    pipeline: they are then that step's output columns, the *stage*.

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
    runs the pass once more per row, at the baseline. Both also predict
    the rows once, all together and before the first batch, to check that
    ``X_context`` is the model's context and that every step's forward
    gives what its ``transform`` gives, so that a wrong one fails after
    the first batch.
    Neither refits, and each pass's autograd graph holds one batch of
    about 64 points, or of one row's ``n_steps`` points if there are more.

    The gradient methods take the gradient as far back as it reaches
    (:mod:`lazy._steps`). Through a :class:`~lazy.pipeline.LazyPipeline`
    it passes each step that has a forward, down to the raw input; a
    ``Tabularizer`` is entered column by column, down to the items of each
    encoder that has a forward, such as the pixels of an image. A step
    that the gradient cannot cross is a barrier: attribution ends at its
    output, whose columns become the features, with a warning that names
    the step and the reason, and ``Explanation.barriers`` lists them.
    Through a :class:`~lazy.models.ensemble.LazyEnsembleModel` that pools
    densities, one backward pass from the pooled density gives each
    member's path its share, and a feature's attribution is the sum over
    the members.

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
            backend, a pipeline that ends in one or an ensemble of them,
            on numeric columns, with no transform or clip that the
            backend's own package applies (``transforms=("none",)`` and
            ``outlier_threshold=None`` always qualify).
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
            of ``X_context``, or of the context's stage or items. A row can
            be given only when the features are the columns of one table.
        at: The name of a step of a :class:`~lazy.pipeline.LazyPipeline`
            before its model: the features are that step's output columns,
            named as the table it returns names them, or ``x0`` onwards
            for a step that returns an array. A removal method runs
            the pipeline to that stage, removes or permutes the stage's
            columns, then runs the rest; a gradient method takes the
            gradient in them, and every later step needs a forward.
            ``None`` for the raw input, or, for a gradient method, as far
            back as the gradient reaches.
        random_state: The seed of the permutations, the shuffles and the
            column of noise that stands in for no feature at all.

    Returns:
        The attributions.

    Raises:
        ValueError: If an argument is not as described above, an option
            is given to a method that does not use it, a gradient
            method's model has no differentiable forward pass, or ``at``
            leaves a barrier between the stage and the model.
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
    if at is not None and method not in _GRADIENT_METHODS:
        model, X, X_context = _staged(
            model, at, X, X_context, y_context, refit=method != "permutation"
        )
    X, named = _frame(X)
    if X.empty:
        raise ValueError("X has no rows to explain")
    y = _inputs.check_target(y, len(X))
    rng = np.random.default_rng(random_state)

    if method in _GRADIENT_METHODS:
        baseline = _check_baseline(baseline)
        context, context_targets, grid = _context(
            X_context, y_context, X, y, y_grid
        )
        n_steps = (
            _count(n_steps, "n_steps", DEFAULT_STEPS)
            if method == "integrated_gradients"
            else 1
        )
        return _gradient_explanation(
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
            at,
            groups,
        )
    names, feature_of = _features(X.columns, groups)
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
        return Explanation(method, STATISTICS, names, values, full, None, at)

    context, context_targets, grid = _context(
        X_context, y_context, X, y, y_grid
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
        return Explanation(
            method, STATISTICS, names, values, table[0], None, at
        )
    if n_permutations is None and n_features <= EXACT_MAX_FEATURES:
        values, full, empty = _exact_shapley(value_function, n_features)
    else:
        n_permutations = _count(
            n_permutations, "n_permutations", DEFAULT_PERMUTATIONS, even=True
        )
        values, full, empty = _sampled_shapley(
            value_function, n_features, n_permutations, rng
        )
    return Explanation(method, STATISTICS, names, values, full, empty, at)


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
    at: str | None = None,
    through_encoders: bool | None = None,
) -> ContextExplanation:
    """Attributes each row's predicted distribution to the context rows.

    ``"gradient"`` gives context row ``r`` gradient times input in its
    features, summed over them: the derivative of an explained row's
    statistic in each feature of ``r``, times that feature's offset from
    the baseline. It runs the model's differentiable forward pass once for
    each batch of 64 rows, and one backward pass through it for each
    statistic of each row, ``3 * n_rows`` in all, each about as costly as
    the forward pass; it also predicts the rows once, all together,
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
        at: For ``"gradient"`` on a :class:`~lazy.pipeline.LazyPipeline`,
            the name of the step in whose output columns each context
            row's gradient is taken; ``None`` for as far back as the
            gradient reaches.
        through_encoders: For ``"gradient"``, whether the gradient goes
            through the encoders of a ``Tabularizer`` to the context's
            items, which encodes the whole context on the autograd graph
            once per batch; ``None`` or ``False`` stops at the encoders'
            outputs.

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
        method,
        baseline=baseline,
        rows=rows,
        random_state=random_state,
        at=at,
        through_encoders=through_encoders,
    )
    baseline = _check_baseline(baseline)
    X, named = _frame(X)
    if X.empty:
        raise ValueError("X has no rows to explain")
    y = _inputs.check_target(y, len(X))
    context, context_targets, grid = _context(
        X_context, y_context, X, y, y_grid
    )
    if method == "gradient":
        chosen = np.arange(len(context))
        values, full = _context_gradients(
            model,
            context,
            context_targets,
            X,
            y,
            named,
            grid,
            baseline,
            at,
            bool(through_encoders),
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
        "gradient": ("baseline", "at", "through_encoders"),
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
) -> tuple[Any, ...]:
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
    scores = STATISTICS[:-1]
    _, terms = metrics._row_terms(y, grid, distribution.on_grid(grid), scores)  # noqa: SLF001 - the metric registry, for torch too.
    return *(-terms[name] for name in scores), distribution.mean()


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


def _staged(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    at: str,
    X: Any,
    X_context: Any,
    y_context: Any,
    refit: bool,
) -> tuple[Any, Any, Any]:
    """A pipeline cut after one of its steps, for a removal method.

    Args:
        model: A :class:`~lazy.pipeline.LazyPipeline`.
        at: The name of a step before its model.
        X: The rows to explain, as the pipeline takes them.
        X_context: The context, in the same form, or ``None``.
        y_context: Its targets, or ``None``.
        refit: Whether the method refits: the steps up to ``at`` are then
            cloned and fitted on the context, and the rest is cloned.

    Returns:
        A tuple (rest, X, X_context): the steps after ``at`` as a
        pipeline, fitted unless ``refit``; and the rows and the context as
        the step ``at`` gives them, the context ``None`` unless ``refit``.

    Raises:
        ValueError: If ``model`` is not a pipeline, or ``at`` names no step
            before its model.
        sklearn.exceptions.NotFittedError: If the pipeline must be fitted
            and is not.
    """
    if not isinstance(model, pipeline.LazyPipeline):
        raise _paths.no_steps(model, at)
    cut = _paths.position(at, [name for name, _ in model.steps[:-1]]) + 1
    head, rest = model[:cut], model[cut:]
    if not refit:
        validation.check_is_fitted(model)
        return rest, head.transform(X), None
    if any(
        getattr(step, "trains_through_model", False) for _, step in head.steps
    ):
        # Such a step is fitted through the model, by the whole pipeline.
        head = sklearn_base.clone(model).fit(X_context, y_context)[:cut]
        context = head.transform(X_context)
    else:
        head = sklearn_base.clone(head)
        context = head.fit_transform(X_context, y_context)
    return sklearn_base.clone(rest), head.transform(X), context


def _gradient_explanation(  # noqa: GS030 - scikit-learn's X, y.
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
    at: str | None,
    groups: Mapping[str, Sequence[str]] | None,
) -> Explanation:
    """Gradient times input, or integrated gradients, of every statistic.

    Element ``j`` of a leaf ``x`` of a row gets ``x_j - b_j`` times the
    mean of the statistic's derivative in ``x_j`` at ``b + (k / m) (x - b)``
    for ``k = 1, ..., m``: at ``x`` alone for ``"gradient"``, whose ``m`` is
    1. Every leaf moves along its own line at once. Each point's statistics
    depend only on its own row, so the gradient of their sum over a batch
    of points is each point's gradient. A batch holds the ``m`` points of
    each of about ``_QUERY_BUDGET / m`` rows.

    Args:
        model: A model fitted on ``X_context`` and ``y_context``.
        method: ``"gradient"`` or ``"integrated_gradients"``.
        X_context: Its context, shape (n_context, n_columns).
        y_context: Its targets, shape (n_context,).
        X: The rows to explain, shape (n, n_columns).
        y: Their true targets, shape (n,).
        named: Whether the caller named the columns; see :func:`_table`.
        grid: The grid on which the CRPS and the NLL are computed.
        baseline: The baseline ``b``, checked, or ``None`` for the mean of
            the context's leaves.
        n_steps: The steps ``m``; 1 for ``"gradient"``.
        at: The stage to start at, or ``None``; see :func:`explain`.
        groups: Features to treat as one each, or ``None``.

    Returns:
        The explanation. Its ``base_values`` are the statistics at the
        baseline for ``"integrated_gradients"``, and ``None`` for
        ``"gradient"``.

    Raises:
        sklearn.exceptions.NotFittedError: If the model is not fitted.
        ValueError: As :func:`lazy._paths.build` raises, if ``X_context``
            and ``y_context`` are not the model's context, if the baseline
            is not valid, or if torch's inference mode is on.
    """
    label = f"method={method!r}"
    path = _paths.build(
        model,
        _table(X, named),
        _table(X_context, named),
        y_context,
        label=label,
        at=at,
    )
    _warn_of_barriers(path, label)
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    differentiable.refuse_inference_mode(label)
    origins = _origins(path.leaves, baseline)
    columns: dict[str, None] = {}
    for leaf in path.leaves:
        columns.update(dict.fromkeys(leaf.names))
    # Before any pass: a wrong groups is refused at once.
    names, feature_of = _features(pd.Index(list(columns)), groups)
    deltas = [
        leaf.queries - origin
        for leaf, origin in zip(path.leaves, origins, strict=True)
    ]
    fractions = np.arange(1, n_steps + 1) / n_steps
    chunk = max(1, _QUERY_BUDGET // n_steps)
    gradients = [np.zeros((len(STATISTICS), *delta.shape)) for delta in deltas]
    full = np.empty((len(STATISTICS), len(y)))
    base = np.empty_like(full)
    predicted = model.predict_distribution(_table(X, named)).mean()
    # Inside torch.no_grad() too, which would build no graph.
    with torch.enable_grad():
        with torch.no_grad():
            context = path.table(
                [torch.tensor(leaf.context) for leaf in path.leaves]
            )
        for start in range(0, len(y), chunk):
            rows = slice(start, start + chunk)
            points = []
            for origin, delta in zip(origins, deltas, strict=True):
                steps = fractions.reshape(-1, *[1] * delta.ndim) * delta[rows]
                line = (origin + steps).reshape(-1, *delta.shape[1:])
                points.append(torch.from_numpy(line).requires_grad_())
            density = path.density(path.table(points), context)
            statistics = _scores(density, np.tile(y[rows], n_steps), grid)
            # The last step, k = m, is the row itself.
            at_points = torch.stack(statistics).detach().numpy()
            full[:, rows] = at_points.reshape(len(STATISTICS), n_steps, -1)[
                :, -1
            ]
            # The three backward passes share one forward pass's graph,
            # which the last of them frees.
            for index, statistic in enumerate(statistics):
                found = torch.autograd.grad(
                    statistic.sum(),
                    points,
                    retain_graph=index < len(statistics) - 1,
                    allow_unused=True,
                )
                for gradient, delta, held in zip(
                    found, deltas, gradients, strict=True
                ):
                    if gradient is not None:
                        held[index, rows] = (
                            gradient.numpy()
                            .reshape(n_steps, -1, *delta.shape[1:])
                            .sum(axis=0)
                        )
            _check_context(method, y_context, full[-1, rows], predicted[rows])
        if method == "integrated_gradients":
            # The baseline is one point; each row scores it at its own
            # target.
            for start in range(0, len(y), _QUERY_BUDGET):
                rows = slice(start, start + _QUERY_BUDGET)
                with torch.no_grad():
                    at_origin = path.table(
                        [
                            torch.tensor(origin).expand(
                                len(y[rows]), *origin.shape
                            )
                            for origin in origins
                        ]
                    )
                    scores = _scores(
                        path.density(at_origin, context), y[rows], grid
                    )
                base[:, rows] = torch.stack(scores).numpy()
    for held, delta in zip(gradients, deltas, strict=True):
        held /= n_steps
        held *= delta
    _, values, item_values = _by_feature(path.leaves, gradients)
    # A feature's attribution is the sum of its columns'.
    one_hot = feature_of[:, None] == np.arange(len(names))
    return Explanation(
        method,
        STATISTICS,
        names,
        values @ one_hot,
        full,
        base if method == "integrated_gradients" else None,
        path.stage,
        path.barriers,
        item_values or None,
    )


def _warn_of_barriers(path: _paths.Path, label: str) -> None:
    """Warns of each step that a gradient could not cross."""
    for barrier in path.barriers:
        _warnings.warn(
            f"{label} stops at the output of the step {barrier.step!r}, a"
            f" barrier to gradients: {barrier.reason}",
            _warnings.LazyWarning,
        )


def _check_baseline(
    baseline: npt.ArrayLike | None,
) -> _typing.FloatArray | None:
    """``baseline`` as a float row, checked; ``None`` stays ``None``.

    Args:
        baseline: The baseline as given.

    Returns:
        The baseline, float64, shape (n,), or ``None``.

    Raises:
        ValueError: If it is not a finite row of numbers.
    """
    if baseline is None:
        return None
    try:
        row = np.asarray(baseline, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"baseline must be numbers: {error}") from error
    if row.ndim != 1 or not np.isfinite(row).all():
        raise ValueError(
            f"baseline must be finite, one number per column: got {row}"
        )
    return row


def _origins(
    leaves: Sequence[_steps.Leaf], baseline: _typing.FloatArray | None
) -> list[_typing.FloatArray]:
    """The baseline of each leaf: the point that stands in for no feature.

    Args:
        leaves: The leaves of a path.
        baseline: A checked row of numbers, or ``None`` for the mean of
            each leaf over the context.

    Returns:
        One array per leaf, of the leaf's trailing shape.

    Raises:
        ValueError: If a baseline is given and the leaves are not one
            table of its width.
    """
    if baseline is None:
        return [leaf.context.mean(axis=0) for leaf in leaves]
    # The members of an ensemble each have a leaf of the same table.
    one_table = all(
        leaf.per_column and leaf.names == leaves[0].names for leaf in leaves
    )
    if not one_table:
        raise ValueError(
            "baseline applies when the features are the columns of one"
            " table; here they are several blocks, so leave it as None"
        )
    n_columns = leaves[0].queries.shape[1]
    if baseline.shape != (n_columns,):
        raise ValueError(
            f"baseline must be finite, of shape ({n_columns},): got {baseline}"
        )
    return [baseline] * len(leaves)


def _by_feature(
    leaves: Sequence[_steps.Leaf], values: Sequence[_typing.FloatArray]
) -> tuple[list[str], _typing.FloatArray, dict[str, _typing.FloatArray]]:
    """Each leaf's attributions, as one column per feature.

    A leaf of a table gives a column per column. A leaf of items gives one
    column, the sum over each item's elements. Leaves that share a name, as
    the members of an ensemble do, add up.

    Args:
        leaves: The leaves of a path.
        values: Each leaf's attributions, shape (n_statistics, n_rows, ...)
            with the leaf's trailing shape.

    Returns:
        A tuple (names, values, item_values): the distinct feature names,
        in the order they first appear; their attributions, shape
        (n_statistics, n_rows, n_names); and the attributions of each
        column of items, by name, with the items' shape.
    """
    index: dict[str, int] = {}
    columns: list[_typing.FloatArray] = []
    item_values: dict[str, _typing.FloatArray] = {}
    for leaf, held in zip(leaves, values, strict=True):
        if leaf.per_column:
            parts = list(np.moveaxis(held, -1, 0))
        else:
            parts = [held.reshape(*held.shape[:2], -1).sum(axis=-1)]
            name = leaf.names[0]
            item_values[name] = (
                item_values[name] + held if name in item_values else held
            )
        for name, part in zip(leaf.names, parts, strict=True):
            position = index.setdefault(name, len(index))
            if position == len(columns):
                columns.append(np.zeros_like(part))
            columns[position] = columns[position] + part
    return list(index), np.stack(columns, axis=-1), item_values


def _context_gradients(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    X_context: pd.DataFrame,
    y_context: _typing.FloatArray,
    X: pd.DataFrame,
    y: _typing.FloatArray,
    named: bool,
    grid: grid_lib.Grid,
    baseline: _typing.FloatArray | None,
    at: str | None,
    through_encoders: bool,
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Gradient times input of every statistic in the context's leaves.

    Context row ``r`` gets the sum over the elements ``j`` of its leaves of
    the statistic's derivative in ``x_rj`` times ``x_rj - b_j``. Every
    row's statistics depend on the whole context, so each statistic of each
    row takes a backward pass of its own, all through one forward pass per
    batch of ``_QUERY_BUDGET`` rows, which keeps one graph in memory.

    Args:
        model: A model fitted on ``X_context`` and ``y_context``.
        X_context: Its context, shape (n_context, n_columns).
        y_context: Its targets, shape (n_context,).
        X: The rows to explain, shape (n, n_columns).
        y: Their true targets, shape (n,).
        named: Whether the caller named the columns; see :func:`_table`.
        grid: The grid on which the CRPS and the NLL are computed.
        baseline: The baseline ``b``, checked, or ``None`` for the mean of
            the context's leaves.
        at: The stage to differentiate in, or ``None``; see
            :func:`explain_context`.
        through_encoders: Whether the gradient goes through the encoders
            of a ``Tabularizer`` to the context's items.

    Returns:
        A tuple (values, full): each context row's attribution, shape
        (n_statistics, n, n_context), and the statistics of the rows, shape
        (n_statistics, n).

    Raises:
        sklearn.exceptions.NotFittedError: If the model is not fitted.
        ValueError: As :func:`lazy._paths.build` raises, if ``X_context``
            and ``y_context`` are not the model's context, if the baseline
            is not valid, or if torch's inference mode is on.
    """
    label = "method='gradient'"
    path = _paths.build(
        model,
        _table(X, named),
        _table(X_context, named),
        y_context,
        label=label,
        at=at,
        items=through_encoders,
    )
    _warn_of_barriers(path, label)
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    differentiable.refuse_inference_mode(label)
    origins = _origins(path.leaves, baseline)
    offsets = [
        torch.tensor(leaf.context - origin)
        for leaf, origin in zip(path.leaves, origins, strict=True)
    ]
    contexts = [
        torch.tensor(leaf.context, requires_grad=True) for leaf in path.leaves
    ]
    n_context = len(y_context)
    values = np.zeros((len(STATISTICS), len(y), n_context))
    full = np.empty((len(STATISTICS), len(y)))
    predicted = model.predict_distribution(_table(X, named)).mean()
    # Inside torch.no_grad() too, which would build no graph.
    with torch.enable_grad():
        for start in range(0, len(y), _QUERY_BUDGET):
            rows = slice(start, start + _QUERY_BUDGET)
            with torch.no_grad():
                query = path.table(
                    [torch.tensor(leaf.queries[rows]) for leaf in path.leaves]
                )
            density = path.density(query, path.table(contexts))
            statistics = torch.stack(_scores(density, y[rows], grid))
            full[:, rows] = statistics.detach().numpy()
            # One backward pass per statistic of a row; the last frees the
            # graph.
            flat = statistics.reshape(-1)
            n_queries = statistics.shape[1]
            for index, statistic in enumerate(flat):
                found = torch.autograd.grad(
                    statistic,
                    contexts,
                    retain_graph=index < len(flat) - 1,
                    allow_unused=True,
                )
                which, row = divmod(index, n_queries)
                for gradient, offset in zip(found, offsets, strict=True):
                    if gradient is not None:
                        values[which, start + row] += (
                            (gradient * offset)
                            .reshape(n_context, -1)
                            .sum(dim=1)
                            .numpy()
                        )
            _check_context(
                "gradient", y_context, full[-1, rows], predicted[rows]
            )
    return values, full


def _check_context(
    method: str,
    y_context: _typing.FloatArray,
    means: _typing.FloatArray,
    predicted: _typing.FloatArray,
) -> None:
    """Refuses a pass that does not predict what the fitted model predicts.

    The forward pass's context is the caller's; predict's is the fit's.
    Their means agree to float32 rounding, far inside this tolerance, when
    they are the same context and every step's forward gives what its
    ``transform`` gives.

    Args:
        method: The gradient method, for the errors.
        y_context: The caller's context targets, shape (n_context,).
        means: Some rows' means from the forward pass on the caller's
            context, shape (n,).
        predicted: Their means as the fitted model predicts them, shape
            (n,).

    Raises:
        ValueError: If ``predicted`` differs from ``means``.
    """
    tolerance = 1e-3 * (np.std(y_context) + np.abs(np.mean(y_context)))
    if not np.allclose(means, predicted, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"method={method!r} needs X_context and y_context to be the"
            " context the model was fitted on: its forward pass on them"
            " predicts means that differ from predict's by up to"
            f" {np.abs(means - predicted).max():.3g}"
        )
