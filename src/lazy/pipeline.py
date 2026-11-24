# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""A scikit-learn pipeline that keeps a model's density methods.

:class:`sklearn.pipeline.Pipeline` chains transformers before an estimator.
It has none of a model's methods beyond ``predict``, ``predict_proba`` and
``score``, takes their ``method`` and ``y_grid`` by name only, and its
``score`` drops ``y_grid``. :class:`LazyPipeline` defines every prediction
method of :class:`~lazy.base.BaseDensityRegressor` with the model's own
signature, so that a model behind preprocessing steps answers as it does
alone::

    model = lazy.make_pipeline(preprocessing.StandardScaler(), LazyModel())
    model.fit(X_train, y_train)

    lo, med, hi = model.predict_quantiles(X_test, [0.16, 0.5, 0.84]).T
    model.score(X_test, y_test, grid)

Each of these methods transforms ``X`` with every step but the last and
calls the last step, the model, on the result. Everything else is
scikit-learn's: ``clone``, ``get_params``, ``set_params``, slicing, caching
with ``memory``, and the ``<step>__<parameter>`` names a search uses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy.typing as npt
import pandas as pd
from sklearn import pipeline as sklearn_pipeline
from sklearn.utils import metaestimators
from sklearn.utils import validation

from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy import metrics

__all__ = ["LazyPipeline", "make_pipeline"]

# What the steps accept as features is theirs to say: a table, or the raw
# items (images, text) that an encoder step turns into one.
_Features = Any


def _last_step_has(name: str) -> Callable[[LazyPipeline], bool]:
    """A check that a pipeline's last step has the method ``name``."""

    def check(pipe: LazyPipeline) -> bool:
        """Whether the last step of ``pipe`` has the method."""
        return hasattr(pipe.steps[-1][1], name)

    return check


class LazyPipeline(sklearn_pipeline.Pipeline):
    """A pipeline whose last step is a density model, with its methods.

    The parameters, attributes and fitting are those of
    :class:`sklearn.pipeline.Pipeline`. Every prediction method of
    :class:`~lazy.base.BaseDensityRegressor` is defined here with the
    model's own signature, and is the last step's method of the same name
    on the features that the earlier steps transformed. The last step may
    be any estimator with those methods, another ``LazyPipeline`` included;
    a method that the last step lacks is absent here too, as in
    scikit-learn. The model's attributes stay on the model:
    ``pipe[-1].grid_``.

    Each of these methods raises :class:`sklearn.exceptions.NotFittedError`
    before ``fit``. Under scikit-learn's metadata routing they pass no
    metadata to the earlier steps, so a transformer that requests metadata
    for ``transform`` needs :class:`sklearn.pipeline.Pipeline`.
    """

    def _features(self, X: _Features) -> _Features:  # noqa: GS030 - scikit-learn's X, y.
        """``X`` as the last step sees it, after every earlier step."""
        validation.check_is_fitted(self)
        for _, step in self.steps[:-1]:
            if step is not None and step != "passthrough":
                X = step.transform(X)
        return X

    @metaestimators.available_if(_last_step_has("predict"))
    def predict(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        method: str = "mode",
        y_grid: grid_lib.GridLike = None,
    ) -> _typing.FloatArray:
        """One value per row, reducing each density by ``method``.

        Args:
            X: Features as the first step takes them, n_samples rows.
            method: One of :data:`~lazy.base.POINT_ESTIMATORS`.
            y_grid: The grid to evaluate on; ``None`` for the model's
                default.

        Returns:
            Point estimates, shape (n_samples,).
        """
        return self.steps[-1][1].predict(self._features(X), method, y_grid)

    @metaestimators.available_if(_last_step_has("predict_proba"))
    def predict_proba(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Conditional densities on ``y_grid``.

        Args:
            X: Features as the first step takes them, n_samples rows.
            y_grid: The grid to evaluate on; ``None`` for the model's
                default.

        Returns:
            Normalized densities, shape (n_samples, n_bins).
        """
        return self.steps[-1][1].predict_proba(self._features(X), y_grid)

    @metaestimators.available_if(_last_step_has("predict_pdf"))
    def predict_pdf(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Alias of ``predict_proba``: densities on ``y_grid``.

        Args:
            X: Features as the first step takes them, n_samples rows.
            y_grid: The grid to evaluate on; ``None`` for the model's
                default.

        Returns:
            Normalized densities, shape (n_samples, n_bins).
        """
        return self.steps[-1][1].predict_pdf(self._features(X), y_grid)

    @metaestimators.available_if(_last_step_has("predict_cdf"))
    def predict_cdf(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Each row's cumulative distribution at the centers of ``y_grid``.

        Args:
            X: Features as the first step takes them, n_samples rows.
            y_grid: The grid whose bin centers to evaluate at; ``None`` for
                the model's default.

        Returns:
            The probability below each bin center, shape (n_samples, n_bins).
        """
        return self.steps[-1][1].predict_cdf(self._features(X), y_grid)

    @metaestimators.available_if(_last_step_has("predict_distribution"))
    def predict_distribution(self, X: _Features) -> distributions.Distribution:  # noqa: GS030 - scikit-learn's X, y.
        """The model's native per-row distributions, on no grid.

        Args:
            X: Features as the first step takes them, n_samples rows.

        Returns:
            One distribution per row of ``X``.
        """
        return self.steps[-1][1].predict_distribution(self._features(X))

    @metaestimators.available_if(_last_step_has("predict_quantiles"))
    def predict_quantiles(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        quantiles: npt.ArrayLike = (0.16, 0.5, 0.84),
    ) -> _typing.FloatArray:
        """Quantiles of each row's native distribution.

        Args:
            X: Features as the first step takes them, n_samples rows.
            quantiles: Cumulative probabilities within [0, 1], shape (k,).

        Returns:
            The value at each level, shape (n_samples, k).
        """
        return self.steps[-1][1].predict_quantiles(self._features(X), quantiles)

    @metaestimators.available_if(_last_step_has("predict_interval"))
    def predict_interval(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, coverage: float = 0.95
    ) -> _typing.FloatArray:
        """The central interval holding ``coverage`` of each row's probability.

        Args:
            X: Features as the first step takes them, n_samples rows.
            coverage: The probability inside the interval, in (0, 1).

        Returns:
            The lower and upper bounds, shape (n_samples, 2).
        """
        return self.steps[-1][1].predict_interval(self._features(X), coverage)

    @metaestimators.available_if(_last_step_has("predict_pit"))
    def predict_pit(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y: npt.ArrayLike
    ) -> _typing.FloatArray:
        """Each row's probability integral transform, ``F_i(y_i)``.

        Args:
            X: Features as the first step takes them, n_samples rows.
            y: Finite target values, shape (n_samples,).

        Returns:
            The PIT values, within [0, 1], shape (n_samples,).
        """
        return self.steps[-1][1].predict_pit(self._features(X), y)

    @metaestimators.available_if(_last_step_has("evaluate"))
    def evaluate(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        y: npt.ArrayLike,
        method: str = "mode",
        y_grid: grid_lib.GridLike = None,
        *,
        scale: metrics.Scale = "none",
    ) -> pd.DataFrame:
        """Scores predictions for ``X`` with the full diagnostic metric set.

        Args:
            X: Features as the first step takes them, n_samples rows.
            y: Finite true target values, shape (n_samples,).
            method: The point estimate to score, one of
                :data:`~lazy.base.POINT_ESTIMATORS`.
            y_grid: The grid to evaluate on; ``None`` for the model's
                default.
            scale: How the point metrics scale residuals; see
                :func:`lazy.metrics.point_metrics`.

        Returns:
            The one-row table of
            :meth:`BaseDensityRegressor.evaluate
            <lazy.base.BaseDensityRegressor.evaluate>`.
        """
        return self.steps[-1][1].evaluate(
            self._features(X), y, method, y_grid, scale=scale
        )

    @metaestimators.available_if(_last_step_has("score"))
    def score(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y: npt.ArrayLike, y_grid: grid_lib.GridLike = None
    ) -> float:
        """Negative conditional-density-estimate loss; higher is better.

        Args:
            X: Features as the first step takes them, n_samples rows.
            y: Finite true target values, shape (n_samples,).
            y_grid: The grid to score on; ``None`` for the model's default.

        Returns:
            Minus :func:`lazy.metrics.cde_loss`.
        """
        return self.steps[-1][1].score(self._features(X), y, y_grid)


def make_pipeline(
    *steps: Any,
    memory: Any = None,
    transform_input: list[str] | None = None,
    verbose: bool = False,
) -> LazyPipeline:
    """A :class:`LazyPipeline` of ``steps``, each named after its class.

    The counterpart of :func:`sklearn.pipeline.make_pipeline`, which names
    the steps the same way.

    Args:
        *steps: The transformers, in order, and then the model.
        memory: Where to cache the fitted transformers; see
            :class:`sklearn.pipeline.Pipeline`.
        transform_input: Names of the ``fit`` metadata that the earlier
            steps transform before the model gets them; see
            :class:`sklearn.pipeline.Pipeline`.
        verbose: Whether to print the time each step takes to fit.

    Returns:
        The unfitted pipeline.
    """
    named = sklearn_pipeline.make_pipeline(*steps).steps
    return LazyPipeline(
        named, transform_input=transform_input, memory=memory, verbose=verbose
    )
