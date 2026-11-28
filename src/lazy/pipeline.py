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

``fit`` adds one thing to scikit-learn's. A step that must train through
the model after it, which a scikit-learn step never sees, declares
``trains_through_model`` and implements ``_fit_through(X, y, later,
model)``. :meth:`LazyPipeline.fit` fits the steps, calls the hook on the
context as the step takes it, then fits the later steps again and the
model on what the trained step gives. Unless ``fit`` is given parameters,
the model is fitted only then, not also before the training. The adaptive
reducer of :class:`~lazy.encoders.Tabularizer` is such a step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base as sklearn_base
from sklearn import pipeline as sklearn_pipeline
from sklearn.utils import metadata_routing
from sklearn.utils import metaestimators
from sklearn.utils import validation

from lazy import _inputs
from lazy import _typing
from lazy import _warnings
from lazy import distributions
from lazy import grid as grid_lib
from lazy import metrics

__all__ = ["LazyPipeline", "make_pipeline"]

# The methods of a pipeline that predict; LazyPipeline routes no metadata
# through them.
_PREDICTING = (
    "decision_function",
    "predict",
    "predict_log_proba",
    "predict_proba",
    "score",
)

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
    scikit-learn. The model's attributes stay on the model,
    ``pipe[-1].grid_``, except two that describe the whole pipeline:
    :attr:`name_` and :attr:`provenance_`.

    Each of these methods raises :class:`sklearn.exceptions.NotFittedError`
    before ``fit``. Under scikit-learn's metadata routing they take no
    metadata (see :meth:`get_metadata_routing`).
    """

    def fit(self, X: _Features, y: Any = None, **params: Any) -> LazyPipeline:  # noqa: GS030 - scikit-learn's X, y.
        """Fits every step, then each step that trains through the model.

        If a step trains through the model, ``y`` is checked first, and,
        unless ``params`` are given, the model is fitted once, after the
        training, on what the trained step gives.

        Args:
            X: Features as the first step takes them, n_samples rows.
            y: The targets, shape (n_samples,).
            **params: Parameters of the steps' ``fit``, as
                :meth:`sklearn.pipeline.Pipeline.fit` takes them. The fits
                that follow a step trained through the model do not get
                them again.

        Returns:
            The pipeline itself.

        Raises:
            ValueError: If a step trains through the model and ``y`` is
                not finite target values, one per row; or as such a step
                raises it: a later step with no forward, say, or a model
                with no differentiable forward pass.
        """
        trained = [
            index
            for index, (_, step) in enumerate(self.steps[:-1])
            if getattr(step, "trains_through_model", False)
        ]
        if not trained:
            return super().fit(X, y, **params)
        # As the model checks it, before any training.
        targets = _inputs.check_target(y, len(X))
        model = self.steps[-1]
        try:
            if not params:
                # The model is fitted once, after the training. Metadata
                # routed to it needs it in place.
                self.steps[-1] = (model[0], "passthrough")
            try:
                super().fit(X, y, **params)
            finally:
                self.steps[-1] = model
            for index in trained:
                later = [
                    inner
                    for _, inner in self.steps[index + 1 : -1]
                    if inner is not None and inner != "passthrough"
                ]
                table = self.steps[index][1]._fit_through(  # noqa: SLF001 - the hook of such a step.
                    self._through(X, index), targets, later, model[1]
                )
                for inner in later:
                    table = inner.fit_transform(table, targets)
            model[1].fit(table, targets)
        except Exception:
            # A failed fit leaves the pipeline unfitted: the steps from the
            # first trained one on are replaced by unfitted copies.
            self.steps[trained[0] :] = [
                (name, sklearn_base.clone(inner))
                if hasattr(inner, "get_params")
                else (name, inner)
                for name, inner in self.steps[trained[0] :]
            ]
            raise
        return self

    def get_metadata_routing(self) -> metadata_routing.MetadataRouter:
        """scikit-learn's routing, without the methods that predict.

        A search or ``cross_validate`` therefore routes metadata to ``fit``
        alone.

        Returns:
            The routing of :class:`sklearn.pipeline.Pipeline`, in which
            ``predict``, ``predict_proba``, ``score`` and the other methods
            that predict route no metadata to any step.
        """
        inherited = super().get_metadata_routing()
        router = metadata_routing.MetadataRouter(owner=inherited.owner)
        for name, pair in inherited:
            mapping = metadata_routing.MethodMapping()
            for route in pair.mapping:
                if route.caller not in _PREDICTING:
                    mapping.add(caller=route.caller, callee=route.callee)
            router.add(method_mapping=mapping, **{name: pair.router})
        return router

    @property
    def name_(self) -> str:
        """The step names and the model's ``name_``, joined by ``+``.

        For example ``"tabularizer+tabpfn:v3.5"``.
        """
        return "+".join([*self._step_names(), self.steps[-1][1].name_])

    @property
    def provenance_(self) -> dict[str, Any]:
        """The step names and the model's ``provenance_``, as a dict.

        Its keys are ``"steps"`` and ``"model"``.

        Raises:
            AttributeError: If the model has no ``provenance_``, as before
                ``fit``.
        """
        return {
            "steps": self._step_names(),
            "model": self.steps[-1][1].provenance_,
        }

    def _step_names(self) -> list[str]:
        """The names of the steps before the model."""
        return [name for name, _ in self.steps[:-1]]

    def _features(self, X: _Features) -> _Features:  # noqa: GS030 - scikit-learn's X, y.
        """``X`` as the last step sees it, after every earlier step.

        Raises:
            ValueError: If ``X`` has no rows; no step has run.
        """
        validation.check_is_fitted(self)
        if np.shape(X)[0] == 0:
            raise ValueError(
                f"Found array with 0 sample(s) (shape={np.shape(X)}) while a "
                f"minimum of 1 is required by {type(self).__name__}."
            )
        return self._through(X, len(self.steps) - 1)

    def _through(self, X: _Features, stop: int) -> _Features:  # noqa: GS030 - scikit-learn's X.
        """``X`` after the first ``stop`` steps, the skipped ones aside."""
        for _, step in self.steps[:stop]:
            if step is not None and step != "passthrough":
                X = step.transform(X)
        return X

    @metaestimators.available_if(_last_step_has("predict"))
    def predict(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        method: str = _warnings.POINT_DEFAULT,
        y_grid: grid_lib.GridLike = None,
    ) -> _typing.FloatArray:
        """One value per row, reducing each density by ``method``.

        Args:
            X: Features as the first step takes them, n_samples rows.
            method: One of :data:`~lazy.base.POINT_ESTIMATORS`. The default
                is ``"mode"`` until lazy 0.8.0, which changes it to
                ``"median"``; leaving it out warns.
            y_grid: The grid to evaluate on; ``None`` for the model's
                default.

        Returns:
            Point estimates, shape (n_samples,).

        Raises:
            ValueError: If ``X`` has no rows; no step has run.
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

        Raises:
            ValueError: If ``X`` has no rows; no step has run.
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

        Raises:
            ValueError: If ``X`` has no rows; no step has run.
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

        Raises:
            ValueError: If ``X`` has no rows; no step has run.
        """
        return self.steps[-1][1].predict_cdf(self._features(X), y_grid)

    @metaestimators.available_if(_last_step_has("predict_distribution"))
    def predict_distribution(self, X: _Features) -> distributions.Distribution:  # noqa: GS030 - scikit-learn's X, y.
        """The model's native per-row distributions, on no grid.

        Args:
            X: Features as the first step takes them, n_samples rows.

        Returns:
            One distribution per row of ``X``.

        Raises:
            ValueError: If ``X`` has no rows; no step has run.
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
            quantiles: Cumulative probabilities within [0, 1], a scalar or
                shape (k,).

        Returns:
            The value at each level, shape (n_samples, k).

        Raises:
            ValueError: If ``X`` has no rows, or ``quantiles`` is not a
                scalar or 1-D array within [0, 1]; no step has run.
        """
        validation.check_is_fitted(self)
        quantiles = _inputs.check_levels(quantiles, name="quantiles")
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

        Raises:
            ValueError: If ``X`` has no rows, or ``coverage`` does not lie
                in (0, 1); no step has run.
        """
        validation.check_is_fitted(self)
        _inputs.check_coverage(coverage, name="coverage")
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

        Raises:
            ValueError: If ``X`` has no rows; no step has run.
        """
        return self.steps[-1][1].predict_pit(self._features(X), y)

    @metaestimators.available_if(_last_step_has("evaluate"))
    def evaluate(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        y: npt.ArrayLike,
        method: str = _warnings.POINT_DEFAULT,
        y_grid: grid_lib.GridLike = None,
        *,
        scale: metrics.Scale = "none",
    ) -> pd.DataFrame:
        """Scores predictions for ``X`` with the full diagnostic metric set.

        Args:
            X: Features as the first step takes them, n_samples rows.
            y: Finite true target values, shape (n_samples,).
            method: The point estimate to score, one of
                :data:`~lazy.base.POINT_ESTIMATORS`. The default is
                ``"mode"`` until lazy 0.8.0, which changes it to
                ``"median"``; leaving it out warns.
            y_grid: The grid to evaluate on; ``None`` for the model's
                default.
            scale: How the point metrics scale residuals; see
                :func:`lazy.metrics.point_metrics`.

        Returns:
            The one-row table of
            :meth:`BaseDensityRegressor.evaluate
            <lazy.base.BaseDensityRegressor.evaluate>`, labeled with the
            model's ``name_``.

        Raises:
            ValueError: If ``X`` has no rows; no step has run.
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

        Raises:
            ValueError: If ``X`` has no rows; no step has run.
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
