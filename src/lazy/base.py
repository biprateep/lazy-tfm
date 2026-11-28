# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The estimator protocol every :mod:`lazy` model implements.

The API is scikit-learn's, with one difference: the natural output of these
models is a conditional density of the target, not a number, so
:meth:`BaseDensityRegressor.predict_proba` is the primary method and
:meth:`BaseDensityRegressor.predict` is a documented reduction of it::

    model = LazyModel("tabfm", n_estimators=4)
    model.fit(X_train, y_train)

    pdfs = model.predict_proba(X_test, y_grid)  # (n_test, n_bins) densities
    y_pred = model.predict(X_test, method="mode")  # (n_test,)
    model.score(X_test, y_test)  # negative CDE loss

``y_grid`` is an argument to the *prediction*, not to the constructor.
Nothing about fitting depends on the output binning -- these models place
their internal bins by the distribution of the context targets, and the grid
only enters at the final, exact rebinning step -- so one fitted model can
answer on as many grids as you like without refitting. The constructor still
accepts ``y_grid`` as a per-model default for when every call would pass the
same thing; a grid given at call time wins, and ``None`` at both levels means
the model's native grid (for an estimator that has none, a uniform grid over
the training targets' range).

Everything scikit-learn expects of an estimator holds: parameters are stored
verbatim by ``__init__`` and never validated there, all validation and all
real work happen in ``fit``, fitted attributes carry a trailing underscore,
and ``get_params`` / ``set_params`` / :func:`sklearn.base.clone` work
unmodified.

A subclass implements :meth:`_fit` and :meth:`_predict_pdf` and nothing else.
Both receive a :class:`pandas.DataFrame` with the columns ``fit`` was given,
and ``_predict_pdf`` is handed the :class:`~lazy.grid.Grid` to answer
on; the base class takes care of validation, grid resolution and
normalization.

Note on "fitting" an in-context model: none of the backends updates any
weights. ``fit`` stores the labeled rows that become the model's *context*,
and the cost of a prediction scales with how many there are. That is an
implementation detail of the backends, not of this interface -- a classical
estimator can implement the same protocol.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator, Sequence
import numbers
from typing import Any, ClassVar, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base as sklearn_base
from sklearn.utils import metadata_routing
from sklearn.utils import validation

from lazy import _inputs
from lazy import _typing
from lazy import _warnings
from lazy import distributions
from lazy import grid as grid_lib
from lazy import metrics

__all__ = [
    "DEFAULT_N_BINS",
    "DEFAULT_PADDING",
    "POINT_ESTIMATORS",
    "BaseDensityRegressor",
    "default_grid",
]

DEFAULT_N_BINS = 200
"""Bins of the default grid of a model without a native grid."""

DEFAULT_PADDING = 0.05
"""How far that grid extends past the training targets, as a fraction of their
range on each side.
"""

POINT_ESTIMATORS = metrics.POINT_ESTIMATORS
"""Reductions of a density to a single value.

The ``method`` argument of :meth:`BaseDensityRegressor.predict` accepts these.
``mode`` and ``peak_mean`` are the DC1 ``z_PEAK`` and ``z_WEIGHT`` definitions
(see :mod:`lazy.metrics`).
"""

# What the public methods accept as features: any table lazy._inputs reads
# (arrays, structured arrays, DataFrames, astropy Tables, to_pandas()
# objects). Tables from optional packages have no common type, hence Any.
_Features: TypeAlias = Any


class RowScores(NamedTuple):
    """Each row's scores of a model's densities, as ``_row_scores`` gives.

    Package-internal, so not in ``__all__``. Arrays have shape (n_samples,).

    Attributes:
        y: The checked true targets.
        point: The point estimates, or ``None`` if none was asked for.
        pit: The PIT values.
        terms: Each named score's terms, by name, lower is better.
    """

    y: _typing.FloatArray
    point: _typing.FloatArray | None
    pit: _typing.FloatArray
    terms: dict[str, _typing.FloatArray]


# predict, score and evaluate reduce each row's density as they go, a block
# of rows at a time, so that no more than about this many bytes of float64
# densities exist at once: all of DC1 on a 5,000-bin native grid would
# otherwise be one 15.6 GB array.
_BLOCK_BYTES = 256 * 2**20


class BaseDensityRegressor(
    sklearn_base.RegressorMixin, sklearn_base.BaseEstimator, abc.ABC
):
    """A regressor that predicts conditional densities, with a scikit-learn API.

    Subclasses must accept a ``y_grid`` parameter and pass it through
    unchanged; see the module docstring. Lazy's models also take
    ``categorical_features``; a subclass without it reads pandas'
    ``category`` columns as categorical.

    Attributes:
        backend: The registered backend name (a key of
            :data:`lazy.ESTIMATORS`), set as a class attribute by each
            backend; ``None`` on a subclass that is not one.
        y_grid: The default output grid as passed to the constructor: a
            :class:`~lazy.grid.Grid`, bin centers, or ``None``.
        grid_: The resolved default output grid, set by :meth:`fit`.
        is_fitted_: ``True`` once :meth:`fit` has run.
        categorical_features: Which columns are categorical:
            ``"from_dtype"`` (the default), for the columns of pandas'
            ``category`` type that are not ordered; ``None``, for none; or
            column names, positions or a boolean mask. A categorical column
            reaches the model as the codes 0 to n_categories - 1 of its
            categories, with NaN for a missing value or a category that
            ``fit`` did not see. A column of pandas' ordered ``category``
            type that is not marked reaches it as the same codes, read as
            numbers in the order of its categories. Integer or boolean
            columns stay numeric unless marked.
        is_categorical_: Which columns are categorical, a boolean array of
            shape (``n_features_in_``,).
        categories_: Each column's categories, in the order of its codes,
            for a column read as codes, or ``None`` for one read as it is.
        n_features_in_: Number of feature columns seen by :meth:`fit`.
        feature_names_in_: Their names, an object array of shape
            (``n_features_in_``,).
    """

    backend: str | None = None

    # Whether fit and the predict methods hand _fit and _predict_* the
    # features as float64, categorical columns as their codes. A model
    # whose members check their own features sets it False, and gets them
    # as to_frame reads them, so that items (images, text) reach a member
    # that encodes them.
    _numeric_features: ClassVar[bool] = True

    # ``y_grid`` is a call-time option, not per-sample metadata, so
    # scikit-learn's metadata routing never passes it on: a router would
    # split a grid as long as the data between folds.
    __metadata_request__predict: ClassVar[dict[str, str]] = {
        "y_grid": metadata_routing.UNUSED
    }
    __metadata_request__predict_proba: ClassVar[dict[str, str]] = {
        "y_grid": metadata_routing.UNUSED
    }
    __metadata_request__score: ClassVar[dict[str, str]] = {
        "y_grid": metadata_routing.UNUSED
    }

    # Set by each subclass's __init__, which scikit-learn requires to store
    # its parameters verbatim; declared here for the type checker only.
    y_grid: grid_lib.GridLike

    def __sklearn_tags__(self) -> sklearn_base.Tags:
        """scikit-learn's tags: a regressor that accepts NaN features.

        Returns:
            A regressor's tags (which require a target), with ``allow_nan``.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    # -- to be provided by subclasses --------------------------------------

    @abc.abstractmethod
    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        """Stores or fits whatever the backend needs; sets ``*_`` attributes.

        Args:
            X: Validated features, shape (n_samples, n_features).
            y: Finite target values, shape (n_samples,).
        """

    @abc.abstractmethod
    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        """Returns un-normalized densities on ``grid``, one row per row of X.

        Args:
            X: Validated features, with the columns ``fit`` saw, shape
                (n_samples, n_features).
            grid: The grid to answer on.

        Returns:
            Non-negative densities at the bin centers, shape
            (n_samples, grid.n_bins).
        """

    def _predict_distribution(
        self, X: pd.DataFrame
    ) -> distributions.Distribution:
        """Returns the native distributions of validated features.

        This default serves estimators that only produce densities on a
        grid: it tabulates them on the default grid as a histogram. Models
        with a native output override it.

        Args:
            X: Validated features, shape (n_samples, n_features).
        """
        grid = self.grid_
        density = grid.normalize(self._predict_pdf(X, grid))
        return distributions.HistogramDistribution(
            grid.edges, density * grid.widths
        )

    def _default_grid(self) -> grid_lib.Grid:
        """The grid to answer on when none is given; called after ``_fit``.

        The constructor's ``y_grid`` (``"native"`` meaning the model's own),
        else :data:`DEFAULT_N_BINS` bins over the training targets' range,
        widened by :data:`DEFAULT_PADDING` of it on each side. Models with a
        native grid override this to make it their default.
        """
        if isinstance(self.y_grid, str) and self.y_grid == grid_lib.NATIVE:
            return self.native_grid
        if self.y_grid is not None:
            return grid_lib.as_grid(self.y_grid)
        return default_grid(*self.y_range_)

    @property
    def native_grid(self) -> grid_lib.Grid:
        """The model's own output grid, available after ``fit``.

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted
                yet.
            AttributeError: If this estimator has no native grid.
        """
        validation.check_is_fitted(
            self,
            msg=(
                f"This {type(self).__name__} instance is not fitted yet, so "
                "it has no native grid; call 'fit' first"
            ),
        )
        native = getattr(self, "native_grid_", None)
        if native is None:
            raise AttributeError(
                f"{type(self).__name__} has no native grid; pass a grid"
            )
        return native

    # -- the public API ----------------------------------------------------

    def fit(self, X: _Features, y: npt.ArrayLike) -> BaseDensityRegressor:  # noqa: GS030 - scikit-learn's X, y.
        """Fits on labeled rows.

        A fit either succeeds or leaves the estimator unfitted: every fitted
        attribute of an earlier fit is dropped before this one starts, the
        inputs are validated in full before anything is recorded, and a fit
        that raises drops whatever it had set.

        Args:
            X: Features, shape (n_samples, n_features): a NumPy array, a
                structured or record array, a pandas DataFrame, an astropy
                Table, or anything with ``to_pandas()``. Missing values are
                NaN. Column names, when the input has them, are remembered
                and enforced at predict time. Categorical columns are those
                ``categorical_features`` names.
            y: Finite target values, shape (n_samples,), with
                n_samples >= 1.

        Returns:
            The fitted estimator itself.

        Raises:
            ImportError: If the backend's package is not installed.
            ValueError: If a column is neither numeric nor categorical, a
                categorical column has no value, or ``categorical_features``
                names a column that ``X`` does not have.
            TypeError: If ``categorical_features`` is of no accepted form.
        """
        self._clear_fitted()
        frame, named = _inputs.to_frame(X)
        self._check_not_empty(frame)
        y = _inputs.check_target(y, len(frame))
        is_categorical = _inputs.categorical_mask(
            getattr(self, "categorical_features", "from_dtype"), frame, named
        )
        categories = _inputs.fit_categories(frame, is_categorical)
        X = (
            _inputs.numeric_frame(frame, named, categories)
            if self._numeric_features
            else frame
        )
        try:
            self.is_categorical_ = is_categorical
            self.categories_ = categories
            self.n_features_in_ = int(X.shape[1])
            if named:
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.y_range_ = (float(np.min(y)), float(np.max(y)))
            self._fit(X, y)
            # After _fit: a model's native grid is known only once it has
            # seen its context.
            self.grid_ = self._default_grid()
        except BaseException:
            self._clear_fitted()
            raise
        self.is_fitted_ = True
        return self

    def predict_proba(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Conditional densities ``p(z | x)`` on ``y_grid``.

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).
            y_grid: A :class:`~lazy.grid.Grid`, an array of at least two
                evenly spaced, increasing bin centers, ``"native"`` for the
                model's own grid, or ``None`` for this model's default.

        Returns:
            Densities, shape (n_samples, n_bins). These are **densities**,
            normalized by the grid's convention (the trapezoid rule over the
            bin centers, or ``sum(p * widths) == 1`` on a histogram grid),
            which is how every metric in :mod:`lazy.metrics` integrates
            them. Multiply by ``grid.widths`` for per-bin probability masses.
            Probability outside the grid is dropped before normalizing. For
            a model with a native distribution (every registered backend)
            this is ``predict_distribution(X).on_grid(grid)``.

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        validation.check_is_fitted(self)
        grid = self._resolve_grid(y_grid)
        return self._densities(self._check_features(X), grid)

    def predict_distribution(self, X: _Features) -> distributions.Distribution:  # noqa: GS030 - scikit-learn's X, y.
        """The model's native per-row distributions, on no grid at all.

        A bar-distribution model answers with a
        :class:`~lazy.distributions.HistogramDistribution` over its own
        buckets, a quantile model with a
        :class:`~lazy.distributions.QuantileDistribution`, bagged or dithered
        members with a :class:`~lazy.distributions.MixtureDistribution`. Each
        gives exact ``pdf``, ``cdf``, ``ppf``, ``rvs``, moments and
        ``to_qp()``.

        Every row's distribution is held at once, ``8 * n_buckets`` bytes per
        row for a bar distribution. :meth:`predict_quantiles`,
        :meth:`predict_interval` and :meth:`predict_pit` hold one of the
        model's chunks of rows at a time instead.

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).

        Returns:
            One distribution per row of ``X``.

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        validation.check_is_fitted(self)
        return self._predict_distribution(self._check_features(X))

    def predict_quantiles(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        quantiles: npt.ArrayLike = (0.16, 0.5, 0.84),
    ) -> _typing.FloatArray:
        """Quantiles of each row's distribution, exactly.

        Computed on the native distribution (:meth:`predict_distribution`),
        never on a grid, so they carry the model's full resolution.

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).
            quantiles: Cumulative probabilities within [0, 1], a scalar or
                shape (k,).

        Returns:
            The value at each level, shape (n_samples, k).

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows, or ``quantiles`` is not a
                scalar or 1-D array within [0, 1]; the model has not run.
        """
        validation.check_is_fitted(self)
        quantiles = _inputs.check_levels(quantiles, name="quantiles")
        return np.concatenate(
            [
                dist.ppf(quantiles)
                for dist in self._distribution_chunks(self._check_features(X))
            ]
        )

    def predict_interval(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, coverage: float = 0.95
    ) -> _typing.FloatArray:
        """The central interval holding ``coverage`` of each row's probability.

        The bounds are the quantiles at ``(1 - coverage) / 2`` and
        ``(1 + coverage) / 2``, computed exactly, as in
        :meth:`predict_quantiles`.

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).
            coverage: The probability inside the interval, in (0, 1); 0.95
                gives the 2.5th and 97.5th percentiles.

        Returns:
            The lower and upper bounds, shape (n_samples, 2).

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        validation.check_is_fitted(self)
        _inputs.check_coverage(coverage, name="coverage")
        return np.concatenate(
            [
                dist.interval(coverage)
                for dist in self._distribution_chunks(self._check_features(X))
            ]
        )

    def predict_pit(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y: npt.ArrayLike
    ) -> _typing.FloatArray:
        """Each row's probability integral transform (PIT), exactly.

        The PIT of a row is its predicted CDF at its own target value,
        ``F_i(y_i)``. Computed on the native distribution
        (:meth:`predict_distribution`), never on a grid, as in
        :meth:`predict_quantiles`; the PIT values of a well-calibrated
        model are uniform on [0, 1] (:func:`lazy.metrics.pit_statistics`).

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).
            y: Finite target values, one per row of ``X``, shape
                (n_samples,).

        Returns:
            The PIT values, within [0, 1], shape (n_samples,).

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        validation.check_is_fitted(self)
        X = self._check_features(X)
        y = _inputs.check_target(y, len(X))
        pits, start = [], 0
        for dist in self._distribution_chunks(X):
            pits.append(dist.pit(y[start : start + len(dist)]))
            start += len(dist)
        return np.concatenate(pits)

    def predict_pdf(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Alias of :meth:`predict_proba`, for when "pdf" reads better.

        Args:
            X: Features, shape (n_samples, n_features).
            y_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            Normalized densities, shape (n_samples, n_bins).

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        return self.predict_proba(X, y_grid)

    def predict_cdf(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Each row's cumulative distribution at the centers of ``y_grid``.

        Computed on the native distribution (:meth:`predict_distribution`),
        never on a grid, as in :meth:`predict_quantiles`: column ``j`` of
        row ``i`` is ``F_i(y_grid.centers[j])``. It includes the probability
        below the grid, which :meth:`predict_proba` drops, so column 0
        need not be zero, nor the last column one. A model that predicts
        only densities answers from its default grid's histogram, so
        ``y_grid`` sets where the CDF is read, not its resolution.

        Args:
            X: Features, shape (n_samples, n_features).
            y_grid: The grid whose bin centers to evaluate at; ``None`` for
                this model's default.

        Returns:
            The probability below each bin center, within [0, 1], shape
            (n_samples, n_bins).

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        validation.check_is_fitted(self)
        centers = self._resolve_grid(y_grid).centers
        return np.concatenate(
            [
                dist.cdf(centers)
                for dist in self._distribution_chunks(self._check_features(X))
            ]
        )

    def predict(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        method: str = _warnings.POINT_DEFAULT,
        y_grid: grid_lib.GridLike = None,
    ) -> _typing.FloatArray:
        """One value per row, reducing each density by ``method``.

        The four definitions disagree exactly when a PDF is multimodal,
        which is the interesting case: ``mean`` lands between two peaks,
        where there is no probability at all, while ``mode`` and
        ``peak_mean`` pick one. If you want several, call
        :meth:`predict_proba` once and pass the result to
        :func:`lazy.metrics.grid_point_estimates` rather than re-running the
        model per definition.

        Rows are predicted and reduced a block at a time, so the densities
        of all of ``X`` never exist at once, whatever the grid.

        Args:
            X: Features, shape (n_samples, n_features).
            method: One of :data:`POINT_ESTIMATORS`: ``"mode"`` (the peak,
                DC1's ``z_PEAK``), ``"peak_mean"`` (DC1's main-peak weighted
                mean), ``"mean"`` or ``"median"``. The default is
                ``"mode"`` until lazy 0.8.0, which changes it to
                ``"median"``; leaving it out warns.
            y_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            Point estimates, shape (n_samples,).

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        validation.check_is_fitted(self)
        method = _check_method(
            _warnings.resolve_default(method, "method", "median")
        )
        grid = self._resolve_grid(y_grid)
        X = self._check_features(X)
        return np.concatenate(
            [
                metrics.grid_point_estimate(grid, density, method)
                for _, density in self._density_blocks(X, grid)
            ]
        )

    def score(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y: npt.ArrayLike, y_grid: grid_lib.GridLike = None
    ) -> float:
        """Negative conditional-density-estimate loss -- higher is better.

        Negated so the scikit-learn convention (greater ``score`` is a better
        model) holds, which is what ``GridSearchCV`` and friends assume.
        Rows are predicted and scored a block at a time, as in
        :meth:`predict`.

        Args:
            X: Features, shape (n_samples, n_features).
            y: Finite true target values, shape (n_samples,).
            y_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            Minus :func:`lazy.metrics.cde_loss`.

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        rows = self._row_scores(X, y, y_grid, ("cde_loss",))
        return metrics._negated_mean(rows.terms["cde_loss"])  # noqa: SLF001 - the metric registry.

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

        A convenience wrapper over :func:`lazy.metrics.summarize`; that
        function is the general-purpose entry point, taking PDFs you already
        have. Rows are predicted and scored a block at a time, as in
        :meth:`predict`, with the same table as a result.

        Args:
            X: Features, shape (n_samples, n_features).
            y: Finite true target values, shape (n_samples,).
            method: The point estimate to score, one of
                :data:`POINT_ESTIMATORS`. The default is ``"mode"`` until
                lazy 0.8.0, which changes it to ``"median"``; leaving it
                out warns.
            y_grid: The grid to evaluate on; ``None`` for this model's
                default.
            scale: How the point metrics (bias, scatter, outlier rates)
                scale residuals. ``"none"``, the default, scores the plain
                ``prediction - y``, in the target's units, outlier
                thresholds included; ``"1+y"`` divides it by ``1 + y``, the
                photometric-redshift convention, which DC1's numbers need.
                See :func:`lazy.metrics.point_metrics`.

        Returns:
            A one-row table labeled with :attr:`name_`; its ``scale``
            column records the choice.

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        validation.check_is_fitted(self)
        method = _check_method(
            _warnings.resolve_default(method, "method", "median")
        )
        if scale not in metrics.SCALES:
            raise ValueError(f"scale must be one of {metrics.SCALES}: {scale=}")
        rows = self._row_scores(X, y, y_grid, metrics.DENSITY_SCORES, method)
        assert rows.point is not None
        return metrics.summarize_scores(
            rows.y,
            rows.point,
            rows.terms["cde_loss"],
            rows.pit,
            point=method,
            label=self.name_,
            scale=scale,
            crps_terms=rows.terms["crps"],
            nll_terms=rows.terms["nll"],
        )

    # -- helpers -----------------------------------------------------------

    def _row_scores(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        y: npt.ArrayLike,
        y_grid: grid_lib.GridLike,
        names: Sequence[str],
        method: str | None = None,
    ) -> RowScores:
        """Each row's terms of the named scores, a block of rows at a time.

        :meth:`score`, :meth:`evaluate` and
        :func:`lazy.metrics.density_scorer` all score through this.

        Args:
            X: Features, shape (n_samples, n_features).
            y: Finite true target values, shape (n_samples,).
            y_grid: The grid to score on; ``None`` for this model's default.
            names: Names of scores, as :data:`lazy.metrics.DENSITY_SCORES`
                lists them.
            method: A checked point estimate to add, or ``None``.

        Returns:
            The rows' scores; ``point`` is ``None`` without ``method``.

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
            ValueError: If ``X`` has no rows.
        """
        validation.check_is_fitted(self)
        grid = self._resolve_grid(y_grid)
        X = self._check_features(X)
        y = _inputs.check_target(y, len(X))
        pits, terms, points = [], [], []
        for rows, density in self._density_blocks(X, grid):
            pit, block = metrics._row_terms(y[rows], grid, density, names)  # noqa: SLF001 - the metric registry.
            pits.append(pit)
            terms.append(block)
            if method is not None:
                points.append(
                    metrics.grid_point_estimate(grid, density, method)
                )
        return RowScores(
            y=y,
            point=np.concatenate(points) if points else None,
            pit=np.concatenate(pits),
            terms={
                name: np.concatenate([block[name] for block in terms])
                for name in names
            },
        )

    def _distribution_chunks(
        self, X: pd.DataFrame
    ) -> Iterator[distributions.Distribution]:
        """The native distributions of validated features, in chunks of rows.

        A model that predicts its queries a chunk at a time overrides this to
        yield each chunk's distributions as they are predicted, so that a
        caller reducing them holds one chunk at a time; this default yields
        every row's at once.

        Args:
            X: Validated features, shape (n_samples, n_features), with
                n_samples >= 1.

        Yields:
            The distributions of consecutive rows of ``X``, in order.
        """
        yield self._predict_distribution(X)

    def _density_blocks(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> Iterator[tuple[slice, _typing.FloatArray]]:
        """Normalized densities of validated features, a block at a time.

        Blocks hold about ``_BLOCK_BYTES`` of densities. When the model
        chunks its own queries (a ``chunk_size`` parameter), a block is a
        whole number of those chunks, so that the backend sees exactly the
        chunks it would have seen without blocking.

        Args:
            X: Validated features, shape (n_samples, n_features).
            grid: The grid to answer on.

        Yields:
            Tuples (rows, density): the slice of ``X`` a block covers, and
            its densities, shape (rows, grid.n_bins).
        """
        size = max(1, _BLOCK_BYTES // (8 * grid.n_bins))
        chunk = getattr(self, "chunk_size", None)
        if isinstance(chunk, numbers.Integral) and not isinstance(chunk, bool):
            chunk = int(chunk)
            if chunk > 0:
                size = max(chunk, size - size % chunk)
        for rows, pdfs in self._pdf_blocks(X, grid, size):
            # Rebound, so that the raw block is freed while the caller works.
            pdfs = self._checked_densities(pdfs, rows.stop - rows.start, grid)
            yield rows, pdfs

    def _pdf_blocks(
        self, X: pd.DataFrame, grid: grid_lib.Grid, size: int
    ) -> Iterator[tuple[slice, _typing.FloatArray]]:
        """Un-normalized densities of validated features, a block at a time.

        This default calls :meth:`_predict_pdf` once per block. A model that
        predicts its queries a chunk at a time overrides it to answer every
        block from one pass over its chunks, under one progress bar.

        Args:
            X: Validated features, shape (n_samples, n_features), with
                n_samples >= 1.
            grid: The grid to answer on.
            size: Rows per block; the last block may hold fewer.

        Yields:
            Tuples (rows, pdfs): the slice of ``X`` a block covers, with an
            explicit start and stop, and its densities, shape
            (rows, grid.n_bins).
        """
        for start in range(0, len(X), size):
            rows = slice(start, min(start + size, len(X)))
            block = X.iloc[rows].reset_index(drop=True)
            yield rows, self._predict_pdf(block, grid)

    def _densities(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        """Normalized densities of validated features on ``grid``.

        Args:
            X: Validated features, shape (n_samples, n_features).
            grid: The grid to answer on.

        Returns:
            Densities, shape (n_samples, grid.n_bins).

        Raises:
            RuntimeError: If the backend returns densities of the wrong
                shape.
        """
        return self._checked_densities(self._predict_pdf(X, grid), len(X), grid)

    def _checked_densities(
        self, pdfs: npt.ArrayLike, n_rows: int, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        """``pdfs`` normalized on ``grid``, once their shape is checked.

        Args:
            pdfs: Un-normalized densities from the backend.
            n_rows: The rows they answer.
            grid: The grid they are on.

        Returns:
            Densities, shape (n_rows, grid.n_bins).

        Raises:
            RuntimeError: If ``pdfs`` is not of shape (n_rows, grid.n_bins).
        """
        pdfs = np.asarray(pdfs, dtype=float)
        if pdfs.shape != (n_rows, grid.n_bins):
            raise RuntimeError(
                f"{type(self).__name__} returned {pdfs.shape}, expected"
                f" {(n_rows, grid.n_bins)}"
            )
        return grid.normalize(pdfs)

    @property
    def name_(self) -> str:
        """The label :meth:`evaluate` puts on its row.

        It reads ``"<backend>:<version>"``, or the class name for a subclass
        that is not a registered backend.

        The version is in the label on purpose: a foundation model's version
        is part of the method, so two rows both called ``tabpfn`` that came
        from different checkpoints would be a quietly wrong comparison.
        """
        if self.backend is None:
            return type(self).__name__
        version = getattr(self, "version", None)
        return f"{self.backend}:{version}" if version else self.backend

    def point_estimates(
        self, pdfs: npt.ArrayLike, y_grid: grid_lib.GridLike = None
    ) -> dict[str, _typing.FloatArray]:
        """Every supported reduction of already-computed densities.

        Cheaper than calling :meth:`predict` once per definition, which would
        re-run the model each time.

        Args:
            pdfs: Densities on ``y_grid``, shape (n_samples, n_bins).
            y_grid: The grid they are on; ``None`` for this model's default.

        Returns:
            A dict mapping each name in :data:`POINT_ESTIMATORS` to point
            values, shape (n_samples,).

        Raises:
            sklearn.exceptions.NotFittedError: If the model is not fitted.
        """
        validation.check_is_fitted(self)
        grid = self._resolve_grid(y_grid)
        return metrics.grid_point_estimates(grid, pdfs)

    @property
    def grid(self) -> grid_lib.Grid:
        """This model's default output grid. Available only after ``fit``."""
        validation.check_is_fitted(self, "grid_")
        return self.grid_

    def _resolve_grid(self, y_grid: grid_lib.GridLike) -> grid_lib.Grid:
        """A call-time grid, else this model's default."""
        if isinstance(y_grid, str) and y_grid == grid_lib.NATIVE:
            return self.native_grid
        if y_grid is not None:
            return grid_lib.as_grid(y_grid)
        default = getattr(self, "grid_", None)
        return default if default is not None else self._default_grid()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restores a pickle, one from before categorical columns included.

        Such a pickle lacks ``categorical_features``, which it gets at its
        default, and, if fitted, ``is_categorical_`` and ``categories_``,
        which it gets as every column numeric, as it was fitted.

        Args:
            state: The pickled ``__dict__``.
        """
        if "categorical_features" in self._get_param_names():
            state.setdefault("categorical_features", "from_dtype")
        if "n_features_in_" in state and "categories_" not in state:
            n_features = state["n_features_in_"]
            state["is_categorical_"] = np.zeros(n_features, dtype=bool)
            state["categories_"] = [None] * n_features
        super().__setstate__(state)

    def _clear_fitted(self) -> None:
        """Drops every fitted attribute, leaving the estimator unfitted.

        Fitted attributes are those scikit-learn's ``check_is_fitted`` looks
        for: names ending in an underscore, not starting with two.
        """
        for name in [
            name
            for name in vars(self)
            if name.endswith("_") and not name.startswith("__")
        ]:
            delattr(self, name)

    def _check_features(self, X: _Features) -> pd.DataFrame:
        """Coerces features to a DataFrame, consistent with those of ``fit``.

        Feature names follow scikit-learn: a predict-time table with the
        same names in another order is reordered, and a named table meeting
        an unnamed fit (or the reverse) is used by position, with a warning.

        Args:
            X: Features, shape (n_samples, n_features), in any form
                :func:`lazy._inputs.to_frame` accepts.

        Returns:
            The features as a DataFrame with a fresh index, columns labeled
            and ordered as at fit time: float64, categorical columns as
            their codes, unless the model reads its features as given.

        Raises:
            ValueError: If ``X`` has no rows or the wrong number of features.
        """
        frame, named = _inputs.to_frame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {frame.shape[1]} features, but "
                f"{type(self).__name__} is expecting {self.n_features_in_} "
                "features as input"
            )
        self._check_not_empty(frame)
        frame = self._align_to_fit(frame, named)
        if not self._numeric_features:
            return frame
        return _inputs.numeric_frame(frame, True, self.categories_)

    def _check_not_empty(self, X: pd.DataFrame) -> None:
        """Refuses a table with no rows, as scikit-learn does.

        Args:
            X: Features as ``_inputs.to_frame`` returns them, shape
                (n_samples, n_features).
        """
        if X.empty:
            raise ValueError(
                f"Found array with 0 sample(s) (shape={X.shape}) while a "
                f"minimum of 1 is required by {type(self).__name__}."
            )

    def _align_to_fit(self, frame: pd.DataFrame, named: bool) -> pd.DataFrame:
        """Labels and orders predict-time columns as ``fit`` saw them.

        Reads ``feature_names_in_`` from this object only, never through a
        wrapper's attribute delegation.
        """
        fitted = vars(self).get("feature_names_in_")
        if fitted is None:
            if named:
                _warnings.warn(
                    f"X has feature names, but {type(self).__name__} was "
                    "fitted without feature names; using columns by position",
                    _warnings.LazyWarning,
                )
            frame.columns = [f"x{i}" for i in range(frame.shape[1])]
            return frame
        if not named:
            _warnings.warn(
                "X does not have valid feature names, but "
                f"{type(self).__name__} was fitted with feature names; using "
                "columns by position",
                _warnings.LazyWarning,
            )
            frame.columns = list(fitted)
            return frame
        names = np.asarray(frame.columns, dtype=object)
        if np.array_equal(names, fitted):
            return frame
        if set(names) == set(fitted):
            return frame[list(fitted)]
        raise ValueError(
            "feature names differ from those seen during fit:\n"
            f"  fitted:    {list(fitted)}\n"
            f"  predicted: {list(names)}"
        )


def default_grid(low: float, high: float) -> grid_lib.Grid:
    """The grid of a model without one, for targets between two values.

    Args:
        low: The smallest target.
        high: The largest target.

    Returns:
        :data:`DEFAULT_N_BINS` uniform bins over ``[low, high]``, widened by
        :data:`DEFAULT_PADDING` of that range on each side.
    """
    pad = DEFAULT_PADDING * (high - low) or DEFAULT_PADDING
    return grid_lib.Grid.linear(
        low - pad, high + pad, DEFAULT_N_BINS, normalization="trapezoid"
    )


def _check_method(method: str) -> str:
    """Validates a point-estimate name, naming the alternatives if wrong."""
    if method not in POINT_ESTIMATORS:
        raise ValueError(
            f"method must be one of {POINT_ESTIMATORS}, got {method!r}"
        )
    return method
