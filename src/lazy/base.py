# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The estimator protocol every :mod:`lazy` model implements.

The API is scikit-learn's, with one difference: the natural output of these
models is a conditional density of the target, not a number, so
:meth:`predict_proba` is the primary method and :meth:`predict` is a
documented reduction of it::

    model = LazyModel("tabfm", n_estimators=4)
    model.fit(X_train, z_train)

    pdfs = model.predict_proba(X_test, y_grid)  # (n_test, n_bins) densities
    z = model.predict(X_test, method="mode")  # (n_test,)
    model.score(X_test, z_test)  # negative CDE loss

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
normalisation.

Note on "fitting" an in-context model: none of the backends updates any
weights. ``fit`` stores the labelled rows that become the model's *context*,
and the cost of a prediction scales with how many there are. That is an
implementation detail of the backends, not of this interface -- a classical
estimator can implement the same protocol.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
import os
from typing import Any, TypeAlias
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base as sklearn_base
from sklearn.utils import validation

from lazy import _inputs
from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy import metrics

__all__ = [
    "DEFAULT_N_BINS",
    "DEFAULT_PADDING",
    "POINT_ESTIMATORS",
    "BaseDensityRegressor",
]

#: Bins of the default grid of a model without a native grid.
DEFAULT_N_BINS = 200

#: How far that grid extends past the training targets, as a fraction of
#: their range on each side.
DEFAULT_PADDING = 0.05

#: Reductions of a density to a single value, accepted by the ``method``
#: argument of :meth:`BaseDensityRegressor.predict`. ``mode`` and
#: ``peak_mean`` are the DC1 ``z_PEAK`` and ``z_WEIGHT`` definitions (see
#: :mod:`lazy.metrics`).
POINT_ESTIMATORS = ("mode", "peak_mean", "mean", "median")

# What the public methods accept as features: any table lazy._inputs reads
# (arrays, structured arrays, DataFrames, astropy Tables, to_pandas()
# objects). Tables from optional packages have no common type, hence Any.
_Features: TypeAlias = Any

# Warnings skip every frame inside this package, so that they point at the
# user's call however deep in the package they are raised.
_PACKAGE_PREFIX = os.path.dirname(os.path.abspath(__file__)) + os.sep

# predict, score and evaluate reduce each row's density as they go, a block
# of rows at a time, so that no more than about this many bytes of float64
# densities exist at once: all of DC1 on a 5,000-bin native grid would
# otherwise be one 15.6 GB array.
_BLOCK_BYTES = 256 * 2**20


class BaseDensityRegressor(sklearn_base.BaseEstimator, abc.ABC):
    """A regressor that predicts conditional densities, with a scikit-learn API.

    Subclasses must accept a ``y_grid`` parameter and pass it through
    unchanged; see the module docstring.

    Attributes:
        backend: The registered backend name (a key of
            :data:`lazy.ESTIMATORS`), set as a class attribute by each
            backend; ``None`` on a subclass that is not one.
        y_grid: The default output grid as passed to the constructor: a
            :class:`~lazy.grid.Grid`, bin centres, or ``None``.
        grid_: The resolved default output grid, set by :meth:`fit`.
        is_fitted_: ``True`` once :meth:`fit` has run.
        n_features_in_: Number of feature columns seen by :meth:`fit`.
        feature_names_in_: Their names, an object array of shape
            (``n_features_in_``,).
    """

    #: The registered backend name (a key of :data:`lazy.ESTIMATORS`), set
    #: as a class attribute by each backend. ``None`` on a subclass that is
    #: not one.
    backend: str | None = None

    # Set by each subclass's __init__, which scikit-learn requires to store
    # its parameters verbatim; declared here for the type checker only.
    y_grid: grid_lib.GridLike

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
        """Returns un-normalised densities on ``grid``, one row per row of X.

        Args:
            X: Validated features, with the columns ``fit`` saw, shape
                (n_samples, n_features).
            grid: The grid to answer on.

        Returns:
            Non-negative densities at the bin centres, shape
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
        low, high = self.y_range_
        pad = DEFAULT_PADDING * (high - low) or DEFAULT_PADDING
        return grid_lib.Grid.linear(low - pad, high + pad, DEFAULT_N_BINS)

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
        """Fits on labelled rows.

        Args:
            X: Features, shape (n_samples, n_features): a NumPy array, a
                structured or record array, a pandas DataFrame, an astropy
                Table, or anything with ``to_pandas()``. Missing values are
                NaN. Column names, when the input has them, are remembered
                and enforced at predict time.
            y: Finite target values, shape (n_samples,), with
                n_samples >= 1.

        Returns:
            The fitted estimator itself.
        """
        X = self._check_features(X, reset=True)
        y = _check_target(y, len(X))
        self.y_range_ = (float(np.min(y)), float(np.max(y)))
        self._fit(X, y)
        # After _fit: a model's native grid is known only once it has seen
        # its context.
        self.grid_ = self._default_grid()
        self.is_fitted_ = True
        return self

    def predict_proba(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Conditional densities ``p(z | x)`` on ``y_grid``.

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).
            y_grid: A :class:`~lazy.grid.Grid`, an array of bin
                centres, ``"native"`` for the model's own grid, or ``None``
                for this model's default.

        Returns:
            Densities, shape (n_samples, n_bins). These are **densities**,
            normalised by the grid's convention (the trapezoid rule over the
            bin centres, or ``sum(p * widths) == 1`` on a histogram grid),
            which is how every metric in :mod:`lazy.metrics` integrates
            them. Multiply by ``grid.widths`` for per-bin probability masses.

        Raises:
            RuntimeError: If the backend returns densities of the wrong
                shape.
        """
        validation.check_is_fitted(self)
        grid = self._resolve_grid(y_grid)
        return self._densities(self._check_features(X, reset=False), grid)

    def predict_distribution(self, X: _Features) -> distributions.Distribution:  # noqa: GS030 - scikit-learn's X, y.
        """The model's native per-row distributions, on no grid at all.

        A bar-distribution model answers with a
        :class:`~lazy.distributions.HistogramDistribution` over its own
        buckets, a quantile model with a
        :class:`~lazy.distributions.QuantileDistribution`, bagged or dithered
        members with a :class:`~lazy.distributions.MixtureDistribution`. Each
        gives exact ``pdf``, ``cdf``, ``ppf``, ``rvs``, moments and
        ``to_qp()``.

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).

        Returns:
            One distribution per row of ``X``. For no rows at all, an empty
            :class:`~lazy.distributions.HistogramDistribution` on the
            default grid, without running the model.
        """
        validation.check_is_fitted(self)
        X = self._check_features(X, reset=False)
        if X.empty:
            return distributions.HistogramDistribution(
                self.grid_.edges, np.zeros((0, self.grid_.n_bins))
            )
        return self._predict_distribution(X)

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
            quantiles: Cumulative probabilities within [0, 1], shape (k,).

        Returns:
            The value at each level, shape (n_samples, k).
        """
        validation.check_is_fitted(self)
        return self.predict_distribution(X).ppf(quantiles)

    def predict_pdf(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Alias of :meth:`predict_proba`, for when "pdf" reads better.

        Args:
            X: Features, shape (n_samples, n_features).
            y_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            Normalised densities, shape (n_samples, n_bins).
        """
        return self.predict_proba(X, y_grid)

    def predict_cdf(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Cumulative distributions on ``y_grid``, at the bin centres.

        Args:
            X: Features, shape (n_samples, n_features).
            y_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            The mass below each bin centre, shape (n_samples, n_bins), by
            the grid's convention (:meth:`lazy.grid.Grid.cdf`): column 0 is
            zero on a trapezoid grid, and half the first bin's mass on a
            histogram grid.
        """
        validation.check_is_fitted(self)
        grid = self._resolve_grid(y_grid)
        return grid.cdf(self.predict_proba(X, grid))

    def predict(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        method: str = "mode",
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
                mean), ``"mean"`` or ``"median"``.
            y_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            Point estimates, shape (n_samples,).
        """
        validation.check_is_fitted(self)
        method = _check_method(method)
        grid = self._resolve_grid(y_grid)
        X = self._check_features(X, reset=False)
        return np.concatenate(
            [
                metrics.grid_point_estimates(grid, density)[method]
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
        """
        validation.check_is_fitted(self)
        grid = self._resolve_grid(y_grid)
        X = self._check_features(X, reset=False)
        self._check_not_empty(X)
        y = _check_target(y, len(X))
        terms = [
            metrics.per_object_scores(y[rows], grid, density)[0]
            for rows, density in self._density_blocks(X, grid)
        ]
        return -float(np.mean(np.concatenate(terms)))

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

        A convenience wrapper over :func:`lazy.metrics.summarize`; that
        function is the general-purpose entry point, taking PDFs you already
        have. Rows are predicted and scored a block at a time, as in
        :meth:`predict`, with the same table as a result.

        Args:
            X: Features, shape (n_samples, n_features).
            y: Finite true target values, shape (n_samples,).
            method: The point estimate to score, one of
                :data:`POINT_ESTIMATORS`.
            y_grid: The grid to evaluate on; ``None`` for this model's
                default.
            scale: How the point metrics (bias, scatter, outlier rates)
                scale residuals. ``"none"``, the default, scores the plain
                ``prediction - y``, in the target's units, outlier
                thresholds included; ``"1+y"`` divides it by ``1 + y``, the
                photometric-redshift convention, which DC1's numbers need.
                See :func:`lazy.metrics.point_metrics`.

        Returns:
            A one-row table labelled with :attr:`name_`; its ``scale``
            column records the choice.
        """
        validation.check_is_fitted(self)
        method = _check_method(method)
        if scale not in metrics.SCALES:
            raise ValueError(f"scale must be one of {metrics.SCALES}: {scale=}")
        grid = self._resolve_grid(y_grid)
        X = self._check_features(X, reset=False)
        self._check_not_empty(X)
        y = _check_target(y, len(X))
        y_pred, cde_terms, pit = [], [], []
        for rows, density in self._density_blocks(X, grid):
            y_pred.append(metrics.grid_point_estimates(grid, density)[method])
            terms, values = metrics.per_object_scores(y[rows], grid, density)
            cde_terms.append(terms)
            pit.append(values)
        return metrics.summarize_scores(
            y,
            np.concatenate(y_pred),
            np.concatenate(cde_terms),
            np.concatenate(pit),
            point=method,
            label=self.name_,
            scale=scale,
        )

    # -- helpers -----------------------------------------------------------

    def _density_blocks(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> Iterator[tuple[slice, _typing.FloatArray]]:
        """Normalised densities of validated features, a block at a time.

        Blocks hold about ``_BLOCK_BYTES`` of densities. When the model
        chunks its own queries (a ``chunk_size`` parameter), a block is a
        whole number of those chunks, so that the backend sees exactly the
        chunks it would have seen without blocking.

        Args:
            X: Validated features, shape (n_samples, n_features).
            grid: The grid to answer on.

        Yields:
            Tuples (rows, density): the slice of ``X`` a block covers, and
            its densities, shape (rows, grid.n_bins). No rows give one
            empty block.
        """
        size = max(1, _BLOCK_BYTES // (8 * grid.n_bins))
        chunk = getattr(self, "chunk_size", None)
        if isinstance(chunk, int) and not isinstance(chunk, bool) and chunk > 0:
            size = max(chunk, size - size % chunk)
        for start in range(0, max(len(X), 1), size):
            rows = slice(start, start + size)
            block = X.iloc[rows].reset_index(drop=True)
            yield rows, self._densities(block, grid)

    def _densities(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        """Normalised densities of validated features on ``grid``.

        Args:
            X: Validated features, shape (n_samples, n_features).
            grid: The grid to answer on.

        Returns:
            Densities, shape (n_samples, grid.n_bins); for no rows, an empty
            array, without running the model.

        Raises:
            RuntimeError: If the backend returns densities of the wrong
                shape.
        """
        if X.empty:
            return np.zeros((0, grid.n_bins))
        pdfs = np.asarray(self._predict_pdf(X, grid), dtype=float)
        if pdfs.shape != (len(X), grid.n_bins):
            raise RuntimeError(
                f"{type(self).__name__} returned {pdfs.shape}, expected"
                f" {(len(X), grid.n_bins)}"
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
        """
        validation.check_is_fitted(self)
        grid = self._resolve_grid(y_grid)
        return metrics.grid_point_estimates(
            grid.centers, pdfs, bin_edges=grid.histogram_edges
        )

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

    def _check_features(self, X: _Features, *, reset: bool) -> pd.DataFrame:
        """Coerces features to a DataFrame, consistent with those of ``fit``.

        Feature names follow scikit-learn: they are recorded as
        ``feature_names_in_`` only when the input carries string column
        names, a predict-time table with the same names in another order is
        reordered, and a named table meeting an unnamed fit (or the reverse)
        is used by position, with a warning.

        Args:
            X: Features, shape (n_samples, n_features), in any form
                :func:`lazy._inputs.as_feature_frame` accepts.
            reset: Whether to record the columns (at fit time) rather than
                enforce them (at predict time).

        Returns:
            The features as a float64 DataFrame with a fresh index, columns
            labelled and ordered as at fit time.
        """
        frame, named = _inputs.as_feature_frame(X)
        if reset:
            # Before anything is recorded, so that a refused fit leaves an
            # unfitted estimator unfitted.
            self._check_not_empty(frame)
            self.n_features_in_ = int(frame.shape[1])
            if named:
                self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
            else:
                self.__dict__.pop("feature_names_in_", None)
            return frame
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {frame.shape[1]} features, but this "
                f"{type(self).__name__} was fitted with {self.n_features_in_}"
            )
        return self._align_to_fit(frame, named)

    def _check_not_empty(self, X: pd.DataFrame) -> None:
        """Refuses a table with no rows, as scikit-learn does.

        Args:
            X: Validated features, shape (n_samples, n_features).
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
                warnings.warn(
                    f"X has feature names, but {type(self).__name__} was "
                    "fitted without feature names; using columns by position",
                    UserWarning,
                    skip_file_prefixes=(_PACKAGE_PREFIX,),
                )
            frame.columns = [f"x{i}" for i in range(frame.shape[1])]
            return frame
        if not named:
            warnings.warn(
                "X does not have valid feature names, but "
                f"{type(self).__name__} was fitted with feature names; using "
                "columns by position",
                UserWarning,
                skip_file_prefixes=(_PACKAGE_PREFIX,),
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


def _check_target(target: npt.ArrayLike, n_rows: int) -> _typing.FloatArray:
    """Validates target values against the feature rows they label.

    Args:
        target: Target values, shape (n_rows,), in any form
            :func:`lazy._inputs.as_target` accepts.
        n_rows: The number of feature rows.

    Returns:
        The values as a float64 array, shape (n_rows,).
    """
    values = _inputs.as_target(target)
    if values.size != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {values.size} values")
    if not np.isfinite(values).all():
        raise ValueError("y contains non-finite values")
    return values


def _check_method(method: str) -> str:
    """Validates a point-estimate name, naming the alternatives if wrong."""
    if method not in POINT_ESTIMATORS:
        raise ValueError(
            f"method must be one of {POINT_ESTIMATORS}, got {method!r}"
        )
    return method
