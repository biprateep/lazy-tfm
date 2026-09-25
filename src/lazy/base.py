# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The estimator protocol every :mod:`lazy` photo-z model implements.

The API is scikit-learn's, with one difference: the natural output of a
photo-z model is a conditional density, not a number, so :meth:`predict_proba`
is the primary method and :meth:`predict` is a documented reduction of it::

    model = LazyModel("tabfm", n_estimators=4)
    model.fit(X_train, z_train)

    pdfs = model.predict_proba(X_test, z_grid)  # (n_test, n_bins) densities
    z = model.predict(X_test, method="z_peak")  # (n_test,)
    model.score(X_test, z_test)  # negative CDE loss

``z_grid`` is an argument to the *prediction*, not to the constructor.
Nothing about fitting depends on the output binning -- these models place
their internal bins by the quantiles of the context redshifts, and the grid
only enters at the final, exact rebinning step -- so one fitted model can
answer on as many grids as you like without refitting. The constructor still
accepts ``z_grid`` as a per-model default for when every call would pass the
same thing; a grid given at call time wins, and ``None`` at both levels means
:data:`lazy.grid.DC1_GRID`.

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

__all__ = ["POINT_ESTIMATORS", "BasePhotoZEstimator"]

#: Reductions of a density to a single redshift, accepted by the ``method``
#: argument of :meth:`BasePhotoZEstimator.predict`. ``z_peak`` and
#: ``z_weight`` are the DC1 ``z_PEAK`` and ``z_WEIGHT`` definitions (see
#: :mod:`lazy.metrics`).
POINT_ESTIMATORS = ("z_peak", "z_weight", "z_mean", "z_median")

# What the public methods accept as features: any table lazy._inputs reads
# (arrays, structured arrays, DataFrames, astropy Tables, to_pandas()
# objects). Tables from optional packages have no common type, hence Any.
_Features: TypeAlias = Any


class BasePhotoZEstimator(sklearn_base.BaseEstimator, abc.ABC):
    """A conditional-density photo-z estimator with a scikit-learn API.

    Subclasses must accept a ``z_grid`` parameter and pass it through
    unchanged; see the module docstring.

    Attributes:
        backend: The registered backend name (a key of
            :data:`lazy.ESTIMATORS`), set as a class attribute by each
            backend; ``None`` on a subclass that is not one.
        z_grid: The default output grid as passed to the constructor: a
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
    z_grid: grid_lib.GridLike

    # -- to be provided by subclasses --------------------------------------

    @abc.abstractmethod
    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        """Stores or fits whatever the backend needs; sets ``*_`` attributes.

        Args:
            X: Validated features, shape (n_samples, n_features).
            y: Finite redshifts, shape (n_samples,).
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

        The constructor's ``z_grid`` (``"native"`` meaning the model's own),
        else :data:`lazy.grid.DC1_GRID`. Models with a native grid override
        this to make it their default.
        """
        if isinstance(self.z_grid, str) and self.z_grid == grid_lib.NATIVE:
            return self.native_grid
        return grid_lib.as_grid(self.z_grid)

    @property
    def native_grid(self) -> grid_lib.Grid:
        """The model's own output grid, available after ``fit``.

        Raises:
            AttributeError: If this estimator has no native grid.
        """
        native = getattr(self, "native_grid_", None)
        if native is None:
            raise AttributeError(
                f"{type(self).__name__} has no native grid; pass a grid"
            )
        return native

    # -- the public API ----------------------------------------------------

    def fit(self, X: _Features, y: npt.ArrayLike) -> BasePhotoZEstimator:  # noqa: GS030 - scikit-learn's X, y.
        """Fits on labelled photometry.

        Args:
            X: Features, shape (n_samples, n_features): a NumPy array, a
                structured or record array, a pandas DataFrame, an astropy
                Table, or anything with ``to_pandas()``. Missing values are
                NaN. Column names, when the input has them, are remembered
                and enforced at predict time.
            y: Finite redshifts, shape (n_samples,).

        Returns:
            The fitted estimator itself.
        """
        X = self._check_features(X, reset=True)
        y = _inputs.as_target(y)
        if y.size != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {y.size} values")
        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite redshifts")
        self._fit(X, y)
        # After _fit: a model's native grid is known only once it has seen
        # its context.
        self.grid_ = self._default_grid()
        self.is_fitted_ = True
        return self

    def predict_proba(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, z_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Conditional densities ``p(z | x)`` on ``z_grid``.

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).
            z_grid: A :class:`~lazy.grid.Grid`, an array of bin
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
        grid = self._resolve_grid(z_grid)
        X = self._check_features(X, reset=False)
        pdfs = np.asarray(self._predict_pdf(X, grid), dtype=float)
        if pdfs.shape != (len(X), grid.n_bins):
            raise RuntimeError(
                f"{type(self).__name__} returned {pdfs.shape}, expected"
                f" {(len(X), grid.n_bins)}"
            )
        return grid.normalize(pdfs)

    def predict_distribution(self, X: _Features) -> distributions.Distribution:  # noqa: GS030 - scikit-learn's X, y.
        """The model's native per-galaxy distributions, on no grid at all.

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
            One distribution per row of ``X``.
        """
        validation.check_is_fitted(self)
        return self._predict_distribution(self._check_features(X, reset=False))

    def predict_quantiles(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        quantiles: npt.ArrayLike = (0.16, 0.5, 0.84),
    ) -> _typing.FloatArray:
        """Redshift quantiles of each galaxy's distribution, exactly.

        Computed on the native distribution (:meth:`predict_distribution`),
        never on a grid, so they carry the model's full resolution.

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).
            quantiles: Cumulative probabilities within [0, 1], shape (k,).

        Returns:
            The redshift at each level, shape (n_samples, k).
        """
        return self.predict_distribution(X).ppf(quantiles)

    def predict_pdf(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, z_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Alias of :meth:`predict_proba`, for when "pdf" reads better.

        Args:
            X: Features, shape (n_samples, n_features).
            z_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            Normalised densities, shape (n_samples, n_bins).
        """
        return self.predict_proba(X, z_grid)

    def predict_cdf(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, z_grid: grid_lib.GridLike = None
    ) -> _typing.FloatArray:
        """Cumulative distributions on ``z_grid``, at the bin centres.

        Args:
            X: Features, shape (n_samples, n_features).
            z_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            The mass below each bin centre, shape (n_samples, n_bins);
            column 0 is zero.
        """
        grid = self._resolve_grid(z_grid)
        return grid.cdf(self.predict_proba(X, grid))

    def predict(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        method: str = "z_peak",
        z_grid: grid_lib.GridLike = None,
    ) -> _typing.FloatArray:
        """One redshift per row, reducing each density by ``method``.

        The four definitions disagree exactly when a PDF is multimodal,
        which is the interesting case: ``z_mean`` lands between two peaks,
        where there is no probability at all, while ``z_peak`` and
        ``z_weight`` pick one. If you want several, call
        :meth:`predict_proba` once and pass the result to
        :func:`lazy.metrics.grid_point_estimates` rather than re-running the
        model per definition.

        Args:
            X: Features, shape (n_samples, n_features).
            method: One of :data:`POINT_ESTIMATORS`: ``"z_peak"`` (the mode,
                DC1's ``z_PEAK``), ``"z_weight"`` (DC1's main-peak weighted
                mean), ``"z_mean"`` or ``"z_median"``.
            z_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            Point redshifts, shape (n_samples,).
        """
        grid = self._resolve_grid(z_grid)
        return self.point_estimates(self.predict_proba(X, grid), grid)[
            _check_method(method)
        ]

    def score(  # noqa: GS030 - scikit-learn's X, y.
        self, X: _Features, y: npt.ArrayLike, z_grid: grid_lib.GridLike = None
    ) -> float:
        """Negative conditional-density-estimate loss -- higher is better.

        Negated so the scikit-learn convention (greater ``score`` is a better
        model) holds, which is what ``GridSearchCV`` and friends assume.

        Args:
            X: Features, shape (n_samples, n_features).
            y: True redshifts, shape (n_samples,).
            z_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            Minus :func:`lazy.metrics.cde_loss`.
        """
        grid = self._resolve_grid(z_grid)
        y = np.asarray(y, dtype=float).ravel()
        return -metrics.cde_loss(
            y,
            grid.centers,
            self.predict_proba(X, grid),
            bin_edges=grid.histogram_edges,
        )

    def evaluate(  # noqa: GS030 - scikit-learn's X, y.
        self,
        X: _Features,
        y: npt.ArrayLike,
        method: str = "z_peak",
        z_grid: grid_lib.GridLike = None,
    ) -> pd.DataFrame:
        """Scores predictions for ``X`` with the full diagnostic metric set.

        A convenience wrapper over :func:`lazy.metrics.summarize`; that
        function is the general-purpose entry point, taking PDFs you already
        have.

        Args:
            X: Features, shape (n_samples, n_features).
            y: True redshifts, shape (n_samples,).
            method: The point estimate to score, one of
                :data:`POINT_ESTIMATORS`.
            z_grid: The grid to evaluate on; ``None`` for this model's
                default.

        Returns:
            A one-row table labelled with :attr:`name_`.
        """
        grid = self._resolve_grid(z_grid)
        return metrics.summarize(
            np.asarray(y, dtype=float).ravel(),
            grid.centers,
            self.predict_proba(X, grid),
            point=_check_method(method),
            label=self.name_,
            bin_edges=grid.histogram_edges,
        )

    # -- helpers -----------------------------------------------------------

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
        self, pdfs: npt.ArrayLike, z_grid: grid_lib.GridLike = None
    ) -> dict[str, _typing.FloatArray]:
        """Every supported reduction of already-computed densities.

        Cheaper than calling :meth:`predict` once per definition, which would
        re-run the model each time.

        Args:
            pdfs: Densities on ``z_grid``, shape (n_samples, n_bins).
            z_grid: The grid they are on; ``None`` for this model's default.

        Returns:
            A dict mapping each name in :data:`POINT_ESTIMATORS` to point
            redshifts, shape (n_samples,).
        """
        grid = self._resolve_grid(z_grid)
        return metrics.grid_point_estimates(
            grid.centers, pdfs, bin_edges=grid.histogram_edges
        )

    @property
    def grid(self) -> grid_lib.Grid:
        """This model's default output grid. Available only after ``fit``."""
        validation.check_is_fitted(self, "grid_")
        return self.grid_

    def _resolve_grid(self, z_grid: grid_lib.GridLike) -> grid_lib.Grid:
        """A call-time grid, this model's default, or the DC1 grid, in order."""
        if isinstance(z_grid, str) and z_grid == grid_lib.NATIVE:
            return self.native_grid
        if z_grid is not None:
            return grid_lib.as_grid(z_grid)
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
                    stacklevel=4,
                )
            frame.columns = [f"x{i}" for i in range(frame.shape[1])]
            return frame
        if not named:
            warnings.warn(
                "X does not have valid feature names, but "
                f"{type(self).__name__} was fitted with feature names; using "
                "columns by position",
                UserWarning,
                stacklevel=4,
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


def _check_method(method: str) -> str:
    """Validates a point-estimate name, naming the alternatives if wrong."""
    if method not in POINT_ESTIMATORS:
        raise ValueError(
            f"method must be one of {POINT_ESTIMATORS}, got {method!r}"
        )
    return method
