"""The estimator protocol every :mod:`lazy` photo-z model implements.

The API is scikit-learn's, with one difference: the natural output of a photo-z
model is a conditional density, not a number, so :meth:`predict_proba` is the
primary method and :meth:`predict` is a documented reduction of it::

    model = LazyModel("tabfm", n_estimators=4)
    model.fit(X_train, z_train)

    pdfs = model.predict_proba(X_test, z_grid)       # (n_test, n_bins) densities
    z = model.predict(X_test, method="z_peak")       # (n_test,)
    model.score(X_test, z_test)                      # negative CDE loss

``z_grid`` is an argument to the *prediction*, not to the constructor. Nothing
about fitting depends on the output binning -- these models place their internal
bins by the quantiles of the context redshifts, and the grid only enters at the
final, exact rebinning step -- so one fitted model can answer on as many grids
as you like without refitting. The constructor still accepts ``z_grid`` as a
per-model default for when every call would pass the same thing; a grid given at
call time wins, and ``None`` at both levels means :data:`lazy.grid.DC1_GRID`.

Everything scikit-learn expects of an estimator holds: parameters are stored
verbatim by ``__init__`` and never validated there, all validation and all real
work happen in ``fit``, fitted attributes carry a trailing underscore, and
``get_params`` / ``set_params`` / :func:`sklearn.base.clone` work unmodified.

A subclass implements :meth:`_fit` and :meth:`_predict_pdf` and nothing else.
Both receive a :class:`pandas.DataFrame` with the columns ``fit`` was given, and
``_predict_pdf`` is handed the :class:`~lazy.grid.RedshiftGrid` to answer on;
the base class takes care of validation, grid resolution and normalisation.

Note on "fitting" an in-context model: none of the backends updates any
weights. ``fit`` stores the labelled rows that become the model's *context*, and
the cost of a prediction scales with how many there are. That is an
implementation detail of the backends, not of this interface -- a classical
estimator can implement the same protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from lazy.grid import RedshiftGrid, as_grid

__all__ = ["POINT_ESTIMATORS", "BasePhotoZEstimator"]

#: Reductions of a density to a single redshift, accepted by the ``method``
#: argument of :meth:`BasePhotoZEstimator.predict`. ``z_peak`` and ``z_weight``
#: are the DC1 ``z_PEAK`` and ``z_WEIGHT`` definitions (see :mod:`lazy.metrics`).
POINT_ESTIMATORS = ("z_peak", "z_weight", "z_mean", "z_median")


class BasePhotoZEstimator(BaseEstimator, ABC):
    """Base class for conditional-density photo-z estimators.

    Subclasses must accept a ``z_grid`` parameter and pass it through
    unchanged; see the module docstring.
    """

    #: The registered backend name (a key of :data:`lazy.ESTIMATORS`), set as a
    #: class attribute by each backend. ``None`` on a subclass that is not one.
    backend: str | None = None

    # -- to be provided by subclasses --------------------------------------

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: NDArray[np.float64]) -> None:
        """Store or fit whatever the backend needs. Sets ``*_`` attributes."""

    @abstractmethod
    def _predict_pdf(
        self, X: pd.DataFrame, grid: RedshiftGrid
    ) -> NDArray[np.float64]:
        """Un-normalised densities on ``grid``, one row per row of ``X``."""

    # -- the public API ----------------------------------------------------

    def fit(self, X, y) -> BasePhotoZEstimator:
        """Fit on labelled photometry.

        Parameters
        ----------
        X
            ``(n_samples, n_features)`` DataFrame or array of features. A
            DataFrame's column names are remembered and enforced at predict
            time.
        y
            ``(n_samples,)`` redshifts.
        """
        X = self._check_features(X, reset=True)
        y = np.asarray(y, dtype=float).ravel()
        if y.size != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {y.size} values")
        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite redshifts")
        self.grid_ = as_grid(self.z_grid)
        self._fit(X, y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X, z_grid=None) -> NDArray[np.float64]:
        """Conditional densities ``p(z | x)`` on ``z_grid``.

        Parameters
        ----------
        X
            ``(n_samples, n_features)`` features, with the columns ``fit`` saw.
        z_grid
            A :class:`~lazy.grid.RedshiftGrid`, an array of bin centres, or
            ``None`` for this model's default.

        Returns
        -------
        ndarray
            ``(n_samples, n_bins)``. These are **densities**, normalised to
            unit trapezoid mass over the bin centres -- the convention every
            metric in :mod:`lazy.metrics` uses. Multiply by ``grid.widths`` for
            per-bin probability masses.
        """
        check_is_fitted(self)
        grid = self._resolve_grid(z_grid)
        X = self._check_features(X, reset=False)
        pdfs = np.asarray(self._predict_pdf(X, grid), dtype=float)
        if pdfs.shape != (len(X), grid.n_bins):
            raise RuntimeError(
                f"{type(self).__name__} returned {pdfs.shape}, expected {(len(X), grid.n_bins)}"
            )
        return grid.normalize(pdfs)

    def predict_pdf(self, X, z_grid=None) -> NDArray[np.float64]:
        """Alias of :meth:`predict_proba`, for when "pdf" reads better than "proba"."""
        return self.predict_proba(X, z_grid)

    def predict_cdf(self, X, z_grid=None) -> NDArray[np.float64]:
        """Cumulative distributions on ``z_grid``, evaluated at the bin centres."""
        grid = self._resolve_grid(z_grid)
        return grid.cdf(self.predict_proba(X, grid))

    def predict(
        self, X, method: str = "z_peak", z_grid=None
    ) -> NDArray[np.float64]:
        """One redshift per row, reducing each density by ``method``.

        Parameters
        ----------
        X
            ``(n_samples, n_features)`` features.
        method
            One of :data:`POINT_ESTIMATORS`: ``"z_peak"`` (the mode, DC1's
            ``z_PEAK``), ``"z_weight"`` (DC1's main-peak weighted mean),
            ``"z_mean"`` or ``"z_median"``.
        z_grid
            The grid to evaluate on; ``None`` for this model's default.

        Notes
        -----
        The four definitions disagree exactly when a PDF is multimodal, which is
        the interesting case: ``z_mean`` lands between two peaks, where there is
        no probability at all, while ``z_peak`` and ``z_weight`` pick one. If you
        want several, call :meth:`predict_proba` once and pass the result to
        :func:`lazy.metrics.grid_point_estimates` rather than re-running the
        model per definition.
        """
        grid = self._resolve_grid(z_grid)
        return self.point_estimates(self.predict_proba(X, grid), grid)[
            _check_method(method)
        ]

    def score(self, X, y, z_grid=None) -> float:
        """Negative conditional-density-estimate loss -- higher is better.

        Negated so the scikit-learn convention (greater ``score`` is a better
        model) holds, which is what ``GridSearchCV`` and friends assume.
        """
        from lazy.metrics import cde_loss

        grid = self._resolve_grid(z_grid)
        y = np.asarray(y, dtype=float).ravel()
        return -cde_loss(y, grid.centers, self.predict_proba(X, grid))

    def evaluate(
        self, X, y, method: str = "z_peak", z_grid=None
    ) -> pd.DataFrame:
        """Score predictions for ``X`` with the full diagnostic metric set.

        A convenience wrapper over :func:`lazy.metrics.summarize`; that function
        is the general-purpose entry point, taking PDFs you already have.
        """
        from lazy.metrics import summarize

        grid = self._resolve_grid(z_grid)
        return summarize(
            np.asarray(y, dtype=float).ravel(),
            grid.centers,
            self.predict_proba(X, grid),
            point=_check_method(method),
            label=self.name_,
        )

    # -- helpers -----------------------------------------------------------

    @property
    def name_(self) -> str:
        """``"<backend>:<version>"`` -- the label :meth:`evaluate` puts on its row.

        The version is in the label on purpose: a foundation model's version is
        part of the method, so two rows both called ``tabpfn`` that came from
        different checkpoints would be a quietly wrong comparison.
        """
        if self.backend is None:
            return type(self).__name__
        version = getattr(self, "version", None)
        return f"{self.backend}:{version}" if version else self.backend

    def point_estimates(
        self, pdfs: ArrayLike, z_grid=None
    ) -> dict[str, NDArray[np.float64]]:
        """Every supported reduction of already-computed densities.

        Cheaper than calling :meth:`predict` once per definition, which would
        re-run the model each time.
        """
        from lazy.metrics import grid_point_estimates

        return grid_point_estimates(self._resolve_grid(z_grid).centers, pdfs)

    @property
    def grid(self) -> RedshiftGrid:
        """This model's default output grid. Available only after ``fit``."""
        check_is_fitted(self, "grid_")
        return self.grid_

    def _resolve_grid(self, z_grid) -> RedshiftGrid:
        """A call-time grid, this model's default, or the DC1 grid, in that order."""
        if z_grid is not None:
            return as_grid(z_grid)
        default = getattr(self, "grid_", None)
        return default if default is not None else as_grid(self.z_grid)

    def _check_features(self, X, *, reset: bool) -> pd.DataFrame:
        """Coerce features to a DataFrame and enforce consistency with ``fit``."""
        if isinstance(X, pd.DataFrame):
            frame = X.reset_index(drop=True)
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError(f"X must be 2D, got shape {array.shape}")
            frame = pd.DataFrame(
                array, columns=[f"x{i}" for i in range(array.shape[1])]
            )
        if frame.shape[1] == 0:
            raise ValueError("X has no feature columns")
        if reset:
            self.n_features_in_ = int(frame.shape[1])
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
            return frame
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {frame.shape[1]} features, but this "
                f"{type(self).__name__} was fitted with {self.n_features_in_}"
            )
        names = np.asarray(frame.columns, dtype=object)
        if not np.array_equal(names, self.feature_names_in_):
            if set(names) == set(self.feature_names_in_):
                return frame[list(self.feature_names_in_)]
            raise ValueError(
                "feature names differ from those seen during fit:\n"
                f"  fitted:    {list(self.feature_names_in_)}\n"
                f"  predicted: {list(names)}"
            )
        return frame


def _check_method(method: str) -> str:
    """Validate a point-estimate name, naming the alternatives when it is wrong."""
    if method not in POINT_ESTIMATORS:
        raise ValueError(
            f"method must be one of {POINT_ESTIMATORS}, got {method!r}"
        )
    return method
