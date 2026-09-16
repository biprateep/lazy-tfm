"""One entry point for every backend, chosen by name.

:class:`LazyModel` is the class most code should use. It takes the backend name
as its first argument and forwards everything else to that backend, so switching
between them is a string change rather than an import change::

    model = LazyModel("tabfm", n_estimators=4, n_dither=3)
    model = LazyModel("tabicl", n_estimators=8)
    model = LazyModel("tabpfn", n_estimators=8)

    model.fit(X_train, z_train)
    pdfs = model.predict_proba(X_test, z_grid)
    z = model.predict(X_test, method="z_peak")

The concrete classes (:class:`lazy.models.tabfm.TabFMHistogram`,
:class:`lazy.models.tabicl.TabICLQuantile`,
:class:`lazy.models.tabpfn.TabPFNBarDistribution`) remain importable and behave
identically -- ``LazyModel`` holds one and delegates to it. Use the concrete
class when you want its parameters documented at your fingertips; use
``LazyModel`` when the backend is a configuration value, which is the usual case
in a benchmark loop or a config file.

Constructing a model is cheap: no weights are downloaded, no GPU is touched and
the backend package is not even imported until :meth:`fit`. An unknown backend
name, or a parameter the backend does not accept, is reported immediately rather
than an hour into a run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lazy.base import BasePhotoZEstimator
from lazy.grid import RedshiftGrid
from lazy.models.registry import ESTIMATORS, list_estimators

__all__ = ["LazyModel"]


class LazyModel(BasePhotoZEstimator):
    """A photo-z model addressed by backend name.

    Parameters
    ----------
    model
        Which backend to use; one of :func:`lazy.list_estimators`.
        ``"tabfm"`` builds the density from a hierarchy of in-context
        classifiers, ``"tabicl"`` from a quantile regression head, ``"tabpfn"``
        from the bucket masses TabPFN-3 predicts natively.
    z_grid
        Default output grid for this model: a :class:`lazy.grid.RedshiftGrid`,
        an array of bin centres, or ``None`` for :data:`lazy.grid.DC1_GRID`.
        Every prediction method takes a ``z_grid`` that overrides it per call.
    **params
        Passed straight to the backend's constructor. See
        :class:`lazy.models.tabfm.TabFMHistogram`,
        :class:`lazy.models.tabicl.TabICLQuantile` and
        :class:`lazy.models.tabpfn.TabPFNBarDistribution` for what each
        accepts; an unrecognised name raises ``TypeError`` here, naming the
        class.

    Notes
    -----
    The backend instance every call is delegated to is kept as ``.estimator``.
    Its own fitted attributes (``inference_``, ``regressor_``, ``bin_prior_``,
    ...) and its parameters are reachable through this model by ordinary
    attribute access, so ``model.n_estimators`` and ``model.inference_`` work
    without reaching into ``.estimator`` yourself.

    Examples
    --------
    >>> model = LazyModel("tabfm", n_estimators=4, n_dither=3)
    >>> model.name_
    'tabfm:v1.0'
    >>> model.estimator.n_dither
    3
    >>> LazyModel("tabicl", n_estimators=16).get_params()["n_estimators"]
    16
    >>> LazyModel("tabpfn", version="v2.5").name_
    'tabpfn:v2.5'
    """

    def __init__(self, model: str = "tabfm", *, z_grid=None, **params):
        if model not in ESTIMATORS:
            raise ValueError(f"unknown model {model!r}; known: {list_estimators()}")
        self.model = model
        self.z_grid = z_grid
        self.estimator = ESTIMATORS[model](z_grid=z_grid, **params)

    # -- naming ------------------------------------------------------------

    @property
    def name_(self) -> str:
        """``"<backend>:<version>"``: the name asked for, and the version it loads.

        Built from ``self.model`` rather than delegated, so the label is the
        name the caller used; the version is appended because a row saying only
        ``tabpfn`` would not say which model produced it. A backend with no
        ``version`` parameter is labelled by name alone.
        """
        version = getattr(self.estimator, "version", None)
        return f"{self.model}:{version}" if version else self.model

    def __repr__(self) -> str:
        params = {
            k: v
            for k, v in self.estimator.get_params().items()
            if k != "z_grid" and v != _default_of(type(self.estimator), k)
        }
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
        return f"LazyModel({self.model!r}{', ' + inner if inner else ''})"

    # -- delegation --------------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: NDArray[np.float64]) -> None:
        self.estimator.fit(X, y)

    def _predict_pdf(self, X: pd.DataFrame, grid: RedshiftGrid) -> NDArray[np.float64]:
        return self.estimator._predict_pdf(X, grid)

    def __getattr__(self, name: str):
        """Expose the backend's fitted attributes without restating them here.

        Only reached for names this object does not define, and deliberately
        not for ``estimator`` itself -- looking that up before ``__init__`` has
        set it would recurse forever.
        """
        if name.startswith("_") or name == "estimator":
            raise AttributeError(name)
        try:
            estimator = self.__dict__["estimator"]
        except KeyError:
            raise AttributeError(name) from None
        try:
            return getattr(estimator, name)
        except AttributeError:
            raise AttributeError(
                f"neither LazyModel nor {type(estimator).__name__} has attribute {name!r}"
            ) from None

    # -- scikit-learn parameter protocol -----------------------------------
    #
    # `BaseEstimator` reads parameter names off `__init__`, which here ends in
    # `**params` and so would hide every backend parameter from `get_params`,
    # breaking `clone` and every search object. Both methods are therefore
    # implemented directly, flattening the backend's parameters into this one's.

    def get_params(self, deep: bool = True) -> dict:
        """This model's parameters: ``model`` plus everything the backend takes."""
        params = {"model": self.model}
        params.update(self.estimator.get_params(deep=deep))
        params["z_grid"] = self.z_grid
        return params

    def set_params(self, **params) -> LazyModel:
        """Set parameters, rebuilding the backend if ``model`` changes."""
        model = params.pop("model", self.model)
        if model not in ESTIMATORS:
            raise ValueError(f"unknown model {model!r}; known: {list_estimators()}")
        if model != self.model:
            # A different backend takes different parameters, so the old ones
            # cannot be carried over; only what is passed in this call survives.
            self.model = model
            self.z_grid = params.pop("z_grid", self.z_grid)
            self.estimator = ESTIMATORS[model](z_grid=self.z_grid, **params)
            return self
        if "z_grid" in params:
            self.z_grid = params["z_grid"]
        self.estimator.set_params(**params)
        return self


def _default_of(cls: type, name: str):
    """The declared default of ``cls.__init__``'s ``name`` parameter, or a sentinel."""
    import inspect

    parameter = inspect.signature(cls.__init__).parameters.get(name)
    return parameter.default if parameter is not None else object()
