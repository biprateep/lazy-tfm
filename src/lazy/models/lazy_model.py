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
identically -- a fitted ``LazyModel`` holds one as ``estimator_`` and delegates
to it. Use the concrete class when you want its parameters documented at your
fingertips; use ``LazyModel`` when the backend is a configuration value, which
is the usual case in a benchmark loop or a config file.

Constructing a model does nothing but store its arguments, as scikit-learn
requires: no weights are downloaded, no GPU is touched, the backend package is
not imported and nothing is validated. The backend is built, and an unknown
backend name or a parameter it does not take is reported, at :meth:`fit`.
"""

from __future__ import annotations

import inspect

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
        accepts; a name the backend does not take raises ``TypeError`` at
        :meth:`fit`, naming the class.

    Attributes
    ----------
    estimator_ : lazy.base.BasePhotoZEstimator
        The fitted backend every prediction is delegated to.

    Notes
    -----
    The backend's parameters are reachable by ordinary attribute access, and
    after ``fit`` so are its fitted attributes (``inference_``,
    ``regressor_``, ...), so ``model.n_estimators`` and ``model.inference_``
    work without reaching into ``estimator_`` yourself.

    ``get_params`` lists ``model``, ``z_grid`` and every parameter of the
    chosen backend, flattened, so :func:`sklearn.base.clone` and the search
    objects work with no prefix: ``GridSearchCV(model, {"n_dither": [1, 3]})``.

    Examples
    --------
    >>> model = LazyModel("tabfm", n_estimators=4, n_dither=3)
    >>> model.name_
    'tabfm:v1.0'
    >>> model.n_dither
    3
    >>> LazyModel("tabicl", n_estimators=16).get_params()["n_estimators"]
    16
    >>> LazyModel("tabpfn", version="v2.5").name_
    'tabpfn:v2.5'
    """

    def __init__(self, model: str = "tabfm", *, z_grid=None, **params):
        self.model = model
        self.z_grid = z_grid
        # The backend's own parameters, exactly as given. They cannot be
        # attributes of their own: scikit-learn allows only names from this
        # signature, and which names a backend takes depends on `model`.
        self._params = params

    # -- naming ------------------------------------------------------------

    @property
    def name_(self) -> str:
        """``"<backend>:<version>"``: the name asked for, and the version it loads.

        Built from ``self.model`` rather than delegated, so the label is the
        name the caller used; the version is appended because a row saying only
        ``tabpfn`` would not say which model produced it. A backend with no
        ``version`` parameter is labelled by name alone.
        """
        version = self.get_params().get("version")
        return f"{self.model}:{version}" if version else str(self.model)

    def __repr__(self) -> str:
        defaults = _backend_defaults(self.model) or {}
        params = {k: v for k, v in self._params.items() if k not in defaults or v != defaults[k]}
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
        return f"LazyModel({self.model!r}{', ' + inner if inner else ''})"

    # -- delegation --------------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: NDArray[np.float64]) -> None:
        self.estimator_ = self._build()
        self.estimator_.fit(X, y)

    def _predict_pdf(self, X: pd.DataFrame, grid: RedshiftGrid) -> NDArray[np.float64]:
        return self.estimator_._predict_pdf(X, grid)

    def _build(self) -> BasePhotoZEstimator:
        """The unfitted backend these parameters describe, validated."""
        if self.model not in ESTIMATORS:
            raise ValueError(f"unknown model {self.model!r}; known: {list_estimators()}")
        cls = ESTIMATORS[self.model]
        unknown = sorted(set(self._params) - set(_backend_defaults(self.model)))
        if unknown:
            raise TypeError(
                f"{cls.__name__} does not take {', '.join(map(repr, unknown))}; "
                f"its parameters are {sorted(_backend_defaults(self.model))}"
            )
        return cls(z_grid=self.z_grid, **self._params)

    def __getattr__(self, name: str):
        """Expose the backend's parameters and fitted attributes.

        Only reached for names this object does not define. Reads ``__dict__``
        directly, so that a half-built object -- during unpickling, say --
        raises ``AttributeError`` rather than recursing.
        """
        state = self.__dict__
        if name.startswith("_") or "model" not in state or "_params" not in state:
            raise AttributeError(name)
        if "estimator_" in state:
            try:
                return getattr(state["estimator_"], name)
            except AttributeError:
                pass
        elif name in state["_params"]:
            return state["_params"][name]
        else:
            defaults = _backend_defaults(state["model"]) or {}
            if name in defaults:
                return defaults[name]
        backend = ESTIMATORS.get(state["model"])
        owner = f"LazyModel nor {backend.__name__}" if backend else "LazyModel"
        raise AttributeError(f"neither {owner} has attribute {name!r}")

    # -- scikit-learn parameter protocol -----------------------------------
    #
    # `BaseEstimator` reads parameter names off `__init__`, which here ends in
    # `**params` and so would hide every backend parameter from `get_params`,
    # breaking `clone` and every search object. Both methods are therefore
    # implemented directly, flattening the backend's parameters into this one's.

    def get_params(self, deep: bool = True) -> dict:
        """This model's parameters: ``model`` plus everything the backend takes."""
        params = {"model": self.model}
        params.update(_backend_defaults(self.model) or {})
        params.update(self._params)
        params["z_grid"] = self.z_grid
        return params

    def set_params(self, **params) -> LazyModel:
        """Set parameters, starting the backend's afresh if ``model`` changes."""
        if "z_grid" in params:
            self.z_grid = params.pop("z_grid")
        model = params.pop("model", self.model)
        if model != self.model:
            # A different backend takes different parameters, so the old ones
            # cannot be carried over; only what is passed in this call survives.
            self.model = model
            self._params = dict(params)
            return self
        defaults = _backend_defaults(self.model)
        if defaults is not None:
            for name in params:
                if name not in defaults:
                    raise ValueError(
                        f"invalid parameter {name!r} for LazyModel({self.model!r}); "
                        f"valid parameters are {sorted(['model', 'z_grid', *defaults])}"
                    )
        self._params = {**self._params, **params}
        return self


def _backend_defaults(model) -> dict | None:
    """``{name: default}`` for a backend's parameters other than ``z_grid``.

    ``None`` for a name that is not a registered backend, which is not an error
    until :meth:`LazyModel.fit`.
    """
    cls = ESTIMATORS.get(model) if isinstance(model, str) else None
    if cls is None:
        return None
    return {
        name: parameter.default
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name not in ("self", "z_grid")
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }
