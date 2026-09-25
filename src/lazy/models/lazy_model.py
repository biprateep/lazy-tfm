# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""One entry point for every backend, chosen by name.

:class:`LazyModel` is the class most code should use. It takes the backend
name as its first argument and forwards everything else to that backend, so
switching between them is a string change rather than an import change::

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
from typing import Any

import pandas as pd

from lazy import _typing
from lazy import base
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import registry

__all__ = ["LazyModel"]


class LazyModel(base.BaseDensityRegressor):
    """A photo-z model addressed by backend name.

    The backend's parameters are reachable by ordinary attribute access, and
    after ``fit`` so are its fitted attributes (``inference_``,
    ``regressor_``, ...), so ``model.n_estimators`` and ``model.inference_``
    work without reaching into ``estimator_`` yourself.

    ``get_params`` lists ``model``, ``z_grid`` and every parameter of the
    chosen backend, flattened, so :func:`sklearn.base.clone` and the search
    objects work with no prefix: ``GridSearchCV(model, {"n_dither": [1, 3]})``.

    Args:
        model: Which backend to use; one of :func:`lazy.list_estimators`.
            ``"tabfm"`` builds the density from a hierarchy of in-context
            classifiers, ``"tabicl"`` from a quantile regression head,
            ``"tabpfn"`` from the bucket masses TabPFN-3 predicts natively.
        z_grid: Default output grid for this model: a
            :class:`lazy.grid.Grid`, an array of bin centres, or
            ``None`` for :data:`lazy.grid.DC1_GRID`. Every prediction method
            takes a ``z_grid`` that overrides it per call.
        **params: Passed straight to the backend's constructor. See
            :class:`lazy.models.tabfm.TabFMHistogram`,
            :class:`lazy.models.tabicl.TabICLQuantile` and
            :class:`lazy.models.tabpfn.TabPFNBarDistribution` for what each
            accepts; a name the backend does not take raises ``TypeError`` at
            :meth:`fit`, naming the class.

    Attributes:
        estimator_: The fitted backend every prediction is delegated to, a
            :class:`lazy.base.BaseDensityRegressor`.

    Examples:
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

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        model: str = "tabfm",
        *,
        z_grid: grid_lib.GridLike = None,
        **params: Any,
    ):
        self.model = model
        self.z_grid = z_grid
        # The backend's own parameters, exactly as given. They cannot be
        # attributes of their own: scikit-learn allows only names from this
        # signature, and which names a backend takes depends on `model`.
        self._params = params

    # -- naming ------------------------------------------------------------

    @property
    def name_(self) -> str:
        """The ``"<backend>:<version>"`` label: the name asked for, and version.

        Built from ``self.model`` rather than delegated, so the label is the
        name the caller used; the version is appended because a row saying
        only ``tabpfn`` would not say which model produced it. A backend with
        no ``version`` parameter is labelled by name alone.
        """
        version = self.get_params().get("version")
        return f"{self.model}:{version}" if version else str(self.model)

    def __repr__(self) -> str:
        defaults = _backend_defaults(self.model) or {}
        params = {
            k: v
            for k, v in self._params.items()
            if k not in defaults or v != defaults[k]
        }
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
        return f"LazyModel({self.model!r}{', ' + inner if inner else ''})"

    # -- delegation --------------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        self.estimator_ = self._build()
        # Unnamed features reach the backend unnamed, so that its fitted
        # attributes (and hence this wrapper's) keep scikit-learn's meaning.
        named = "feature_names_in_" in vars(self)
        self.estimator_.fit(X if named else X.to_numpy(), y)

    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        # Validation already ran in this wrapper, so the backend's hook is
        # called directly rather than through its public method.
        return self.estimator_._predict_pdf(X, grid)  # noqa: SLF001 - delegate.

    def _predict_distribution(
        self, X: pd.DataFrame
    ) -> distributions.Distribution:
        return self.estimator_._predict_distribution(X)  # noqa: SLF001 - delegate.

    def _default_grid(self) -> grid_lib.Grid:
        # The backend has already resolved its default (native or not).
        return self.estimator_.grid_

    def _build(self) -> base.BaseDensityRegressor:
        """Returns the unfitted backend these parameters describe, validated.

        Raises:
            ValueError: If ``model`` is not a registered backend.
            TypeError: If the backend does not take one of the parameters.
        """
        if self.model not in registry.ESTIMATORS:
            raise ValueError(
                f"unknown model {self.model!r}; "
                f"known: {registry.list_estimators()}"
            )
        cls = registry.ESTIMATORS[self.model]
        defaults = _constructor_defaults(cls)
        unknown = sorted(set(self._params) - set(defaults))
        if unknown:
            raise TypeError(
                f"{cls.__name__} does not take "
                f"{', '.join(map(repr, unknown))}; "
                f"its parameters are {sorted(defaults)}"
            )
        return cls(z_grid=self.z_grid, **self._params)

    def __getattr__(self, name: str) -> Any:
        """Exposes the backend's parameters and fitted attributes.

        Only reached for names this object does not define. Reads
        ``__dict__`` directly, so that a half-built object -- during
        unpickling, say -- raises ``AttributeError`` rather than recursing.

        Args:
            name: The attribute looked up.

        Returns:
            The fitted backend's attribute after ``fit``; before it, the
            backend parameter as given, or its default.

        Raises:
            AttributeError: If neither this object nor its backend has it.
        """
        state = self.__dict__
        if (
            name.startswith("_")
            or "model" not in state
            or "_params" not in state
        ):
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
        backend = registry.ESTIMATORS.get(state["model"])
        if backend is None:
            raise AttributeError(f"LazyModel has no attribute {name!r}")
        raise AttributeError(
            f"neither LazyModel nor {backend.__name__} has attribute {name!r}"
        )

    # -- scikit-learn parameter protocol -----------------------------------
    #
    # `BaseEstimator` reads parameter names off `__init__`, which here ends in
    # `**params` and so would hide every backend parameter from `get_params`,
    # breaking `clone` and every search object. Both methods are therefore
    # implemented directly, flattening the backend's parameters into this
    # one's.

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Returns ``model`` and ``z_grid`` plus everything the backend takes.

        Args:
            deep: Accepted for scikit-learn compatibility and ignored: the
                backend's parameters are always flattened into this model's.

        Returns:
            Parameter name to value, the backend's defaults filled in.
        """
        del deep  # Unused; there are no nested estimators to descend into.
        params: dict[str, Any] = {"model": self.model}
        params.update(_backend_defaults(self.model) or {})
        params.update(self._params)
        params["z_grid"] = self.z_grid
        return params

    def set_params(self, **params: Any) -> LazyModel:
        """Sets parameters, starting the backend's afresh if ``model`` changes.

        Args:
            **params: New values for ``model``, ``z_grid`` or any parameter
                of the backend.

        Returns:
            This model.
        """
        if "z_grid" in params:
            self.z_grid = params.pop("z_grid")
        model = params.pop("model", self.model)
        if model != self.model:
            # A different backend takes different parameters, so the old ones
            # cannot be carried over; only what is passed in this call
            # survives.
            self.model = model
            self._params = dict(params)
            return self
        defaults = _backend_defaults(self.model)
        if defaults is not None:
            for name in params:
                if name not in defaults:
                    valid = sorted(["model", "z_grid", *defaults])
                    raise ValueError(
                        f"invalid parameter {name!r} for "
                        f"LazyModel({self.model!r}); "
                        f"valid parameters are {valid}"
                    )
        self._params = {**self._params, **params}
        return self


def _constructor_defaults(
    cls: type[base.BaseDensityRegressor],
) -> dict[str, Any]:
    """Returns ``{name: default}`` for ``cls``'s parameters but ``z_grid``."""
    return {
        name: parameter.default
        for name, parameter in inspect.signature(
            cls.__init__
        ).parameters.items()
        if name not in ("self", "z_grid")
        and parameter.kind
        not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }


def _backend_defaults(model: object) -> dict[str, Any] | None:
    """Returns ``{name: default}`` for a backend's parameters but ``z_grid``.

    Args:
        model: The backend name, which may not be registered.

    Returns:
        The defaults, or ``None`` for a name that is not a registered backend,
        which is not an error until :meth:`LazyModel.fit`.
    """
    cls = registry.ESTIMATORS.get(model) if isinstance(model, str) else None
    if cls is None:
        return None
    return _constructor_defaults(cls)
