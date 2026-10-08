# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Pools of different models: one density from several backends.

:class:`LazyEnsembleModel` fits several models on the same context and
combines their predicted distributions into one, with equal weights::

    model = LazyEnsembleModel(["tabpfn", "limix", "tabfm"])
    model.fit(X_train, z_train)
    pdfs = model.predict_proba(X_test)  # the pooled densities

Three pools are available. ``"geometric"`` (the default) is the normalized
product of the members' densities, each raised to its weight (a log pool);
``"linear"`` is their mixture; ``"quantile"`` averages their quantile
functions (Vincentization). The members are evaluated exactly, each on its
own native distribution, and pooled on one fine internal grid
(``native_grid_``) that covers the training targets with a margin on either
side, at least as finely as the finest member resolves them.

Every member is a full model, so a pool costs about the sum of its members.
The pooled model is itself a :class:`~lazy.base.BaseDensityRegressor`, so it
answers ``predict_proba``, ``predict_quantiles``, ``predict_pit``, ``score``
and ``evaluate`` like any single model, and scikit-learn's ``clone`` and
search objects reach the members' parameters as ``<member>__<parameter>``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import inspect
import math
import numbers
import os
from typing import Any, Literal, TypeAlias, TypeGuard
import warnings

import numpy as np
import pandas as pd
from sklearn import base as sklearn_base
from sklearn import utils as sklearn_utils

from lazy import _typing
from lazy import base
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _ensemble
from lazy.models import lazy_model
from lazy.models import registry

__all__ = [
    "LOG_FLOOR",
    "MAX_POOL_BINS",
    "MIN_POOL_BINS",
    "POOLINGS",
    "POOL_PADDING",
    "QUANTILE_LEVELS",
    "SMALL_CONTEXT_ROWS",
    "EnsembleSizeWarning",
    "LazyEnsembleModel",
]

#: The accepted values of ``pooling``.
POOLINGS: tuple[str, ...] = ("geometric", "linear", "quantile")

Pooling: TypeAlias = Literal["geometric", "linear", "quantile"]

#: What a member may be given as: a backend name, or an estimator.
Member: TypeAlias = str | base.BaseDensityRegressor

#: How far the pooling grid extends past the training targets, as a fraction
#: of their range on each side; the native grid of TabICL is padded alike.
POOL_PADDING = 0.25

#: The fewest bins the automatic pooling grid has.
MIN_POOL_BINS = 200

#: The most bins the automatic pooling grid has, which bounds the memory of
#: a pooled prediction at 80 kB per row in float64.
MAX_POOL_BINS = 10_000

#: The floor added to every member's density before the geometric pool takes
#: its logarithm, as a fraction of the uniform density over the pooling grid.
LOG_FLOOR = 1e-10

#: The cumulative levels the quantile pool averages the members at: 0, 0.001,
#: ..., 1, the ends included, so that the pool has no point masses.
QUANTILE_LEVELS: _typing.FloatArray = np.linspace(0.0, 1.0, 1001)

#: Contexts with fewer rows than this make a pool of several members warn.
SMALL_CONTEXT_ROWS = 3_000

# A bagged member's mixture is inverted exactly on the union of its
# components' edges, unless there are more than this many; it is then read
# at the pooling grid's resolution, so that its quantiles fit in memory.
_EXACT_PPF_MAX_EDGES = 20_000

# The parameters the ensemble passes on to members that keep their default.
_SHARED_PARAMS = ("random_state", "device")

# Warnings skip every frame inside this package, so that they point at the
# user's call.
_PACKAGE_PREFIX = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
)


class EnsembleSizeWarning(UserWarning):
    """The context is small enough for one weak member to spoil the pool."""


class LazyEnsembleModel(base.BaseDensityRegressor):
    """Several models fitted on the same context, their densities pooled.

    Each member is cloned and fitted on the same rows at :meth:`fit`, and
    every prediction pools the members' distributions with equal weights.
    Construction stores the arguments and nothing else, as scikit-learn
    requires; the members are checked and built at :meth:`fit`.

    Members are named after their backend (``"tabpfn"``, ``"limix"``, ...),
    with ``_2``, ``_3``, ... appended to repeats, and a member that is not a
    registered backend after its class, in lower case. ``get_params`` lists
    every member's parameters as ``<name>__<parameter>``, as scikit-learn's
    ``VotingRegressor`` does, so ``clone`` works and a search can tune a
    member: ``GridSearchCV(model, {"tabpfn__n_estimators": [4, 8]})``.

    Args:
        models: The members, a list of backend names (one of
            :func:`lazy.list_estimators`, built as ``LazyModel(name)`` at
            its defaults), configured
            :class:`~lazy.models.lazy_model.LazyModel` instances, or other
            :class:`~lazy.base.BaseDensityRegressor` estimators, such as a
            concrete backend.
        pooling: ``"geometric"``, the normalized product of the members'
            densities each raised to its weight (a log pool); ``"linear"``,
            their mixture; or ``"quantile"``, the average of their quantile
            functions at :data:`QUANTILE_LEVELS`, clipped to the pooling
            grid.
        pool_bins: The number of bins of the pooling grid, or ``None`` to
            choose it: as fine as the finest member's median bin width
            inside the grid's range, within :data:`MIN_POOL_BINS` and
            :data:`MAX_POOL_BINS`.
        chunk_size: Query rows pooled at a time, each member being called
            once per chunk (and chunking further by its own ``chunk_size``);
            0 pools every row at once. The members' distributions of one
            chunk are what bounds the memory of a prediction.
        y_grid: Default output grid, as for
            :class:`~lazy.models.lazy_model.LazyModel`: ``None`` is the
            pooling grid (``native_grid_``).
        random_state: The seed given to every member whose own
            ``random_state`` is still its default (0); a member given
            another value keeps it.
        device: The device given to every member whose own ``device`` is
            still its default (``"auto"``); a member given another value
            keeps it.

    Attributes:
        estimators_: The fitted members, a list, in the order of
            ``models``.
        named_estimators_: The same members by name, a
            :class:`sklearn.utils.Bunch`.
        weights_: Each member's weight in the pool, shape (n_members,);
            ``1 / n_members`` each.
        native_grid_: The pooling grid, a histogram-normalized
            :class:`~lazy.grid.Grid` of equal bins.
        provenance_: What answered: each member's own ``provenance_`` by
            name, and the pooling, weights and pooling grid.

    Examples:
        >>> model = LazyEnsembleModel(["tabpfn", "tabfm"])
        >>> model.name_
        'geometric(tabpfn:v3.5, tabfm:v1.0)'
        >>> model.get_params()["tabfm__n_estimators"]
        8
        >>> model.set_params(tabfm__n_estimators=4).models[1]
        LazyModel('tabfm', n_estimators=4)
    """

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        models: Sequence[Member],
        *,
        pooling: Pooling = "geometric",
        pool_bins: int | None = None,
        chunk_size: int = _ensemble.UNIFORM_DEFAULTS["chunk_size"],
        y_grid: grid_lib.GridLike = None,
        random_state: int | None = 0,
        device: str = "auto",
    ):
        self.models = models
        self.pooling = pooling
        self.pool_bins = pool_bins
        self.chunk_size = chunk_size
        self.y_grid = y_grid
        self.random_state = random_state
        self.device = device

    # -- naming ------------------------------------------------------------

    @property
    def name_(self) -> str:
        """The ``"<pooling>(<member>, ...)"`` label that ``evaluate`` uses."""
        items = _as_items(self.models)
        if not items:
            return type(self).__name__
        labels = ", ".join(_label(item) for item in items)
        return f"{self.pooling}({labels})"

    # -- scikit-learn parameter protocol -----------------------------------

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Returns this model's parameters and, if deep, its members'.

        Args:
            deep: Whether to add each member, by name, and its parameters as
                ``<name>__<parameter>``. A member given by name is listed as
                ``LazyModel(name)``.

        Returns:
            Parameter name to value.
        """
        params = super().get_params(deep=False)
        if not deep:
            return params
        for name, item in _named_items(self.models):
            member = _as_estimator(item)
            params[name] = member
            for key, value in member.get_params(deep=True).items():
                params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params: Any) -> LazyEnsembleModel:
        """Sets this model's parameters, members' ones included.

        ``<name>=estimator`` replaces a member, and
        ``<name>__<parameter>=value`` sets one of its parameters; a member
        given by name becomes ``LazyModel(name)`` with that parameter set.
        ``models`` itself is replaced by a new list, never changed in place.

        Args:
            **params: New values, by the names :meth:`get_params` lists.

        Returns:
            This model.

        Raises:
            ValueError: If a name is not a parameter of this model or of a
                member.
        """
        own = super().get_params(deep=False)
        if "models" in params:
            self.models = params.pop("models")
        named = _named_items(self.models)
        names = [name for name, _ in named]
        items = [item for _, item in named]
        nested: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            name, _, sub = key.partition("__")
            if key in own:
                setattr(self, key, value)
            elif name in names and not sub:
                items[names.index(name)] = value
            elif name in names:
                nested.setdefault(name, {})[sub] = value
            else:
                raise ValueError(
                    f"invalid parameter {key!r} for {type(self).__name__}; "
                    f"valid parameters are {sorted(self.get_params(deep=True))}"
                )
        for name, sub_params in nested.items():
            index = names.index(name)
            items[index] = _as_estimator(items[index]).set_params(**sub_params)
        if any(
            new is not old for new, (_, old) in zip(items, named, strict=True)
        ):
            self.models = items
        return self

    # -- fitting -----------------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        pooling = _check_pooling(self.pooling)
        _check_options(self.pool_bins, self.chunk_size)
        members = [
            (name, _configure(item, self.random_state, self.device))
            for name, item in _check_models(self.models)
        ]
        _warn_if_small(len(X), len(members))
        # Unnamed features reach the members unnamed, as in LazyModel.
        named = "feature_names_in_" in vars(self)
        context = X if named else X.to_numpy()
        for _, member in members:
            member.fit(context, y)
        self.estimators_ = [member for _, member in members]
        self.named_estimators_ = sklearn_utils.Bunch(**dict(members))
        self.weights_ = np.full(len(members), 1.0 / len(members))
        self.pooling_ = pooling
        self.native_grid_ = _pool_grid(
            self.y_range_,
            [_member_grid(member) for member in self.estimators_],
            self.pool_bins,
        )
        self.provenance_ = self._provenance(members)

    def _provenance(
        self, members: list[tuple[str, base.BaseDensityRegressor]]
    ) -> dict[str, Any]:
        """What answered, for ``provenance_``."""
        grid = self.native_grid_
        names = [name for name, _ in members]
        recipe: dict[str, Any] = {
            "pooling": self.pooling_,
            "weights": dict(zip(names, self.weights_.tolist(), strict=True)),
            "pool_grid": {
                "y_min": grid.y_min,
                "y_max": grid.y_max,
                "n_bins": grid.n_bins,
                "bin_width": float(grid.widths[0]),
            },
            "members": {
                name: getattr(member, "provenance_", None)
                for name, member in members
            },
        }
        if self.pooling_ == "geometric":
            recipe["log_floor"] = LOG_FLOOR
        if self.pooling_ == "quantile":
            recipe["quantile_levels"] = int(QUANTILE_LEVELS.size)
        return recipe

    def _default_grid(self) -> grid_lib.Grid:
        """The constructor's grid, else the pooling grid."""
        if self.y_grid is None:
            return self.native_grid
        return super()._default_grid()

    # -- predicting --------------------------------------------------------

    def _predict_distribution(
        self, X: pd.DataFrame
    ) -> distributions.Distribution:
        return distributions.concatenate(list(self._pooled_chunks(X)))

    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        blocks = [dist.on_grid(grid) for dist in self._pooled_chunks(X)]
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _pooled_chunks(
        self, X: pd.DataFrame
    ) -> Iterator[distributions.Distribution]:
        """The pooled distributions of validated features, chunk by chunk.

        Args:
            X: Validated features, shape (n_samples, n_features), with
                n_samples >= 1.

        Yields:
            One pooled distribution per chunk of rows, in order.
        """
        named = "feature_names_in_" in vars(self)
        size = int(self.chunk_size) or len(X)
        for start in range(0, len(X), size):
            block = X.iloc[start : start + size]
            # Through each member's public method, which checks the features
            # against what the member was fitted on.
            query = block if named else block.to_numpy()
            parts = [
                member.predict_distribution(query)
                for member in self.estimators_
            ]
            yield _pool(parts, self.weights_, self.pooling_, self.native_grid_)


def _as_items(models: object) -> list[Any]:
    """The members as a list, or an empty one if ``models`` is not a list."""
    if isinstance(models, list | tuple):
        return list(models)
    return []


def _base_name(item: object) -> str:
    """A member's name before repeats are numbered."""
    if isinstance(item, str):
        return item
    if isinstance(item, lazy_model.LazyModel):
        return str(item.model)
    backend = getattr(item, "backend", None)
    if isinstance(backend, str):
        return backend
    return type(item).__name__.lower()


def _named_items(models: object) -> list[tuple[str, Any]]:
    """Each member with its name, repeats numbered ``_2``, ``_3``, ...

    Args:
        models: The ``models`` parameter, which may not be a list yet.

    Returns:
        A list of tuples (name, item), in the order of ``models``.
    """
    seen: dict[str, int] = {}
    named = []
    for item in _as_items(models):
        name = _base_name(item)
        count = seen[name] = seen.get(name, 0) + 1
        named.append((name if count == 1 else f"{name}_{count}", item))
    return named


def _as_estimator(item: Any) -> Any:
    """A member as an estimator: a backend name becomes ``LazyModel(name)``."""
    return lazy_model.LazyModel(item) if isinstance(item, str) else item


def _label(item: object) -> str:
    """A member's label in :attr:`LazyEnsembleModel.name_`."""
    estimator = _as_estimator(item)
    if isinstance(estimator, base.BaseDensityRegressor):
        return estimator.name_
    return repr(item)


def _check_pooling(pooling: object) -> Pooling:
    """Validates ``pooling``."""
    if pooling == "geometric" or pooling == "linear" or pooling == "quantile":
        return pooling
    raise ValueError(f"pooling must be one of {POOLINGS}: {pooling=}")


def _check_options(pool_bins: object, chunk_size: object) -> None:
    """Validates ``pool_bins`` and ``chunk_size``."""
    if pool_bins is not None and (not _is_integer(pool_bins) or pool_bins < 2):
        raise ValueError(f"pool_bins must be None or an int >= 2: {pool_bins=}")
    if not _is_integer(chunk_size) or chunk_size < 0:
        raise ValueError(
            "chunk_size must be a non-negative int (0 means one pass): "
            f"{chunk_size=}"
        )


def _is_integer(value: object) -> TypeGuard[int]:
    """Whether ``value`` is a Python or NumPy integer, and not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(
        value, bool | np.bool_
    )


def _check_models(models: object) -> list[tuple[str, Member]]:
    """Validates ``models``; returns its items with their names.

    Raises:
        TypeError: If an item is neither a name nor a density regressor.
        ValueError: If ``models`` is not a non-empty list, or names an
            unknown backend.
    """
    if not isinstance(models, list | tuple) or not models:
        raise ValueError(
            "models must be a non-empty list of backend names or estimators: "
            f"{models=}"
        )
    for item in models:
        if isinstance(item, str):
            if item not in registry.ESTIMATORS:
                raise ValueError(
                    f"unknown model {item!r} in models; "
                    f"known: {registry.list_estimators()}"
                )
        elif not isinstance(item, base.BaseDensityRegressor):
            raise TypeError(
                "models must hold backend names or lazy estimators, not "
                f"{type(item).__name__}: {item!r}"
            )
    return _named_items(models)


def _configure(
    item: Member, random_state: int | None, device: str
) -> base.BaseDensityRegressor:
    """An unfitted copy of a member, with the ensemble's shared settings.

    ``random_state`` and ``device`` replace the member's own only where the
    member's value equals its default, which is how a member that was not
    given one is told apart from one that was (``clone`` writes every
    default out, so whether it was passed explicitly cannot be known).

    Args:
        item: A backend name or an estimator.
        random_state: The ensemble's seed.
        device: The ensemble's device.

    Returns:
        A new, unfitted estimator.
    """
    member = (
        lazy_model.LazyModel(item)
        if isinstance(item, str)
        else sklearn_base.clone(item)
    )
    current = member.get_params(deep=False)
    defaults = _default_params(member)
    shared = dict(zip(_SHARED_PARAMS, (random_state, device), strict=True))
    updates = {
        name: value
        for name, value in shared.items()
        if name in current
        and name in defaults
        and _equal(current[name], defaults[name])
    }
    return member.set_params(**updates) if updates else member


def _default_params(member: base.BaseDensityRegressor) -> dict[str, Any]:
    """The constructor defaults of a member's parameters."""
    if isinstance(member, lazy_model.LazyModel):
        return lazy_model.LazyModel(member.model).get_params()
    return {
        name: parameter.default
        for name, parameter in inspect.signature(
            type(member).__init__
        ).parameters.items()
        if parameter.default is not parameter.empty
    }


def _equal(first: object, second: object) -> bool:
    """Whether two parameter values are equal, False if undecidable."""
    if type(first) is not type(second):
        return False
    try:
        return bool(first == second)
    except (TypeError, ValueError):  # An array, say, with no single truth.
        return False


def _warn_if_small(n_rows: int, n_members: int) -> None:
    """Warns when a small context meets a pool of several members."""
    if n_members < 2 or n_rows >= SMALL_CONTEXT_ROWS:
        return
    warnings.warn(
        f"The context has {n_rows:,} rows. On small contexts an equal pool "
        "can be dragged down by one weak member, and do no better than the "
        "best single model, or worse; 'Ensembles of models' in the "
        "documentation gives the measurements. Consider fewer members, or "
        "the best single model.",
        EnsembleSizeWarning,
        skip_file_prefixes=(_PACKAGE_PREFIX,),
    )


def _member_grid(member: base.BaseDensityRegressor) -> grid_lib.Grid:
    """A fitted member's native grid, or its default grid if it has none."""
    native = getattr(member, "native_grid_", None)
    return native if isinstance(native, grid_lib.Grid) else member.grid_


def _pool_grid(
    y_range: tuple[float, float],
    member_grids: Sequence[grid_lib.Grid],
    pool_bins: int | None,
) -> grid_lib.Grid:
    """The equal-bin grid the members are pooled on.

    It spans the training targets padded by :data:`POOL_PADDING` of their
    range on each side, cut to the union of the members' grids where that
    is narrower, since no member places mass beyond its own grid.

    Args:
        y_range: The training targets' minimum and maximum.
        member_grids: Each member's native (or default) grid.
        pool_bins: The number of bins, or ``None`` to choose it.

    Returns:
        A histogram-normalized grid of equal bins.
    """
    low, high = y_range
    pad = POOL_PADDING * (high - low) or POOL_PADDING * max(abs(low), 1.0)
    start, stop = low - pad, high + pad
    union_start = min(grid.y_min for grid in member_grids)
    union_stop = max(grid.y_max for grid in member_grids)
    if max(start, union_start) < min(stop, union_stop):
        start, stop = max(start, union_start), min(stop, union_stop)
    if pool_bins is None:
        width = min(_median_width(grid, start, stop) for grid in member_grids)
        n_bins = MAX_POOL_BINS
        if math.isfinite(width):
            n_bins = math.ceil((stop - start) / width)
        pool_bins = min(max(n_bins, MIN_POOL_BINS), MAX_POOL_BINS)
    return grid_lib.Grid.linear(
        start, stop, pool_bins, normalization="histogram"
    )


def _median_width(grid: grid_lib.Grid, start: float, stop: float) -> float:
    """The median width of a grid's bins that overlap [start, stop]."""
    edges = grid.edges
    inside = (edges[1:] > start) & (edges[:-1] < stop)
    if not inside.any():
        return math.inf
    return float(np.median(grid.widths[inside]))


def _pool(
    parts: Sequence[distributions.Distribution],
    weights: _typing.FloatArray,
    pooling: Pooling,
    grid: grid_lib.Grid,
) -> distributions.Distribution:
    """The members' distributions of the same rows, pooled.

    Args:
        parts: One distribution per member, all with the same n_rows rows.
        weights: The members' weights, shape (n_members,), summing to one.
        pooling: ``"geometric"``, ``"linear"`` or ``"quantile"``.
        grid: The pooling grid.

    Returns:
        A :class:`~lazy.distributions.HistogramDistribution` on the pooling
        grid, or for the quantile pool a
        :class:`~lazy.distributions.QuantileDistribution` at
        :data:`QUANTILE_LEVELS`.
    """
    if pooling == "quantile":
        locs = sum(
            weight * _quantiles(part, grid)
            for weight, part in zip(weights, parts, strict=True)
        )
        return distributions.QuantileDistribution(QUANTILE_LEVELS, locs)
    edges = grid.edges
    if pooling == "linear":
        # The mixture restricted to the grid: the mass each member puts in
        # each bin, averaged before the row is renormalized.
        masses = sum(
            weight * part.histogramize(edges).masses
            for weight, part in zip(weights, parts, strict=True)
        )
        return distributions.HistogramDistribution(edges, masses)
    floor = LOG_FLOOR / (grid.y_max - grid.y_min)
    log_pool = sum(
        weight * np.log(part.histogramize(edges).masses / grid.widths + floor)
        for weight, part in zip(weights, parts, strict=True)
    )
    pooled = np.exp(log_pool - np.max(log_pool, axis=1, keepdims=True))
    return distributions.HistogramDistribution(edges, pooled * grid.widths)


def _quantiles(
    part: distributions.Distribution, grid: grid_lib.Grid
) -> _typing.FloatArray:
    """A member's quantiles at :data:`QUANTILE_LEVELS`, shape (n_rows, k).

    Each is clipped to the pooling grid, where the levels 0 and 1 (the ends
    of a member's support, which reach far into the tails for a bar
    distribution) would otherwise stretch the pool's outermost thousandths
    across the whole range of the buckets. Exact within the grid, except for
    a mixture whose components' edges number more than
    ``_EXACT_PPF_MAX_EDGES`` (bagged buckets, each bag placing its own),
    which is read on the pooling grid's edges and its own outermost two.
    """
    if (
        isinstance(part, distributions.MixtureDistribution)
        and part.bins.size > _EXACT_PPF_MAX_EDGES
    ):
        bins = part.bins
        edges = np.unique(np.concatenate([bins[[0, -1]], grid.edges]))
        part = part.histogramize(edges)
    return np.clip(part.ppf(QUANTILE_LEVELS), grid.y_min, grid.y_max)
