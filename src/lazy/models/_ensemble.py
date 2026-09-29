# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The uniform feature layer every foundation-model backend is built on.

Every backend supports the same features through the same parameters, and
means the same thing by them:

=====================  =====================================================
``kv_cache``           cache the context's keys and values (exact); default
                       True
``n_estimators``       ensemble members
``feature_shuffle``    each member sees the feature columns permuted
``transforms``         per-member feature transforms (:mod:`._transforms`)
``bag_size``           each member sees a random subset of context rows
``z_grid``             the default output grid; None means the model's
                       native grid
=====================  =====================================================

A backend translates each feature to its model's own machinery where the
model has it -- so the model's tuned default recipe (``transforms="auto"``)
and its batched ensembling are used unchanged -- and the missing pieces are
scaffolded here: rows, transforms and column permutations applied per member
(:mod:`._members`), members combined, the queries chunked. A new backend
subclasses :class:`ContextEnsembleEstimator`, declares what its model has
natively, and implements four small methods.

Typical usage example:

  class MyBackend(ContextEnsembleEstimator):
      backend = "mine"
      native_output = "histogram"
      ...
      def _fit_group(self, X, y, group): ...
      def _predict_group(self, handle, X): ...
"""

from __future__ import annotations

import abc
from collections.abc import Iterator, Mapping
import types
from typing import Any, ClassVar, Literal
import warnings

import numpy as np
import pandas as pd

from lazy import _typing
from lazy import base
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _device
from lazy.models import _hub
from lazy.models import _members
from lazy.models import _progress
from lazy.models import _transforms

__all__ = [
    "UNIFORM_DEFAULTS",
    "UNIFORM_PARAMS",
    "ContextEnsembleEstimator",
    "ContextSizeWarning",
    "PerformanceWarning",
]

#: The parameters every backend's constructor must take, keyword-only.
UNIFORM_PARAMS: tuple[str, ...] = (
    "version",
    "n_estimators",
    "transforms",
    "feature_shuffle",
    "bag_size",
    "kv_cache",
    "z_grid",
    "device",
    "random_state",
    "chunk_size",
    "progress",
    "verbose",
)

#: The uniform parameters whose default is the same on every backend.
UNIFORM_DEFAULTS: dict[str, Any] = {
    "transforms": _transforms.AUTO,
    "feature_shuffle": True,
    "bag_size": None,
    "kv_cache": True,
    "z_grid": None,
    "device": "auto",
    "progress": "auto",
    "verbose": False,
}


class ContextSizeWarning(UserWarning):
    """The context is larger than the model was pretrained for."""


class PerformanceWarning(UserWarning):
    """The prediction will be correct but slower than it needs to be."""


class ContextEnsembleEstimator(base.BaseDensityRegressor, abc.ABC):
    """A pretrained in-context model behind the uniform feature layer.

    Subclasses declare their model's capabilities as class attributes and
    implement :meth:`_import_backend`, :meth:`_fit_group`,
    :meth:`_predict_group` and :meth:`_native_grid`; everything else --
    validation, checkpoints, bagging, scaffolded transforms, member
    combination, chunking, progress and the native grid -- lives here.

    Attributes:
        display_name: The model's name in progress bars and warnings.
        extra: The pip extra that installs the model.
        native_output: ``"histogram"`` or ``"quantiles"``.
        native_transforms: Uniform transform names the model implements,
            mapped to its own tokens.
        auto_tokens: The model's own default recipe as tokens, cycled by
            single-member groups under ``transforms="auto"``.
        supports_native_bagging: Whether the model subsamples rows per
            member itself.
        member_combination: How groups combine: ``"mixture"`` of densities
            or ``"quantile_average"`` of quantile functions.
        kv_cache_modes: The accepted ``kv_cache`` values.
        kv_cache_rtol: How closely cached and uncached outputs agree.
        recommended_max_context: The context size above which the model
            degrades without bagging, or None.
        exact_chunking: Whether chunking the queries is exact bit for bit;
            if not, a query's answer still does not depend on the others in
            its chunk, but differs from an unchunked one by float rounding.
        chunks_queries: Whether the base class chunks the queries, or the
            model does it itself.
        accepts_auto_estimators: Whether ``n_estimators="auto"`` is allowed.
    """

    display_name: ClassVar[str] = ""
    extra: ClassVar[str] = ""
    native_output: ClassVar[Literal["histogram", "quantiles"]] = "histogram"
    native_transforms: ClassVar[Mapping[str, Any]] = {}
    auto_tokens: ClassVar[tuple[Any, ...] | None] = None
    supports_native_bagging: ClassVar[bool] = False
    member_combination: ClassVar[Literal["mixture", "quantile_average"]] = (
        "mixture"
    )
    kv_cache_modes: ClassVar[tuple[bool | str, ...]] = (True, False)
    kv_cache_rtol: ClassVar[float] = 1e-5
    recommended_max_context: ClassVar[int | None] = None
    exact_chunking: ClassVar[bool] = True
    chunks_queries: ClassVar[bool] = True
    accepts_auto_estimators: ClassVar[bool] = False

    # Set by each backend's __init__; declared for the type checker only.
    version: str
    n_estimators: int | str
    transforms: str | tuple[str, ...]
    feature_shuffle: bool
    bag_size: int | float | None
    kv_cache: bool | str
    device: str
    random_state: int
    chunk_size: int
    progress: _progress.Progress
    verbose: bool

    # -- the per-backend interface -----------------------------------------

    @abc.abstractmethod
    def _import_backend(self) -> types.ModuleType:
        """Imports the model's package, naming the extra if it is missing."""

    def _check_backend_params(self) -> None:
        """Validates the backend's own parameters; raises ValueError."""
        return

    @abc.abstractmethod
    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> Any:
        """Fits one group of members; returns a handle for prediction.

        Args:
            X: The group's context features, already restricted to its rows,
                transformed and permuted, shape (n_rows, n_features').
            y: Their target values, shape (n_rows,).
            group: The members to serve and how.
        """

    @abc.abstractmethod
    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.Distribution:
        """Predicts one group's distributions for prepared query features."""

    @abc.abstractmethod
    def _native_grid(self) -> grid_lib.Grid:
        """The model's own output grid; called at the end of ``_fit``."""

    def _load_checkpoint(self) -> None:
        """Fetches the pinned weights; sets ``checkpoint_`` and provenance."""
        spec = _hub.get_checkpoint(self.backend or "", self.version)
        self.checkpoint_ = spec.download()
        self.provenance_ = spec.provenance(device=self.device_)

    def _progress_postfix(
        self, dist: distributions.Distribution
    ) -> dict[str, Any]:
        """Extra progress-bar fields for a predicted chunk."""
        del dist  # Unused.
        return {}

    def _recommended_max_context(self) -> int | None:
        """The context size above which the model wants bagging."""
        return self.recommended_max_context

    # -- fitting -----------------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        self._check_uniform_params()
        self._import_backend()
        self._check_backend_params()
        _progress.check_progress(self.progress)
        self.device_ = _device.resolve_device(self.device)
        self._load_checkpoint()
        n_rows, n_features = X.shape
        self.n_context_ = int(n_rows)
        self.bag_rows_ = _members.resolve_bag_size(self.bag_size, n_rows)
        self.bagging_ = self.bag_rows_ < n_rows
        self._warn_if_context_too_large(n_rows)
        self.kv_cache_: bool | str = self.kv_cache
        self.member_groups_ = _members.plan(
            n_estimators=self._n_members(),
            transforms=_transforms.parse(self.transforms),
            native_transforms=self.native_transforms,
            feature_shuffle=self.feature_shuffle,
            bag_rows=self.bag_rows_,
            n_rows=n_rows,
            n_features=n_features,
            random_state=self.random_state,
            supports_native_bagging=self.supports_native_bagging,
            auto_tokens=self.auto_tokens,
        )
        features = X.to_numpy(dtype=np.float64)
        self.transformers_: list[_transforms.ScaffoldTransform | None] = []
        self.handles_: list[Any] = []
        for group in self.member_groups_:
            rows = slice(None) if group.rows is None else group.rows
            context = features[rows]
            transformer = (
                None
                if group.scaffold is None
                else _transforms.ScaffoldTransform(
                    group.scaffold, group.seed
                ).fit(context)
            )
            self.transformers_.append(transformer)
            prepared = self._prepare(context, group, transformer)
            self._log(
                f"fitting group {group.index} ({group.n_members} members) on "
                f"{len(prepared)} context rows ({self.device_})"
            )
            self.handles_.append(self._fit_group(prepared, y[rows], group))
        if len(self.handles_) == 1:
            self.regressor_ = self.handles_[0]
        self.native_grid_ = self._native_grid()
        self.provenance_ = {**self.provenance_, **self._recipe()}

    def _check_uniform_params(self) -> None:
        """Validates the uniform parameters, before anything is imported."""
        known = _hub.list_versions(self.backend or "")
        if known and self.version not in known:
            raise ValueError(
                f"unknown version {self.version!r} for {self.backend!r}; "
                f"known: {known}"
            )
        if not isinstance(self.chunk_size, int) or self.chunk_size < 0:
            raise ValueError(
                f"chunk_size must be a non-negative int (0 means one pass): "
                f"{self.chunk_size=}"
            )
        valid_cache = isinstance(self.kv_cache, bool | str) and any(
            self.kv_cache == mode and type(self.kv_cache) is type(mode)
            for mode in self.kv_cache_modes
        )
        if not valid_cache:
            raise ValueError(
                f"kv_cache must be one of {self.kv_cache_modes}: "
                f"{self.kv_cache=}"
            )
        if not isinstance(self.feature_shuffle, bool):
            raise ValueError(
                f"feature_shuffle must be a bool: {self.feature_shuffle=}"
            )
        _transforms.parse(self.transforms)
        _members.resolve_bag_size(self.bag_size, 1)
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise ValueError(
                f"random_state must be an int: {self.random_state=}"
            )
        self._n_members()

    def _n_members(self) -> int:
        """The ensemble size, validated; ``"auto"`` stays upstream's choice."""
        n_estimators = self.n_estimators
        if n_estimators == "auto" and self.accepts_auto_estimators:
            needs_count = self.bag_size is not None or _transforms.parse(
                self.transforms
            )
            if needs_count:
                raise ValueError(
                    "n_estimators='auto' leaves the count to the model, so "
                    "bag_size and transforms need an explicit n_estimators"
                )
            return 1
        if (
            isinstance(n_estimators, bool)
            or not isinstance(n_estimators, int)
            or n_estimators < 1
        ):
            raise ValueError(
                f"n_estimators must be a positive int: {self.n_estimators=}"
            )
        return n_estimators

    def _warn_if_context_too_large(self, n_rows: int) -> None:
        """Warns when the context exceeds what the model handles unbagged."""
        limit = self._recommended_max_context()
        if self.bagging_ or limit is None or n_rows <= limit:
            return
        needed = -(-n_rows // limit)
        warnings.warn(
            f"{self.display_name} degrades on contexts larger than about "
            f"{limit:,} rows, and this one has {n_rows:,}. Turn on bagging: "
            f"pass bag_size={limit} with n_estimators >= {needed} so the "
            "members together cover the context.",
            ContextSizeWarning,
            stacklevel=4,
        )

    def _recipe(self) -> dict[str, Any]:
        """What the ensemble was, for ``provenance_``."""
        members: list[_members.MemberSpec] = []
        for group in self.member_groups_:
            members.extend(group.members)
        return {
            "n_estimators": self.n_estimators,
            "transforms": [
                "auto" if m.transform is None else m.transform.name
                for m in sorted(members, key=lambda m: m.index)
            ],
            "feature_shuffle": self.feature_shuffle,
            "bag_rows": self.bag_rows_ if self.bagging_ else None,
            "kv_cache": self.kv_cache_,
            "groups": len(self.member_groups_),
        }

    # -- predicting ----------------------------------------------------------

    def _prepare(
        self,
        features: _typing.FloatArray,
        group: _members.MemberGroup,
        transformer: _transforms.ScaffoldTransform | None,
    ) -> _typing.FloatArray:
        """A group's view of some features: transformed, then permuted."""
        if transformer is not None:
            features = transformer.transform(features)
        if group.permutation is not None:
            permutation = group.permutation
            if features.shape[1] != permutation.size:
                # "+original" doubled the columns; permute both halves alike.
                halves = features.shape[1] // permutation.size
                permutation = np.concatenate(
                    [permutation + k * permutation.size for k in range(halves)]
                )
            features = features[:, permutation]
        return features

    def _predict_chunk(
        self, features: _typing.FloatArray
    ) -> distributions.Distribution:
        """Every group's prediction for some query features, combined."""
        parts = [
            self._predict_group(handle, self._prepare(features, group, fitted))
            for handle, group, fitted in zip(
                self.handles_,
                self.member_groups_,
                self.transformers_,
                strict=True,
            )
        ]
        return self._combine(parts)

    def _combine(
        self, parts: list[distributions.Distribution]
    ) -> distributions.Distribution:
        """Groups' predictions as one, weighted by their members."""
        if len(parts) == 1:
            return parts[0]
        weights = np.array([g.n_members for g in self.member_groups_], float)
        weights /= weights.sum()
        if self.member_combination == "quantile_average":
            quantiles = [
                part
                for part in parts
                if isinstance(part, distributions.QuantileDistribution)
            ]
            if len(quantiles) != len(parts):
                raise TypeError("quantile_average needs quantile outputs")
            return distributions.QuantileDistribution.average(
                quantiles, weights
            )
        return _mixture(parts, weights)

    def _chunks(self, X: pd.DataFrame) -> Iterator[distributions.Distribution]:
        """The predicted distributions, chunk by chunk, with a progress bar."""
        features = X.to_numpy(dtype=np.float64)
        if not self.chunks_queries:
            yield self._predict_chunk(features)
            return
        size = self.chunk_size if self.chunk_size > 0 else len(features)
        with _progress.bar(
            self.progress,
            total=len(features),
            desc=f"{self.display_name} {self.version}",
            unit="row",
        ) as progress:
            progress.set_postfix(context=self.n_context_)
            for start in range(0, len(features), max(size, 1)):
                stop = min(start + size, len(features))
                self._log(f"rows {start}:{stop} of {len(features)}")
                dist = self._predict_chunk(features[start:stop])
                progress.set_postfix(
                    context=self.n_context_,
                    **self._progress_postfix(dist),
                    refresh=False,
                )
                progress.update(stop - start)
                yield dist

    def _predict_distribution(
        self, X: pd.DataFrame
    ) -> distributions.Distribution:
        return distributions.concatenate(list(self._chunks(X)))

    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        # Chunk by chunk, so the full native container never exists at once.
        blocks = [dist.on_grid(grid) for dist in self._chunks(X)]
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _default_grid(self) -> grid_lib.Grid:
        """The constructor's grid, else this model's native grid."""
        if self.z_grid is None:
            return self.native_grid
        return super()._default_grid()

    def _log(self, message: str) -> None:
        """Prints a log line when ``verbose``."""
        if self.verbose:
            print(f"[{type(self).__name__}] {message}", flush=True)


def _mixture(
    parts: list[distributions.Distribution], weights: _typing.FloatArray
) -> distributions.Distribution:
    """Histogram parts as one distribution: summed when their buckets agree."""
    components: list[distributions.HistogramDistribution] = []
    component_weights: list[float] = []
    for part, weight in zip(parts, weights, strict=True):
        if isinstance(part, distributions.MixtureDistribution):
            components.extend(part.components)
            component_weights.extend(weight * part.weights)
        elif isinstance(part, distributions.HistogramDistribution):
            components.append(part)
            component_weights.append(float(weight))
        else:
            raise TypeError(
                f"cannot mix a {type(part).__name__}; mixtures take histograms"
            )
    bins = components[0].bins
    if all(np.array_equal(c.bins, bins) for c in components):
        masses = np.zeros_like(components[0].probabilities)
        for weight, component in zip(
            component_weights, components, strict=True
        ):
            masses += weight * component.probabilities
        return distributions.HistogramDistribution(bins, masses)
    return distributions.MixtureDistribution(
        tuple(components), np.asarray(component_weights)
    )
