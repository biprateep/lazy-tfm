# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The uniform feature layer every foundation-model backend is built on.

Every backend takes the same parameters (:data:`UNIFORM_PARAMS`), with the
same defaults (:data:`UNIFORM_DEFAULTS`; only ``version`` differs), and means
the same thing by them:

=======================  ===================================================
``n_estimators``         exactly this many ensemble members
``transforms``           per-member feature transforms (:mod:`._transforms`);
                         ``"auto"`` is the model's own recipe, written out by
                         the backend rather than inherited from the installed
                         upstream package
``feature_shuffle``      each member sees the feature columns permuted
``bag_size``             each member sees a random subset of context rows
``kv_cache``             cache the context's keys and values
``y_grid``               the default output grid; None is the native grid
``random_state``         the ensemble's seed
``chunk_size``           query rows per forward pass
``softmax_temperature``  divides the output logits; ``"auto"`` is the
                         checkpoint's calibrated value
``mixed_precision``      the model's reduced-precision path on a GPU
``outlier_threshold``    the soft clip at that many standard deviations
=======================  ===================================================

A backend translates each parameter to its model's own machinery where the
model has it, and the missing pieces are scaffolded here: rows, transforms,
outlier clipping and column permutations applied per member
(:mod:`._members`), members combined, the queries chunked. Every upstream
setting that changes a prediction is either driven by one of these parameters
or pinned by the backend, so no upstream default decides an answer silently.
A new backend subclasses :class:`ContextEnsembleEstimator`, declares what its
model has natively, and implements four small methods, under the contract
the class docstring sets out. The "Adding a backend" page of the
documentation gives a skeleton module and everything else a backend needs.

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
from collections.abc import Callable, Hashable, Iterator, Mapping
import numbers
import os
import sys
import types
from typing import Any, ClassVar, Literal, TypeGuard
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
    "AUTO",
    "REQUIRED_PROVENANCE",
    "UNIFORM_DEFAULTS",
    "UNIFORM_PARAMS",
    "ContextEnsembleEstimator",
    "ContextSizeWarning",
    "PerformanceWarning",
    "import_extra",
]

# Warnings skip every frame inside this package, so that they point at the
# user's call however deep in the package (or LazyModel) they are raised.
_PACKAGE_PREFIX = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
)

#: The parameters every backend's constructor must take, keyword-only.
UNIFORM_PARAMS: tuple[str, ...] = (
    "version",
    "n_estimators",
    "transforms",
    "feature_shuffle",
    "bag_size",
    "kv_cache",
    "y_grid",
    "device",
    "random_state",
    "chunk_size",
    "softmax_temperature",
    "mixed_precision",
    "outlier_threshold",
    "progress",
    "verbose",
)

#: Every uniform parameter's default, the same on every backend; ``version``
#: alone is per backend, because the checkpoints are.
UNIFORM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 8,
    "transforms": _transforms.AUTO,
    "feature_shuffle": True,
    "bag_size": None,
    "kv_cache": True,
    "y_grid": None,
    "device": "auto",
    "random_state": 0,
    "chunk_size": 8_192,
    "softmax_temperature": "auto",
    "mixed_precision": True,
    "outlier_threshold": "auto",
    "progress": "auto",
    "verbose": False,
}

#: The ``softmax_temperature`` and ``outlier_threshold`` value that defers to
#: the model's own setting.
AUTO = "auto"

#: The keys every fitted backend's ``provenance_`` carries: which weights and
#: code answered (:data:`lazy.models._hub.PROVENANCE_KEYS`, recorded by
#: ``_load_checkpoint``) and the ensemble's resolved settings (``_recipe``).
#: A backend may record more.
REQUIRED_PROVENANCE: tuple[str, ...] = (
    *_hub.PROVENANCE_KEYS,
    "n_estimators",
    "transforms",
    "feature_shuffle",
    "random_state",
    "bag_rows",
    "kv_cache",
    "chunk_size",
    "softmax_temperature",
    "mixed_precision",
    "outlier_threshold",
    "groups",
)

#: The distributions ``_predict_group`` may answer with, by ``native_output``.
#: A histogram model may answer with a mixture of histograms (TabFM's
#: dithered bins).
_NATIVE_TYPES: dict[str, tuple[type[distributions.Distribution], ...]] = {
    "histogram": (
        distributions.HistogramDistribution,
        distributions.MixtureDistribution,
    ),
    "quantiles": (distributions.QuantileDistribution,),
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

    The contract a subclass meets is the following. Its constructor takes every
    name in :data:`UNIFORM_PARAMS` as a keyword, with the default in
    :data:`UNIFORM_DEFAULTS`, stores each verbatim and takes no ``**kwargs``, so
    that ``cls()`` builds a working estimator (``register`` checks this).
    ``_fit`` splits the ensemble into groups of members that one call of the
    model serves together, and :meth:`_fit_group` returns, for each, a *handle*:
    any object, kept in ``handles_`` (and in ``regressor_`` when there is one
    group) and passed back unchanged to :meth:`_predict_group` for every chunk
    of queries. Besides returning it, a fit may set fitted attributes of its
    own, lower ``kv_cache_`` to the mode that actually ran and add keys to
    ``provenance_``, but it may not change the planned members, their rows or
    their seeds. :meth:`_check_backend_params` may likewise record the backend's
    own resolved parameters.

    Every answer of :meth:`_predict_group` is checked to be a distribution of
    the kind ``native_output`` declares with one row per query, and each row
    must depend on its own query alone; a prediction leaves the handle as it
    found it. After ``_fit``, ``provenance_`` holds every key in
    :data:`REQUIRED_PROVENANCE`, which is checked: :meth:`_load_checkpoint`
    records the checkpoint's and :meth:`_recipe` the ensemble's.

    A fitted estimator pickles and predicts the same after unpickling; a network
    shared through :mod:`lazy.models._weights` is kept through
    :meth:`_memoized_network`, which ``__getstate__`` drops, so it is fetched
    again on first use rather than pickled. The device is resolved
    into ``device_`` (``"cpu"``, ``"cuda"``, ``"cuda:<index>"`` or ``"mps"``)
    before the checkpoint loads; a backend runs there, in float32 unless
    ``mixed_precision_`` is set (only on CUDA), and raises on a device its model
    does not support.

    Shared code never names a backend. What a backend needs beyond the
    contract above is a hook the package calls on every registered class:
    :meth:`prefetch` for downloads beside the weights,
    :meth:`clear_upstream_caches` for upstream's own caches, :meth:`setup`
    for ``lazy setup``, and ``cpu_note`` for the CPU warning. Each defaults
    to doing nothing.

    Attributes:
        display_name: The model's name in progress bars and warnings.
        extra: The pip extra that installs the model.
        native_output: ``"histogram"`` or ``"quantiles"``.
        native_transforms: Uniform transform names the model implements,
            mapped to its own tokens.
        auto_tokens: The model's own default recipe as tokens, cycled by
            single-member groups under ``transforms="auto"``.
        supports_native_bagging: Whether the model can run several
            members on different rows in one call, given each member's bag
            (``MemberGroup.member_rows``); the rows are always lazy's, and
            a member with a scaffolded transform is a group of its own.
        member_combination: How groups combine: ``"mixture"`` of densities
            or ``"quantile_average"`` of quantile functions.
        kv_cache_modes: The accepted ``kv_cache`` values.
        kv_cache_rtol: How closely cached and uncached outputs agree on a
            CPU; under mixed precision on a GPU they agree to that
            precision's rounding instead.
        recommended_max_context: The context size above which the model
            degrades without bagging, or None.
        exact_chunking: Whether chunking the queries is exact bit for bit
            on a CPU; if not, a query's answer still does not depend on the
            others in its chunk, but differs from an unchunked one by float
            rounding. Under mixed precision on a GPU it differs by that
            precision's rounding on every model.
        chunks_queries: Whether the base class chunks the queries, or the
            model does it itself.
        has_softmax: Whether the model's output passes through a softmax,
            so that ``softmax_temperature`` means something; a model without
            one accepts only ``"auto"``.
        native_outlier_clipping: Whether the model applies the soft outlier
            clip itself, given a threshold; if not, the base class applies
            it to the features before the model sees them.
        cpu_friendly: Whether the model runs at a usable speed on a CPU; if
            not, a fit that falls back to the CPU under ``device="auto"``
            warns.
        cpu_friendly_versions: Versions that run at a usable speed on a CPU
            although the model as a whole does not (a smaller checkpoint,
            say); they fit on a CPU without the warning, which suggests them.
        cpu_note: What the CPU warning adds after "runs slowly on CPU", as a
            clause starting with a space; empty for nothing.
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
    has_softmax: ClassVar[bool] = True
    native_outlier_clipping: ClassVar[bool] = False
    cpu_friendly: ClassVar[bool] = False
    cpu_friendly_versions: ClassVar[tuple[str, ...]] = ()
    cpu_note: ClassVar[str] = ""

    # Set by each backend's __init__; declared for the type checker only.
    version: str
    n_estimators: int
    transforms: str | tuple[str, ...]
    feature_shuffle: bool
    bag_size: int | float | None
    kv_cache: bool | str
    device: str
    random_state: int | None
    chunk_size: int
    softmax_temperature: float | str
    mixed_precision: bool
    outlier_threshold: float | str | None
    progress: _progress.Progress
    verbose: bool

    # -- the per-backend interface -----------------------------------------

    @abc.abstractmethod
    def _import_backend(self) -> types.ModuleType:
        """Imports the model's package, naming the extra if it is missing.

        :func:`import_extra` does this for a package that imports by name.
        """

    def _before_fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        """Prepares a fit; runs once the backend's package is imported.

        It runs before any parameter is validated, so it may set what
        :meth:`_check_backend_params` reads (TabFM clips the context targets
        to its bins' support here). The default does nothing.

        Args:
            X: The whole context, shape (n_rows, n_features).
            y: Its target values, shape (n_rows,).
        """
        del X, y  # Unused: nothing to prepare.

    def _after_fit(self) -> None:
        """Finishes a fit that succeeded, once ``provenance_`` is checked.

        It may record more fitted attributes and provenance keys, but under
        the same rules as :meth:`_fit_group`. The default does nothing.
        """
        return

    def _check_backend_params(self) -> None:
        """Validates the backend's own parameters; raises ValueError.

        Runs after the uniform parameters are validated and before the
        checkpoint loads. It may record the backend's resolved parameters as
        fitted attributes (TabFM fixes its bins and inference path here).
        """
        return

    @abc.abstractmethod
    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> Any:
        """Fits one group of members; returns a handle for prediction.

        Called once per group, in order, after the checkpoint is loaded. It
        may set fitted attributes of its own, lower ``kv_cache_`` to the
        mode that actually ran and add keys to ``provenance_``; it may not
        change ``member_groups_`` or anything the plan fixed.

        Args:
            X: The group's context features, already restricted to its rows,
                transformed and permuted, shape (n_rows, n_features').
            y: Their target values, shape (n_rows,).
            group: The members to serve and how.

        Returns:
            The handle: any object, which the base class keeps in
            ``handles_`` and passes back unchanged to
            :meth:`_predict_group`. It is pickled with the estimator.
        """

    @abc.abstractmethod
    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.Distribution:
        """Predicts one group's distributions for prepared query features.

        Args:
            handle: What :meth:`_fit_group` returned for the group.
            X: Query features prepared as the group's context was, shape
                (n_queries, n_features').

        Returns:
            One row per query, each depending on its own query alone: a
            :class:`~lazy.distributions.HistogramDistribution` or a
            :class:`~lazy.distributions.MixtureDistribution` of them when
            ``native_output`` is ``"histogram"``, a
            :class:`~lazy.distributions.QuantileDistribution` when it is
            ``"quantiles"``. The base class checks both and raises
            otherwise. A prediction leaves the handle as it found it.
        """

    @abc.abstractmethod
    def _native_grid(self) -> grid_lib.Grid:
        """The model's own output grid; called at the end of ``_fit``."""

    def _load_checkpoint(self) -> None:
        """Fetches the pinned weights; sets ``checkpoint_`` and provenance.

        An override, say one that defers the download to the first
        prediction, must still set ``provenance_`` to a dict with every key
        in :data:`lazy.models._hub.PROVENANCE_KEYS`, as
        ``Checkpoint.provenance`` gives it.
        """
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

    def _auto_softmax_temperature(self) -> float | None:
        """The checkpoint's calibrated temperature, or None without a softmax.

        Called after the checkpoint is loaded, so a backend can read the
        value from it.
        """
        return None

    def _temperature(self) -> float:
        """The resolved ``softmax_temperature`` of a model with a softmax."""
        if self.softmax_temperature_ is None:
            raise RuntimeError(
                f"{type(self).__name__} has a softmax but no calibrated "
                "temperature: _auto_softmax_temperature returned None"
            )
        return self.softmax_temperature_

    def _auto_outlier_threshold(self) -> float | None:
        """The clip threshold of the model's own recipe, or None for none.

        What ``outlier_threshold="auto"`` resolves to under
        ``transforms="auto"``; under any explicit ``transforms`` it resolves
        to None, so an explicit recipe is all the model sees.
        """
        return None

    @classmethod
    def _pinned_recipe(cls, version: str) -> Mapping[str, Any]:
        """The model's own recipe at ``version``, as the backend pins it.

        Every upstream setting the backend fixes for that version rather than
        leaving to the installed package: what ``transforms="auto"`` runs,
        and the settings it passes under any ``transforms``. A recipe is read
        from upstream for one checkpoint, so a version gains one only when
        someone has checked it; tests/lazy/test_backend_completeness.py
        reports a pinned checkpoint without one.

        Args:
            version: A model version with a pinned checkpoint.

        Returns:
            The recipe's settings by name.

        Raises:
            KeyError: If the backend pins no recipe for ``version``.
        """
        raise KeyError(f"{cls.__name__} pins no recipe for {version!r}")

    # -- hooks the rest of the package calls on every registered backend ----

    @classmethod
    def prefetch(cls, version: str, *, download: bool = True) -> None:
        """Fetches what the model needs beside its checkpoint's weights.

        :func:`lazy.download_checkpoint` calls it before fetching the
        weights, and :func:`lazy.is_cached` with ``download=False``, so that
        a model with more to fetch (source code that is not on PyPI, say)
        can be warmed up before going offline. The default fetches nothing.

        Args:
            version: The model version whose weights are being fetched.
            download: Whether to fetch what is missing; if False, only check
                that it is already there.

        Raises:
            ImportError: If something the model needs is missing and
                ``download`` is False, or cannot be fetched.
        """
        del version, download  # Unused: nothing to fetch.

    @classmethod
    def clear_upstream_caches(cls) -> None:
        """Empties the caches the upstream package keeps of its own.

        :func:`lazy.clear_model_cache` calls it, so that the memory held by
        an upstream cache of checkpoints or networks comes back with the
        package's own. The default has nothing to empty.
        """
        return

    @classmethod
    def setup(cls, *, dry_run: bool) -> bool:
        """Finishes installing the model where its pip extra cannot.

        ``lazy setup`` calls it on every registered backend. It installs or
        fetches what PyPI cannot provide, prints a line saying what it did,
        and does nothing when the extra is not installed or the install is
        already complete, so that it is safe to run again. The default has
        nothing to do and prints nothing.

        Args:
            dry_run: Say what would be done, and change nothing.

        Returns:
            Whether the model is complete, or would be after a dry run;
            True for a model whose extra is not installed.
        """
        del dry_run  # Unused: nothing to do.
        return True

    # -- fitting -----------------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        # Set below only when one group serves the whole ensemble, so a
        # refit into several groups must not keep the previous fit's.
        self.__dict__.pop("regressor_", None)
        # A missing backend package is the likeliest reason a first fit
        # fails, so it is reported before any parameter complaint.
        self._import_backend()
        self._before_fit(X, y)
        self._check_uniform_params()
        self._check_backend_params()
        _progress.check_progress(self.progress)
        self.random_state_ = _resolve_seed(self.random_state)
        self.device_ = _device.resolve_device(self.device)
        # Reduced precision exists only on a GPU; on a CPU every backend runs
        # in float32.
        self.mixed_precision_ = bool(self.mixed_precision) and str(
            self.device_
        ).startswith("cuda")
        self._warn_if_slow_on_cpu()
        self._load_checkpoint()
        self.softmax_temperature_: float | None = (
            self._auto_softmax_temperature()
            if self.softmax_temperature == AUTO
            else float(self.softmax_temperature)
        )
        self.outlier_threshold_ = self._resolve_outlier_threshold()
        n_rows, n_features = X.shape
        self.n_context_ = int(n_rows)
        self.bag_rows_ = _members.resolve_bag_size(self.bag_size, n_rows)
        self.bagging_ = self.bag_rows_ < n_rows
        self._warn_if_bags_too_small()
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
            random_state=self.random_state_,
            supports_native_bagging=self.supports_native_bagging,
            auto_tokens=self.auto_tokens,
        )
        features = X.to_numpy(dtype=np.float64)
        self.transformers_: list[_transforms.ScaffoldTransform | None] = []
        self.clippers_: list[_transforms.SoftClip | None] = []
        self.handles_: list[Any] = []
        threshold = self.outlier_threshold_
        scaffold_clip = (
            threshold is not None and not self.native_outlier_clipping
        )
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
            clipper = None
            if scaffold_clip and threshold is not None:
                transformed = (
                    context
                    if transformer is None
                    else transformer.transform(context)
                )
                clipper = _transforms.SoftClip(threshold).fit(transformed)
            self.clippers_.append(clipper)
            prepared = self._prepare(context, group, transformer, clipper)
            self._log(
                f"fitting group {group.index} ({group.n_members} members) on "
                f"{len(prepared)} context rows ({self.device_})"
            )
            self.handles_.append(self._fit_group(prepared, y[rows], group))
        if len(self.handles_) == 1:
            self.regressor_ = self.handles_[0]
        self.native_grid_ = self._native_grid()
        self.provenance_ = {
            **getattr(self, "provenance_", {}),
            **self._recipe(),
        }
        self._check_provenance()
        self._after_fit()

    def _check_provenance(self) -> None:
        """Raises unless ``provenance_`` holds :data:`REQUIRED_PROVENANCE`."""
        missing = [
            key for key in REQUIRED_PROVENANCE if key not in self.provenance_
        ]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__}.provenance_ lacks {missing}: "
                "_load_checkpoint records the checkpoint's keys "
                "(Checkpoint.provenance) and _recipe the ensemble's"
            )

    def _check_uniform_params(self) -> None:
        """Validates the uniform parameters, before anything is loaded."""
        known = _hub.list_versions(self.backend or "")
        if known and self.version not in known:
            raise ValueError(
                f"unknown version {self.version!r} for {self.backend!r}; "
                f"known: {known}"
            )
        if not _is_integer(self.chunk_size) or self.chunk_size < 0:
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
        if self.random_state is not None and not _is_integer(self.random_state):
            raise ValueError(
                f"random_state must be an int or None: {self.random_state=}"
            )
        self._n_members()
        temperature = self.softmax_temperature
        if temperature != AUTO:
            if not _is_positive_real(temperature):
                raise ValueError(
                    "softmax_temperature must be 'auto' or a positive number: "
                    f"{self.softmax_temperature=}"
                )
            if not self.has_softmax:
                raise ValueError(
                    f"{self.display_name} has no softmax output, so "
                    "softmax_temperature must be 'auto': "
                    f"{self.softmax_temperature=}"
                )
        if not isinstance(self.mixed_precision, bool):
            raise ValueError(
                f"mixed_precision must be a bool: {self.mixed_precision=}"
            )
        threshold = self.outlier_threshold
        if not (
            threshold is None
            or threshold == AUTO
            or _is_positive_real(threshold)
        ):
            raise ValueError(
                "outlier_threshold must be 'auto', None or a positive number "
                f"of standard deviations: {self.outlier_threshold=}"
            )

    def _resolve_outlier_threshold(self) -> float | None:
        """The clip threshold in force: ``"auto"`` follows ``transforms``."""
        if self.outlier_threshold is None:
            return None
        if self.outlier_threshold == AUTO:
            if _transforms.parse(self.transforms) is None:
                return self._auto_outlier_threshold()
            return None
        return float(self.outlier_threshold)

    def _n_members(self) -> int:
        """The ensemble size, validated."""
        n_estimators = self.n_estimators
        if not _is_integer(n_estimators) or n_estimators < 1:
            raise ValueError(
                f"n_estimators must be a positive int: {self.n_estimators=}"
            )
        return int(n_estimators)

    def _warn_if_slow_on_cpu(self) -> None:
        """Warns when ``"auto"`` fell back to the CPU for a GPU model."""
        # Imported here: the registry imports every backend, hence this.
        from lazy.models import registry  # noqa: PLC0415

        auto = str(self.device).strip().lower() == "auto"
        friendly_here = (
            self.cpu_friendly or self.version in self.cpu_friendly_versions
        )
        if friendly_here or not auto or self.device_ != "cpu":
            return
        friendly = []
        for name, cls in sorted(registry.ESTIMATORS.items()):
            if getattr(cls, "cpu_friendly", False):
                friendly.append(f"LazyModel({name!r})")
            friendly.extend(
                f"LazyModel({name!r}, version={version!r})"
                for version in getattr(cls, "cpu_friendly_versions", ())
            )
        models = " or ".join(friendly)
        suggestion = (
            f"For CPU work prefer a CPU-friendly model: {models}. "
            if friendly
            else ""
        )
        warnings.warn(
            f"PyTorch sees no GPU, so device='auto' runs {self.display_name} "
            f"on the CPU. {self.display_name} runs slowly on CPU"
            f"{self.cpu_note}. {suggestion}Pass "
            "device='cpu' to run it there anyway without this warning; see "
            "'Supported models' in the documentation.",
            PerformanceWarning,
            skip_file_prefixes=(_PACKAGE_PREFIX,),
        )

    def _warn_if_bags_too_small(self) -> None:
        """Warns when bags are so small an int was likely meant as a float."""
        if not self.bagging_ or self.bag_rows_ >= _members.MIN_BAG_ROWS:
            return
        warnings.warn(
            f"bag_size={self.bag_size!r} gives each member only "
            f"{self.bag_rows_} context row(s). An int bag_size is a row "
            "count and a float one a fraction of the context: bag_size=1 is "
            "one row, bag_size=1.0 all of them.",
            UserWarning,
            skip_file_prefixes=(_PACKAGE_PREFIX,),
        )

    def _warn_if_context_too_large(self, n_rows: int) -> None:
        """Warns when a member's context exceeds what the model handles."""
        limit = self._recommended_max_context()
        if limit is None or self.bag_rows_ <= limit:
            return
        needed = -(-n_rows // limit)
        seen = (
            f"each bag has {self.bag_rows_:,} of the context's {n_rows:,}"
            if self.bagging_
            else f"this one has {n_rows:,}"
        )
        warnings.warn(
            f"{self.display_name} degrades on contexts larger than about "
            f"{limit:,} rows, and {seen}. Bag the context in smaller pieces: "
            f"pass bag_size={limit} with n_estimators >= {needed} so the "
            "members together cover it.",
            ContextSizeWarning,
            skip_file_prefixes=(_PACKAGE_PREFIX,),
        )

    def _recipe(self) -> dict[str, Any]:
        """What the ensemble was, for ``provenance_``."""
        members: list[_members.MemberSpec] = []
        for group in self.member_groups_:
            members.extend(group.members)
        return {
            "n_estimators": int(self.n_estimators),
            "transforms": [
                "auto" if m.transform is None else m.transform.name
                for m in sorted(members, key=lambda m: m.index)
            ],
            "feature_shuffle": self.feature_shuffle,
            "random_state": self.random_state_,
            "bag_rows": self.bag_rows_ if self.bagging_ else None,
            "kv_cache": self.kv_cache_,
            "chunk_size": int(self.chunk_size),
            "softmax_temperature": self.softmax_temperature_,
            "mixed_precision": self.mixed_precision_,
            "outlier_threshold": self.outlier_threshold_,
            "groups": len(self.member_groups_),
        }

    # -- predicting ----------------------------------------------------------

    def _prepare(
        self,
        features: _typing.FloatArray,
        group: _members.MemberGroup,
        transformer: _transforms.ScaffoldTransform | None,
        clipper: _transforms.SoftClip | None = None,
    ) -> _typing.FloatArray:
        """A group's view of some features: transformed, clipped, permuted."""
        if transformer is not None:
            features = transformer.transform(features)
        if clipper is not None:
            features = clipper.transform(features)
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
        clippers = getattr(self, "clippers_", None) or [None] * len(
            self.handles_
        )
        parts = [
            self._check_group_output(
                self._predict_group(
                    handle, self._prepare(features, group, fitted, clipper)
                ),
                len(features),
            )
            for handle, group, fitted, clipper in zip(
                self.handles_,
                self.member_groups_,
                self.transformers_,
                clippers,
                strict=True,
            )
        ]
        return self._combine(parts)

    def _check_group_output(
        self, part: object, n_rows: int
    ) -> distributions.Distribution:
        """``_predict_group``'s answer, once it is checked against the contract.

        Args:
            part: What ``_predict_group`` returned.
            n_rows: How many query rows it was given.

        Returns:
            ``part`` itself, unchanged.

        Raises:
            TypeError: If ``part`` is not a distribution of the kind
                ``native_output`` declares.
            RuntimeError: If it does not have one row per query.
        """
        kinds = _NATIVE_TYPES[self.native_output]
        if not isinstance(part, kinds):
            expected = " or ".join(kind.__name__ for kind in kinds)
            raise TypeError(
                f"{type(self).__name__}._predict_group returned a "
                f"{type(part).__name__}, but native_output="
                f"{self.native_output!r} needs a {expected}"
            )
        if len(part) != n_rows:
            raise RuntimeError(
                f"{type(self).__name__}._predict_group returned {len(part)} "
                f"rows for {n_rows} query rows"
            )
        return part

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
        chunk_size = int(self.chunk_size)
        size = chunk_size if chunk_size > 0 else len(features)
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
        if X.shape[0] == 0:
            # The empty answer still carries the model's buckets or levels,
            # which only a prediction reveals: predict one placeholder row
            # and keep none of it.
            placeholder = np.zeros((1, X.shape[1]))
            return self._predict_chunk(placeholder)[:0]
        return distributions.concatenate(list(self._chunks(X)))

    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        if X.shape[0] == 0:
            return np.empty((0, grid.n_bins))
        # Chunk by chunk, so the full native container never exists at once.
        blocks = [dist.on_grid(grid) for dist in self._chunks(X)]
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _default_grid(self) -> grid_lib.Grid:
        """The constructor's grid, else this model's native grid."""
        if self.y_grid is None:
            return self.native_grid
        return super()._default_grid()

    # -- the network ---------------------------------------------------------

    def _memoized_network(
        self, memo_key: Hashable, get: Callable[[], Any]
    ) -> Any:
        """The loaded network, kept on the instance until ``memo_key`` changes.

        A backend that loads its network outside the fit (lazily, on the
        first prediction, say) keeps it here rather than in an attribute of
        its own: the reference is dropped on pickling and on cloning, and an
        unpickled estimator calls ``get`` again on first use.

        Args:
            memo_key: Everything that decides which network the instance
                needs, such as its version, device and precision; a change
                calls ``get`` again, so that no prediction can run on the
                previous one.
            get: Loads or finds the network, usually through
                :func:`lazy.models._weights.network`; called only when the
                instance holds none for ``memo_key``.

        Returns:
            The network.
        """
        cached_key, network = getattr(self, "_network_cache", (None, None))
        if network is None or cached_key != memo_key:
            network = get()
            self._network_cache = (memo_key, network)
        return network

    def __getstate__(self) -> dict[str, Any]:
        """Pickles without the memoized network, which reloads on first use."""
        state = dict(super().__getstate__())
        state.pop("_network_cache", None)
        return state

    def _log(self, message: str) -> None:
        """Prints a log line when ``verbose``."""
        if self.verbose:
            print(f"[{type(self).__name__}] {message}", flush=True)


def import_extra(
    module: str, *, needed_by: str, extra: str
) -> types.ModuleType:
    """Imports a backend's optional package, naming the extra if it is missing.

    Only the package itself missing is a missing extra; anything it fails to
    import in turn is reported as it is.

    Args:
        module: The package to import, such as ``"tabicl"``.
        needed_by: The class that needs it, for the message.
        extra: The pip extra of lazy-tfm that installs it.

    Returns:
        The imported package.

    Raises:
        ImportError: Naming the extra when ``module`` is not installed, or
            upstream's own when something it imports is missing or broken.
    """
    try:
        # What `import module` runs, so that it fails as the statement would.
        __import__(module)
    except ImportError as error:
        if (error.name or "").partition(".")[0] != module.partition(".")[0]:
            raise
        raise ImportError(
            f"{needed_by} needs the {module} backend: "
            f"pip install 'lazy-tfm[{extra}]'"
        ) from error
    return sys.modules[module]


def _resolve_seed(random_state: int | None) -> int:
    """The ensemble's seed: ``random_state``, or fresh entropy for None.

    Every member's seed is derived from it (:func:`._members.member_seed`);
    a drawn one is kept below 2**31, as an explicit one usually is.
    """
    if random_state is None:
        return int(np.random.SeedSequence().entropy % 2**31)  # type: ignore[operator]  # entropy is an int when drawn
    return int(random_state)


def _is_positive_real(value: object) -> bool:
    """Whether ``value`` is a finite real number above zero, and not a bool."""
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool | np.bool_)
        and bool(np.isfinite(float(value)))
        and float(value) > 0
    )


def _is_integer(value: object) -> TypeGuard[int]:
    """Whether ``value`` is a Python or NumPy integer, and not a bool.

    Typed as a guard for ``int``, which NumPy integers stand in for.
    """
    return isinstance(value, numbers.Integral) and not isinstance(
        value, bool | np.bool_
    )


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
