# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Target distributions from TabICLv2's quantile regression head.

TabICLv2 (Qu et al. 2026) is an in-context tabular foundation model whose
regressor answers with a *distribution*: 999 quantiles of the predictive CDF
per query row, not a single number. :class:`TabICLQuantile` returns them as a
:class:`~lazy.distributions.QuantileDistribution`, whose densities on any
grid come from evaluating the quantile CDF at the bin edges and differencing
(:meth:`lazy.grid.Grid.from_quantiles`): every bin gets exactly the
mass the quantiles place inside it.

The uniform features map onto TabICL's own machinery: ``kv_cache`` onto its
key/value cache, ``feature_shuffle`` onto its Latin-square column shuffles,
``outlier_threshold`` onto its soft outlier clip and ``mixed_precision`` onto
its float16 autocast. ``transforms="auto"`` is TabICL's own recipe, pinned
here (:data:`AUTO_NORM_METHODS`, :data:`AUTO_FEAT_SHUFFLE_METHOD`,
:data:`AUTO_OUTLIER_THRESHOLD`): members alternate between no transform and
a Yeo-Johnson power transform, and every column is soft-clipped at four
standard deviations. Of the uniform transforms only ``none`` is TabICL's
own; the others are applied by lazy, since TabICL's methods of the same names
differ from them (see ``native_transforms``). TabICL cannot subsample rows,
so ``bag_size`` is scaffolded: one single-member regressor per bag, their
quantile functions averaged, as TabICL averages its own members.

What TabICL always does to the features, under any ``transforms``: missing
values are filled with the context column's mean; columns with a single
value in the context are dropped; every column is z-scored on the context
and clipped to +-100 standard deviations. Under ``transforms="auto"`` each
member then applies its norm method and the 4-sigma clip; under an explicit
``transforms`` the clip is off unless ``outlier_threshold`` is a number.

Compared with :class:`lazy.models.tabfm.TabFMHistogram`, this backbone is far
smaller (a ~100 MB checkpoint rather than ~6.6 GB) and much faster, at the cost
of a density whose resolution is set by the spacing of the quantiles rather
than by the data.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterator, Mapping
import contextlib
import dataclasses
import functools
import math
import types
from typing import Any

import numpy as np

from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _device
from lazy.models import _ensemble
from lazy.models import _hub
from lazy.models import _locks
from lazy.models import _members
from lazy.models import _progress
from lazy.models import _quantile_head
from lazy.models import _weights

__all__ = [
    "AUTO_FEAT_SHUFFLE_METHOD",
    "AUTO_NORM_METHODS",
    "AUTO_OUTLIER_THRESHOLD",
    "NATIVE_PAD",
    "NATIVE_QUANTILE_BINS",
    "TabICLQuantile",
    "quantile_levels",
]

#: Bins of the native grid across the training targets' range: as fine as
#: the 999 quantiles resolve. The padding adds bins of the same width.
NATIVE_QUANTILE_BINS = _quantile_head.NATIVE_QUANTILE_BINS

#: How far the native grid extends beyond the training targets on each
#: side, as a fraction of their range: TabICL's quantiles extrapolate.
NATIVE_PAD = _quantile_head.NATIVE_PAD

#: The cumulative probabilities TabICL's quantile outputs sit at.
quantile_levels = _quantile_head.quantile_levels

# TabICLv2's own recipe, ``transforms="auto"``, written out here so that the
# installed upstream version cannot change it: the values TabICLRegressor
# 2.2.0 defaults to, each passed explicitly.

#: The norm methods members cycle through under ``transforms="auto"``.
AUTO_NORM_METHODS: tuple[str, ...] = ("none", "power")

#: How members' columns are permuted when ``feature_shuffle=True``.
AUTO_FEAT_SHUFFLE_METHOD = "latin"

#: The soft outlier clip of the recipe, in standard deviations.
AUTO_OUTLIER_THRESHOLD = 4.0

#: The threshold that turns upstream's clip off: its bounds become infinite,
#: and it then returns every value unchanged.
_CLIP_OFF = math.inf

#: Every other TabICLRegressor argument, pinned. None of them changes an
#: answer beyond float rounding, but none is left to upstream either.
_PINNED_REGRESSOR_ARGS: dict[str, Any] = {
    # Members per forward pass.
    "batch_size": 8,
    # FlashAttention-3 needs its own package on a Hopper GPU, and swaps the
    # attention kernel where it runs; off, so that the kernel never depends
    # on what happens to be installed.
    "use_fa3": False,
    # Where the column embeddings are kept between layers: memory only.
    "offload_mode": "auto",
    "disk_offload_dir": None,
    # The checkpoint is handed over by path; a missing file is an error,
    # never a download of upstream's default.
    "allow_auto_download": False,
    # PyTorch's own thread count.
    "n_jobs": None,
    # Built by upstream from the arguments here.
    "inference_config": None,
    "verbose": False,
}

#: The versions the recipe above was read from upstream for. A new version
#: joins once its recipe is checked against upstream's.
_RECIPE_VERSIONS: tuple[str, ...] = ("v2",)


class TabICLQuantile(
    _quantile_head.QuantileHead, _ensemble.ContextEnsembleEstimator
):
    """Target distributions from the quantiles TabICLv2's regressor predicts.

    Args:
        version: Which pinned TabICL checkpoint to load; see
            :func:`lazy.list_versions`. Recorded in ``provenance_``.
        n_estimators: Ensemble members, exactly; costs scale linearly, and 8
            is the value the benchmarks use. A TabICL regressor pairs each
            of its column shuffles with each norm method, and has only one
            shuffle per column (a Latin square), so where it would run fewer
            members than asked, further regressors with seeds of their own
            run the rest; with a single column order (``feature_shuffle=
            False``, or one usable column) members with the same transform
            are identical, so each runs once and is weighted by its count.
            Member ``i`` takes transform ``i`` of the cycle, as TabICL's
            own do.
        transforms: Per-member feature transforms: ``"auto"`` (TabICL's own
            recipe, ``none`` and ``power`` alternating, with the 4-sigma
            clip), a recipe name, a transform name or a sequence of them;
            see :mod:`lazy.models._transforms`. Only ``none`` maps onto
            TabICL's norm methods; every other name is applied by lazy and
            TabICL is run with no norm method of its own. An explicit
            recipe therefore differs from ``"auto"`` even when it names the
            same transforms: TabICL's ``power`` is fitted on z-scored
            columns.
        feature_shuffle: Whether members see the columns in different orders
            (TabICL's Latin-square shuffles, ``feat_shuffle_method="latin"``;
            ``"none"`` when False). A regressor running one member keeps the
            columns in their given order.
        bag_size: Context rows per member: an int is a row count (1 means
            one row), a float a fraction in (0, 1] (1.0 means all rows), and
            None all of them. Scaffolded: one regressor per bag, with lazy's
            own column permutation when ``feature_shuffle=True``, its
            transforms, clip and target standardisation fitted on the bag.
        kv_cache: Cache the context's keys and values at fit, so each chunk
            of queries skips the context forward pass: ``True`` (TabICL's
            ``"kv"`` cache), ``"repr"`` (cached row representations, far
            smaller, re-running the in-context layers) or ``False``. Exact
            on a CPU, to float32 rounding; on a GPU with
            ``mixed_precision=True`` the cache is stored in float16, and the
            cached and uncached answers agree to float16 rounding (about
            1e-3 of the quantiles' spread).
        y_grid: Default output grid: a :class:`lazy.grid.Grid`, an
            array of bin centres, ``"native"``, or None for the native grid
            (equal-width bins, 1,000 across the training targets' range,
            which extend a quarter of that range beyond it on each side).
            Densities on a grid carry only the mass inside it, renormalised;
            tabulating a row that puts more than 1% of its probability
            outside the native grid warns.
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"mps"``, ``"cpu"``, or a ``torch.device``.
        random_state: Seed for the ensemble. A group's first regressor is
            seeded by the group's first member,
            ``SeedSequence([random_state, i])``, and every other seed is
            derived from that one. None draws a fresh seed at fit, recorded as
            ``random_state_`` and in ``provenance_``.
        chunk_size: Query rows predicted at a time, to bound peak memory
            (999 quantiles per row is about 8 kB); ``0`` does them in one
            pass. TabICL builds its keys and values from the context rows
            alone, so a row's answer never depends on the other rows in its
            chunk beyond float rounding: the batch shape can change the
            kernels (on a CPU, a one-row chunk differs by a few parts in a
            million of the quantiles' spread), and with
            ``mixed_precision=True`` on a GPU the answers agree to float16
            rounding.
        softmax_temperature: Only ``"auto"``: TabICL's quantile head has no
            softmax for a temperature to divide, so any other value raises
            ValueError. Recorded in ``provenance_`` as None.
        mixed_precision: On CUDA, TabICL's float16 autocast for every
            context size (``use_amp=True``; upstream's own ``"auto"`` would
            turn it on only from 1,024 context rows or 60 features). False,
            or any device other than CUDA, runs in float32
            (``use_amp=False``). FlashAttention-3 is pinned off either way.
        outlier_threshold: TabICL's soft outlier clip, applied to each
            member's z-scored and transformed columns, at this many standard
            deviations of the context (``outlier_threshold`` upstream).
            ``"auto"`` is 4.0 under ``transforms="auto"`` and off under an
            explicit ``transforms``; None is off (upstream's bounds are made
            infinite, which leaves every value unchanged).
        progress: A progress bar over the query rows: ``"auto"`` shows it
            on a terminal or in a notebook, ``True`` always, ``False`` never.
        verbose: Print log messages to stdout.

    Attributes:
        grid_: The resolved default output grid.
        native_grid_: The native grid, histogram-normalised: 1,500 bins over
            the training targets' range (all of them, bagged or not) padded
            by a quarter of it on each side (a constant target is padded by
            1% of its value, and at least by 0.01).
        checkpoint_: The pinned checkpoint file, a :class:`pathlib.Path`.
        provenance_: Which weights, code and ensemble answered, as a dict.
        regressor_: The fitted ``tabicl.TabICLRegressor``, when one serves
            the whole ensemble.
        handles_: Per member group, the fitted regressors that run it, with
            the members each stands for (also in ``provenance_``).
        n_quantiles_: How many quantiles the backbone returns per row.
        n_context_: Context rows ``fit`` was given.

    Examples:
        >>> est = TabICLQuantile(n_estimators=8, chunk_size=50_000)
        >>> est.chunk_size
        50000
    """

    backend = "tabicl"
    display_name = "TabICL"
    method_note = "Quantiles of TabICLv2's regression head."
    extra = "tabicl"
    native_output = "quantiles"
    # TabICL's other norm methods are not the uniform transforms of the same
    # names: it fits them on z-scored columns (power, robust), with sklearn's
    # defaults (quantile: 1,000 quantiles of a 10,000-row subsample), with
    # its own noise and a trailing rescaling (quantile_rtdl), and nothing
    # rescales their output. lazy applies those names itself.
    native_transforms = {"none": "none"}
    auto_tokens = AUTO_NORM_METHODS
    supports_native_bagging = False
    member_combination = "quantile_average"
    kv_cache_modes = (True, False, "repr")
    # A quantile head: there is no softmax for a temperature to divide.
    has_softmax = False
    native_outlier_clipping = True
    # On a CPU, where the conformance tests run, the cache agrees to float32
    # rounding (about 1e-6 of the quantiles' spread), and so does chunking:
    # most chunk sizes give the same bits, but a one-row chunk takes a
    # different matrix-multiply path in the cached decode, about 6e-6 of the
    # spread away. On a GPU with mixed_precision=True the cache is stored in
    # float16 and the batch shapes change the kernels: both then agree to
    # float16 rounding, about 1e-3 of the spread.
    kv_cache_rtol = 1e-4
    exact_chunking = False
    cpu_friendly = True

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v2",
        n_estimators: int = 8,
        transforms: str | tuple[str, ...] = "auto",
        feature_shuffle: bool = True,
        bag_size: int | float | None = None,
        kv_cache: bool | str = True,
        y_grid: grid_lib.GridLike = None,
        device: str = "auto",
        random_state: int | None = 0,
        chunk_size: int = 8_192,
        softmax_temperature: float | str = "auto",
        mixed_precision: bool = True,
        outlier_threshold: float | str | None = "auto",
        progress: _progress.Progress = "auto",
        verbose: bool = False,
    ):
        self.version = version
        self.n_estimators = n_estimators
        self.transforms = transforms
        self.feature_shuffle = feature_shuffle
        self.bag_size = bag_size
        self.kv_cache = kv_cache
        self.y_grid = y_grid
        self.device = device
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.softmax_temperature = softmax_temperature
        self.mixed_precision = mixed_precision
        self.outlier_threshold = outlier_threshold
        self.progress = progress
        self.verbose = verbose

    def _import_backend(self) -> types.ModuleType:
        return _ensemble.import_extra(
            "tabicl", needed_by=type(self).__name__, extra=self.extra
        )

    def _auto_outlier_threshold(self) -> float | None:
        return AUTO_OUTLIER_THRESHOLD

    @classmethod
    def _pinned_recipe(cls, version: str) -> Mapping[str, Any]:
        if version not in _RECIPE_VERSIONS:
            return super()._pinned_recipe(version)
        return types.MappingProxyType(
            {
                "norm_methods": AUTO_NORM_METHODS,
                "feat_shuffle_method": AUTO_FEAT_SHUFFLE_METHOD,
                "outlier_threshold": AUTO_OUTLIER_THRESHOLD,
                **_PINNED_REGRESSOR_ARGS,
            }
        )

    def _after_fit(self) -> None:
        # The base class keeps the one group's handle; regressor_ is the
        # TabICLRegressor itself, when one serves the whole ensemble.
        handle = self.__dict__.pop("regressor_", None)
        if handle is not None and len(handle.regressors) == 1:
            self.regressor_ = handle.regressors[0]
        # Every regressor runs on the one shared network, whose head fixes
        # how many quantiles a row gets.
        network = self.handles_[0].regressors[0].model_
        self.n_quantiles_ = int(network.num_quantiles)

    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> _RegressorGroup:
        # TabICL works in float32, which cannot resolve a narrow spread about
        # a large offset; standardising in float64 first keeps it. Each
        # group is standardised on its own rows, a bagged member on its bag.
        offset, scale = _quantile_head.target_scaling(y)
        handle = self._fit_members(
            # The features stay float64: upstream fits its scaling and power
            # transform in float64, and casts to float32 only for the model.
            X,
            ((y - offset) / scale).astype(np.float32),
            group,
        )
        return dataclasses.replace(handle, offset=offset, scale=scale)

    def _fit_members(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> _RegressorGroup:
        """Regressors that together run exactly the group's members.

        Upstream pairs each of its column shuffles with each norm method and
        keeps the first ``n_estimators`` pairs, so it runs fewer members
        than asked when there are too few shuffles: a Latin square has one
        per column, and ``feat_shuffle_method="none"`` (or one member) has
        one. The shortfall is made up by further regressors with seeds of
        their own, which start a fresh cycle of the norm methods. When
        there is only one shuffle, every member with a norm method is the
        same member, so each norm method runs once and is weighted by the
        members it stands for.

        Args:
            X: The group's prepared context features, float64, shape
                ``(n_rows, n_features)``.
            y: Their standardised targets, float32, shape ``(n_rows,)``.
            group: The members to run.

        Returns:
            The fitted regressors, with the members each stands for.

        Raises:
            RuntimeError: If upstream ran members other than the planned
                ones, which would make the weights wrong.
        """
        methods = _norm_methods(
            AUTO_NORM_METHODS
            if group.native_transforms is None
            else group.native_transforms
        )
        shuffle = AUTO_FEAT_SHUFFLE_METHOD if group.feature_shuffle else "none"
        n_members = group.n_members
        planned = collections.Counter(
            methods[i % len(methods)] for i in range(n_members)
        )

        def fit(n: int, norm_methods: list[str], seed: int) -> Any:
            regressor = self._regressor(n, norm_methods, shuffle, seed)
            with self._shared_network(regressor):
                regressor.fit(X, y)
            return _order_members(regressor, norm_methods)

        # Each regressor, with the members it stands for by norm method.
        fitted: list[tuple[Any, collections.Counter[str]]] = []
        if shuffle != "none" or n_members <= len(methods):
            first = fit(n_members, methods, group.seed)
            fitted.append((first, _members_run(first)))
            if fitted[0][1].total() < n_members and _n_shuffles(first) == 1:
                fitted = []
        if not fitted:
            fitted = [
                (fit(1, [m], group.seed), collections.Counter({m: planned[m]}))
                for m in methods
            ]
        while (remaining := n_members - sum(c.total() for _, c in fitted)) > 0:
            seed = _extra_seed(group.seed, len(fitted))
            regressor = fit(remaining, methods, seed)
            fitted.append((regressor, _members_run(regressor)))
        handle = _RegressorGroup(
            regressors=tuple(regressor for regressor, _ in fitted),
            members=tuple(dict(counts) for _, counts in fitted),
            feat_shuffle_method=shuffle,
        )
        if handle.members_by_method() != dict(planned):
            raise RuntimeError(
                f"TabICL ran the members {handle.members_by_method()} where "
                f"{dict(planned)} were planned"
            )
        return handle

    def _regressor(
        self,
        n_estimators: int,
        norm_methods: list[str],
        shuffle: str,
        seed: int,
    ) -> Any:
        """A ``TabICLRegressor`` with every argument set here.

        Nothing is left to upstream's defaults: the uniform parameters drive
        what they mean, and the rest of the recipe is pinned
        (:data:`AUTO_NORM_METHODS`, :data:`AUTO_FEAT_SHUFFLE_METHOD`,
        :data:`AUTO_OUTLIER_THRESHOLD`, :data:`_PINNED_REGRESSOR_ARGS`).
        """
        tabicl = self._import_backend()
        threshold = self.outlier_threshold_
        return tabicl.TabICLRegressor(
            n_estimators=n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=shuffle,
            outlier_threshold=_CLIP_OFF if threshold is None else threshold,
            kv_cache=self.kv_cache,
            # The pinned checkpoint is handed over by path, never left to
            # TabICL's own default, which upstream is free to bump: this way
            # `CHECKPOINTS` is the authority on which weights answer.
            model_path=str(self.checkpoint_),
            checkpoint_version=self.checkpoint_.name,
            device=self.device_,
            # Upstream's "auto" switches float16 on only for contexts of
            # 1,024 rows or 60 features; mixed precision means it always.
            use_amp=self.mixed_precision_,
            random_state=seed,
            **_PINNED_REGRESSOR_ARGS,
        )

    @contextlib.contextmanager
    def _shared_network(self, regressor: Any) -> Iterator[None]:
        """Has a regressor's fit load its network through the shared cache.

        Upstream's ``fit`` builds a fresh ``TabICL`` from the checkpoint
        every time. Its network holds nothing a fit needs to keep (the
        key/value caches are the regressor's ``model_kv_cache_``), so every
        fit with the same checkpoint and device runs on one, through
        :mod:`lazy.models._weights`. The loader is swapped on this one
        regressor, for the duration of its fit, so a pickled regressor
        reloads from the checkpoint as upstream's own does. The weights
        stay in float32: ``use_amp`` autocasts and casts nothing.

        Upstream does keep the cache it records last on the network, so the
        fit holds the network's lock (:func:`lazy.models._locks.
        network_lock`) throughout.

        Args:
            regressor: An unfitted ``TabICLRegressor``.

        Yields:
            Nothing; the regressor's own loader is back on exit.
        """
        load = getattr(type(regressor), "_load_model", None)
        if load is None:  # Not upstream's regressor: nothing to share.
            yield
            return
        cache_key = _weights.key(
            self.backend,
            self.version,
            self.checkpoint_,
            self.device_,
            "float32",
            load,
        )
        network, config, path = _weights.network(
            cache_key, functools.partial(_load_network, regressor, load)
        )

        def shared() -> None:
            regressor.model_ = network
            regressor.model_config_ = dict(config)
            regressor.model_path_ = path

        regressor._load_model = shared  # noqa: SLF001 - upstream's loader.
        try:
            with _locks.network_lock(network):
                yield
        finally:
            del regressor._load_model  # noqa: SLF001 - upstream's loader.

    def _attach_network(self, regressor: Any) -> None:
        """Hands an unpickled regressor the shared network, on first use.

        A pickle leaves the network out (:meth:`__getstate__`), and the
        checkpoint is found again through the hub, where it is on this
        machine, rather than at the pickled path. A fitted regressor that
        holds its network already is left as it is.

        Args:
            regressor: A fitted ``TabICLRegressor``.
        """
        if hasattr(regressor, "model_") or not hasattr(
            type(regressor), "_load_model"
        ):
            return
        if self.__dict__.pop("_relocate_checkpoint", False):
            spec = _hub.get_checkpoint(self.backend, self.version)
            self.checkpoint_ = spec.download()
        regressor.model_path = str(self.checkpoint_)
        with self._shared_network(regressor):
            regressor._load_model()  # noqa: SLF001 - the shared loader.
            # As upstream's fit does: a network just loaded is on the CPU.
            regressor.model_.to(regressor.device_)

    def _predict_group(
        self, handle: _RegressorGroup, X: _typing.FloatArray
    ) -> distributions.QuantileDistribution:
        offset, scale = handle.offset, handle.scale
        total = sum(handle.weights)
        # Upstream averages its members' quantiles; across regressors, each
        # average counts for the members it stands for.
        answers = []
        for regressor in handle.regressors:
            self._attach_network(regressor)
            with _lending(regressor):
                answers.append(
                    np.asarray(
                        regressor.predict(X, output_type="raw_quantiles"),
                        dtype=np.float64,
                    )
                )
        quantiles = sum(
            (weight / total) * answer
            for answer, weight in zip(answers, handle.weights, strict=True)
        )
        quantiles = np.asarray(quantiles) * scale + offset
        if quantiles.shape[1] != self.n_quantiles_:
            raise RuntimeError(
                f"{self.display_name} returned {quantiles.shape[1]} quantiles "
                f"per row, not {self.n_quantiles_}"
            )
        return distributions.QuantileDistribution(
            quantile_levels(self.n_quantiles_), quantiles
        )

    def _recipe(self) -> dict[str, Any]:
        threshold = self.outlier_threshold_
        return {
            **super()._recipe(),
            # What upstream was asked for, and the members each regressor of
            # each group ran.
            "tabicl": {
                "outlier_threshold": threshold,
                "use_amp": self.mixed_precision_,
                **{
                    name: _PINNED_REGRESSOR_ARGS[name]
                    for name in ("batch_size", "use_fa3")
                },
                "groups": [handle.provenance() for handle in self.handles_],
            },
        }

    def __getstate__(self) -> dict[str, Any]:
        """Pickles the regressors without their network.

        Upstream's own pickle of a regressor leaves the network out but
        reloads it as soon as it is unpickled, from the absolute checkpoint
        path and on the device it was fitted on, one network per regressor.
        Each regressor is pickled as the state upstream's pickle holds (its
        key/value cache, on the CPU, and its context included) and restored
        by :meth:`__setstate__` without a network, which the first
        prediction attaches, shared (:meth:`_attach_network`).
        """
        state = super().__getstate__()
        if "handles_" in state:
            state["handles_"] = [
                dataclasses.replace(
                    handle,
                    regressors=tuple(
                        _PickledRegressor.of(regressor)
                        for regressor in handle.regressors
                    ),
                )
                for handle in state["handles_"]
            ]
            state.pop("regressor_", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restores a pickle on this machine, not the one it was made on.

        ``device="auto"`` is resolved again, so a model fitted on a GPU
        predicts on the CPU where there is none, and the checkpoint is found
        again through the hub on next use rather than at the pickled path.
        """
        super().__setstate__(state)
        if "device_" in state and str(self.device).strip().lower() == "auto":
            self.device_ = _device.resolve_device(self.device)
            self.mixed_precision_ = bool(
                self.mixed_precision
            ) and self.device_.startswith("cuda")
            moved: dict[str, Any] = {
                "device": self.device_,
                "mixed_precision": self.mixed_precision_,
            }
            recipe = self.provenance_.get("tabicl")
            if isinstance(recipe, dict):
                moved["tabicl"] = {**recipe, "use_amp": self.mixed_precision_}
            self.provenance_ = {**self.provenance_, **moved}
        if "checkpoint_" in state:
            self._relocate_checkpoint = True
        if "handles_" in state:
            self.handles_ = [
                dataclasses.replace(
                    handle,
                    regressors=tuple(
                        self._unpickled_regressor(pickled)
                        for pickled in handle.regressors
                    ),
                )
                for handle in self.handles_
            ]
            if (
                len(self.handles_) == 1
                and len(self.handles_[0].regressors) == 1
            ):
                self.regressor_ = self.handles_[0].regressors[0]

    def _unpickled_regressor(self, pickled: _PickledRegressor) -> Any:
        """A pickled regressor, restored as upstream restores one, but lazily.

        Everything upstream's unpickling does but load the network: the
        device is this model's, and the inference config and the key/value
        cache follow it.

        Args:
            pickled: The regressor as :meth:`__getstate__` left it.

        Returns:
            The fitted ``TabICLRegressor``, without a network.
        """
        cls: Any = pickled.cls
        regressor = cls.__new__(cls)
        state = dict(pickled.state)
        # Upstream's record of the versions it was pickled with; lazy's own
        # is in provenance_.
        state.pop("_persistence_metadata", None)
        regressor.__dict__.update(state)
        regressor.device = self.device_
        regressor.use_amp = self.mixed_precision_
        # Upstream's own steps of unpickling, in its order.
        regressor._resolve_device()  # noqa: SLF001 - upstream's.
        regressor._build_inference_config()  # noqa: SLF001 - upstream's.
        regressor._move_cache_to_device()  # noqa: SLF001 - upstream's.
        return regressor


@dataclasses.dataclass(frozen=True)
class _PickledRegressor:
    """A fitted ``TabICLRegressor`` as a pickle holds it.

    Attributes:
        cls: The regressor's class.
        state: What upstream's ``__getstate__`` returns: everything but the
            network, the device and the checkpoint path.
    """

    cls: type[Any]
    state: dict[str, Any]

    @classmethod
    def of(cls, regressor: Any) -> _PickledRegressor:
        """The pickled form of a fitted regressor."""
        return cls(type(regressor), regressor.__getstate__())


@dataclasses.dataclass(frozen=True)
class _RegressorGroup:
    """The fitted regressors that together run one member group.

    Attributes:
        regressors: The ``tabicl.TabICLRegressor`` instances.
        members: For each regressor, the members it stands for, as counts by
            norm method.
        feat_shuffle_method: How upstream permuted the members' columns.
        offset: The mean the group's targets were standardised by.
        scale: Their standard deviation, or 1 for a constant target.
    """

    regressors: tuple[Any, ...]
    members: tuple[dict[str, int], ...]
    feat_shuffle_method: str
    offset: float = 0.0
    scale: float = 1.0

    @property
    def weights(self) -> tuple[int, ...]:
        """How many members each regressor's averaged answer stands for."""
        return tuple(sum(counts.values()) for counts in self.members)

    def members_by_method(self) -> dict[str, int]:
        """The group's members, counted by norm method."""
        total: collections.Counter[str] = collections.Counter()
        for counts in self.members:
            total.update(counts)
        return dict(total)

    def provenance(self) -> dict[str, Any]:
        """What ran, for ``provenance_``."""
        return {
            "feat_shuffle_method": self.feat_shuffle_method,
            "regressors": [
                {"random_state": int(regressor.random_state), "members": counts}
                for regressor, counts in zip(
                    self.regressors, self.members, strict=True
                )
            ],
        }


def _load_network(
    regressor: Any, load: Callable[[Any], None]
) -> tuple[Any, dict[str, Any], Any]:
    """Upstream's loader, run on ``regressor``.

    Args:
        regressor: The ``TabICLRegressor`` whose settings name the
            checkpoint.
        load: Upstream's ``TabICLRegressor._load_model``.

    Returns:
        The network, its config and the checkpoint path, as a tuple
        (network, config, path).
    """
    load(regressor)
    return regressor.model_, regressor.model_config_, regressor.model_path_


@contextlib.contextmanager
def _lending(regressor: Any) -> Iterator[None]:
    """Holds a fitted regressor's network for one prediction.

    Upstream hands the network the regressor's cache and reads it back from
    the network; on a network other models share, a prediction holds the
    network's lock, so that two threads never answer from each other's
    context, and leaves no cache behind.

    Args:
        regressor: A fitted ``TabICLRegressor``.

    Yields:
        Nothing; the network holds no cache on exit.
    """
    network = getattr(regressor, "model_", None)
    if network is None:  # Not upstream's regressor: nothing shared.
        yield
        return
    with _locks.network_lock(network):
        try:
            yield
        finally:
            _release_network_cache(regressor)


def _release_network_cache(regressor: Any) -> None:
    """Drops the key/value cache upstream leaves on a shared network.

    A cached predict hands the network the regressor's cache and leaves it
    there; on a network other models share, it would keep a cache alive
    after its model is gone.
    """
    clear = getattr(getattr(regressor, "model_", None), "clear_cache", None)
    if clear is not None:
        clear()


def _members_run(regressor: Any) -> collections.Counter[str]:
    """The members a fitted regressor runs, counted by norm method."""
    configs = regressor.ensemble_generator_.ensemble_configs_
    return collections.Counter(
        {method: len(members) for method, members in configs.items()}
    )


def _n_shuffles(regressor: Any) -> int:
    """How many distinct column orders a fitted regressor's members see."""
    configs = regressor.ensemble_generator_.ensemble_configs_
    return len(
        {
            tuple(int(i) for i in shuffle)
            for members in configs.values()
            for shuffle, _ in members
        }
    )


def _order_members(regressor: Any, norm_methods: list[str]) -> Any:
    """Puts a fitted regressor's members in ``norm_methods`` order.

    Upstream groups its members by norm method in the order of a Python
    ``set`` of the method names, which changes with the interpreter's string
    hashing (``PYTHONHASHSEED``), and averages them in that order: the same
    fit then differs between processes in the last float32 digits. The
    members and their caches are keyed by method, so fixing the order
    changes nothing else.

    Args:
        regressor: A fitted ``tabicl.TabICLRegressor``.
        norm_methods: The norm methods it was given.

    Returns:
        The regressor, modified in place.
    """
    generator = regressor.ensemble_generator_
    configs = generator.ensemble_configs_
    generator.ensemble_configs_ = collections.OrderedDict(
        (method, configs[method])
        for method in norm_methods
        if method in configs
    )
    if set(generator.ensemble_configs_) != set(configs):
        raise RuntimeError(
            f"TabICL ran norm methods {sorted(configs)} it was not given: "
            f"{norm_methods}"
        )
    return regressor


def _extra_seed(seed: int, index: int) -> int:
    """The seed of a group's ``index``-th regressor beyond its first.

    Drawn from the group's seed, so that it stays clear of the seeds the
    other groups use, and below 2**31, so that it is a valid seed upstream.
    """
    sequence = np.random.SeedSequence([seed % 2**32, index])
    return int(sequence.generate_state(1)[0] % 2**31)


def _norm_methods(tokens: tuple[str, ...]) -> list[str]:
    """TabICL's ``norm_methods`` for the members' planned transforms.

    Upstream pairs member ``k`` with norm method ``k % len(norm_methods)``,
    and a norm method listed twice gives two members the same feature
    shuffle as well, so they are one member counted twice. The distinct
    tokens in order therefore run the plan only when it cycles through them.

    Args:
        tokens: TabICL's token for each member's transform, in member order.

    Returns:
        The distinct tokens, in order.

    Raises:
        ValueError: If the plan repeats a token within a cycle, as
            ``("power", "power", "none")`` would. No plan the uniform
            parameters make does: only ``none`` is native, and the auto
            recipe alternates.
    """
    methods = list(dict.fromkeys(tokens))
    cycled = tuple(methods[i % len(methods)] for i in range(len(tokens)))
    if cycled != tuple(tokens):
        raise ValueError(
            "TabICL runs each of its transforms once per cycle over the "
            f"members, so it cannot weight them as {tuple(tokens)}; list "
            "each transform once in transforms"
        )
    return methods
