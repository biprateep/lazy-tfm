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
standard deviations. The ``transforms`` it has (``none``, ``power``,
``quantile``, ``quantile_rtdl``, ``robust``) map onto its ``norm_methods``.
TabICL cannot subsample rows, so ``bag_size`` is scaffolded: one
single-member regressor per bag, their quantile functions averaged, as
TabICL averages its own members.

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
import dataclasses
import math
import os
import types
from typing import Any
import warnings

import numpy as np
import pandas as pd

from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _ensemble
from lazy.models import _members
from lazy.models import _progress

__all__ = [
    "AUTO_FEAT_SHUFFLE_METHOD",
    "AUTO_NORM_METHODS",
    "AUTO_OUTLIER_THRESHOLD",
    "NATIVE_PAD",
    "NATIVE_QUANTILE_BINS",
    "TabICLQuantile",
    "quantile_levels",
]

# Warnings skip every frame inside this package, so that they point at the
# user's call however deep in the package (or LazyModel) they are raised.
_PACKAGE_PREFIX = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
)

#: Bins of the native grid across the training targets' range: as fine as
#: the 999 quantiles resolve. The padding adds bins of the same width.
NATIVE_QUANTILE_BINS = 1000

#: How far the native grid extends beyond the training targets on each
#: side, as a fraction of their range: TabICL's quantiles extrapolate.
NATIVE_PAD = 0.25

#: The probability a row may put outside the native grid before
#: tabulating it there warns.
_OUTSIDE_TOLERANCE = 0.01

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


def quantile_levels(n_quantiles: int) -> _typing.FloatArray:
    """Returns the cumulative probabilities TabICL's quantile outputs sit at.

    The levels are interior points of ``[0, 1]``: the model never claims to
    know where the 0th or 100th percentile is.

    Args:
        n_quantiles: How many quantiles the model returns per row.

    Returns:
        The evenly spaced levels, shape ``(n_quantiles,)``.

    Examples:
        >>> quantile_levels(3).tolist()
        [0.25, 0.5, 0.75]
    """
    return np.linspace(0.0, 1.0, int(n_quantiles) + 2)[1:-1]


class TabICLQuantile(_ensemble.ContextEnsembleEstimator):
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
            see :mod:`lazy.models._transforms`.
        feature_shuffle: Whether members see the columns in different orders
            (TabICL's Latin-square shuffles, ``feat_shuffle_method="latin"``;
            ``"none"`` when False). A regressor running one member keeps the
            columns in their given order.
        bag_size: Context rows per member: an int is a row count (1 means
            one row), a float a fraction in (0, 1] (1.0 means all rows), and
            None all of them. Scaffolded: one regressor per bag, with lazy's
            own column permutation when ``feature_shuffle=True``.
        kv_cache: Cache the context's keys and values at fit, so each chunk
            of queries skips the context forward pass: ``True`` (TabICL's
            ``"kv"`` cache), ``"repr"`` (cached row representations, far
            smaller, re-running the in-context layers) or ``False``. Exact
            either way, up to floating-point rounding.
        z_grid: Default output grid: a :class:`lazy.grid.Grid`, an
            array of bin centres, ``"native"``, or None for the native grid
            (equal-width bins, 1,000 across the training targets' range,
            which extend a quarter of that range beyond it on each side).
            Densities on a grid carry only the mass inside it, renormalised;
            tabulating a row that puts more than 1% of its probability
            outside the native grid warns.
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"mps"``, ``"cpu"``, or a ``torch.device``.
        random_state: Seed for the ensemble: TabICL's ``random_state`` for
            a group's first regressor, from which every other seed is
            derived. None draws a fresh seed at fit, recorded as
            ``random_state_`` and in ``provenance_``.
        chunk_size: Query rows predicted at a time, to bound peak memory
            (999 quantiles per row is about 8 kB); ``0`` does them in one
            pass. Exact: TabICL builds its keys and values from the context
            rows alone, so a row's answer never depends on the other rows in
            its chunk.
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
            the training targets' range padded by a quarter of it on each
            side (a constant target is padded by 1% of its value, and at
            least by 0.01).
        checkpoint_: The pinned checkpoint file, a :class:`pathlib.Path`.
        provenance_: Which weights, code and ensemble answered, as a dict.
        regressor_: The fitted ``tabicl.TabICLRegressor``, when one serves
            the whole ensemble.
        handles_: Per member group, the fitted regressors that run it, with
            the members each stands for (also in ``provenance_``).
        n_quantiles_: How many quantiles the backbone returned.
        n_context_: Context rows ``fit`` was given.

    Examples:
        >>> est = TabICLQuantile(n_estimators=8, chunk_size=50_000)
        >>> est.chunk_size
        50000
    """

    backend = "tabicl"
    display_name = "TabICL"
    extra = "tabicl"
    native_output = "quantiles"
    native_transforms = {
        "none": "none",
        "power": "power",
        "quantile": "quantile",
        "quantile_rtdl": "quantile_rtdl",
        "robust": "robust",
    }
    auto_tokens = AUTO_NORM_METHODS
    supports_native_bagging = False
    member_combination = "quantile_average"
    kv_cache_modes = (True, False, "repr")
    # A quantile head: there is no softmax for a temperature to divide.
    has_softmax = False
    native_outlier_clipping = True
    kv_cache_rtol = 1e-3  # The cache is stored in fp16 under autocast.
    cpu_friendly = True

    # The training targets' range, recorded by _fit_group for the grid.
    _support: tuple[float, float]
    # The (offset, scale) the targets are standardised by before TabICL's
    # float32 sees them, recorded by _fit_group.
    _target_scaling: tuple[float, float]

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v2",
        n_estimators: int = 8,
        transforms: str | tuple[str, ...] = "auto",
        feature_shuffle: bool = True,
        bag_size: int | float | None = None,
        kv_cache: bool | str = True,
        z_grid: grid_lib.GridLike = None,
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
        self.z_grid = z_grid
        self.device = device
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.softmax_temperature = softmax_temperature
        self.mixed_precision = mixed_precision
        self.outlier_threshold = outlier_threshold
        self.progress = progress
        self.verbose = verbose

    def _import_backend(self) -> types.ModuleType:
        try:
            import tabicl  # noqa: PLC0415 - an optional, heavy extra.
        except ImportError as error:
            # Only the backend itself missing is a missing extra; anything
            # it fails to import in turn is reported as it is.
            if (error.name or "").partition(".")[0] != "tabicl":
                raise
            raise ImportError(
                "TabICLQuantile needs the tabicl backend: "
                "pip install 'lazy-tfm[tabicl]'"
            ) from error
        return tabicl

    def _auto_outlier_threshold(self) -> float | None:
        return AUTO_OUTLIER_THRESHOLD

    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        super()._fit(X, y)
        # The base class keeps the one group's handle; regressor_ is the
        # TabICLRegressor itself, when one serves the whole ensemble.
        handle = self.__dict__.pop("regressor_", None)
        if handle is not None and len(handle.regressors) == 1:
            self.regressor_ = handle.regressors[0]

    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> _RegressorGroup:
        # TabICL works in float32, which cannot resolve a narrow spread about
        # a large offset; standardising in float64 first keeps it. Every
        # group shares the first group's scaling, as any one will do.
        if group.index == 0:
            self._target_scaling = _target_scaling(y)
        offset, scale = self._target_scaling
        handle = self._fit_members(
            X.astype(np.float32),
            ((y - offset) / scale).astype(np.float32),
            group,
        )
        low, high = float(y.min()), float(y.max())
        if group.index > 0:
            low, high = min(low, self._support[0]), max(high, self._support[1])
        self._support = (low, high)
        return handle

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
            X: The group's prepared context features, float32, shape
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
            return _order_members(regressor.fit(X, y), norm_methods)

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

    def _predict_group(
        self, handle: _RegressorGroup, X: _typing.FloatArray
    ) -> distributions.QuantileDistribution:
        offset, scale = getattr(self, "_target_scaling", (0.0, 1.0))
        total = sum(handle.weights)
        # Upstream averages its members' quantiles; across regressors, each
        # average counts for the members it stands for.
        quantiles = sum(
            (weight / total)
            * np.asarray(
                regressor.predict(
                    X.astype(np.float32), output_type="raw_quantiles"
                ),
                dtype=np.float64,
            )
            for regressor, weight in zip(
                handle.regressors, handle.weights, strict=True
            )
        )
        quantiles = np.asarray(quantiles) * scale + offset
        self.n_quantiles_ = int(quantiles.shape[1])
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

    def _native_grid(self) -> grid_lib.Grid:
        low, high = self._support
        if high > low:
            pad = NATIVE_PAD * (high - low)
            n_bins = round(NATIVE_QUANTILE_BINS * (1.0 + 2.0 * NATIVE_PAD))
        else:
            pad = 0.01 * max(abs(high), 1.0)
            n_bins = NATIVE_QUANTILE_BINS
        return grid_lib.Grid.linear(
            low - pad, high + pad, n_bins, normalization="histogram"
        )

    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        blocks = []
        outside = 0
        for dist in self._chunks(X):
            blocks.append(dist.on_grid(grid))
            if grid == self.native_grid_ and isinstance(
                dist, distributions.QuantileDistribution
            ):
                outside += int(
                    np.count_nonzero(
                        _mass_outside(dist, grid) > _OUTSIDE_TOLERANCE
                    )
                )
        if outside:
            warnings.warn(
                f"{outside} of {len(X)} rows put more than "
                f"{_OUTSIDE_TOLERANCE:.0%} of their probability outside the "
                f"native grid [{grid.z_min:g}, {grid.z_max:g}], and their "
                "densities there are renormalised. Pass a wider z_grid, or "
                "use predict_distribution, which has full support.",
                UserWarning,
                skip_file_prefixes=(_PACKAGE_PREFIX,),
            )
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _progress_postfix(
        self, dist: distributions.Distribution
    ) -> dict[str, Any]:
        del dist  # Unused: the count is recorded when predicting.
        return {"quantiles": getattr(self, "n_quantiles_", None)}


@dataclasses.dataclass(frozen=True)
class _RegressorGroup:
    """The fitted regressors that together run one member group.

    Attributes:
        regressors: The ``tabicl.TabICLRegressor`` instances.
        members: For each regressor, the members it stands for, as counts by
            norm method.
        feat_shuffle_method: How upstream permuted the members' columns.
    """

    regressors: tuple[Any, ...]
    members: tuple[dict[str, int], ...]
    feat_shuffle_method: str

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
        ValueError: If the plan repeats a transform within a cycle, as
            ``transforms=("power", "power", "none")`` does.
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


def _target_scaling(y: _typing.FloatArray) -> tuple[float, float]:
    """The (offset, scale) that standardise the targets: mean and std.

    Args:
        y: The context targets, shape ``(n,)``.

    Returns:
        A tuple ``(offset, scale)``; the scale is 1 for a constant target.
    """
    scale = float(np.std(y))
    return float(np.mean(y)), scale if scale > 0 else 1.0


def _mass_outside(
    dist: distributions.QuantileDistribution, grid: grid_lib.Grid
) -> _typing.FloatArray:
    """Each row's probability outside the grid, to one quantile level.

    Args:
        dist: The predicted quantiles.
        grid: The grid they are tabulated on.

    Returns:
        The probability below ``grid.z_min`` plus that above ``grid.z_max``,
        shape ``(n_rows,)``: a lower bound, short by at most one quantile
        level on each side.
    """
    levels = np.r_[0.0, dist.quants, 1.0]
    n_levels = dist.quants.size
    below = np.count_nonzero(dist.locs < grid.z_min, axis=1)
    above = np.count_nonzero(dist.locs > grid.z_max, axis=1)
    return levels[below] + (1.0 - levels[n_levels + 1 - above])
