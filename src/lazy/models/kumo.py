# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Target distributions from NVIDIA Kumo Tabular's quantile regression head.

Kumo Tabular (NVIDIA 2026) is an in-context tabular foundation model whose
regressor predicts 999 quantiles of the target per query row, at the levels
0.001 to 0.999. It ships in three sizes, which are this backend's versions
(:data:`SIZES`), in NVIDIA's ``structured-data-models`` package (imported as
``sdm``), which the ``kumo`` extra installs. :class:`KumoQuantile` returns the
quantiles as a :class:`~lazy.distributions.QuantileDistribution`.

Upstream runs every member through a *recipe* of preprocessing steps, which
is written out here (:func:`_recipe`, :data:`AUTO_TRANSFORMS`,
:data:`AUTO_OUTLIER_THRESHOLD`, :data:`AUTO_TRIM_FRACTION`,
:data:`_PINNED_RECIPE`) rather than taken from the installed package, so that
an upgrade of it cannot change what ``"auto"`` runs. The uniform features map
onto the recipe: ``transforms="auto"`` cycles the members through upstream's
four transforms, ``feature_shuffle`` is its Latin-square column shuffle,
``outlier_threshold`` its two-pass soft clip, ``kv_cache`` its fitted context
cache and ``mixed_precision`` the float16 autocast NVIDIA runs it in. Of the
uniform transforms only ``none`` is upstream's own; lazy applies the others.
Under ``"auto"`` the members' quantiles combine as upstream combines them,
by a mean that drops the highest and lowest fifth at every level, taken here
over every member of the ensemble
(``member_combination="quantile_trimmed_average"``), so that bagged members,
which run one at a time, combine the same way. The trim belongs to the recipe
as the clip does, so under an explicit ``transforms`` it is dropped with it,
and the quantiles are a plain mean over every member.

What upstream always does to the features, under any ``transforms``: columns
constant in the context are dropped, every column is z-scored on the context
and clipped at 100 standard deviations, each member flips the sign of a
random half of the columns, and at most 500 columns are kept. The target is
z-scored and its sign flipped at random per member, and the answers are
mapped back. The random sign flips of the columns and of the target are part
of how the network is run rather than of the recipe, so they stay under an
explicit ``transforms``. Kumo has no imputation step: missing values reach
the network, which embeds them itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import functools
from importlib import util
import pathlib
import types
import typing
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from lazy import _typing
from lazy import _warnings
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _device
from lazy.models import _ensemble
from lazy.models import _hub
from lazy.models import _install
from lazy.models import _locks
from lazy.models import _members
from lazy.models import _progress
from lazy.models import _quantile_head
from lazy.models import _transforms
from lazy.models import _weights

__all__ = [
    "AUTO_FEAT_SHUFFLE_METHOD",
    "AUTO_OUTLIER_THRESHOLD",
    "AUTO_TRANSFORMS",
    "AUTO_TRIM_FRACTION",
    "REPOSITORY_COMMIT",
    "SIZES",
    "KumoQuantile",
]

REPOSITORY_COMMIT = "2b76468cb5b4ac1f4bc59f7f07f0506a4a2e8298"
"""The structured-data-models commit LAZY's recipe was read from and tested on.
"""

_REQUIREMENT = "structured-data-models>=0.2.1"
"""The ``kumo`` extra's requirement. Release 0.2.1 is the first with Kumo
Tabular, and its ``sdm/models/kumo/tabular`` is the code at
:data:`REPOSITORY_COMMIT`.
"""

_VALIDATED_RELEASE = "structured-data-models==0.2.1"
"""What the deprecated ``lazy setup kumo`` installs: the validated release."""

SIZES: dict[str, str] = {
    "v1-large": "large",
    "v1-medium": "medium",
    "v1-small": "small",
}
"""The model size each version loads; the sizes share one recipe."""

# Kumo Tabular's own recipe, ``transforms="auto"``, written out here so that
# the installed upstream version cannot change it: the regression branch of
# sdm.models.kumo.tabular.recipe.default_recipe at REPOSITORY_COMMIT.

AUTO_TRANSFORMS: tuple[str, ...] = (
    "identity",
    "power",
    "robust",
    "rank_gaussian",
)
"""The transforms members cycle through under ``transforms="auto"``, by
upstream's position in the ensemble: none, a Yeo-Johnson power transform, a
robust scaling with a soft clip at 3, and a Gaussian rank transform.
"""

AUTO_FEAT_SHUFFLE_METHOD = "latin"
"""How members' columns are permuted when ``feature_shuffle=True``."""

AUTO_OUTLIER_THRESHOLD = 4.0
"""The two-pass soft outlier clip of the recipe, in standard deviations."""

AUTO_TRIM_FRACTION = 0.2
"""The fraction of members dropped from each tail of every quantile; an explicit
``transforms`` drops none.
"""

_PINNED_RECIPE: dict[str, Any] = {
    # Standardize's floor on the standard deviation.
    "standardize_eps": 1e-6,
    # The hard clip after z-scoring, in standard deviations.
    "z_clip": 100.0,
    "robust_soft_clip": 3.0,
    "rank_gaussian_knots": 8192,
    # FlipSign's probability of negating a column, or the target.
    "flip_probability": 0.5,
    "max_columns": 500,
    # Members per forward pass: upstream's default, one at a time.
    "estimator_batch_size": 1,
    # The precision NVIDIA's benchmarks and model card run on a GPU.
    "autocast_dtype": "float16",
}
"""The rest of the recipe, and how upstream is run, pinned. ``max_columns``,
``z_clip`` and ``flip_probability`` act under any ``transforms``;
``robust_soft_clip`` and ``rank_gaussian_knots`` belong to the auto transforms.
"""

_N_QUANTILES = 999
"""The quantiles upstream's regression head returns per row, at the levels 0.001
to 0.999 (``num_quantiles`` of sdm.models.KumoTabular at REPOSITORY_COMMIT, the
same for every size).
"""

# Features in float64 and targets in float32, as upstream is handed them.
_Array: TypeAlias = npt.NDArray[np.floating[Any]]

_RECIPE_VERSIONS: tuple[str, ...] = tuple(SIZES)
"""The versions the recipe above was read from upstream for."""


class KumoQuantile(
    _quantile_head.QuantileHead, _ensemble.ContextEnsembleEstimator
):
    """Target distributions from the quantiles Kumo Tabular predicts.

    Args:
        version: Which size to load: ``"v1-large"`` (216 million
            parameters), ``"v1-medium"`` (62 million) or ``"v1-small"`` (28
            million); see :func:`lazy.list_versions`. Recorded in
            ``provenance_``.
        n_estimators: Ensemble members, exactly; costs scale linearly.
            NVIDIA's benchmarks run 8 with the small and medium sizes and 16
            with the large one; 8 is the package-wide default.
        transforms: Per-member feature transforms: ``"auto"`` (Kumo's own
            recipe, cycling none, a power transform, a robust scaling and a
            Gaussian rank transform, with the 4-sigma clip), a recipe name,
            a transform name or a sequence of them; see
            :mod:`lazy.models._transforms`. Only ``none`` is Kumo's own; lazy
            applies every other name, and Kumo then runs with no transform of
            its own. An explicit ``transforms`` also drops the recipe's 20%
            trim, so the members' quantiles are a plain mean; the random sign
            flips of the columns and of the target stay under any
            ``transforms``.
        feature_shuffle: Whether members see the columns in different orders
            (upstream's Latin-square shuffle). A bagged member, which runs on
            its own, has its columns permuted by lazy instead.
        bag_size: Context rows per member: an int is a row count (1 means
            one row), a float a fraction in (0, 1] (1.0 means all rows), and
            None all of them. Scaffolded: each bag runs as an ensemble of one,
            with its preprocessing and target standardization fitted on the
            bag.
        kv_cache: Fit upstream's context cache (``fit`` then ``predict``),
            so each chunk of queries skips the context's forward pass; False
            runs the whole forward pass, recipe included, for every chunk.
            On a CPU the two agree to float rounding; on a GPU with
            ``mixed_precision=True`` they agree to float16 rounding.
        y_grid: Default output grid: a :class:`lazy.grid.Grid`, an array of
            bin centres, ``"native"``, or None for the native grid (1,000
            equal bins across the training targets' range, extended by a
            quarter of that range on each side). Tabulating a row that puts
            more than 1% of its probability outside the native grid warns.
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"cpu"``, or a ``torch.device``. Apple's MPS is not supported,
            since the recipe works in float64, which MPS lacks: a fit on it
            raises ValueError before the checkpoint loads.
        random_state: Seed for the ensemble. A group of members runs on a
            ``torch.Generator`` seeded by its first member,
            ``SeedSequence([random_state, i])``, which draws the members'
            column shuffles and sign flips. None draws a fresh seed at fit,
            recorded as ``random_state_`` and in ``provenance_``.
        chunk_size: Query rows predicted at a time, to bound peak memory;
            ``0`` does them in one pass. Every query attends to the context
            rows alone, so a row's answer never depends on the other rows in
            its chunk beyond float rounding: the batch shape can change the
            kernels (on a CPU, a one-row chunk differs by a few parts in a
            million of the quantiles' spread).
        softmax_temperature: Only ``"auto"``: the quantile head has no
            softmax, so any other value raises ValueError. Recorded in
            ``provenance_`` as None.
        mixed_precision: On CUDA, run upstream under float16 autocast, as
            NVIDIA does. False, or any device other than CUDA, runs in
            float32.
        outlier_threshold: Upstream's two-pass soft clip of each member's
            transformed columns, at this many standard deviations of the
            context. ``"auto"`` is 4.0 under ``transforms="auto"`` and off
            under an explicit ``transforms``; None is off.
        progress: A progress bar over the query rows: ``"auto"`` shows it
            on a terminal or in a notebook, ``True`` always, ``False`` never.
        verbose: Print log messages to stdout.

    Attributes:
        grid_: The resolved default output grid.
        native_grid_: The native grid, histogram-normalized: 1,500 bins over
            the training targets' range (all of them, bagged or not) padded
            by a quarter of it on each side.
        checkpoint_: The pinned checkpoint file, a :class:`pathlib.Path`.
        provenance_: Which weights, code and ensemble answered, as a dict.
        handles_: Per member group, its fitted context.
        trim_fraction_: The fraction of members dropped from each tail of
            every quantile: 0.2 under ``transforms="auto"``, 0 under an
            explicit ``transforms``.
        n_quantiles_: How many quantiles the backbone returns per row.
        n_context_: Context rows ``fit`` was given.

    Examples:
        >>> est = KumoQuantile(version="v1-small", n_estimators=16)
        >>> est.version, est.n_estimators
        ('v1-small', 16)
    """

    backend = "kumo"
    display_name = "Kumo Tabular"
    method_note = (
        "Quantiles of Kumo Tabular's regression head, trimmed across members."
    )
    extra = "kumo"
    native_output = "quantiles"
    # Kumo's power, robust and rank transforms are fitted on z-scored and
    # clipped columns and are not the uniform transforms of those names;
    # lazy applies those names itself.
    native_transforms = {"none": "identity"}
    auto_tokens = AUTO_TRANSFORMS
    supports_native_bagging = False
    member_combination = "quantile_trimmed_average"
    member_trim_fraction = AUTO_TRIM_FRACTION
    kv_cache_modes = (True, False)
    has_softmax = False
    native_outlier_clipping = True
    # NVIDIA's training stopped at tables of 60,000 rows; unlike LimiX's
    # limit, it is not where the model was seen to degrade.
    recommended_max_context = 60_000
    context_limit_note = (
        "was pretrained on tables of up to {limit:,} rows, beyond which its "
        "accuracy has not been measured"
    )
    # Not bit for bit on a CPU: a one-row chunk takes a different
    # matrix-multiply path, cached or not, and lands about 6e-6 of the
    # quantiles' spread away.
    exact_chunking = False
    # On a 4-core x86 CPU the cached and uncached densities on the DC1 grid
    # differ by up to 2.2e-4 of a density, and 1.2e-4 of the peak.
    kv_cache_rtol = 1e-3
    cpu_friendly_versions = ("v1-small",)

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v1-large",
        n_estimators: int = 8,
        transforms: str | tuple[str, ...] = "auto",
        feature_shuffle: bool = True,
        bag_size: int | float | None = None,
        kv_cache: bool = True,
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
        sdm = _ensemble.import_extra(
            "sdm",
            needed_by=type(self).__name__,
            extra=self.extra,
        )
        if not hasattr(getattr(sdm, "models", None), "KumoTabular"):
            raise ImportError(
                f"{type(self).__name__} needs NVIDIA's structured-data-models, "
                f"but the 'sdm' installed at {getattr(sdm, '__file__', '?')} "
                "has no models.KumoTabular; uninstall the package that "
                "provides it, then pip install 'lazy-tfm[kumo]'"
            )
        return sdm

    @classmethod
    def setup(cls, *, dry_run: bool, requested: bool = False) -> bool:
        """Installs structured-data-models when named, which is deprecated.

        The ``kumo`` extra installs the package, so unnamed this does
        nothing. Named, ``lazy setup kumo`` warns, and installs
        :data:`_VALIDATED_RELEASE` when ``sdm.models.KumoTabular`` does not
        import (:func:`lazy.models._install.install`, with pip or
        uv).

        Args:
            dry_run: Say what would be installed, and install nothing.
            requested: Whether the backend was named on the command line.

        Returns:
            Whether the package is installed, or would be after a dry run;
            True when the backend was not named.
        """
        if not requested:
            return True
        _warnings.warn(
            "lazy setup kumo is deprecated and will be removed in 0.7.0: "
            "pip install 'lazy-tfm[kumo]' now installs "
            "structured-data-models.",
            FutureWarning,
        )
        try:
            # The check every fit makes, KumoTabular included.
            cls()._import_backend()  # noqa: SLF001 - a classmethod of the same class.
        except ImportError:
            pass
        else:
            print("Kumo Tabular: structured-data-models is installed.")
            return True
        if util.find_spec("pyarrow") is None:
            # The install leaves dependencies to the extra.
            print(
                "Kumo Tabular: install its dependencies first: pip install "
                "'lazy-tfm[kumo]'."
            )
            return False
        return _install.install(
            "Kumo Tabular",
            _VALIDATED_RELEASE,
            what="structured-data-models",
            check="from sdm.models import KumoTabular",
            dry_run=dry_run,
        )

    @classmethod
    def _pinned_recipe(cls, version: str) -> Mapping[str, Any]:
        if version not in _RECIPE_VERSIONS:
            return super()._pinned_recipe(version)
        return types.MappingProxyType(
            {
                "size": SIZES[version],
                "transforms": AUTO_TRANSFORMS,
                "feat_shuffle_method": AUTO_FEAT_SHUFFLE_METHOD,
                "outlier_threshold": AUTO_OUTLIER_THRESHOLD,
                "trim_fraction": AUTO_TRIM_FRACTION,
                **_PINNED_RECIPE,
            }
        )

    def _auto_outlier_threshold(self) -> float:
        return AUTO_OUTLIER_THRESHOLD

    def _check_backend_params(self) -> None:
        # The trim is part of Kumo's recipe, so it follows transforms as
        # outlier_threshold="auto" does: an explicit recipe drops it.
        self.trim_fraction_ = (
            float(self._pinned_recipe(self.version)["trim_fraction"])
            if _transforms.parse(self.transforms) is None
            else 0.0
        )

    def _member_trim_fraction(self) -> float:
        return self.trim_fraction_

    def _load_checkpoint(self) -> None:
        # The recipe casts to float64 and the context goes to the device in
        # float64, which MPS cannot hold; better said before any download.
        if self.device_ == "mps":
            raise ValueError(
                f"{self.display_name} does not run on device='mps': its "
                "preprocessing works in float64, which MPS does not "
                "support. Use device='cpu' or a CUDA device."
            )
        super()._load_checkpoint()

    def _after_fit(self) -> None:
        self.n_quantiles_ = _N_QUANTILES

    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> _KumoGroup:
        # Upstream z-scores the target in its own dtype, and the network
        # reads it in float32, which cannot resolve a narrow spread about a
        # large offset; standardizing in float64 first keeps it. Each group
        # is standardized on its own rows, a bagged member on its bag.
        offset, scale = _quantile_head.target_scaling(y)
        tokens = _group_tokens(group)
        handle = _KumoGroup(
            n_members=group.n_members,
            seed=group.seed,
            recipe=self._upstream_recipe(group),
            # Upstream's round robin: member k runs token k % len(tokens).
            transforms=tuple(
                tokens[k % len(tokens)] for k in range(group.n_members)
            ),
            feat_shuffle_method=(
                AUTO_FEAT_SHUFFLE_METHOD if group.feature_shuffle else "none"
            ),
            offset=offset,
            scale=scale,
        )
        # The context is kept under either cache setting, so that a pickle,
        # which leaves the cache out, can record it again.
        handle.context = (X, ((y - offset) / scale).astype(np.float32))
        if self.kv_cache:
            handle.cache = self._record(handle)
        return handle

    def _record(self, handle: _KumoGroup) -> Any:
        """Fits upstream on a group's context; returns its recorded cache.

        Args:
            handle: The group, with its context.

        Returns:
            Upstream's fitted context: its fitted recipe and the members'
            keys and values. The shared network is left holding none.

        Raises:
            RuntimeError: If the group kept no context, which _fit_group
                always sets.
        """
        if handle.context is None:
            raise RuntimeError("the group kept no context to record from")
        network = self._network()
        x, target = self._tensors(*handle.context)
        with _lending(network), self._autocast():
            network.fit(
                x=x,
                y=target,
                recipe=handle.recipe,
                num_estimators=handle.n_members,
                estimator_batch_size=_PINNED_RECIPE["estimator_batch_size"],
                generator=self._generator(handle.seed),
            )
            return network._cache  # noqa: SLF001 - upstream's cache.

    def _predict_members(
        self, handle: _KumoGroup, X: _typing.FloatArray
    ) -> list[distributions.QuantileDistribution]:
        if self.kv_cache and handle.cache is None:
            # Unpickled: the cache was left out, and the same context and
            # seed record the same one again, once. The lock is the
            # estimator's, since a group compares by value and so is not
            # hashable.
            with _locks.network_lock(self):
                if handle.cache is None:
                    handle.cache = self._record(handle)
        cache = handle.cache
        network = self._network()
        (queries,) = self._tensors(X)
        with self._autocast():
            if cache is None:
                assert handle.context is not None  # Set by _fit_group.
                x, target = self._tensors(*handle.context)
                out = network(
                    x_context=x,
                    y_context=target,
                    x_query=queries,
                    recipe=handle.recipe,
                    num_estimators=handle.n_members,
                    estimator_batch_size=_PINNED_RECIPE["estimator_batch_size"],
                    generator=self._generator(handle.seed),
                )
            else:
                with _lending(network, cache):
                    out = network.predict(queries)
        # Shape (n_members, n_queries, n_quantiles), each member's sorted.
        quantiles = out.numerical.double().cpu().numpy()
        if quantiles.shape[-1] != self.n_quantiles_:
            raise RuntimeError(
                f"{self.display_name} returned {quantiles.shape[-1]} "
                f"quantiles per row, not {self.n_quantiles_}"
            )
        levels = _quantile_head.quantile_levels(self.n_quantiles_)
        return [
            distributions.QuantileDistribution(
                levels, member * handle.scale + handle.offset
            )
            for member in quantiles
        ]

    def _predict_group(
        self, handle: _KumoGroup, X: _typing.FloatArray
    ) -> distributions.QuantileDistribution:
        return distributions.QuantileDistribution.trimmed_average(
            self._predict_members(handle, X), self._member_trim_fraction()
        )

    def _upstream_recipe(self, group: _members.MemberGroup) -> Any:
        """The group's recipe, as upstream's preprocessing objects."""
        return _recipe(
            self._import_backend().processing,
            _group_tokens(group),
            feature_shuffle=group.feature_shuffle,
            outlier_threshold=self.outlier_threshold_,
        )

    def _network(self) -> Any:
        """The loaded network, cached on the instance per version and device.

        It comes from :mod:`lazy.models._weights`, so every fit with the same
        checkpoint and device runs on one network: the fitted context lives
        in each group's handle, which is handed to the network only for the
        duration of a prediction. The weights are float32 whatever
        ``mixed_precision`` says, which only autocasts.
        """
        if self.__dict__.pop("_relocate_checkpoint", False):
            spec = _hub.get_checkpoint(self.backend, self.version)
            self.checkpoint_ = spec.download()
        return self._memoized_network(
            (self.version, self.device_), self._shared_network
        )

    def _shared_network(self) -> Any:
        """Finds or loads the shared network; see :meth:`_network`."""
        return _weights.network(
            _weights.key(
                self.backend,
                self.version,
                self.checkpoint_,
                self.device_,
                "float32",
                _load_network,
            ),
            functools.partial(
                _load_network,
                self._import_backend(),
                self.checkpoint_,
                SIZES[self.version],
                self.device_,
            ),
        )

    def _tensors(self, *arrays: _Array) -> tuple[Any, ...]:
        """Arrays as tensors on the device; a 1-D target becomes a column.

        Copied, since the arrays may be read-only views.
        """
        import torch  # noqa: PLC0415 - torch comes with the extra.

        return tuple(
            torch.tensor(
                array if array.ndim == 2 else array[:, None],
                device=self.device_,
            )
            for array in arrays
        )

    def _generator(self, seed: int) -> Any:
        """A ``torch.Generator`` on the device, seeded with ``seed``."""
        import torch  # noqa: PLC0415 - torch comes with the extra.

        return torch.Generator(device=self.device_).manual_seed(seed)

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        """Upstream's float16 autocast under mixed precision, else nothing."""
        import torch  # noqa: PLC0415 - torch comes with the extra.

        dtype = getattr(torch, _PINNED_RECIPE["autocast_dtype"])
        return torch.amp.autocast(
            "cuda", dtype=dtype, enabled=self.mixed_precision_
        )

    def _autocast_dtype(self) -> str | None:
        """The autocast dtype in force, for ``provenance_``; None for none."""
        if not self.mixed_precision_:
            return None
        return str(_PINNED_RECIPE["autocast_dtype"])

    def _recipe(self) -> dict[str, Any]:
        return {
            **super()._recipe(),
            "kumo": {
                "size": SIZES[self.version],
                "outlier_threshold": self.outlier_threshold_,
                "autocast": self._autocast_dtype(),
                "trim_fraction": self._member_trim_fraction(),
                "groups": [handle.provenance() for handle in self.handles_],
            },
        }

    def __getstate__(self) -> dict[str, Any]:
        """Pickles without the members' keys and values, which re-record.

        The cache can run to gigabytes and lives on the device it was fitted
        on, so a pickle keeps each group's context instead, and the first
        prediction records the cache again, on the device then in use.
        """
        state = super().__getstate__()
        if "handles_" in state:
            state["handles_"] = [
                dataclasses.replace(handle, cache=None)
                for handle in state["handles_"]
            ]
            if "regressor_" in state:
                state["regressor_"] = state["handles_"][0]
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
            recipe = self.provenance_.get("kumo")
            if isinstance(recipe, dict):
                moved["kumo"] = {**recipe, "autocast": self._autocast_dtype()}
            self.provenance_ = {**self.provenance_, **moved}
        if "checkpoint_" in state:
            self._relocate_checkpoint = True


@dataclasses.dataclass
class _KumoGroup:
    """One member group's fitted context.

    Attributes:
        n_members: The members upstream runs for the group.
        seed: The seed of the group's ``torch.Generator``.
        recipe: The group's unfitted upstream recipe.
        transforms: Upstream's transform token for each member, in member
            order.
        feat_shuffle_method: How upstream permutes the members' columns,
            ``"latin"`` or ``"none"``.
        offset: The mean the group's targets were standardized by.
        scale: Their standard deviation, or 1 for a constant target.
        cache: Upstream's fitted context (its recipe and the members' keys
            and values) when ``kv_cache=True``, else None; a pickle leaves
            it out.
        context: The prepared context features and standardized targets,
            kept under either ``kv_cache``: without the cache every
            prediction runs on them, and with it an unpickled model records
            the cache from them again.
    """

    n_members: int
    seed: int
    recipe: Any
    transforms: tuple[str, ...] = ()
    feat_shuffle_method: str = "none"
    offset: float = 0.0
    scale: float = 1.0
    cache: Any = None
    context: tuple[_typing.FloatArray, _Array] | None = None

    def provenance(self) -> dict[str, Any]:
        """What ran, for ``provenance_``."""
        return {
            "seed": int(self.seed),
            "members": self.n_members,
            "transforms": list(self.transforms),
            "feat_shuffle_method": self.feat_shuffle_method,
        }


@contextlib.contextmanager
def _lending(network: Any, cache: Any = None) -> Iterator[None]:
    """Holds the shared network for one fit or one prediction.

    Upstream keeps the context of its last ``fit`` on the network itself; on
    a network that other fits share, it is lent for one call, under the
    network's lock, so that two threads never see each other's context.

    Args:
        network: The shared ``sdm.models.KumoTabular``.
        cache: A group's fitted context to lend, or None for a fit.

    Yields:
        Nothing; the network holds no context on exit.
    """
    with _locks.network_lock(network):
        network._cache = cache  # noqa: SLF001 - upstream's cache.
        try:
            yield
        finally:
            network._cache = None  # noqa: SLF001 - upstream's cache.


def _recipe(
    sp: types.ModuleType,
    tokens: tuple[str, ...],
    *,
    feature_shuffle: bool,
    outlier_threshold: float | None,
) -> Any:
    """Upstream's regression recipe for a group, from the pinned settings.

    Args:
        sp: The ``sdm.processing`` module.
        tokens: The distinct transforms the group's members cycle through,
            in order (:data:`AUTO_TRANSFORMS` under ``"auto"``).
        feature_shuffle: Whether to shuffle the members' columns.
        outlier_threshold: The soft clip in standard deviations, or None.

    Returns:
        An ``sdm.processing.Recipe`` whose output keeps every member.
    """
    import torch  # noqa: PLC0415 - torch comes with the extra.

    pinned = _PINNED_RECIPE
    features = [
        sp.StypeDispatch(
            numerical=[
                _numerical(sp, tokens, outlier_threshold),
                sp.FlipSign(pinned["flip_probability"]),
            ]
        )
    ]
    if feature_shuffle:
        features.append(sp.ShuffleColumns(method=AUTO_FEAT_SHUFFLE_METHOD))
    features += [
        sp.SelectColumns(pinned["max_columns"], method="first"),
        sp.Cast(torch.float32),
    ]
    return sp.Recipe(
        features=features,
        target=[
            sp.StypeDispatch(
                numerical=[
                    sp.Standardize(),
                    sp.FlipSign(pinned["flip_probability"]),
                ]
            )
        ],
        # No average: the members are combined by lazy, over the ensemble.
        output=[sp.SortQuantiles()],
    )


def _numerical(
    sp: types.ModuleType,
    tokens: tuple[str, ...],
    outlier_threshold: float | None,
) -> Any:
    """Upstream's numerical pipeline: scaling, the transforms and the clip.

    Args:
        sp: The ``sdm.processing`` module.
        tokens: The transforms the members cycle through, in order.
        outlier_threshold: The soft clip in standard deviations, or None.

    Returns:
        An ``sdm.processing.Sequential``.
    """
    import torch  # noqa: PLC0415 - torch comes with the extra.

    pinned = _PINNED_RECIPE
    steps = [
        sp.Cast(torch.float64),
        sp.DropConstantColumns(),
        sp.Standardize(eps=pinned["standardize_eps"]),
        sp.Clip(-pinned["z_clip"], pinned["z_clip"]),
        sp.Choice(
            *(_transform(sp, token) for token in tokens),
            method="round_robin",
        ),
    ]
    if outlier_threshold is not None:
        steps.append(sp.ClipSigma(threshold=outlier_threshold))
    return sp.Sequential(*steps)


def _transform(sp: types.ModuleType, token: str) -> Any:
    """One of upstream's per-member transforms, by its token."""
    if token == "identity":
        return sp.Identity()
    if token == "power":
        return sp.PowerTransform()
    if token == "robust":
        return [
            sp.RobustScale(),
            sp.ClipSoft(_PINNED_RECIPE["robust_soft_clip"]),
        ]
    if token == "rank_gaussian":
        return sp.RankGaussian(max_knots=_PINNED_RECIPE["rank_gaussian_knots"])
    raise ValueError(f"Kumo Tabular has no transform {token!r}")


def _group_tokens(group: _members.MemberGroup) -> tuple[str, ...]:
    """The distinct transforms a group's members cycle through, in order."""
    if group.native_transforms is None:
        return AUTO_TRANSFORMS
    return _members.cycle_tokens(
        group.native_transforms,
        "Kumo Tabular cycles its transforms over the members, so it "
        "cannot run {tokens}",
    )


# Hidden from the type checker, which would otherwise type every misspelt
# attribute of this module as a str.
if not typing.TYPE_CHECKING:

    def __getattr__(name: str) -> str:
        """The deprecated ``REPOSITORY_BUILD``, with a FutureWarning.

        Args:
            name: The attribute looked up.

        Returns:
            The git requirement of :data:`REPOSITORY_COMMIT`.

        Raises:
            AttributeError: For any other name.
        """
        if name != "REPOSITORY_BUILD":
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            )
        _warnings.warn(
            "lazy.models.kumo.REPOSITORY_BUILD is deprecated and will be "
            "removed in 0.7.0: the kumo extra installs "
            "structured-data-models from PyPI.",
            FutureWarning,
        )
        return (
            "structured-data-models @ git+https://github.com/NVIDIA/"
            f"structured-data-models@{REPOSITORY_COMMIT}"
        )


def _load_network(
    sdm: types.ModuleType, path: pathlib.Path, size: str, device: str
) -> Any:
    """Builds the regression network of a size and loads its weights.

    The network is built without memory (on PyTorch's meta device) and the
    checkpoint's tensors become its parameters, as upstream loads its own;
    the checkpoint is the pinned file rather than upstream's download.

    Args:
        sdm: The ``sdm`` package.
        path: The pinned ``regressor.pt``.
        size: ``"small"``, ``"medium"`` or ``"large"``.
        device: Where the weights go.

    Returns:
        The ``sdm.models.KumoTabular``, in evaluation mode.
    """
    import torch  # noqa: PLC0415 - torch comes with the extra.

    network = sdm.models.KumoTabular(
        task="regression", size=size, pretrained=False, device="meta"
    )
    state = torch.load(path, map_location=device, weights_only=True)
    network.models["regression"].load_state_dict(state, assign=True)
    return network.eval()
