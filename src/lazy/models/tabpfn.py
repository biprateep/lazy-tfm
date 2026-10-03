# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Target distributions from TabPFN's bar distribution.

TabPFN (Prior Labs) is an in-context tabular foundation model, and its
regressor does not answer with a number: internally it is a classifier over a
fixed set of *buckets* of the target, and its native output is the probability
mass in each one -- a ``FullSupportBarDistribution``, in the upstream name.
The bucket borders are a fixed array stored in the checkpoint (5,000 buckets
on v3), stretched and shifted by the context targets' mean and standard
deviation: uniform over the bulk, widening far into both tails.

That is already a conditional density estimate, so
:class:`TabPFNBarDistribution` does no post-processing: its predictions are
:class:`~lazy.distributions.HistogramDistribution` objects over those buckets,
mapped onto any grid by exact, mass-conserving integration
(:meth:`lazy.grid.Grid.rebin`), and its native grid is the buckets
themselves, in full. The bucket masses rather than quantiles, because upstream
computes its quantiles by inverting this same piecewise-uniform CDF.

The uniform parameters map onto TabPFN's own machinery: ``kv_cache`` onto its
fit-time key/value cache (at full precision, so exact on a CPU),
``feature_shuffle`` onto its ``FEATURE_SHIFT_METHOD``, ``outlier_threshold``
onto its soft clip, ``OUTLIER_REMOVAL_STD``,
``softmax_temperature`` onto its own and ``mixed_precision`` onto its
``inference_precision``. No setting is left to TabPFN: each version's own
recipe is written out below (:data:`_AUTO_RECIPES`) and handed over in full,
so neither the checkpoint's stored config nor the installed tabpfn package
decides what runs, and every column is numeric (upstream would take a column
with fewer than four distinct values for a category).

``bag_size`` runs one regressor per member, fitted on that member's bag
alone. TabPFN's own row subsampling (``SUBSAMPLE_SAMPLES``) would fit each
member's preprocessing on its rows but standardise the target, and place the
buckets, with the whole context's mean and standard deviation, and it checks
the whole context against the size limits; a regressor per bag fits all of
it on the bag, as every other model does, and only the bag meets the limits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import contextlib
import copy
import functools
import itertools
import os
import pathlib
import types
from typing import Any

import numpy as np

from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _ensemble
from lazy.models import _members
from lazy.models import _progress
from lazy.models import _weights

__all__ = [
    "TabPFNBarDistribution",
    "bucket_masses",
    "path_for_tabpfn",
]


def _native_transforms() -> dict[str, tuple[str, bool]]:
    """Uniform transform names TabPFN implements, as (name, append_original).

    Only ``none``: TabPFN's other transforms are not lazy's, so they are
    scaffolded. Its ``power`` is the same Yeo-Johnson but fills missing
    values with the column mean first; its ``quantile_norm`` takes
    ``n // 5`` quantiles, not ``min(1000, n)``; its ``quantile_uni``
    agrees below 100,000 rows, but above them caps the quantiles at 20,000
    and subsamples the rows; and its ``robust`` scales to unit variance,
    not by the IQR.
    """
    return {"none": ("none", False), "none+original": ("none", True)}


#: Versions whose architecture has no quantised key/value cache; upstream
#: would quietly fall back to full precision.
_UNQUANTISED_VERSIONS = ("v2", "v2.5", "v2.6")


def _preprocessor(
    name: str,
    categorical: str,
    original: bool | str,
    max_features: int,
    svd: str | None,
) -> dict[str, Any]:
    """One upstream ``PreprocessorConfig``, every field written out."""
    return {
        "name": name,
        "categorical_name": categorical,
        "append_original": original,
        "max_features_per_estimator": max_features,
        "global_transformer_name": svd,
        "max_onehot_cardinality": None,
        "differentiable": False,
    }


#: Settings every version's own recipe shares.
_SHARED_RECIPE: dict[str, Any] = {
    "FEATURE_SHIFT_METHOD": "shuffle",
    "FINGERPRINT_FEATURE": True,
    "FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT": 50,
    "SAMPLE_SUBSAMPLING_METHOD": "auto",
    "FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM": True,
    "PASSTHROUGH_INF": False,
}

#: Each version's own recipe (``transforms="auto"``): every
#: ``InferenceConfig`` field that changes a regression prediction on numeric
#: features, as the checkpoint stores it (v2.6 and later) or, for v2 and
#: v2.5, which predate stored configs, as tabpfn 9.0.0's
#: ``InferenceConfig.get_default`` writes it. Copied here so that a tabpfn
#: upgrade cannot change what ``"auto"`` runs. ``OUTLIER_REMOVAL_STD`` is the
#: resolved value (upstream's ``"auto"`` is None for a regressor), and
#: ``SOFTMAX_TEMPERATURE`` the checkpoint's calibrated one; both are what
#: ``"auto"`` resolves to, and are handed over as such.
_AUTO_RECIPES: dict[str, dict[str, Any]] = {
    "v2": {
        **_SHARED_RECIPE,
        "PREPROCESS_TRANSFORMS": (
            _preprocessor(
                "quantile_uni",
                "ordinal_very_common_categories_shuffled",
                True,
                500,
                "svd",
            ),
            _preprocessor("safepower", "onehot", False, 500, None),
        ),
        "SOFTMAX_TEMPERATURE": 0.9,
        "OUTLIER_REMOVAL_STD": None,
        "POLYNOMIAL_FEATURES": "no",
        "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None, "safepower"),
        "ENABLE_GPU_PREPROCESSING": False,
        "FEATURE_SUBSAMPLING_METHOD": "random",
        "FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT": "auto",
    },
    "v2.5": {
        **_SHARED_RECIPE,
        "PREPROCESS_TRANSFORMS": (
            _preprocessor("quantile_uni_coarse", "numeric", "auto", 500, None),
            _preprocessor(
                "squashing_scaler_default",
                "ordinal_very_common_categories_shuffled",
                False,
                500,
                "svd_quarter_components",
            ),
        ),
        "SOFTMAX_TEMPERATURE": 0.9,
        "OUTLIER_REMOVAL_STD": None,
        "POLYNOMIAL_FEATURES": "no",
        "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None, "safepower"),
        "ENABLE_GPU_PREPROCESSING": False,
        "FEATURE_SUBSAMPLING_METHOD": "random",
        "FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT": "auto",
    },
    "v2.6": {
        **_SHARED_RECIPE,
        "PREPROCESS_TRANSFORMS": (
            _preprocessor("quantile_uni", "numeric", False, 680, None),
            _preprocessor(
                "quantile_uni",
                "ordinal_very_common_categories_shuffled",
                "auto",
                500,
                "svd_quarter_components",
            ),
        ),
        "SOFTMAX_TEMPERATURE": 0.9,
        "OUTLIER_REMOVAL_STD": None,
        "POLYNOMIAL_FEATURES": 10,
        "REGRESSION_Y_PREPROCESS_TRANSFORMS": ("none",),
        "ENABLE_GPU_PREPROCESSING": False,
        "FEATURE_SUBSAMPLING_METHOD": "balanced",
        "FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT": "auto",
    },
    "v3": {
        **_SHARED_RECIPE,
        "PREPROCESS_TRANSFORMS": (
            _preprocessor(
                "squashing_scaler_default",
                "ordinal_very_common_categories_shuffled",
                False,
                500,
                "svd_quarter_components",
            ),
            _preprocessor("quantile_uni", "numeric", "auto", 500, None),
        ),
        "SOFTMAX_TEMPERATURE": 0.9,
        "OUTLIER_REMOVAL_STD": None,
        "POLYNOMIAL_FEATURES": "no",
        "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None, "safepower"),
        "ENABLE_GPU_PREPROCESSING": True,
        "FEATURE_SUBSAMPLING_METHOD": "auto",
        "FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT": 150,
    },
    "v3.5": {
        **_SHARED_RECIPE,
        "PREPROCESS_TRANSFORMS": (
            _preprocessor("none", "ordinal_shuffled", False, 768, None),
        ),
        "SOFTMAX_TEMPERATURE": 1.0,
        "OUTLIER_REMOVAL_STD": 12.0,
        "POLYNOMIAL_FEATURES": "no",
        "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None, "safepower"),
        "ENABLE_GPU_PREPROCESSING": True,
        "FEATURE_SUBSAMPLING_METHOD": "balanced",
        "FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT": 150,
    },
}
# The fast variant is the same recipe on a smaller checkpoint.
_AUTO_RECIPES["v3.5-fast"] = _AUTO_RECIPES["v3.5"]


def _upstream_settings(
    version: str,
    group: _members.MemberGroup,
    outlier_threshold: float | None,
    n_estimators: int,
) -> dict[str, Any]:
    """The ``inference_config`` a member group hands TabPFN, as plain data.

    Every prediction-changing field is set here, so none is left to the
    checkpoint or the installed tabpfn: the version's own recipe, or for a
    group of explicit transforms that recipe stripped of its extras. A
    bagged member under the recipe runs the one preprocessor and target
    transform upstream gives that member of the whole ensemble
    (:func:`_recipe_slot`), so the bagged ensemble keeps the recipe's mix.

    Args:
        version: The TabPFN version.
        group: The member group.
        outlier_threshold: The resolved ``outlier_threshold``, or None.
        n_estimators: The size of the whole ensemble.

    Returns:
        ``InferenceConfig`` field names to values, the preprocessors as
        dicts of ``PreprocessorConfig`` fields. ``SUBSAMPLE_SAMPLES`` is
        always None: a bagged member's regressor is given its rows alone.
    """
    settings = {
        field: value
        for field, value in _AUTO_RECIPES[version].items()
        # Handed over as the estimator's own argument instead.
        if field != "SOFTMAX_TEMPERATURE"
    }
    if group.native_transforms is not None:
        # An explicit recipe: the named transforms and none of the extras,
        # with the version's own limit on features per member. Upstream
        # gives each listed config an equal share of the members, repeats
        # included, so one cycle of the plan keeps its weights.
        limit = max(
            config["max_features_per_estimator"]
            for config in settings["PREPROCESS_TRANSFORMS"]
        )
        settings["PREPROCESS_TRANSFORMS"] = tuple(
            _preprocessor(name, "numeric", original, limit, None)
            for name, original in _cycle(group.native_transforms)
        )
        settings["FINGERPRINT_FEATURE"] = False
        settings["POLYNOMIAL_FEATURES"] = "no"
        settings["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = (None,)
    elif group.rows is not None:
        preprocessor, target = _recipe_slot(
            settings, group.members[0].index, n_estimators
        )
        settings["PREPROCESS_TRANSFORMS"] = (preprocessor,)
        settings["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = (target,)
    # Every column is numeric: upstream would otherwise take one with fewer
    # than four distinct values in over 100 rows for a category and encode
    # it. A column needs one value to count as numeric (none declared
    # categorical, so no other threshold applies), and one with fewer is
    # constant, which upstream drops anyway.
    settings["MIN_UNIQUE_FOR_NUMERICAL_FEATURES"] = 1
    settings["OUTLIER_REMOVAL_STD"] = outlier_threshold
    if not group.feature_shuffle:
        settings["FEATURE_SHIFT_METHOD"] = None
    settings["SUBSAMPLE_SAMPLES"] = None
    return settings


def _recipe_slot(
    settings: Mapping[str, Any], index: int, n_estimators: int
) -> tuple[dict[str, Any], str | None]:
    """The preprocessor and target transform upstream gives one member.

    Upstream pairs every preprocessor with every target transform and hands
    the pairs out in equal blocks, in order, with any remainder going to the
    first pairs (``generate_regression_ensemble_configs``).

    Args:
        settings: The recipe, with ``PREPROCESS_TRANSFORMS`` and
            ``REGRESSION_Y_PREPROCESS_TRANSFORMS``.
        index: The member's position in the ensemble.
        n_estimators: The size of the whole ensemble.

    Returns:
        A tuple ``(preprocessor, target_transform)``.

    Examples:
        >>> recipe = {
        ...     "PREPROCESS_TRANSFORMS": ("a", "b"),
        ...     "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None, "safepower"),
        ... }
        >>> [_recipe_slot(recipe, i, 5)[0] for i in range(5)]
        ['a', 'a', 'b', 'b', 'a']
        >>> [_recipe_slot(recipe, i, 5)[1] for i in range(5)]
        [None, 'safepower', None, 'safepower', None]
    """
    pairs = list(
        itertools.product(
            settings["PREPROCESS_TRANSFORMS"],
            settings["REGRESSION_Y_PREPROCESS_TRANSFORMS"],
        )
    )
    per_pair = n_estimators // len(pairs)
    if index < per_pair * len(pairs):
        return pairs[index // per_pair]
    return pairs[index - per_pair * len(pairs)]


def _cache_options(kv_cache: bool | str) -> dict[str, Any]:
    """TabPFN's fit mode and cache precision for a ``kv_cache`` value.

    Upstream quantises its cache to int8 by default on v3 and later, which
    changes the outputs; ``True`` asks for full precision, so caching is
    exact.
    """
    if kv_cache is False:
        return {"fit_mode": "fit_preprocessors"}
    precision = "auto" if kv_cache is True else kv_cache
    return {"fit_mode": "fit_with_cache", "kv_cache_precision": precision}


class TabPFNBarDistribution(_ensemble.ContextEnsembleEstimator):
    """Target distributions from the bucket masses TabPFN predicts.

    Args:
        version: Which TabPFN to run, in upstream's own vocabulary: ``"v2"``,
            ``"v2.5"``, ``"v2.6"``, ``"v3"``, ``"v3.5"`` (the default) or
            ``"v3.5-fast"``. Each is a separately pinned checkpoint
            (:func:`lazy.list_versions`), so sweeping this parameter compares
            model versions on equal terms, and ``provenance_`` records which
            one answered. They differ in more than accuracy: ``"v2"`` is the
            only one under a commercial-use licence, and it and ``"v2.5"``
            declare much smaller context limits than v3 (see
            ``ignore_pretraining_limits``).
        n_estimators: Ensemble members, each a forward pass over a differently
            preprocessed view of the data; costs scale linearly. Exactly this
            many run.
        transforms: Per-member feature transforms: ``"auto"`` (the version's
            own recipe, pinned in :data:`_AUTO_RECIPES`: its preprocessors,
            including any SVD components and original columns they append,
            the fingerprint feature, polynomial features on v2.6, the target
            transforms, and on v3.5 a 12-sigma soft clip), a recipe name, a
            transform name or a sequence of them; see
            :mod:`lazy.models._transforms`. An explicit value is all the
            model sees: no fingerprint or polynomial features, no SVD, the
            identity target transform, and outlier clipping only if
            ``outlier_threshold`` asks for it. TabPFN runs only ``none``
            natively (its other transforms are not the package's) and lazy
            applies the rest. What runs whatever the recipe: the removal of
            constant columns and TabPFN's internal standardisation.
        feature_shuffle: Whether members see the columns in different orders
            (TabPFN's own feature shuffling).
        bag_size: Context rows per member: an int is a row count (1 means
            one row), a float a fraction in (0, 1] (1.0 means all rows), and
            None all of them. Each member is then a regressor of its own,
            fitted on its bag: its preprocessing, outlier clip, target
            standardisation and buckets all come from those rows, and only
            the bag must lie within the limits ``ignore_pretraining_limits``
            describes, so bagging is how TabPFN takes a context larger than
            them.
        kv_cache: Cache the context's keys and values at fit, so each chunk
            of queries skips the context forward pass: ``True`` (exact, full
            precision), ``"int8"`` or ``"fp8"`` (quantised: smaller, not
            exact; v3 and later only), or ``False``. Worth its memory
            whenever the query set is much larger than the context.
        y_grid: Default output grid: a :class:`lazy.grid.Grid`, an
            array of bin centres, ``"native"``, or None for the native grid
            (the bar distribution's own buckets, in full).
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"mps"``, ``"cpu"``, or a ``torch.device``.
        random_state: Seed for the ensemble. None draws a fresh seed at fit,
            recorded as ``random_state_`` and in ``provenance_``.
        softmax_temperature: Temperature on the bucket logits, which sets how
            sharp the densities are. ``"auto"`` takes the version's
            calibrated value (0.9, and 1.0 on v3.5 and v3.5-fast), recorded in
            ``provenance_``; lower sharpens, higher broadens.
        mixed_precision: On CUDA, run in TabPFN's autocast (float16) path;
            otherwise, and always on a CPU, in float32 (upstream would choose
            bfloat16 on CPUs with fast bfloat16).
        outlier_threshold: TabPFN's soft clip of each feature at this many
            standard deviations of the context (``OUTLIER_REMOVAL_STD``).
            ``"auto"`` is the version's own under ``transforms="auto"``
            (12.0 on v3.5 and v3.5-fast, none on the others) and none under
            an explicit recipe; None is none.
        ignore_pretraining_limits: Pass ``True`` to run a context larger than
            the row count the checkpoint declares it was pretrained for, or
            more than upstream allows on a CPU, which otherwise raises.
            Predictions beyond that limit are extrapolation, so this is
            opt-in -- though TabPFN-3 declares a million rows. Under
            bagging only each member's bag is checked, so a ``bag_size``
            within the limit runs any context without it.
        chunk_size: Query rows predicted at a time, to bound peak memory;
            ``0`` does them in one pass. A row's answer never depends on the
            other rows in its chunk: TabPFN's attention builds its keys and
            values from the context rows alone and its preprocessors are
            fitted on the context. Bit for bit on a CPU; under mixed
            precision on a GPU, to float16 rounding.
        progress: A progress bar over the query rows: ``"auto"`` shows it
            on a terminal or in a notebook, ``True`` always, ``False`` never.
        verbose: Print log messages to stdout.

    Attributes:
        grid_: The resolved default output grid.
        native_grid_: The bar distribution's buckets in the target's units, a
            histogram-normalised grid, set at fit; under bagging, the union
            of every member's buckets.
        checkpoint_: The pinned checkpoint file, a :class:`pathlib.Path`.
        provenance_: Which weights, code and ensemble answered, as a dict.
        regressor_: The fitted ``tabpfn.TabPFNRegressor``, when one serves
            the whole ensemble.
        handles_: Every fitted regressor, one per member group.
        borders_: The first member group's bucket borders in the target's
            units, shape (``n_buckets_`` + 1,); fixed at fit by its context
            targets' mean and spread, which under bagging is its bag's.
        n_buckets_: How many buckets the bar distribution has.
        n_context_: Context rows ``fit`` was given.

    Examples:
        >>> est = TabPFNBarDistribution(n_estimators=8, chunk_size=50_000)
        >>> est.chunk_size
        50000
        >>> TabPFNBarDistribution(version="v2.5").name_
        'tabpfn:v2.5'
    """

    backend = "tabpfn"
    display_name = "TabPFN"
    extra = "tabpfn"
    native_output = "histogram"
    native_transforms = _native_transforms()
    # A regressor per bag: upstream's own subsampling standardises the
    # target on the whole context (see the module docstring).
    supports_native_bagging = False
    native_outlier_clipping = True
    # The fast checkpoint runs DC1's 1,000 + 1,000 rows on a CPU in
    # about 11 s with eight members (TabPFN-3.5 26 s, TabICL 7 s), within the
    # 5,000 context rows upstream allows on a CPU.
    cpu_friendly_versions = ("v3.5-fast",)
    kv_cache_modes = (True, False, "int8", "fp8")
    kv_cache_rtol = 1e-5

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v3.5",
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
        ignore_pretraining_limits: bool = False,
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
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.progress = progress
        self.verbose = verbose

    def _import_backend(self) -> types.ModuleType:
        try:
            import tabpfn  # noqa: PLC0415 - an optional, heavy extra.
        except ImportError as error:
            # Only the backend itself missing is a missing extra; anything
            # it fails to import in turn is reported as it is.
            if (error.name or "").partition(".")[0] != "tabpfn":
                raise
            raise ImportError(
                "TabPFNBarDistribution needs the tabpfn backend: "
                "pip install 'lazy-tfm[tabpfn]'"
            ) from error
        return tabpfn

    def _check_backend_params(self) -> None:
        if (
            self.kv_cache in ("int8", "fp8")
            and self.version in _UNQUANTISED_VERSIONS
        ):
            raise ValueError(
                f"TabPFN {self.version} has no quantised key/value cache, so "
                f"kv_cache={self.kv_cache!r} would run at full precision; "
                "use kv_cache=True, or version='v3' or later"
            )

    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> Any:
        # The pinned checkpoint is handed over by path, never left to
        # TabPFN's own default (the newest version it knows, which moves with
        # the package): this way `CHECKPOINTS` is the authority on which
        # TabPFN answers, and `version` means something.
        # Every other argument that changes a prediction is passed too, at
        # the value upstream defaults to today, so that a later default
        # cannot change it.
        tabpfn = self._import_backend()
        regressor = tabpfn.TabPFNRegressor(
            n_estimators=group.n_members,
            model_path=path_for_tabpfn(self.checkpoint_),
            device=self.device_,
            random_state=group.seed,
            categorical_features_indices=None,
            softmax_temperature=self._temperature(),
            average_before_softmax=False,
            tuning_config=None,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            show_progress_bar=False,
            inference_precision=self._inference_precision(),
            inference_config=self._inference_config(group),
            **_cache_options(self.kv_cache),
        )
        try:
            with self._shared_network(tabpfn):
                regressor.fit(X, y)
        except (ValueError, RuntimeError) as error:
            remedied = _with_bagging_remedy(error, regressor, len(X), self)
            if remedied is error:
                raise
            raise remedied from error
        if group.index == 0:
            self.borders_ = _bucket_borders(regressor)
            self.n_buckets_ = int(self.borders_.size - 1)
        return regressor

    def _fit(self, X: Any, y: _typing.FloatArray) -> None:
        # The loaded networks of this fit, shared by its regressors.
        self._networks: dict[tuple[Any, ...], Any] = {}
        try:
            super()._fit(X, y)
        finally:
            self.__dict__.pop("_networks", None)

    @contextlib.contextmanager
    def _shared_network(self, tabpfn: types.ModuleType) -> Iterator[None]:
        """Loads the network once, for every group's regressor and every fit.

        Each ``TabPFNRegressor`` loads its own copy of the network at fit,
        so the regressors of a bagged ensemble (one per member) would hold
        ``n_estimators`` copies on the device, and every refit would read
        the checkpoint again. Upstream's networks hold no state between
        calls (the key/value cache lives in the inference engine, and the
        architectures ignore ``cache_trainset_representation``), so the
        regressors share one: upstream's loader is wrapped, for the duration
        of a fit, to hand back the network loaded first for the same
        arguments, in this fit or, through :mod:`lazy.models._weights`, an
        earlier one. Upstream moves the network to the device, and casts it
        to a forced precision, in place, so both are part of the key. The
        bar distribution and the configs, which a fit may move and rescale,
        are copied.

        Args:
            tabpfn: The imported ``tabpfn`` package.

        Yields:
            Nothing; the loader is restored on exit.
        """
        base = tabpfn.base
        load = base.initialize_tabpfn_model
        networks = self.__dict__.setdefault("_networks", {})
        precision = self._inference_precision()

        def shared(**kwargs: Any) -> Any:
            local = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
            if local not in networks:
                networks[local] = self._load_network(load, precision, kwargs)
            models, configs, bardist, config = networks[local]
            return list(models), *copy.deepcopy((configs, bardist, config))

        base.initialize_tabpfn_model = shared
        try:
            yield
        finally:
            base.initialize_tabpfn_model = load

    def _load_network(
        self,
        load: Callable[..., Any],
        precision: Any,
        arguments: dict[str, Any],
    ) -> Any:
        """Upstream's loader, through the process-wide cache when it can be.

        Args:
            load: Upstream's ``initialize_tabpfn_model``.
            precision: The ``inference_precision`` the regressors run at.
            arguments: What upstream called the loader with.

        Returns:
            What ``load`` returns.
        """
        path = arguments.get("model_path")
        if not isinstance(path, (str, os.PathLike)):
            # Not one checkpoint file: nothing to key it by.
            return load(**arguments)
        # The overrides only reconcile several checkpoints' configs, so one
        # file loads the same network whatever they are.
        cache_key = _weights.key(
            self.backend,
            self.version,
            path,
            self.device_,
            precision,
            arguments.get("which"),
            arguments.get("fit_mode"),
            load,
        )
        return _weights.network(cache_key, functools.partial(load, **arguments))

    def _inference_precision(self) -> Any:
        """Upstream's ``inference_precision``: autocast, or float32.

        Upstream's own default, ``"auto"``, autocasts on a GPU (float16) and
        on a CPU with fast bfloat16 (AMX, AVX512-BF16, Zen 4); here only
        ``mixed_precision_`` autocasts, which is never on a CPU.
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        return "autocast" if self.mixed_precision_ else torch.float32

    def _auto_softmax_temperature(self) -> float:
        return float(_AUTO_RECIPES[self.version]["SOFTMAX_TEMPERATURE"])

    def _auto_outlier_threshold(self) -> float | None:
        return _AUTO_RECIPES[self.version]["OUTLIER_REMOVAL_STD"]

    def _inference_config(self, group: _members.MemberGroup) -> dict[str, Any]:
        """The ``inference_config`` a group hands TabPFN, ready to pass."""
        import tabpfn.preprocessing.configs  # noqa: PLC0415 - optional extra.

        settings = _upstream_settings(
            self.version, group, self.outlier_threshold_, self._n_members()
        )
        settings["PREPROCESS_TRANSFORMS"] = [
            tabpfn.preprocessing.configs.PreprocessorConfig(**fields)
            for fields in settings["PREPROCESS_TRANSFORMS"]
        ]
        return settings

    def _recipe(self) -> dict[str, Any]:
        # What each group handed upstream; a bagged member's rows follow
        # from random_state.
        upstream = [
            _upstream_settings(
                self.version, group, self.outlier_threshold_, self._n_members()
            )
            for group in self.member_groups_
        ]
        return {
            **super()._recipe(),
            "inference_precision": (
                "autocast" if self.mixed_precision_ else "float32"
            ),
            "inference_config": upstream,
        }

    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.HistogramDistribution:
        _, masses = bucket_masses(handle.predict(X, output_type="full"))
        return distributions.HistogramDistribution(
            _bucket_borders(handle), masses
        )

    def _native_grid(self) -> grid_lib.Grid:
        # Every member group's buckets: under bagging each member places
        # its own, from its bag's targets.
        edges = np.unique(
            np.concatenate([_bucket_borders(h) for h in self.handles_])
        )
        if edges.size < 3:
            # A constant target has one bucket; a grid needs two, and
            # halving it changes no density.
            edges = np.r_[edges[0], edges.mean(), edges[-1]]
        return grid_lib.Grid.from_edges(edges, normalization="histogram")

    def _progress_postfix(
        self, dist: distributions.Distribution
    ) -> dict[str, Any]:
        del dist  # Unused: the bucket count is fixed at fit.
        return {"buckets": self.n_buckets_}


def _with_bagging_remedy(
    error: Exception,
    regressor: Any,
    n_rows: int,
    estimator: TabPFNBarDistribution,
) -> Exception:
    """Upstream's size-limit error, saying that bagging is the remedy.

    Upstream checks the context it is given against the rows the checkpoint
    was pretrained for and against its CPU cap, and names only
    ``ignore_pretraining_limits``; a member's context is its bag, so a
    smaller ``bag_size`` is the other way out.

    Args:
        error: What ``TabPFNRegressor.fit`` raised.
        regressor: The regressor that raised it.
        n_rows: The context rows it was given.
        estimator: The estimator fitting it.

    Returns:
        ``error`` itself when it is not a limit on rows, else an error of
        the same type that says how to bag below the limit.
    """
    message = str(error)
    config = getattr(regressor, "inference_config_", None)
    if "Number of samples" in message:
        limit = getattr(config, "MAX_NUMBER_OF_SAMPLES", None)
    elif "Running on CPU with more than" in message:
        limit = getattr(config, "MAX_CPU_SAMPLES", None)
    else:
        return error
    if not isinstance(limit, int):
        return error
    needed = -(-estimator.n_context_ // limit)
    seen = (
        f"each member's bag has {n_rows:,} rows"
        if estimator.bagging_
        else f"the context has {n_rows:,} rows"
    )
    return type(error)(
        f"{message}\nTabPFN {estimator.version} takes at most {limit:,} "
        f"context rows here, and {seen}. Bag the context below the limit "
        f"instead: pass bag_size={limit} (or fewer) with n_estimators >= "
        f"{needed}, so that each member's bag fits and the members together "
        "cover the context."
    )


def _cycle(tokens: tuple[Any, ...]) -> tuple[Any, ...]:
    """The shortest prefix of ``tokens`` that repeats to give all of them.

    Upstream shares the members out among its preprocessing configs in
    equal blocks (an explicit recipe has one target transform, the
    identity), so handing it one cycle of the planned transforms, repeats
    included, keeps their weights: exactly when the member count is a
    multiple of the cycle's length, and to within one member otherwise.

    Args:
        tokens: TabPFN's token for each member's transform, in member order.

    Returns:
        The cycle, e.g. ``("power", "power", "none")`` for the tokens of
        ``transforms=("power", "power", "none")``.

    Examples:
        >>> _cycle(("a", "b", "a", "b", "a"))
        ('a', 'b')
        >>> _cycle(("a", "a", "b", "a", "a", "b"))
        ('a', 'a', 'b')
    """
    for length in range(1, len(tokens)):
        if all(token == tokens[i % length] for i, token in enumerate(tokens)):
            return tokens[:length]
    return tokens


def _bucket_borders(regressor: Any) -> _typing.FloatArray:
    """The fitted regressor's bucket borders in the target's units.

    Upstream keeps them as ``raw_space_bardist_``, but in float32, which
    cannot resolve a narrow spread about a large offset (for targets
    ``1e4 + 0.01 * noise`` most buckets come out zero-width). So they are
    rebuilt here in float64 exactly as upstream builds them, from the
    z-normalised borders and the context targets' mean and standard
    deviation; the buckets, and so the masses in them, are the same ones.

    For a constant target upstream skips the model altogether, predicts one
    bucket around the constant, and keeps that bucket (already in the
    target's units) as ``znorm_space_bardist_``.

    Args:
        regressor: A fitted ``tabpfn.TabPFNRegressor``.

    Returns:
        The borders, made non-decreasing, shape ``(n_buckets + 1,)``.
    """
    znorm = regressor.znorm_space_bardist_.borders.detach().cpu().numpy()
    borders = np.asarray(znorm, dtype=np.float64)
    if not getattr(regressor, "is_constant_target_", False):
        borders = borders * float(regressor.y_train_std_) + float(
            regressor.y_train_mean_
        )
    return np.maximum.accumulate(borders)


def path_for_tabpfn(path: pathlib.Path) -> pathlib.Path:
    """Returns the path to hand TabPFN, not always the one we downloaded.

    TabPFN decides a checkpoint's format from its *resolved* path
    (``model_loading.load_model`` does ``str(path.resolve())``, and
    ``checkpoint.Checkpoint`` then reads the suffix). In the Hugging Face
    cache a snapshot file is a symlink to a content-addressed blob with no
    extension, so resolving it throws the suffix away: a ``.safetensors``
    checkpoint -- which from ``v3.5`` on is how TabPFN ships -- is then handed
    to ``torch.load`` and dies with an ``UnpicklingError``. The ``.ckpt``
    versions are unharmed, being torch archives already.

    A hardlink beside the blob gives the same bytes a name whose suffix does
    survive ``resolve()``. Nothing is copied -- it is one more directory entry
    for an inode that is already there -- and it lives inside the cache entry
    it belongs to, so deleting the model from the HF cache takes it too.

    The rewrite is conditional on the suffix actually being lost, so it stops
    happening by itself if upstream resolves the path differently, or when
    ``HF_HUB_DISABLE_SYMLINKS`` is set and the snapshot is a real file.

    Args:
        path: The downloaded checkpoint, as
            :meth:`lazy.models._hub.Checkpoint.download` returns it.

    Returns:
        ``path`` itself when its suffix survives resolution; otherwise a
        hardlink to the same blob whose resolved name keeps the suffix.

    Raises:
        RuntimeError: If the hardlink cannot be created, for instance on a
            file system without hardlinks.
    """
    blob = path.resolve()
    if blob.suffix == path.suffix:
        return path
    link = blob.parent.parent / "lazy-suffixed" / blob.name[:12] / path.name
    if not link.exists():
        try:
            link.parent.mkdir(parents=True, exist_ok=True)
            os.link(blob, link)
        except OSError as error:
            raise RuntimeError(
                f"cannot give {path.name} a name TabPFN can read its format "
                f"from ({error}). Set HF_HUB_DISABLE_SYMLINKS=1 and "
                "re-download so the cached file is a real one, or use a "
                "version whose checkpoint is a .ckpt, such as version='v3'."
            ) from error
    return link


def bucket_masses(
    output: Mapping[str, Any],
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Returns the bucket borders and masses of a TabPFN full prediction.

    ``output["logits"]`` are log-probabilities over the buckets of
    ``output["criterion"]``, whose ``borders`` are in the raw target's units
    -- the target, here. A softmax turns the first into per-bucket probability
    mass, which is what every upstream reduction (the mean, the quantiles)
    integrates over too, so this is the model's own density and not a reading
    of it.

    Args:
        output: What ``TabPFNRegressor.predict(..., output_type="full")``
            returns; only its ``"criterion"`` and ``"logits"`` are read.

    Returns:
        A tuple ``(borders, masses)``: the bucket borders in the target's
        units, made non-decreasing, shape ``(n_buckets + 1,)``; and the
        probability mass in each bucket, shape ``(n_rows, n_buckets)``.

    Raises:
        RuntimeError: If the logits do not match the bar distribution's
            buckets.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    criterion, logits = output["criterion"], output["logits"]
    borders = np.maximum.accumulate(
        np.asarray(criterion.borders.detach().cpu().numpy(), dtype=np.float64)
    )
    masses = torch.softmax(logits.detach().double(), dim=-1).cpu().numpy()
    if masses.ndim != 2 or masses.shape[1] != borders.size - 1:
        raise RuntimeError(
            f"TabPFN returned logits of shape {tuple(masses.shape)} for a bar "
            f"distribution with {borders.size - 1} buckets"
        )
    return borders, masses
