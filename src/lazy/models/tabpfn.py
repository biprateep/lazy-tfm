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
(:meth:`~lazy.distributions.HistogramDistribution.on_grid`), and its native
grid is the buckets themselves, in full. The bucket masses rather than
quantiles, because upstream computes its quantiles by inverting this same
piecewise-uniform CDF.

The uniform parameters map onto TabPFN's own machinery: ``kv_cache`` onto its
fit-time key/value cache (at full precision, so exact to float rounding on
a CPU),
``feature_shuffle`` onto its ``FEATURE_SHIFT_METHOD``, ``outlier_threshold``
onto its soft clip, ``OUTLIER_REMOVAL_STD``,
``softmax_temperature`` onto its own and ``mixed_precision`` onto its
``inference_precision``. No setting is left to TabPFN: each version's own
recipe is written out below (:data:`_RECIPES`) and handed over in full,
so neither the checkpoint's stored config nor the installed tabpfn package
decides what runs, and every column is numeric (upstream would take a column
with fewer than four distinct values for a category).

``bag_size`` runs one regressor per member, fitted on that member's bag
alone. TabPFN's own row subsampling (``SUBSAMPLE_SAMPLES``) would fit each
member's preprocessing on its rows but standardize the target, and place the
buckets, with the whole context's mean and standard deviation, and it checks
the whole context against the size limits; a regressor per bag fits all of
it on the bag, as every other model does, and only the bag meets the limits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import inspect
import io
import itertools
import os
import pathlib
import pickle
import sys
import types
from typing import Any

import numpy as np

from lazy import _typing
from lazy import _warnings
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _bucket_head
from lazy.models import _ensemble
from lazy.models import _members
from lazy.models import _progress

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


_UNQUANTISED_VERSIONS = ("v2", "v2.5", "v2.6")
"""Versions whose architecture has no quantised key/value cache; upstream would
quietly fall back to full precision.
"""


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


_SHARED_RECIPE: dict[str, Any] = {
    "FEATURE_SHIFT_METHOD": "shuffle",
    "FINGERPRINT_FEATURE": True,
    "FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT": 50,
    "SAMPLE_SUBSAMPLING_METHOD": "auto",
    "FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM": True,
    "PASSTHROUGH_INF": False,
}
"""Settings every version's own recipe shares."""

_RECIPES: dict[str, dict[str, Any]] = {
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
"""Each version's own recipe (``transforms="auto"``): every ``InferenceConfig``
field that changes a regression prediction on numeric features, as the
checkpoint stores it (v2.6 and later) or, for v2 and v2.5, which predate stored
configs, as tabpfn 9.0.0's ``InferenceConfig.get_default`` writes it. Copied
here so that a tabpfn upgrade cannot change what ``"auto"`` runs.
``OUTLIER_REMOVAL_STD`` is the resolved value (upstream's ``"auto"`` is None for
a regressor), and ``SOFTMAX_TEMPERATURE`` the checkpoint's calibrated one; both
are what ``"auto"`` resolves to, and are handed over as such.
"""
# The fast variant is the same recipe on a smaller checkpoint.
_RECIPES["v3.5-fast"] = _RECIPES["v3.5"]


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
        for field, value in _RECIPES[version].items()
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
            for name, original in _shortest_period(group.native_transforms)
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
    exact. Without the cache, upstream ignores the precision.
    """
    fit_mode = "fit_preprocessors" if kv_cache is False else "fit_with_cache"
    precision = kv_cache if isinstance(kv_cache, str) else "auto"
    return {"fit_mode": fit_mode, "kv_cache_precision": precision}


class TabPFNBarDistribution(_ensemble.ContextEnsembleEstimator):
    """Target distributions from the bucket masses TabPFN predicts.

    The parameters every backend shares mean what :doc:`/guide/interface`
    states; each entry below adds only what differs on TabPFN.

    Args:
        version: Which TabPFN to run, in upstream's own vocabulary: ``"v2"``,
            ``"v2.5"``, ``"v2.6"``, ``"v3"``, ``"v3.5"`` (the default) or
            ``"v3.5-fast"``. Each is a separately pinned checkpoint, so
            sweeping this parameter compares model versions on equal terms.
            They differ in more than accuracy: ``"v2"`` is the only one under
            a commercial-use license, and it and ``"v2.5"`` declare much
            smaller context limits than v3 (see
            ``ignore_pretraining_limits``).
        n_estimators: Ensemble members, each a forward pass over a
            differently preprocessed view of the data.
        transforms: Per-member feature transforms. ``"auto"`` is the
            version's own recipe, pinned in :data:`_RECIPES`: its
            preprocessors, including any SVD components and original columns
            they append, the fingerprint feature, polynomial features on
            v2.6, the target transforms, and on v3.5 a 12-sigma soft clip.
            An explicit value has no fingerprint or polynomial features, no
            SVD and the identity target transform. TabPFN runs only ``none``
            natively (its other transforms are not the package's) and lazy
            applies the rest. What runs whatever the recipe: the removal of
            constant columns and TabPFN's internal standardization.
        feature_shuffle: TabPFN's own feature shuffling.
        bag_size: Context rows per member. Each member is then a regressor
            of its own, fitted on its bag: its preprocessing, outlier clip,
            target standardization and buckets all come from those rows, and
            only the bag must lie within the limits
            ``ignore_pretraining_limits`` describes, so bagging is how TabPFN
            takes a context larger than them.
        kv_cache: Cache the context's keys and values at fit: ``True``
            (exact, full precision), ``"int8"`` or ``"fp8"`` (quantised:
            smaller, not exact; v3 and later only), or ``False``.
        y_grid: Default output grid; the native grid is the bar
            distribution's own buckets, in full.
        device: Where to run.
        random_state: The ensemble's seed.
        softmax_temperature: Temperature on the bucket logits. ``"auto"`` is
            the version's calibrated value: 0.9, and 1.0 on v3.5 and
            v3.5-fast.
        mixed_precision: On CUDA, TabPFN's autocast (float16) path; float32
            on a CPU, where upstream would choose bfloat16 on CPUs with fast
            bfloat16.
        outlier_threshold: TabPFN's own soft clip (``OUTLIER_REMOVAL_STD``).
            Under ``transforms="auto"``, ``"auto"`` is 12.0 on v3.5 and
            v3.5-fast and none on the others.
        ignore_pretraining_limits: Pass ``True`` to run a context larger than
            the row count the checkpoint declares it was pretrained for, or
            more than upstream allows on a CPU, which otherwise raises.
            Predictions beyond that limit are extrapolation, so this is
            opt-in -- though TabPFN-3 declares a million rows. Under
            bagging only each member's bag is checked, so a ``bag_size``
            within the limit runs any context without it.
        chunk_size: Query rows per forward pass. TabPFN's attention builds
            its keys and values from the context rows alone and its
            preprocessors are fitted on the context, so a row's answer does
            not depend on its chunk, to float rounding on a CPU.
        progress: A progress bar over the query rows.
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
    method_note = "Bucket masses of TabPFN's bar distribution."
    extra = "tabpfn"
    upstream_package = "tabpfn"
    native_output = "histogram"
    native_transforms = _native_transforms()
    # A regressor per bag: upstream's own subsampling standardizes the
    # target on the whole context (see the module docstring).
    supports_native_bagging = False
    native_outlier_clipping = True
    # Only the fast checkpoint is quick enough on a CPU, where upstream
    # allows 5,000 context rows.
    cpu_friendly_versions = ("v3.5-fast",)
    cpu_note = (
        " and refuses more than 5,000 context rows there (1,000 before v3)"
        " unless ignore_pretraining_limits=True"
    )
    kv_cache_modes = (True, False, "int8", "fp8")
    kv_cache_rtol = 1e-5
    # Not bit for bit on every CPU: on a 4-core x86 machine, 7-row chunks
    # and one pass differ in the last bits of most densities.
    exact_chunking = False
    recipes = _RECIPES
    softmax_temperature_key = "SOFTMAX_TEMPERATURE"
    outlier_threshold_key = "OUTLIER_REMOVAL_STD"

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

    @classmethod
    def clear_upstream_caches(cls) -> None:
        """Empties the checkpoint caches TabPFN keeps of its own.

        TabPFN holds the last checkpoint it read in memory, beside the network
        built from it, and, when ``$TABPFN_MODEL_CACHE_SIZE`` asks, the built
        networks too. Both are private, so each is emptied only if it is
        there.
        """
        model_loading = sys.modules.get("tabpfn.model_loading")
        if model_loading is None:
            return
        raw = getattr(model_loading, "_load_checkpoint_cached", None)
        if raw is not None and hasattr(raw, "cache_clear"):
            raw.cache_clear()
        built = getattr(model_loading, "clear_built_model_cache", None)
        if built is not None:
            built()

    def _check_backend_params(self) -> None:
        if not isinstance(self.ignore_pretraining_limits, bool):
            raise ValueError(
                "ignore_pretraining_limits must be a bool: "
                f"{self.ignore_pretraining_limits=}"
            )
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
        # The pinned checkpoint is handed over loaded, never left to
        # TabPFN's own default (the newest version it knows, which moves with
        # the package): this way `CHECKPOINTS` is the authority on which
        # TabPFN answers, and `version` means something.
        # Every other argument is passed too, at the value upstream defaults
        # to today, so that a later default cannot change a prediction.
        # auto_scale_n_estimators acts only on n_estimators="auto", and
        # n_jobs does nothing; both are deprecated upstream.
        tabpfn = self._import_backend()
        regressor = tabpfn.TabPFNRegressor(
            n_estimators=group.n_members,
            model_path=self._model_specs(tabpfn),
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
            memory_saving_mode="auto",
            keep_cache_on_device=True,
            n_preprocessing_jobs=1,
            n_jobs=None,
            auto_scale_n_estimators=True,
            differentiable_input=False,
            eval_metric=None,
            **_cache_options(self.kv_cache),
        )
        try:
            regressor.fit(X, y)
        except (ValueError, RuntimeError) as error:
            remedied = _with_bagging_remedy(error, regressor, len(X), self)
            if remedied is error:
                raise
            raise remedied from error
        # Upstream reads model_path only at fit, so a saved regressor names
        # the checkpoint, as one given the path does, not the spec's copies.
        regressor.model_path = path_for_tabpfn(self.checkpoint_)
        if group.index == 0:
            self.borders_ = _bucket_borders(regressor)
            self.n_buckets_ = int(self.borders_.size - 1)
        return regressor

    def _model_specs(self, tabpfn: types.ModuleType) -> Any:
        """The shared network, as the model spec of one upstream regressor.

        A ``TabPFNRegressor`` given a checkpoint path loads its own copy of
        the network, so the regressors of a bagged ensemble (one per member)
        would hold ``n_estimators`` copies on the device, and every refit
        would read the checkpoint again. Given a model spec as
        ``model_path``, it runs on the spec's network by reference, so every
        regressor runs on the one :meth:`_network` loads. Upstream's
        networks hold no state between calls (the key/value cache lives in
        the inference engine), so they can share one. A fit moves and
        rescales the bar distribution, and may change the configs, so each
        regressor gets its own copies of them.

        Args:
            tabpfn: The imported ``tabpfn`` package.

        Returns:
            A ``tabpfn.base`` model spec.
        """
        models, architectures, bardist, config = self._network()
        (model,) = models
        # tabpfn 9.2 replaced RegressorModelSpecs with one ModelSpecs class.
        specs = getattr(tabpfn.base, "RegressorModelSpecs", None)
        if specs is None:
            specs = tabpfn.base.ModelSpecs
        return specs(model, *copy.deepcopy((architectures[0], config, bardist)))

    def _network_loader(self) -> tuple[Any, Callable[..., Any]]:
        """The regressors' precision and upstream's loader.

        A regressor moves the network to the device, and casts it to a
        forced precision, in place, so both are part of the network's key.
        """
        tabpfn = self._import_backend()
        return self._inference_precision(), tabpfn.base.initialize_tabpfn_model

    def _read_network(
        self, load: Callable[..., Any], checkpoint: pathlib.Path
    ) -> Any:
        """Loads the checkpoint onto ``device_``, as its own network.

        Upstream's built-model cache (``$TABPFN_MODEL_CACHE_SIZE``) shares a
        network by reference, and a regressor moves and casts it in place,
        so each of lazy's keys must get a network of its own. From tabpfn
        9.2 the cache is keyed by the devices and forced dtype, passed here
        as the regressor computes them; before, by the checkpoint alone,
        and it is skipped under ``fit_mode="fit_with_cache"``, which every
        architecture otherwise ignores.

        Returns:
            Upstream's models, architecture configs, bar distribution and
            inference config.
        """
        tabpfn = self._import_backend()
        accepted = inspect.signature(load).parameters
        options: dict[str, Any] = {}
        if "fit_mode" in accepted:
            options["fit_mode"] = "fit_with_cache"
        if "devices" in accepted:
            precision = self._inference_precision()
            options |= {
                "devices": tabpfn.utils.infer_devices(self.device_),
                # As upstream's determine_precision: autocast forces none.
                "force_inference_dtype": (
                    None if precision == "autocast" else precision
                ),
            }
        loaded = load(
            model_path=path_for_tabpfn(checkpoint), which="regressor", **options
        )
        for model in loaded[0]:
            model.to(self.device_)
        return loaded

    def _inference_precision(self) -> Any:
        """Upstream's ``inference_precision``: autocast, or float32.

        Upstream's own default, ``"auto"``, autocasts on a GPU (float16) and
        on a CPU with fast bfloat16 (AMX, AVX512-BF16, Zen 4); here only
        ``mixed_precision_`` autocasts, which is never on a CPU.
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        return "autocast" if self.mixed_precision_ else torch.float32

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
            **self._relocated_provenance(),
            "inference_config": upstream,
        }

    def _relocated_provenance(self) -> dict[str, Any]:
        return {
            "inference_precision": (
                "autocast" if self.mixed_precision_ else "float32"
            )
        }

    def __getstate__(self) -> dict[str, Any]:
        """Pickles the regressors without the network, loadable anywhere.

        The regressors share one network (:meth:`_model_specs`), which is
        left out and attached again on unpickling. The rest of each
        regressor, its key/value cache included, goes through
        ``torch.save``, so that :meth:`__setstate__` can load its tensors
        on the CPU, which a plain pickle of CUDA tensors cannot. The live
        model is only read.

        Returns:
            The state to pickle, with ``handles_`` as bytes.
        """
        state = super().__getstate__()
        if "handles_" in state:
            state["handles_"] = _without_networks(state["handles_"])
            state.pop("regressor_", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restores a pickle, its regressors on this model's device.

        Regressors written by :meth:`__getstate__` load on the CPU and get
        the network from :meth:`_network`, as a fit's do, which needs the
        pinned checkpoint on this machine and downloads it if it is missing.
        A ``handles_`` of whole regressors, each with its network, loads as
        it is; regressors pickled together share that network, which
        ``to`` moves in place, so they should share a device. Upstream's
        ``to`` then moves each regressor's network and key/value cache and
        sets its precision again, from ``inference_precision``, which is
        therefore set first.

        Args:
            state: The pickled ``__dict__``.
        """
        super().__setstate__(state)
        if "handles_" not in state:
            return
        if isinstance(self.handles_, bytes):
            import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

            self.handles_ = torch.load(
                io.BytesIO(self.handles_),
                map_location="cpu",
                weights_only=False,
            )
            (network,), *_ = self._network()
            for regressor in self.handles_:
                regressor.models_ = [network]
                # A constant target runs no network, so has no engine.
                engine = getattr(regressor, "executor_", None)
                if engine is not None:
                    engine._set_models([network])  # noqa: SLF001 - upstream's.
            if len(self.handles_) == 1:
                self.regressor_ = self.handles_[0]
        precision = self._inference_precision()
        for regressor in self.handles_:
            regressor.inference_precision = precision
            regressor.to(self.device_)

    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.HistogramDistribution:
        borders, masses = bucket_masses(
            handle.predict(X, output_type="full"), handle
        )
        return distributions.HistogramDistribution(borders, masses)

    def _native_grid(self) -> grid_lib.Grid:
        return _bucket_head.native_grid(map(_bucket_borders, self.handles_))

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


def _shortest_period(tokens: tuple[Any, ...]) -> tuple[Any, ...]:
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
        >>> _shortest_period(("a", "b", "a", "b", "a"))
        ('a', 'b')
        >>> _shortest_period(("a", "a", "b", "a", "a", "b"))
        ('a', 'a', 'b')
    """
    for length in range(1, len(tokens)):
        if all(token == tokens[i % length] for i, token in enumerate(tokens)):
            return tokens[:length]
    return tokens


def _without_networks(regressors: list[Any]) -> bytes:
    """Fitted regressors as ``torch.save`` writes them, without the networks.

    Each network unpickles as ``None``, wherever a regressor refers to it.

    Args:
        regressors: Fitted ``tabpfn.TabPFNRegressor`` instances.

    Returns:
        What ``torch.load`` reads back.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    networks: set[int] = set()
    for regressor in regressors:
        networks.update(id(network) for network in regressor.models_)

    class Pickler(pickle.Pickler):
        """Pickles each of ``networks`` as ``None``."""

        def reducer_override(self, obj: Any) -> Any:
            """Writes ``obj`` as ``None`` if it is one of ``networks``.

            Args:
                obj: An object being pickled.

            Returns:
                A reduction to ``None`` for a network, or
                ``NotImplemented``, which pickles ``obj`` as usual.
            """
            if id(obj) in networks:
                return type(None), ()
            return NotImplemented

    buffer = io.BytesIO()
    torch.save(
        regressors,
        buffer,
        # torch.save reads only these two of a pickle module.
        pickle_module=types.SimpleNamespace(__name__="pickle", Pickler=Pickler),
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )
    return buffer.getvalue()


_IDENTITY = (-0.0, 1.0)
"""The ``(mean, std)`` for borders already in the target's units: adding
-0.0, unlike 0.0, keeps every float exactly, a border of -0.0 included.
"""


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
    if getattr(regressor, "is_constant_target_", False):
        return _bucket_head.bucket_edges(borders, *_IDENTITY)
    return _bucket_head.bucket_edges(
        borders, float(regressor.y_train_mean_), float(regressor.y_train_std_)
    )


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
    output: Mapping[str, Any], regressor: Any = None
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Returns the bucket borders and masses of a TabPFN full prediction.

    ``output["logits"]`` are log-probabilities over the buckets of
    ``output["criterion"]``, whose ``borders`` are in the raw target's units
    -- the target, here. A softmax turns the first into per-bucket probability
    mass, which is what every upstream reduction (the mean, the quantiles)
    integrates over too, so this is the model's own density and not a reading
    of it.

    Upstream keeps those borders in float32, which cannot resolve a narrow
    spread about a large offset (for targets ``1e4 + 0.01 * noise`` most
    buckets come out zero-width, holding most of the mass). Given the
    regressor, the borders are the same buckets rebuilt in float64, as
    :class:`TabPFNBarDistribution` uses them; without it, upstream's are
    returned, with a warning when some have collapsed.

    Args:
        output: What ``TabPFNRegressor.predict(..., output_type="full")``
            returns; only its ``"criterion"`` and ``"logits"`` are read.
        regressor: The fitted ``tabpfn.TabPFNRegressor`` that returned
            ``output``, to rebuild the borders from; ``None`` for
            upstream's float32 borders.

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
    if regressor is not None:
        borders = _bucket_borders(regressor)
    else:
        borders = _bucket_head.bucket_edges(
            np.asarray(
                criterion.borders.detach().cpu().numpy(), dtype=np.float64
            ),
            *_IDENTITY,
        )
        if borders.size > 1 and np.any(np.diff(borders) == 0):
            _warnings.warn(
                "some of TabPFN's float32 bucket borders coincide, so those "
                "buckets have zero width; pass the fitted regressor, "
                "bucket_masses(output, regressor), for float64 borders",
                RuntimeWarning,
            )
    masses = torch.softmax(logits.detach().double(), dim=-1).cpu().numpy()
    if masses.ndim != 2 or masses.shape[1] != borders.size - 1:
        raise RuntimeError(
            f"TabPFN returned logits of shape {tuple(masses.shape)} for a bar "
            f"distribution with {borders.size - 1} buckets"
        )
    return borders, masses
