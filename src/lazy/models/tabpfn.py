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

The uniform features all map onto TabPFN's own machinery: ``kv_cache`` onto
its fit-time key/value cache (at full precision, so exact), the
``transforms`` it has onto its ``PREPROCESS_TRANSFORMS``, ``feature_shuffle``
onto its ``FEATURE_SHIFT_METHOD``, and ``bag_size`` onto its per-member row
subsampling, ``SUBSAMPLE_SAMPLES``, handed the package's own bags.
"""

from __future__ import annotations

from collections.abc import Mapping
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

__all__ = [
    "TabPFNBarDistribution",
    "bucket_masses",
    "path_for_tabpfn",
]


def _native_transforms() -> dict[str, tuple[str, bool]]:
    """Uniform transform names TabPFN implements, as (name, append_original)."""
    names = {
        "none": "none",
        "power": "power",
        "quantile": "quantile_norm",
        "quantile_uniform": "quantile_uni",
        "robust": "robust",
    }
    native: dict[str, tuple[str, bool]] = {}
    for uniform, upstream in names.items():
        native[uniform] = (upstream, False)
        native[f"{uniform}+original"] = (upstream, True)
    return native


#: Versions whose architecture has no quantised key/value cache; upstream
#: would quietly fall back to full precision.
_UNQUANTISED_VERSIONS = ("v2", "v2.5", "v2.6")


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
            preprocessed view of the data; costs scale linearly. ``"auto"``
            defers to the count the checkpoint names.
        transforms: Per-member feature transforms: ``"auto"`` (the
            checkpoint's own tuned recipe), a recipe name, a transform name or
            a sequence of them; see :mod:`lazy.models._transforms`. TabPFN
            implements all but ``quantile_rtdl`` itself. Its per-member target
            transforms always stay the checkpoint's.
        feature_shuffle: Whether members see the columns in different orders
            (TabPFN's own feature shuffling).
        bag_size: Context rows per member: an int is a row count (1 means
            one row), a float a fraction in (0, 1] (1.0 means all rows), and
            None all of them. Native: TabPFN's own per-member
            row subsampling, handed the package's bags.
        kv_cache: Cache the context's keys and values at fit, so each chunk
            of queries skips the context forward pass: ``True`` (exact, full
            precision), ``"int8"`` or ``"fp8"`` (quantised: smaller, not
            exact; v3 and later only), or ``False``. Worth its memory
            whenever the query set is much larger than the context.
        z_grid: Default output grid: a :class:`lazy.grid.Grid`, an
            array of bin centres, ``"native"``, or None for the native grid
            (the bar distribution's own buckets, in full).
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"mps"``, ``"cpu"``, or a ``torch.device``.
        random_state: Seed for the ensemble. None draws a fresh seed at fit,
            recorded as ``random_state_`` and in ``provenance_``.
        softmax_temperature: Temperature on the bucket logits, which sets how
            sharp the densities are. ``"auto"`` takes the checkpoint's own
            value, which is the one it was evaluated with; lower sharpens,
            higher broadens.
        ignore_pretraining_limits: Pass ``True`` to run a context larger than
            the row count the checkpoint declares it was pretrained for, which
            otherwise raises. Predictions beyond that limit are extrapolation,
            so this is opt-in -- though TabPFN-3 declares a million rows.
        chunk_size: Query rows predicted at a time, to bound peak memory;
            ``0`` does them in one pass. Exact: TabPFN's attention builds its
            keys and values from the context rows alone and its preprocessors
            are fitted on the context, so a row's answer never depends on the
            other rows in its chunk.
        progress: A progress bar over the query rows: ``"auto"`` shows it
            on a terminal or in a notebook, ``True`` always, ``False`` never.
        verbose: Print log messages to stdout.

    Attributes:
        grid_: The resolved default output grid.
        native_grid_: The bar distribution's buckets in the target's units, a
            histogram-normalised grid, set at fit.
        checkpoint_: The pinned checkpoint file, a :class:`pathlib.Path`.
        provenance_: Which weights, code and ensemble answered, as a dict.
        regressor_: The fitted ``tabpfn.TabPFNRegressor``, when one serves
            the whole ensemble.
        handles_: Every fitted regressor, one per member group.
        borders_: The bucket borders in the target's units, shape
            (``n_buckets_`` + 1,);
            fixed at fit by the context targets' mean and spread.
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
    supports_native_bagging = True
    kv_cache_modes = (True, False, "int8", "fp8")
    kv_cache_rtol = 1e-5
    accepts_auto_estimators = True

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v3.5",
        n_estimators: int | str = 8,
        transforms: str | tuple[str, ...] = "auto",
        feature_shuffle: bool = True,
        bag_size: int | float | None = None,
        kv_cache: bool | str = True,
        z_grid: grid_lib.GridLike = None,
        device: str = "auto",
        random_state: int | None = 42,
        softmax_temperature: float | str = "auto",
        ignore_pretraining_limits: bool = False,
        chunk_size: int = 16_384,
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
        self.softmax_temperature = softmax_temperature
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.chunk_size = chunk_size
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
        tabpfn = self._import_backend()
        regressor = tabpfn.TabPFNRegressor(
            n_estimators=(
                self.n_estimators
                if self.n_estimators == "auto"
                else group.n_members
            ),
            model_path=path_for_tabpfn(self.checkpoint_),
            device=self.device_,
            random_state=group.seed,
            softmax_temperature=self.softmax_temperature,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            show_progress_bar=False,
            inference_config=self._inference_config(group) or None,
            **_cache_options(self.kv_cache),
        )
        regressor.fit(X, y)
        self.borders_ = _bucket_borders(regressor)
        self.n_buckets_ = int(self.borders_.size - 1)
        return regressor

    def _inference_config(self, group: _members.MemberGroup) -> dict[str, Any]:
        """The upstream overrides a group needs; empty for the defaults."""
        overrides: dict[str, Any] = {}
        if group.native_transforms is not None:
            import tabpfn.preprocessing.configs  # noqa: PLC0415 - optional extra.

            configs = tabpfn.preprocessing.configs

            # Upstream gives each listed config an equal share of the
            # members, repeats included, so one cycle of the plan keeps its
            # weights. The settings are those the paper's recipe runs used.
            overrides["PREPROCESS_TRANSFORMS"] = [
                configs.PreprocessorConfig(
                    name,
                    append_original=original,
                    categorical_name="ordinal_shuffled",
                    max_features_per_estimator=768,
                )
                for name, original in _cycle(group.native_transforms)
            ]
        if not group.feature_shuffle:
            overrides["FEATURE_SHIFT_METHOD"] = None
        if group.member_rows is not None:
            overrides["SUBSAMPLE_SAMPLES"] = list(group.member_rows)
        return overrides

    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.HistogramDistribution:
        _, masses = bucket_masses(handle.predict(X, output_type="full"))
        return distributions.HistogramDistribution(
            _bucket_borders(handle), masses
        )

    def _native_grid(self) -> grid_lib.Grid:
        edges = np.unique(self.borders_)
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


def _cycle(tokens: tuple[Any, ...]) -> tuple[Any, ...]:
    """The shortest prefix of ``tokens`` that repeats to give all of them.

    Upstream shares the members out among its preprocessing configs (and,
    within each, its target transforms) in equal blocks, so handing it one
    cycle of the planned transforms, repeats included, keeps their weights:
    exactly when the member count is a multiple of the cycle's length times
    the number of target transforms, and to within one block otherwise.

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
