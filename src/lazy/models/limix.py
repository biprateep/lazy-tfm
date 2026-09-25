# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Photo-z distributions from LimiX-2's bucket probabilities.

LimiX-2 (Stable AI) is an in-context tabular foundation model whose regressor
is, underneath, a classifier over 5,000 buckets of the standardised target:
buckets fixed in the checkpoint, stretched and shifted by the context
redshifts' mean and standard deviation. Upstream's ``predict`` averages its
ensemble members' bucket probabilities and returns their mean; the density is
there and is thrown away one line before the return.
:class:`LimiXBarDistribution` keeps it, as a
:class:`~lazy.distributions.HistogramDistribution` over those buckets, and
its native grid is the buckets themselves.

LimiX is not on PyPI. Its code is loaded from a checkout or a ``pip install``
of the repository (:mod:`lazy.models._limix_source`), and three parts of its
inference path are rebuilt here so that a query's answer depends only on the
context and never on the other queries in its chunk:

* each member's feature preprocessing is fitted on the context alone
  (:mod:`lazy.models._limix_preprocess`), with upstream's own classes and
  seeds;
* the feature positional embedding is drawn from a dedicated generator;
* the key/value cache (``kv_cache``, default on) is a port: the context goes
  through the network once, at fit, and each chunk of queries attends to it
  (:mod:`lazy.models._limix_stream`). Exact: equal to upstream's forward pass
  to float rounding.

The uniform features: ``transforms`` map onto LimiX's own rebalancing step
(``quantile_rtdl`` is scaffolded), ``feature_shuffle`` onto its column
shuffler, and ``bag_size`` is scaffolded as the paper's bagged LimiX-2 was --
one member per bag, each standardising the target on its own bag, members
mixed as densities. LimiX-2 was pretrained on contexts of up to about 20,000
rows, and a :class:`~lazy.models.ContextSizeWarning` says so when a larger
one arrives unbagged.

Built with StableAI LimiX. The code and weights are released under the Stable
AI Technology Co., Ltd. License 1.0 (Apache-2.0 with attribution terms);
see :data:`lazy.CHECKPOINTS`.
"""

from __future__ import annotations

import types
from typing import Any
import warnings

import numpy as np
import pandas as pd

from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _ensemble
from lazy.models import _limix_preprocess
from lazy.models import _limix_source
from lazy.models import _limix_stream
from lazy.models import _members
from lazy.models import _progress

__all__ = ["LimiXBarDistribution"]

# The largest share of free GPU memory the key/value caches may take.
_CACHE_MEMORY_FRACTION = 0.6


class LimiXBarDistribution(_ensemble.ContextEnsembleEstimator):
    """Redshift distributions from LimiX-2's bucket probabilities.

    Args:
        version: Which pinned LimiX checkpoint to load; see
            :func:`lazy.list_versions`. Recorded in ``provenance_``.
        n_estimators: Ensemble members; upstream's recipe has 8.
        transforms: Per-member feature transforms: ``"auto"`` (upstream's
            recipe: four quantile members with the original columns and SVD
            components, four power members), a recipe name, a transform name
            or a sequence of them; see :mod:`lazy.models._transforms`.
        feature_shuffle: Whether members see the columns in different orders
            (upstream's column shuffler).
        bag_size: Context rows per member: an int count, a float fraction in
            (0, 1], or None for all of them. Above about 20,000 context rows
            LimiX-2 needs it: pass ``bag_size=20_000`` with enough members to
            cover the context.
        kv_cache: Run the context through the network once, at fit, and let
            each chunk of queries attend to the result. Exact; costs about
            2 GB of GPU memory per member at 20,000 context rows. Falls back
            to the uncached path, with a warning, when that does not fit.
        z_grid: Default output grid: a :class:`lazy.grid.RedshiftGrid`, an
            array of bin centres, ``"native"``, or None for the native grid
            (the 5,000 buckets, in full).
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"cpu"``.
        random_state: Seed for the ensemble (upstream's default is 0).
        chunk_size: Query rows predicted at a time, to bound peak memory;
            ``0`` does them in one pass. A query's answer does not depend on
            the others in its chunk, up to float rounding.
        progress: A progress bar over the query galaxies: ``"auto"`` shows it
            on a terminal or in a notebook, ``True`` always, ``False`` never.
        verbose: Print log messages to stdout.
        softmax_temperature: Temperature on the bucket logits; 0.9 is
            upstream's default.
        mixed_precision: Run under autocast on CUDA, as upstream does.

    Attributes:
        grid_: The resolved default output grid.
        native_grid_: The native grid: the buckets, mapped to redshift with
            the whole context's mean and standard deviation.
        checkpoint_: The pinned checkpoint file, a :class:`pathlib.Path`.
        provenance_: Which weights, code and ensemble answered, as a dict.
        handles_: Every fitted member group.
        borders_: The buckets' borders in standardised units, shape
            (``n_buckets_`` + 1,).
        n_buckets_: How many buckets the network predicts.
        n_context_: Context rows ``fit`` was given.

    Examples:
        >>> est = LimiXBarDistribution(bag_size=20_000, n_estimators=32)
        >>> est.recommended_max_context
        20000
    """

    backend = "limix"
    display_name = "LimiX"
    extra = "limix"
    native_output = "histogram"
    native_transforms = _limix_preprocess.NATIVE_TRANSFORMS
    auto_tokens = _limix_preprocess.AUTO_TOKENS
    supports_native_bagging = False
    member_combination = "mixture"
    kv_cache_modes = (True, False)
    kv_cache_rtol = 1e-4  # Float rounding: the cached rows batch differently.
    recommended_max_context = 20_000
    exact_chunking = False  # Exact up to float rounding, not bit for bit.

    # The whole context's target mean and standard deviation, for the grid.
    _scale: tuple[float, float]

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v2",
        n_estimators: int = 8,
        transforms: str | tuple[str, ...] = "auto",
        feature_shuffle: bool = True,
        bag_size: int | float | None = None,
        kv_cache: bool = True,
        z_grid: grid_lib.GridLike = None,
        device: str = "auto",
        random_state: int = 0,
        chunk_size: int = 8_192,
        progress: _progress.Progress = "auto",
        verbose: bool = False,
        softmax_temperature: float = 0.9,
        mixed_precision: bool = True,
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
        self.progress = progress
        self.verbose = verbose
        self.softmax_temperature = softmax_temperature
        self.mixed_precision = mixed_precision

    # -- the per-backend interface -------------------------------------------

    def _import_backend(self) -> types.ModuleType:
        try:
            import torch  # noqa: PLC0415, F401 - an optional, heavy extra.
        except ImportError as error:
            raise ImportError(
                "LimiXBarDistribution needs the limix extra: "
                "pip install 'lazy-tfm[limix]'"
            ) from error
        return _limix_source.load().loading

    def _check_backend_params(self) -> None:
        temperature = self.softmax_temperature
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or temperature <= 0
        ):
            raise ValueError(
                f"softmax_temperature must be a positive number: "
                f"{self.softmax_temperature=}"
            )
        if not isinstance(self.mixed_precision, bool):
            raise ValueError(
                f"mixed_precision must be a bool: {self.mixed_precision=}"
            )

    def _load_checkpoint(self) -> None:
        super()._load_checkpoint()
        source = _limix_source.load().source
        self.provenance_ = {
            **self.provenance_,
            "source_commit": source.commit,
            "source_origin": source.origin,
            "attribution": "Built with StableAI LimiX",
        }
        network = self._network()
        self.borders_ = np.asarray(
            network._reg_borders.detach().cpu().numpy(),  # noqa: SLF001 - no public accessor upstream.
            dtype=np.float64,
        )
        self.n_buckets_ = int(self.borders_.size - 1)

    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        self._scale = _standardisation(y)
        self._cache_bytes = 0
        super()._fit(X, y)

    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> Any:
        mean, std = _standardisation(y)
        tokens = group.native_transforms or tuple(
            self.auto_tokens[i % len(self.auto_tokens)]
            for i in range(group.n_members)
        )
        seeds = _limix_preprocess.member_seeds(group.seed, group.n_members)
        members = []
        for token, member_seed in zip(tokens, seeds, strict=True):
            pipeline = _limix_preprocess.MemberPipeline(
                token, member_seed, shuffle=group.feature_shuffle
            ).fit(X)
            member = _limix_stream.Member(
                pipeline.transform(X), (y - mean) / std, seed=group.seed
            )
            members.append(
                {"pipeline": pipeline, "member": member, "cache": None}
            )
        handle = {"members": members, "mean": mean, "std": std}
        if self.kv_cache_:
            self._prefill(handle)
        return handle

    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.HistogramDistribution:
        masses = np.zeros((len(X), self.n_buckets_))
        for entry in handle["members"]:
            masses += self._member_probabilities(entry, X)
        masses /= len(handle["members"])
        edges = self.borders_ * handle["std"] + handle["mean"]
        return distributions.HistogramDistribution(edges, masses)

    def _native_grid(self) -> grid_lib.RedshiftGrid:
        mean, std = self._scale
        return grid_lib.RedshiftGrid.from_edges(
            np.unique(self.borders_ * std + mean), normalization="histogram"
        )

    def _progress_postfix(
        self, dist: distributions.Distribution
    ) -> dict[str, Any]:
        del dist  # Unused: the bucket count is fixed at fit.
        return {"buckets": self.n_buckets_}

    # -- the network ---------------------------------------------------------

    def _network(self) -> Any:
        """The loaded network, cached on the instance per version and device.

        Dropped on pickling: an unpickled estimator reloads it from the local
        Hugging Face cache on next use.
        """
        key = (self.version, self.device_)
        cached_key, network = getattr(self, "_network_cache", (None, None))
        if network is None or cached_key != key:
            self._log(f"loading LimiX network from {self.checkpoint_}")
            network = _limix_stream.load_network(self.checkpoint_, self.device_)
            self._network_cache = (key, network)
        return network

    def _prefill(self, handle: dict[str, Any]) -> None:
        """Caches each member's context, or falls back when it cannot."""
        network = self._network()
        reason = None
        if not _limix_stream.cache_supported(network):
            reason = "this checkpoint's layer layout is not supported"
        for entry in handle["members"]:
            if reason is not None:
                break
            needed = self._cache_bytes + _limix_stream.cache_bytes(
                network,
                len(entry["member"].y),
                entry["member"].x.shape[1],
                _cache_dtype(network, mixed_precision=self.mixed_precision),
            )
            free = _free_device_memory(network)
            if free is not None and needed > _CACHE_MEMORY_FRACTION * free:
                reason = (
                    f"the caches need {needed / 1e9:.1f} GB and the device "
                    f"has {free / 1e9:.1f} GB free"
                )
                break
            entry["cache"] = _limix_stream.prefill(
                network,
                entry["member"],
                mixed_precision=self.mixed_precision,
            )
            self._cache_bytes = needed
        if reason is not None:
            self._drop_caches(handle)
            warnings.warn(
                f"LimiX is predicting without its key/value cache ({reason}),"
                " so every chunk of queries re-runs the context. The answers"
                " are the same; pass kv_cache=False to silence this.",
                _ensemble.PerformanceWarning,
                stacklevel=6,
            )

    def _drop_caches(self, current: dict[str, Any]) -> None:
        """Turns the cache off and frees every member's, ``current``'s too."""
        self.kv_cache_ = False
        for handle in [*getattr(self, "handles_", []), current]:
            for entry in handle["members"]:
                entry["cache"] = None

    def _member_probabilities(
        self, entry: dict[str, Any], X: _typing.FloatArray
    ) -> _typing.FloatArray:
        """One member's bucket probabilities for some query rows."""
        network = self._network()
        member = entry["member"]
        queries = entry["pipeline"].transform(X)
        if self.kv_cache_:
            if entry["cache"] is None:  # dropped on pickling
                entry["cache"] = _limix_stream.prefill(
                    network, member, mixed_precision=self.mixed_precision
                )
            logits = _limix_stream.decode(
                network,
                member,
                entry["cache"],
                queries,
                mixed_precision=self.mixed_precision,
            )
        else:
            logits = _limix_stream.forward(
                network, member, queries, mixed_precision=self.mixed_precision
            )
        scaled = logits.astype(np.float64) / self.softmax_temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def __getstate__(self) -> dict[str, Any]:
        """Pickles without the network or the caches, which reload."""
        state = self.__dict__.copy()
        state.pop("_network_cache", None)
        if "handles_" in state:
            state["handles_"] = [
                {
                    **handle,
                    "members": [
                        {**entry, "cache": None} for entry in handle["members"]
                    ],
                }
                for handle in state["handles_"]
            ]
            state.pop("regressor_", None)
        return state


def _standardisation(y: _typing.FloatArray) -> tuple[float, float]:
    """Upstream's target scaling, as a tuple (mean, sample std or 1)."""
    y = np.asarray(y, dtype=np.float64)
    std = float(y.std(ddof=1)) if y.size > 1 else 0.0
    return float(y.mean()), std if std > 0 else 1.0


def _cache_dtype(network: Any, *, mixed_precision: bool) -> Any:
    """The dtype the cache is held in: autocast's on CUDA, else float32."""
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    device = next(network.parameters()).device
    if mixed_precision and device.type == "cuda":
        return torch.get_autocast_dtype("cuda")
    return torch.float32


def _free_device_memory(network: Any) -> int | None:
    """Free bytes on the network's GPU, or None on CPU."""
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    device = next(network.parameters()).device
    if device.type != "cuda":
        return None
    free, _ = torch.cuda.mem_get_info(device)
    return int(free)
