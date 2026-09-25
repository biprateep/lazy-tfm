# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Photo-z distributions from TabICLv2's quantile regression head.

TabICLv2 (Qu et al. 2026) is an in-context tabular foundation model whose
regressor answers with a *distribution*: 999 quantiles of the predictive CDF
per query row, not a single number. :class:`TabICLQuantile` returns them as a
:class:`~lazy.distributions.QuantileDistribution`, whose densities on any
grid come from evaluating the quantile CDF at the bin edges and differencing
(:meth:`lazy.grid.RedshiftGrid.from_quantiles`): every bin gets exactly the
mass the quantiles place inside it.

The uniform features map onto TabICL's own machinery: ``kv_cache`` onto its
key/value cache, ``feature_shuffle`` onto its Latin-square column shuffles,
and the ``transforms`` it has (``none``, ``power``, ``quantile``,
``quantile_rtdl``, ``robust``) onto its ``norm_methods``. TabICL cannot
subsample rows, so ``bag_size`` is scaffolded: one single-member regressor
per bag, their quantile functions averaged, as TabICL averages its own
members.

Compared with :class:`lazy.models.tabfm.TabFMHistogram`, this backbone is far
smaller (a ~100 MB checkpoint rather than ~6.6 GB) and much faster, at the cost
of a density whose resolution is set by the spacing of the quantiles rather
than by the data.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np

from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _ensemble
from lazy.models import _members
from lazy.models import _progress

__all__ = ["NATIVE_QUANTILE_BINS", "TabICLQuantile", "quantile_levels"]

#: Bins of the native grid: as fine as the 999 quantiles resolve.
NATIVE_QUANTILE_BINS = 1000


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
    """Redshift distributions from the quantiles TabICLv2's regressor predicts.

    Args:
        version: Which pinned TabICL checkpoint to load; see
            :func:`lazy.list_versions`. Recorded in ``provenance_``.
        n_estimators: Ensemble members. Costs scale linearly; 8 is the value
            the benchmarks use.
        transforms: Per-member feature transforms: ``"auto"`` (TabICL's own
            ``none``/``power`` recipe), a recipe name, a transform name or a
            sequence of them; see :mod:`lazy.models._transforms`.
        feature_shuffle: Whether members see the columns in different orders
            (TabICL's Latin-square shuffles).
        bag_size: Context rows per member: an int count, a float fraction in
            (0, 1], or None for all of them. Scaffolded: one regressor per
            bag.
        kv_cache: Cache the context's keys and values at fit, so each chunk
            of queries skips the context forward pass: ``True`` (TabICL's
            ``"kv"`` cache), ``"repr"`` (cached row representations, far
            smaller, re-running the in-context layers) or ``False``. Exact
            either way, up to floating-point rounding.
        z_grid: Default output grid: a :class:`lazy.grid.RedshiftGrid`, an
            array of bin centres, ``"native"``, or None for the native grid
            (1,000 bins spanning the training redshifts).
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"cpu"``.
        random_state: Seed for the ensemble.
        chunk_size: Query rows predicted at a time, to bound peak memory
            (999 quantiles per row is about 8 kB); ``0`` does them in one
            pass. Exact: TabICL builds its keys and values from the context
            rows alone, so a row's answer never depends on the other rows in
            its chunk.
        progress: A progress bar over the query galaxies: ``"auto"`` shows it
            on a terminal or in a notebook, ``True`` always, ``False`` never.
        verbose: Print log messages to stdout.

    Attributes:
        grid_: The resolved default output grid.
        native_grid_: The native grid, 1,000 histogram-normalised bins over
            the training redshifts padded by 2% (not below zero).
        checkpoint_: The pinned checkpoint file, a :class:`pathlib.Path`.
        provenance_: Which weights, code and ensemble answered, as a dict.
        regressor_: The fitted ``tabicl.TabICLRegressor``, when one serves
            the whole ensemble.
        handles_: Every fitted regressor, one per member group.
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
    auto_tokens = ("none", "power")
    supports_native_bagging = False
    member_combination = "quantile_average"
    kv_cache_modes = (True, False, "repr")
    kv_cache_rtol = 1e-3  # The cache is stored in fp16 under autocast.

    # The training redshifts' range, recorded by _fit_group for the grid.
    _support: tuple[float, float]

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
        random_state: int = 42,
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
        self.chunk_size = chunk_size
        self.progress = progress
        self.verbose = verbose

    def _import_backend(self) -> types.ModuleType:
        try:
            import tabicl  # noqa: PLC0415 - an optional, heavy extra.
        except ImportError as error:
            raise ImportError(
                "TabICLQuantile needs the tabicl backend: "
                "pip install 'lazy-tfm[tabicl]'"
            ) from error
        return tabicl

    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> Any:
        # The pinned checkpoint is handed over by path, never left to
        # TabICL's own default, which upstream is free to bump: this way
        # `CHECKPOINTS` is the authority on which weights answer.
        options: dict[str, Any] = {}
        if group.native_transforms is not None:
            # Upstream pairs members with norm methods round robin, as the
            # planner assigned them, so the distinct tokens in order suffice.
            options["norm_methods"] = list(
                dict.fromkeys(group.native_transforms)
            )
        if not group.feature_shuffle:
            options["feat_shuffle_method"] = "none"
        tabicl = self._import_backend()
        regressor = tabicl.TabICLRegressor(
            n_estimators=group.n_members,
            device=self.device_,
            kv_cache=self.kv_cache,
            random_state=group.seed,
            model_path=str(self.checkpoint_),
            verbose=False,
            **options,
        )
        regressor.fit(X.astype(np.float32), y.astype(np.float32))
        low, high = float(y.min()), float(y.max())
        if group.index > 0:
            low, high = min(low, self._support[0]), max(high, self._support[1])
        self._support = (low, high)
        return regressor

    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.QuantileDistribution:
        quantiles = np.asarray(
            handle.predict(X.astype(np.float32), output_type="raw_quantiles"),
            dtype=np.float64,
        )
        self.n_quantiles_ = int(quantiles.shape[1])
        return distributions.QuantileDistribution(
            quantile_levels(self.n_quantiles_), quantiles
        )

    def _native_grid(self) -> grid_lib.RedshiftGrid:
        low, high = self._support
        pad = 0.02 * (high - low) if high > low else 0.01
        start = max(low - pad, 0.0) if low >= 0 else low - pad
        return grid_lib.RedshiftGrid.linear(
            start, high + pad, NATIVE_QUANTILE_BINS, normalization="histogram"
        )

    def _progress_postfix(
        self, dist: distributions.Distribution
    ) -> dict[str, Any]:
        del dist  # Unused: the count is recorded when predicting.
        return {"quantiles": getattr(self, "n_quantiles_", None)}
