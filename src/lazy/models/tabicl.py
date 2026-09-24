# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Photo-z densities from TabICLv2's quantile regression head.

TabICLv2 (Qu et al. 2026) is an in-context tabular foundation model whose
regressor answers with a *distribution*: 999 quantiles of the predictive CDF
per query row, not a single number. That is already a conditional density
estimate in disguise, so :class:`TabICLQuantile` needs no hierarchy and no
post-processing -- it evaluates the quantile CDF at the output grid's bin
edges and differences it, which hands every bin exactly the mass the quantiles
place inside it.

Differentiating at the bin centres instead (``np.gradient``) would smear any
feature narrower than a bin into its neighbours and would not conserve mass,
so :meth:`lazy.grid.RedshiftGrid.from_quantiles` does the edge evaluation.

Compared with :class:`lazy.models.tabfm.TabFMHistogram`, this backbone is far
smaller (a ~100 MB checkpoint rather than ~6.6 GB) and much faster, at the cost
of a density whose resolution is set by the spacing of the quantiles rather
than by the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lazy import _typing
from lazy import base
from lazy import grid as grid_lib
from lazy.models import _device
from lazy.models import _hub
from lazy.models import _progress

__all__ = ["TabICLQuantile", "quantile_levels"]


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


class TabICLQuantile(base.BasePhotoZEstimator):
    """Conditional density from the quantiles TabICLv2's regressor predicts.

    Args:
        version: Which pinned TabICL checkpoint to load; see
            :func:`lazy.list_versions`. Recorded in ``provenance_``.
        n_estimators: TabICL ensemble members. Costs scale linearly; 8 is the
            value the benchmarks use.
        z_grid: Output grid: a :class:`lazy.grid.RedshiftGrid`, an array of
            bin centres, or ``None`` for :data:`lazy.grid.DC1_GRID`.
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"cpu"``.
        random_state: Seed for TabICL's ensemble construction.
        chunk_size: Query rows predicted at a time, to bound peak memory: 999
            float quantiles per row is roughly 8 kB, so a survey-sized query
            set in one pass is gigabytes of intermediate before any of it
            becomes a density. ``0`` does it in one pass. Chunking is
            numerically exact -- TabICL's in-context stage builds its keys and
            values from the context rows only, so a query row's prediction
            never depends on which other query rows share its chunk -- and is
            verified bit-identical in the test suite, so the default is
            bounded rather than fast-and-hopeful.
        progress: A progress bar over the query galaxies: ``"auto"`` (the
            default) shows it on a terminal or in a notebook and not when
            output goes to a file, ``True`` always, ``False`` never. It
            advances one chunk at a time and shows the context size and how
            many quantiles each galaxy is given.
        verbose: Print log messages to stdout.

    Attributes:
        grid_: The resolved output grid, a :class:`lazy.grid.RedshiftGrid`.
        checkpoint_: The pinned checkpoint file the weights were loaded from,
            a :class:`pathlib.Path`.
        provenance_: Which weights and which code answered, as a dict:
            backend, version, repository, revision, package versions and
            device. See :meth:`lazy.models._hub.Checkpoint.provenance`.
        regressor_: The fitted ``tabicl.TabICLRegressor``.
        n_quantiles_: How many quantiles the backbone actually returned.
        n_context_: Context rows ``fit`` was given.

    Examples:
        >>> est = TabICLQuantile(n_estimators=8, chunk_size=50_000)
        >>> est.chunk_size
        50000
    """

    backend = "tabicl"

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v2",
        n_estimators: int = 8,
        z_grid: grid_lib.GridLike = None,
        device: str = "auto",
        random_state: int = 42,
        chunk_size: int = 16_384,
        progress: _progress.Progress = "auto",
        verbose: bool = False,
    ):
        self.version = version
        self.n_estimators = n_estimators
        self.z_grid = z_grid
        self.device = device
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.progress = progress
        self.verbose = verbose

    # -- estimator protocol -------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        try:
            import tabicl  # noqa: PLC0415 - an optional, heavy extra.
        except ImportError as error:
            raise ImportError(
                "TabICLQuantile needs the tabicl backend: "
                "pip install 'lazy-photoz[tabicl]'"
            ) from error

        if self.chunk_size < 0:
            raise ValueError(
                "chunk_size must be non-negative (0 means one pass)"
            )
        _progress.check_progress(self.progress)
        self.device_ = _device.resolve_device(self.device)

        # Fetch the weights ourselves and hand over the path, rather than
        # letting TabICL pick its own default. Its default is a separate
        # constant that upstream is free to bump, and if it did, the
        # checkpoint `lazy.download_checkpoint("tabicl")` pre-fetches would
        # stop being the one actually loaded -- silently breaking offline runs
        # and silently un-pinning the weights. This way there is one
        # checkpoint and one download path, and `CHECKPOINTS` is the authority
        # on both.
        spec = _hub.get_checkpoint("tabicl", self.version)
        self.checkpoint_ = spec.download()
        self.provenance_ = spec.provenance(device=self.device_)
        self._log(
            f"fitting TabICL on {len(X)} context rows ({self.device_}), "
            f"{self.checkpoint_.name}"
        )
        regressor = tabicl.TabICLRegressor(
            n_estimators=self.n_estimators,
            device=self.device_,
            kv_cache=False,
            random_state=self.random_state,
            model_path=str(self.checkpoint_),
            verbose=False,
        )
        regressor.fit(
            X.to_numpy(dtype=np.float32), np.asarray(y, dtype=np.float32)
        )
        self.regressor_ = regressor
        self.n_context_ = len(X)

    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.RedshiftGrid
    ) -> _typing.FloatArray:
        size = self.chunk_size if self.chunk_size > 0 else len(X)
        blocks: list[_typing.FloatArray] = []
        with _progress.bar(
            self.progress,
            total=len(X),
            desc=f"TabICL {self.version}",
            unit="gal",
        ) as progress:
            progress.set_postfix(context=self.n_context_)
            for start in range(0, len(X), size):
                stop = min(start + size, len(X))
                self._log(f"rows {start}:{stop} of {len(X)}")
                quantiles = self.regressor_.predict(
                    X.iloc[start:stop].to_numpy(dtype=np.float32),
                    output_type="raw_quantiles",
                )
                quantiles = np.asarray(quantiles, dtype=np.float64)
                self.n_quantiles_ = int(quantiles.shape[1])
                blocks.append(
                    grid.from_quantiles(
                        quantiles, quantile_levels(self.n_quantiles_)
                    )
                )
                del quantiles
                progress.set_postfix(
                    context=self.n_context_,
                    quantiles=self.n_quantiles_,
                    refresh=False,
                )
                progress.update(stop - start)
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[TabICLQuantile] {message}", flush=True)
