# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""FlexZBoost: a basis expansion of p(z) with boosted-tree coefficients.

FlexCode (Izbicki & Lee 2017) writes the conditional density as an
orthonormal series, f(z | x) = sum_j beta_j(x) phi_j(z), whose coefficients
beta_j(x) = E[phi_j(z) | x] are conditional means, so that any regression
gives them. FlexZBoost (Schmidt et al. 2020) uses xgboost for the regression
and a cosine or Fourier basis on [0, 2], and post-processes the series in
three steps: it lowers the curve until its positive part integrates to one,
removes isolated bumps of little probability, and sharpens the density by a
power alpha. We take the basis from the ``flexcode`` package, so that the
targets are the package's own, and write the rest again. xgboost fits all the
coefficients in one booster, either with one tree per coefficient, as
``flexcode`` does through scikit-learn, or with trees whose leaves hold all
the coefficients at once (``multi_strategy="multi_output_tree"``), which
gave the lower CDE loss in our pilots on DC1. The post-processing works on
the 4,000 bins of the common grid, vectorised over rows, and agrees with
``flexcode``'s own functions to their tolerance (1e-6). ``FM4PZ_THREADS``
sets the threads of xgboost and of the post-processing.

The fit parameters are the feature set, the basis and its number of terms,
the bump threshold and the booster's settings. Every fold then chooses, with
no further fitting, the number of trees (two checkpoints of one booster,
which stands in for early stopping), the number of terms kept and the
sharpening exponent.

Typical usage example:

  model = METHOD.fit(params, split.train, split.train.redshift, seed=0)
  raw = METHOD.predict(model, split.test)
  masses = METHOD.masses(raw, post)
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent import futures
import dataclasses
import functools
import itertools
import os
from typing import Any

from flexcode import basis_functions
from lazy import datasets
import numpy as np
import numpy.typing as npt
import optuna
import threadpoolctl
import xgboost

from fm4pz import config
from fm4pz import cv
from fm4pz import features

__all__ = [
    "BASES",
    "BUMPS",
    "CHECKPOINTS",
    "FEATURE_SETS",
    "METHOD",
    "SHARPEN",
    "STRATEGIES",
    "TERMS",
    "FlexZBoost",
    "Model",
    "Raw",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

FEATURE_SETS: tuple[str, ...] = ("i_colors", "common12", "mags_errs")
BASES: tuple[str, ...] = ("cosine", "Fourier")
STRATEGIES: tuple[str, ...] = ("multi_output_tree", "one_output_per_tree")

#: Bump thresholds (probability) a trial may propose; None removes none.
BUMPS: tuple[float | None, ...] = (None, 0.001, 0.003, 0.01, 0.03)

#: Fractions of the booster's rounds at which predictions are kept.
CHECKPOINTS: tuple[float, ...] = (0.5, 1.0)

#: Fractions of the fitted basis functions a fold may keep: the smallest won
#: in 9 of 10 pilot trials at 1,000 and 10,000 galaxies, the largest at the
#: full size.
TERMS: tuple[float, ...] = (0.25, 0.4, 0.6, 0.8, 1.0)

#: Sharpening exponents every fold evaluates. The CDE loss on the common
#: grid changes by 0.2-0.4 per 0.1 in alpha near its optimum (1.05 to 1.4 in
#: the pilots), hence the fine steps there.
SHARPEN: tuple[float, ...] = (
    *(round(0.75 + 0.025 * k, 3) for k in range(31)),
    1.6,
    1.75,
    2.0,
    2.5,
)

# Rows post-processed at a time: blocks small enough to stay in the cache
# make the post-processing a third faster, and they share out the threads.
_ROWS = 256

# The most leaves a tree may grow, a bound on the cost of the deepest trees;
# the best trees of the pilots at the full size grew about 1,200.
_MAX_LEAVES = 2_048

# Histograms xgboost keeps between levels. Its default (65,536) let one fit
# of the deepest trees take 18.7 GB; 512 keeps it near 3 GB and is faster.
_CACHED_NODES = 512

# The basis lives on u = (z - Z_MIN) / _Z_SPAN in [0, 1].
_Z_SPAN = config.Z_MAX - config.Z_MIN


def _threads() -> int:
    """The thread budget of this process, from ``FM4PZ_THREADS``."""
    return int(os.environ.get("FM4PZ_THREADS", "1"))


@functools.cache
def _bin_means(basis: str, n_terms: int) -> FloatArray:
    """The average of each basis function over each bin of the common grid.

    The 4,000 bins have width 1/4,000 in u. A sine or cosine integrates in
    closed form, so the averages are exact; they differ from the values at
    the bin centres by a relative (pi k / 4,000)^2 / 24 at frequency k, which
    is below 2e-4 for the 100 terms we allow.

    Args:
        basis: ``"cosine"`` or ``"Fourier"``, as ``flexcode`` names them.
        n_terms: Basis functions, the constant included.

    Returns:
        The averages, shape (n_terms, 4000), in density per unit u.

    Raises:
        ValueError: If ``basis`` is not one of :data:`BASES`.
    """
    if basis not in BASES:
        raise ValueError(f"unknown basis: {basis=}")
    edges = np.linspace(0.0, 1.0, config.FINE_BINS + 1)
    out = np.empty((n_terms, config.FINE_BINS))
    out[0] = 1.0
    for j in range(1, n_terms):
        if basis == "cosine":
            # sqrt(2) cos(pi j u).
            omega, cosine = np.pi * j, True
        else:
            # sqrt(2) sin(2 pi k u) at odd j and sqrt(2) cos(2 pi k u) at
            # even j, with k = (j + 1) // 2; an even number of terms ends on
            # a sine, as in flexcode.
            omega, cosine = 2.0 * np.pi * ((j + 1) // 2), j % 2 == 0
        if cosine:
            primitive = np.sin(omega * edges) / omega
        else:
            primitive = -np.cos(omega * edges) / omega
        out[j] = np.sqrt(2.0) * np.diff(primitive) * config.FINE_BINS
    out.setflags(write=False)
    return out


@dataclasses.dataclass
class _Support:
    """The positive bins of n densities on the common grid.

    Attributes:
        values: Density per unit u in each positive bin, shape (nnz,).
        rows: The row of each, ascending, shape (nnz,).
        cols: The bin of each, ascending within a row, shape (nnz,).
        n_rows: Rows, n.
    """

    values: FloatArray
    rows: IntArray
    cols: IntArray
    n_rows: int

    def keep(self, mask: npt.NDArray[np.bool_]) -> _Support:
        """The bins where ``mask`` holds."""
        return _Support(
            self.values[mask], self.rows[mask], self.cols[mask], self.n_rows
        )

    def area(self) -> FloatArray:
        """The integral of each row over u in [0, 1], shape (n,)."""
        return (
            np.bincount(self.rows, weights=self.values, minlength=self.n_rows)
            / config.FINE_BINS
        )

    def scatter(self, out: FloatArray, scale: float) -> None:
        """Writes the values times ``scale`` into zeros, shape (n, 4000)."""
        out[self.rows, self.cols] = self.values * scale


def _concatenate(parts: list[_Support], n_rows: int) -> _Support:
    """The bins of several supports, sorted by row (stably)."""
    rows = np.concatenate([part.rows for part in parts])
    order = np.argsort(rows, kind="stable")
    return _Support(
        np.concatenate([part.values for part in parts])[order],
        rows[order],
        np.concatenate([part.cols for part in parts])[order],
        n_rows,
    )


def _uniform(rows: IntArray, n_rows: int) -> _Support:
    """Uniform densities in the given rows."""
    return _Support(
        np.ones(rows.size * config.FINE_BINS),
        np.repeat(rows, config.FINE_BINS),
        np.tile(np.arange(config.FINE_BINS), rows.size),
        n_rows,
    )


def _normalize(support: _Support) -> _Support:
    """FlexCode's normalisation, for every row at once.

    A row whose positive part integrates to less than one is scaled up, a
    row whose positive part integrates to one or more is lowered by the
    level L at which the integral of (f - L)_+ is one and clipped at zero,
    and a row with no positive part becomes uniform. ``flexcode`` finds L by
    bisection to a tolerance of 1e-6 in the integral. We use Newton's method
    instead, which on this convex, piecewise linear function of L climbs to
    the exact level from below in a few steps; since L only grows, a bin
    that falls below it is dropped for good, and each step works on fewer
    bins.

    Args:
        support: The positive bins of the densities, per unit u.

    Returns:
        The positive bins of the normalised densities, every row
        integrating to one.
    """
    n_rows = support.n_rows
    area = support.area()
    active = area >= 1.0
    level = np.zeros(n_rows)
    values, rows = support.values, support.rows
    counts = np.bincount(rows, minlength=n_rows)
    for _ in range(200):
        keep = active[rows] & (values > level[rows])
        if not keep.all():
            counts -= np.bincount(rows[~keep], minlength=n_rows)
            values, rows = values[keep], rows[keep]
        if values.size == 0:
            break
        # The rows are contiguous, so each sums from its first position; the
        # zero appended is the sum of the empty rows at the end.
        starts = np.cumsum(counts) - counts
        sums = np.add.reduceat(np.append(values, 0.0), starts)
        sums[counts == 0] = 0.0
        excess = (sums - counts * level) / config.FINE_BINS - 1.0
        active &= excess > 1e-12
        level[active] += excess[active] * config.FINE_BINS / counts[active]
    scale = np.where((area > 0.0) & (area < 1.0), area, 1.0)
    out = support.keep(support.values > level[support.rows])
    out.values = (out.values - level[out.rows]) / scale[out.rows]
    empty = np.flatnonzero(area <= 0.0)
    if empty.size:
        out = _concatenate([out, _uniform(empty, n_rows)], n_rows)
    return out


def _remove_bumps(support: _Support, threshold: float) -> _Support:
    """FlexCode's bump removal, for every row at once.

    A bump is a run of adjacent bins with positive density. Every bump whose
    probability is below ``threshold`` is set to zero, and the rows are
    normalised again (a row that loses every bump becomes uniform).

    Args:
        support: The positive bins of normalised densities, per unit u.
        threshold: The smallest probability a bump keeps.

    Returns:
        The positive bins of the normalised densities without the bumps.
    """
    start = np.ones(support.values.size, dtype=bool)
    start[1:] = (support.rows[1:] != support.rows[:-1]) | (
        support.cols[1:] != support.cols[:-1] + 1
    )
    run = np.cumsum(start) - 1
    area = np.bincount(run, weights=support.values) / config.FINE_BINS
    return _normalize(support.keep(area[run] >= threshold))


@dataclasses.dataclass
class _Sorted:
    """Normalised densities, their bins sorted by decreasing value per row.

    Attributes:
        log_values: Logarithm of the density per unit u in each positive
            bin, shape (nnz,).
        rows: The row of each, ascending, shape (nnz,).
        cols: The bin of each, shape (nnz,).
        n_rows: Rows, n.
        starts: Position of each row's first bin, shape (n,).
        rank: One plus each bin's position within its row, shape (nnz,).
    """

    log_values: FloatArray
    rows: IntArray
    cols: IntArray
    n_rows: int
    starts: IntArray
    rank: FloatArray


def _by_value(support: _Support) -> _Sorted:
    """The bins of normalised densities, sorted for :func:`_sharpen`.

    Args:
        support: The positive bins of normalised densities, every row
            holding at least one.

    Returns:
        The bins, sorted by decreasing value within each row.
    """
    counts = np.bincount(support.rows, minlength=support.n_rows)
    starts = np.cumsum(counts) - counts
    # One key orders the rows and, within each row, the values from the
    # largest down. It resolves values to 1e-11 of the row's peak and sorts
    # five times faster than np.lexsort; a closer tie cannot move the level.
    peak = np.maximum.reduceat(support.values, starts)
    key = support.rows + 0.5 * (1.0 - support.values / peak[support.rows])
    order = np.argsort(key)
    rows = support.rows[order]
    return _Sorted(
        log_values=np.log(support.values[order]),
        rows=rows,
        cols=support.cols[order],
        n_rows=support.n_rows,
        starts=starts,
        rank=(np.arange(1, rows.size + 1) - starts[rows]).astype(np.float64),
    )


def _sharpen(density: _Sorted, alpha: float) -> _Support:
    """FlexCode's sharpening: the density to the power alpha, normalised.

    A power keeps the order of the bins within a row, so with the bins
    sorted once by decreasing value, the level of :func:`_normalize` follows
    without a search. If s_1 >= s_2 >= ... are the powered values of a row
    and C_k their running sum, the level is L = (C_k - 4000) / k for the
    largest k with s_k > (C_k - 4000) / k, as in the projection onto the
    simplex (Duchi et al. 2008), here on bins of width 1/4,000. A row whose
    powered values integrate to less than one (alpha below one) is scaled
    up instead, as ``flexcode`` does.

    Args:
        density: Normalised densities, sorted by :func:`_by_value`.
        alpha: The exponent; one returns the densities unchanged.

    Returns:
        The positive bins of the sharpened, normalised densities.
    """
    rows, starts = density.rows, density.starts
    values = np.exp(alpha * density.log_values)
    # The running sum within each row, from the running sum over all rows.
    running = np.cumsum(values)
    running -= np.concatenate([[0.0], running])[starts][rows]
    ends = np.append(starts[1:], rows.size) - 1
    area = running[ends] / config.FINE_BINS
    levels = (running - config.FINE_BINS) / density.rank
    above = np.add.reduceat(
        (values > levels).view(np.int8), starts, dtype=np.int64
    )
    level = np.where(area >= 1.0, levels[starts + above - 1], 0.0)
    scale = np.where(area < 1.0, area, 1.0)
    keep = values > level[rows]
    kept = rows[keep]
    return _Support(
        (values[keep] - level[kept]) / scale[kept],
        kept,
        density.cols[keep],
        density.n_rows,
    )


def _checkpoints(n_estimators: int) -> tuple[int, ...]:
    """The boosting rounds at which predictions are kept."""
    return tuple(sorted({max(1, round(f * n_estimators)) for f in CHECKPOINTS}))


def _terms(max_basis: int) -> tuple[int, ...]:
    """The numbers of terms a fold may keep."""
    return tuple(sorted({max(2, round(f * max_basis)) for f in TERMS}))


@dataclasses.dataclass
class Model:
    """A fitted FlexZBoost.

    Attributes:
        booster: The booster of every coefficient but the constant's.
        inputs: The feature set it was fitted on.
        basis: ``"cosine"`` or ``"Fourier"``.
        max_basis: Basis functions, the constant included.
        bump: The bump threshold, None for none.
        mean: The training mean of each fitted coefficient, the booster's
            base margin, shape (max_basis - 1,).
        checkpoints: Boosting rounds at which predictions are kept.
    """

    booster: xgboost.Booster
    inputs: str
    basis: str
    max_basis: int
    bump: float | None
    mean: FloatArray
    checkpoints: tuple[int, ...]


@dataclasses.dataclass
class Raw:
    """Predicted coefficients, and a cache for :meth:`FlexZBoost.masses`.

    Consecutive post-processing choices that differ only in the sharpening
    exponent share the normalised density before sharpening, which is kept
    here with the choice it belongs to.

    Attributes:
        coefs: Coefficients at every checkpoint, the constant (one) first,
            shape (n, n_checkpoints, max_basis), float32.
        checkpoints: Boosting rounds of the checkpoints.
        basis: ``"cosine"`` or ``"Fourier"``.
        bump: The bump threshold, None for none.
        cache_key: The (n_trees, n_terms) of ``cache``.
        cache: The normalised density without bumps, before sharpening, in
            blocks of rows.
    """

    coefs: npt.NDArray[np.float32]
    checkpoints: tuple[int, ...]
    basis: str
    bump: float | None
    cache_key: tuple[int, int] | None = dataclasses.field(
        default=None, repr=False
    )
    cache: list[_Sorted] | None = dataclasses.field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.coefs)


def _unsharpened(raw: Raw, n_trees: int, n_terms: int, begin: int) -> _Sorted:
    """The normalised density without bumps, before sharpening.

    Args:
        raw: The predictions.
        n_trees: The checkpoint.
        n_terms: Basis functions kept, the constant included.
        begin: The first of the block of rows, at most :data:`_ROWS`.

    Returns:
        The positive bins of the block, sorted for :func:`_sharpen`.
    """
    k = raw.checkpoints.index(n_trees)
    coefs = raw.coefs[begin : begin + _ROWS, k, :n_terms]
    series = coefs.astype(np.float64) @ _bin_means(raw.basis, n_terms)
    rows, cols = np.nonzero(series > 0.0)
    support = _normalize(_Support(series[rows, cols], rows, cols, len(series)))
    if raw.bump is not None:
        support = _remove_bumps(support, raw.bump)
    return _by_value(support)


def _blocks(raw: Raw) -> range:
    """The first row of each block of rows."""
    return range(0, len(raw), _ROWS)


class FlexZBoost(cv.Method):
    """FlexZBoost on the ``flexcode`` basis and one xgboost booster."""

    name = "flexzboost"

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Proposes the inputs, the basis, the bumps and the booster.

        The bump threshold is proposed here rather than evaluated on every
        fold, because it would multiply the cost of every sharpening
        exponent and it rarely helps. The ranges of the booster keep the
        cost of a fit bounded: a tree with vector leaves costs time in
        proportion to its nodes times the coefficients, so that one of
        depth 16 with leaves of one galaxy took 18 minutes for 200 trees on
        1,000 galaxies.

        Args:
            trial: The Optuna trial.

        Returns:
            The fit parameters.
        """
        return {
            "feature_set": trial.suggest_categorical(
                "feature_set", FEATURE_SETS
            ),
            "basis": trial.suggest_categorical("basis", BASES),
            "max_basis": trial.suggest_int("max_basis", 10, 100, step=5),
            "bump": trial.suggest_categorical("bump", BUMPS),
            "multi_strategy": trial.suggest_categorical(
                "multi_strategy", STRATEGIES
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", 20, 300, log=True
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.03, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 2.0, 300.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-2, 100.0, log=True
            ),
        }

    def post_grid(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The checkpoints, the terms kept and the sharpening exponents.

        The exponent varies fastest: :meth:`masses` keeps the density before
        sharpening, which the exponents of one (checkpoint, terms) share.

        Args:
            params: The fit parameters.

        Returns:
            One mapping per choice, ``n_trees``, ``n_terms`` and
            ``sharpen``.
        """
        return [
            {"n_trees": n_trees, "n_terms": n_terms, "sharpen": alpha}
            for n_trees, n_terms, alpha in itertools.product(
                _checkpoints(int(params["n_estimators"])),
                _terms(int(params["max_basis"])),
                SHARPEN,
            )
        ]

    def fit(
        self,
        params: Mapping[str, Any],
        train: datasets.Catalog,
        z_train: FloatArray,
        seed: int,
    ) -> Model:
        """Fits the booster to the basis functions of the true redshifts.

        The constant's coefficient is one for every galaxy and is not
        fitted. The booster starts from the training mean of each
        coefficient, its base margin, so that the trees of one checkpoint
        add to those of the one before.

        Args:
            params: The fit parameters.
            train: The training catalogue.
            z_train: Its redshifts, shape (n_train,), within [0, 2].
            seed: The booster's seed.

        Returns:
            The fitted :class:`Model`.
        """
        x = features.build(train, params["feature_set"])
        u = (np.asarray(z_train, dtype=np.float64) - config.Z_MIN) / _Z_SPAN
        target = basis_functions.evaluate_basis(
            u[:, None], int(params["max_basis"]), params["basis"]
        )[:, 1:]
        mean = target.mean(axis=0)
        dtrain = xgboost.DMatrix(
            x, label=target, base_margin=np.tile(mean, (len(x), 1))
        )
        booster = xgboost.train(
            {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "multi_strategy": params["multi_strategy"],
                "eta": params["learning_rate"],
                "max_depth": params["max_depth"],
                "min_child_weight": params["min_child_weight"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "lambda": params["reg_lambda"],
                "max_leaves": _MAX_LEAVES,
                "max_cached_hist_node": _CACHED_NODES,
                "nthread": _threads(),
                "seed": seed,
            },
            dtrain,
            num_boost_round=int(params["n_estimators"]),
        )
        return Model(
            booster=booster,
            inputs=params["feature_set"],
            basis=params["basis"],
            max_basis=int(params["max_basis"]),
            bump=params["bump"],
            mean=mean,
            checkpoints=_checkpoints(int(params["n_estimators"])),
        )

    def predict(self, model: Model, catalog: datasets.Catalog) -> Raw:
        """The coefficients at every checkpoint.

        Each checkpoint adds the trees since the one before, so the booster
        is walked once.

        Args:
            model: What :meth:`fit` returned.
            catalog: The rows to predict.

        Returns:
            The :class:`Raw` predictions, 4 * 3 * max_basis bytes a row.
        """
        x = features.build(catalog, model.inputs)
        n_fit = model.max_basis - 1
        dmatrix = xgboost.DMatrix(x, base_margin=np.zeros((len(x), n_fit)))
        coefs = np.empty(
            (len(x), len(model.checkpoints), model.max_basis), np.float32
        )
        coefs[:, :, 0] = 1.0
        total = np.tile(model.mean, (len(x), 1))
        begin = 0
        for k, end in enumerate(model.checkpoints):
            total += model.booster.predict(
                dmatrix, output_margin=True, iteration_range=(begin, end)
            ).reshape(len(x), n_fit)
            coefs[:, k, 1:] = total
            begin = end
        return Raw(
            coefs=coefs,
            checkpoints=model.checkpoints,
            basis=model.basis,
            bump=model.bump,
        )

    def masses(self, raw: Raw, post: Mapping[str, Any]) -> FloatArray:
        """Evaluates the series on the common grid and post-processes it.

        The series is averaged exactly over each bin, normalised, cleared of
        bumps and sharpened, and the density in each bin times its width is
        the bin's mass.

        Args:
            raw: What :meth:`predict` returned, for n rows.
            post: ``n_trees`` (a checkpoint), ``n_terms`` (at most the
                fitted number) and ``sharpen`` (None for none).

        Returns:
            Masses on the common grid, shape (n, 4000), rows summing to one.
        """
        key = (int(post["n_trees"]), int(post["n_terms"]))
        alpha = 1.0 if post["sharpen"] is None else float(post["sharpen"])
        out = np.zeros((len(raw), config.FINE_BINS))

        def sharpen(begin: int, block: _Sorted) -> None:
            _sharpen(block, alpha).scatter(
                out[begin : begin + _ROWS], 1.0 / config.FINE_BINS
            )

        with futures.ThreadPoolExecutor(_threads()) as pool:
            if raw.cache is None or raw.cache_key != key:
                raw.cache = None  # Free the old cache before the new one.
                with threadpoolctl.threadpool_limits(1):
                    raw.cache = list(
                        pool.map(
                            functools.partial(_unsharpened, raw, *key),
                            _blocks(raw),
                        )
                    )
                raw.cache_key = key
            if _threads() > 1:
                list(pool.map(sharpen, _blocks(raw), raw.cache))
            else:
                for begin, block in zip(_blocks(raw), raw.cache, strict=True):
                    sharpen(begin, block)
        return out


METHOD = FlexZBoost()
