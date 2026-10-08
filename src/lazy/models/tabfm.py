# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Densities from TabFM's in-context classifier, as a bin hierarchy.

TabFM (Google, 2026) is an in-context tabular foundation model: it is never
trained on your data, it reads labelled rows as context and answers queries
against them. Its classifier handles at most ten classes, which is far short
of the resolution a conditional density needs, so :class:`TabFMHistogram`
builds the density as a two-level hierarchy of *equal-mass* bins of the target:

* one classifier over ``n_coarse_bins`` quantile bins of the context values;
* one classifier per coarse bin over ``n_fine_bins`` quantile bins inside it,
  with only that bin's context rows as context.

Then ``P(bin | x) = P(coarse | x) * P(fine | coarse, x)``: a
:class:`~lazy.distributions.HistogramDistribution` over the equal-mass bins.
With ``n_dither > 1`` the whole hierarchy is repeated with the bin edges
shifted in quantile space and the results mixed with equal weights (a
:class:`~lazy.distributions.MixtureDistribution`) -- an averaged shifted
histogram, which removes the edge artifacts a single binning leaves behind.
The bins span the constructor grid's range, or the training values'; they
never depend on the grid a prediction is asked on. The native grid is the
union of every dither's edges.

Under ``"auto"`` (the default for both levels) the bin counts follow the
context: :func:`auto_bin_count` gives each level about ``sqrt(n / 5)`` bins,
so a final bin holds about five context rows, and every context of 500 rows
or more gets 10 x 10. Fewer rows per bin than that make the histogram
overconfident, with spikes where the few rows of a bin happen to fall. With
``"grid"`` at both levels, a constructor ``y_grid`` of at most 100 bins
replaces the equal-mass bins altogether: its own bins are the fine classes,
grouped into runs of at most ten for the coarse level, and the density lives
on that grid as it is.

The uniform features map onto TabFM's own machinery: ``kv_cache`` onto the
streaming prefill/decode path (:mod:`lazy.models._icl_stream`),
``feature_shuffle`` and the transforms it has onto its classifier's own
shuffles and ``norm_methods``, ``bag_size`` onto its members' own rows (lazy's
bags, handed to each classifier's members), ``outlier_threshold`` onto its soft
clip and ``mixed_precision`` onto its bfloat16 weights. The rest of TabFM's
recipe is pinned here, so the installed tabfm's defaults decide nothing.

Because the bins are equal-mass they are narrow where the targets are crowded,
so in the busy part of the distribution they are routinely *narrower* than
the output bin.
Mapping them onto the output grid is therefore done by exact, mass-conserving
integration (:meth:`lazy.grid.Grid.rebin`), not by sampling the density
at the output bin centres, which would drop whole bins and lose probability.

The ten-class limit constrains the hierarchy, never the output: ``y_grid`` can
have any number of bins, at any spacing, over any range.

There is deliberately no post-processing stage. Probability sharpening and
Gaussian smoothing were both tried and neither is used: TabFM's class
probabilities are already well calibrated in sharpness, and a Gaussian narrower
than the output bin cannot change a density tabulated on it.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Mapping
import copy
import functools
from importlib import util
import inspect
import math
import numbers
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import types
from typing import Any
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _ensemble
from lazy.models import _hub
from lazy.models import _icl_stream
from lazy.models import _members
from lazy.models import _progress
from lazy.models import _weights

__all__ = [
    "AUTO_BINS",
    "GRID_BINS",
    "REPOSITORY_BUILD",
    "TabFMHistogram",
    "TabFMPerformanceWarning",
    "auto_bin_count",
    "grid_groups",
    "prior_shift_em",
    "quantile_edges",
]

# Warnings skip every frame inside this package, so that they point at the
# user's call however deep in the package (or LazyModel) they are raised.
_PACKAGE_PREFIX = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
)


class TabFMPerformanceWarning(_ensemble.PerformanceWarning):
    """The prediction will be correct but far slower than it needs to be.

    Its own category so that it can be silenced on purpose --
    ``warnings.filterwarnings("ignore", category=TabFMPerformanceWarning)`` --
    without hiding anything else, and so that a test suite can assert it was
    raised.
    """


MAX_CLASSES = 10
"""TabFM's classifier ceiling, and hence the ceiling on each hierarchy level."""

#: The ``n_coarse_bins`` and ``n_fine_bins`` value that sizes equal-mass
#: bins from the context.
AUTO_BINS = "auto"

#: The ``n_coarse_bins`` and ``n_fine_bins`` value that takes the bins from
#: the constructor's grid.
GRID_BINS = "grid"

#: Context rows per final bin that ``"auto"`` aims at.
ROWS_PER_BIN = 5

#: The most bins a constructor grid can have for its bins to be the classes:
#: ten groups of ten.
MAX_GRID_BINS = MAX_CLASSES * MAX_CLASSES

# TabFM v1.0's own recipe, ``transforms="auto"``, written out here so that the
# installed upstream version cannot change it: the values TabFMClassifier
# defaults to, each passed explicitly.

#: The norm methods members cycle through under ``transforms="auto"``.
AUTO_NORM_METHODS: tuple[str, ...] = ("none", "power")

#: The soft outlier clip of the recipe, in standard deviations.
AUTO_OUTLIER_THRESHOLD = 4.0

#: Columns each member sees under the recipe; a wider table is subsampled.
AUTO_MAX_NUM_FEATURES = 500

#: The softmax temperature of the recipe: TabFMClassifier's default, the
#: value TabFM v1.0 was released with.
AUTO_SOFTMAX_TEMPERATURE = 0.9

#: Every other TabFMClassifier argument, pinned to the recipe's value.
_PINNED_CLASSIFIER_ARGS: dict[str, Any] = {
    # Each member sees the class labels cyclically shifted, undone on its
    # logits, which are then averaged before one softmax.
    "class_shift": True,
    "average_logits": True,
    # No calibration, no NNLS member weights, no appended feature crosses or
    # SVD features: the default TabFMClassifier, not its "ensemble" preset.
    "binary_calibration_method": None,
    "multiclass_calibration_method": None,
    "enable_nnls": False,
    "n_feature_crosses": 0,
    "n_svd_features": 0,
    # Inert, with every column numeric and the steps above off; pinned so that
    # nothing is left to upstream.
    "permute_categorical": False,
    "cat_encoder_mode": "appearance",
    "total_svd_pool": None,
    "num_folds_for_cv": 5,
    "nnls_beta": 0.75,
    "calibration_lambda": 1e-2,
    "min_rows_for_single_val_split": 2000,
    "verbose": False,
}

#: The versions the recipe above was read from upstream for. A new version
#: joins once its recipe is checked against upstream's.
_RECIPE_VERSIONS: tuple[str, ...] = ("v1.0",)


#: The repository build of TabFM, with the KV-cache API, at the commit lazy
#: was validated on. PyPI refuses direct URLs in an extra, so ``lazy setup``
#: installs it (:meth:`TabFMHistogram.setup`); pyproject.toml pins the same
#: commit.
REPOSITORY_BUILD = (
    "tabfm[pytorch] @ git+https://github.com/google-research/tabfm"
    "@fbb665569425fd2f490c6576b3af967876fe11ff"
)

# What `lazy setup` runs in a fresh interpreter to check the TabFM it just
# installed: the running process has already imported the old one.
_CHECK_BUILD = (
    "from lazy.models import _icl_stream; "
    "raise SystemExit(not _icl_stream.streaming_available())"
)

_SLOW_PATH_WARNING = (
    "the installed tabfm has no KV-cache API (TabFM.prefill and "
    "TabFM.decode; the PyPI releases 1.0.0 and 1.0.1 lack it), so "
    "TabFMHistogram is falling back to upstream predict_proba, which "
    "re-encodes the training context for every chunk of query rows -- "
    "measured at roughly 26x the compute per query row (13.7 ms vs 0.53 ms "
    "per member-row). The answers agree up to rounding; only the runtime "
    "differs, so a large prediction will simply take far longer than "
    "expected. The repository build has the API; install it with\n"
    "    lazy setup\n"
    "Pass kv_cache=False to accept the slow path and silence this."
)


def auto_bin_count(n_context: int) -> int:
    """Bins per hierarchy level under ``"auto"``, for a context of this size.

    ``floor(sqrt(n_context / 5))``, between 2 and 10: the two levels then make
    about five context rows per final bin, and 500 rows or more give the
    full 10 x 10.

    Args:
        n_context: Context rows, at least 0.

    Returns:
        The bin count of each level.

    Examples:
        >>> [auto_bin_count(n) for n in (10, 100, 277, 499, 500, 10_000)]
        [2, 4, 7, 9, 10, 10]
    """
    # isqrt of the floor is the floor of the square root, exactly.
    root = math.isqrt(int(n_context) // ROWS_PER_BIN)
    return min(MAX_CLASSES, max(2, root))


def grid_groups(n_bins: int) -> _typing.IntArray:
    """Where the coarse groups of a matched grid start and end.

    The fewest consecutive runs of at most ten bins, as even in length as
    they can be: 25 bins make runs of 9, 8 and 8.

    Args:
        n_bins: The grid's bins, from 1 to 100.

    Returns:
        The group boundaries as bin indices, shape ``(n_groups + 1,)``, from
        0 to ``n_bins``.

    Examples:
        >>> grid_groups(25).tolist()
        [0, 9, 17, 25]
        >>> grid_groups(100).tolist()[:3]
        [0, 10, 20]
    """
    n_groups = -(-n_bins // MAX_CLASSES)
    sizes = [len(run) for run in np.array_split(np.arange(n_bins), n_groups)]
    return np.cumsum([0, *sizes])


def quantile_edges(
    values: npt.ArrayLike,
    n_bins: int,
    lo: float,
    hi: float,
    shift: float = 0.0,
) -> _typing.FloatArray:
    """Equal-mass bin edges over ``[lo, hi]``.

    Args:
        values: The values whose quantiles set the edges, shape
            ``(n,)``.
        n_bins: Number of bins.
        lo: The first edge, replacing the lowest quantile.
        hi: The last edge, replacing the highest quantile.
        shift: Moves the interior edges in quantile space, in bins, which is
            what dithering varies between repeats.

    Returns:
        Non-decreasing edges, shape ``(n_bins + 1,)``. With no values at all
        there are no quantiles to take, and the edges are evenly spaced.

    Examples:
        >>> quantile_edges(np.linspace(0, 1, 101), 2, 0.0, 1.0).tolist()
        [0.0, 0.5, 1.0]
        >>> quantile_edges([], 2, 0.0, 1.0).tolist()
        [0.0, 0.5, 1.0]
    """
    values = np.asarray(values, dtype=float)
    if not values.size:
        return np.linspace(lo, hi, n_bins + 1)
    levels = np.clip(
        np.linspace(0.0, 1.0, n_bins + 1) + shift / n_bins, 0.0, 1.0
    )
    edges = np.quantile(values, levels)
    edges[0], edges[-1] = lo, hi
    return np.maximum.accumulate(edges)


def prior_shift_em(
    probs: npt.ArrayLike,
    prior_train: npt.ArrayLike,
    n_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Label-shift correction of classifier posteriors (Saerens et al. 2002).

    A classifier's posteriors carry the target distribution of its context.
    When the context and the queries come from different populations (in
    photo-z, a spectroscopic training set and a photometric sample), those
    distributions differ, and the posteriors inherit
    the wrong prior. This estimates the query set's own bin prior by EM and
    re-weights the posteriors by the ratio.

    Args:
        probs: Classifier posteriors, shape ``(n_query, n_bins)``.
        prior_train: The bin prior the posteriors carry (the context's bin
            fractions), shape ``(n_bins,)``; normalized here.
        n_iter: Maximum number of EM iterations.
        tol: Stop once no prior entry changes by more than this.

    Returns:
        A tuple ``(posteriors, prior)``: the re-weighted posteriors, shape
        ``(n_query, n_bins)``, and the estimated query-set prior, shape
        ``(n_bins,)``.
    """
    posteriors = np.asarray(probs, dtype=float)
    p_train = np.maximum(np.asarray(prior_train, dtype=float), 1e-12)
    p_train = p_train / p_train.sum()
    pi = p_train.copy()
    for _ in range(n_iter):
        post = posteriors * (pi / p_train)
        post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-300)
        new_pi = post.mean(axis=0)
        converged = np.abs(new_pi - pi).max() < tol
        pi = new_pi
        if converged:
            break
    post = posteriors * (pi / p_train)
    post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-300)
    return post, pi


class TabFMHistogram(_ensemble.ContextEnsembleEstimator):
    """Target distributions from TabFM's classifier over a bin hierarchy.

    Each in-context stage -- the coarse level and every coarse bin's fine
    level -- is one ``TabFMClassifier`` of ``n_estimators`` members. Within a
    classifier the members' class-shift-corrected logits are averaged and
    divided by the temperature before one softmax (TabFM's own
    ``average_logits``); the levels multiply into bin probabilities; a
    group of members (one per scaffolded transform) and the dithers are then
    density mixtures, weighted by members and equally respectively. Every
    dither runs its classifiers with the same seeds -- the coarse level the
    group's seed, fine level ``j`` that seed plus ``1 + j`` -- so dithers
    differ only by their bin edges, as in the paper's runs. The bin edges
    are the whole context's, for every group and every bag.

    Every ``TabFMClassifier`` argument is set by lazy, none left to the
    installed tabfm. What TabFM still does to the features under any
    ``transforms``: mean-impute missing values (dropping an all-missing
    column), drop constant columns, and standardise each column (clipped
    to +-100); under ``transforms="auto"`` the recipe adds the Yeo-Johnson
    members, the 4-sigma soft clip and, above 500 columns, a random
    500-column subset per member.

    Args:
        version: Which pinned TabFM checkpoint to load; see
            :func:`lazy.list_versions`. Recorded in ``provenance_``.
        n_estimators: Ensemble members per classifier, exactly; each sees
            its own column order and class shift. More members cost
            linearly more time and reduce member noise.
        transforms: Per-member feature transforms: ``"auto"`` (TabFM's own
            recipe, pinned in lazy: members alternate ``none`` and ``power``),
            a recipe name, a transform name or a sequence of them; see
            :mod:`lazy.models._transforms`. ``none`` and ``power`` are
            TabFM's own ``norm_methods``, which equal lazy's (TabFM applies
            them after its standardisation); the rest are scaffolded, each as
            a hierarchy of its own, because TabFM's namesakes differ (its
            ``robust`` scales to unit variance, its ``quantile`` subsamples
            10,000 rows, its ``quantile_rtdl`` draws other noise). Under an
            explicit value no column subsampling and, unless
            ``outlier_threshold`` is a number, no soft clip.
        feature_shuffle: Whether members see the columns in different orders
            (TabFM's ``feat_shuffle_method="random"``, else ``"none"``).
            Without it every member sees every column in the original order,
            however wide the table.
        bag_size: Context rows per member: an int is a row count (1 means
            one row), a float a fraction in (0, 1] (1.0 means all rows), and
            None all of them. Member ``i`` sees lazy's bag ``i``, as on every
            model: its coarse-level member that bag, its member of fine
            level ``j`` the rows of the bag in coarse bin ``j``. A member of
            TabFM's own transforms keeps its place in the classifier (its
            column order, class shift and the logit average), with its
            standardisation, norm method and clip fitted on those rows; a
            member with a scaffolded transform is a hierarchy of its own on
            its bag. Members are run one at a time, whatever
            ``member_batch_size`` says, because their contexts differ in
            length.
        kv_cache: Prefill each member's context once and decode the queries
            against the cache, chunk by chunk (``True``; needs TabFM's
            repository build, and falls back with a
            :class:`TabFMPerformanceWarning` without it), or re-encode the
            context for every chunk of queries (``False``). Each classifier
            is fitted and its context prefilled inside every prediction
            call, so the cache serves the chunks of one call, not later
            calls. In float32 the two paths agree to float rounding. In
            bfloat16 on CUDA, whose kernels round differently for different
            batch shapes, they differ as any two bfloat16 runs do, each as
            far from a float32 run: a typical row's density by about 3% of
            its peak (at most 4.5% in our tests), single bins by up to a
            quarter of the peak. In our tests the CRPS and NLL moved about
            as much as with a change of seed, and ``mixed_precision=False``
            removes the difference at two to three times the time.
        y_grid: Default output grid: a :class:`lazy.grid.Grid`, an
            array of bin centres, ``"native"``, or None for the native grid
            (the union of every dither's bin edges). A constructor grid also
            sets the range the bins span; without one they span the training
            values. With ``"grid"`` bins, a constructor grid of at most 100
            bins is matched: its bins become the classes, see
            ``n_coarse_bins``. A grid passed to a prediction only rebins
            the answer.
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"mps"``, ``"cpu"``, or a ``torch.device``.
        random_state: Seed of the ensemble: a group's classifiers are seeded
            from its first member ``i``, ``SeedSequence([random_state, i])``
            (see the seeds above). None draws a fresh seed at fit, recorded
            as ``random_state_`` and in ``provenance_``.
        chunk_size: Query rows per forward pass, on both paths: decoded
            against the cache at a time, or handed to the upstream
            ``predict_proba`` at a time when ``kv_cache=False``; ``0`` does
            them in one pass. The in-context stage builds its keys and values
            from the context rows only, so a row's answer never depends on
            the other rows in its chunk, but the kernels batch differently:
            on the CPU in float32 it changes by float rounding (measured up
            to 2e-6 of the peak density; bit-identical in some cases), and
            in bfloat16 on CUDA by the rounding ``kv_cache`` describes.
        softmax_temperature: Divides the member-averaged logits before the
            softmax. ``"auto"`` is TabFM v1.0's calibrated 0.9, deliberately
            not 1.0; a positive number overrides it. Recorded in
            ``provenance_``.
        mixed_precision: On CUDA, run TabFM in bfloat16, the precision it
            was designed for (its float32 weights cast at load); False keeps
            float32. A CPU always runs float32, whatever this says.
            Recorded, resolved, in ``provenance_``.
        outlier_threshold: TabFM's own two-pass soft clip of each
            standardised column, at this many standard deviations
            (:class:`lazy.models._transforms.SoftClip`). ``"auto"`` is 4.0
            under ``transforms="auto"`` and off under an explicit
            ``transforms``; None is off (TabFM is passed an infinite
            threshold, which makes its clip the identity).
        n_coarse_bins: Classes at the first hierarchy level: an int from 2
            to 10, ``"auto"`` or ``"grid"``. An int gives equal-mass bins,
            and so does ``"auto"``, with :func:`auto_bin_count` of the
            context size (about five context rows per final bin; 10 from 500
            rows up). ``"grid"``, which both levels must take together,
            needs a constructor ``y_grid`` of at most 100 bins: the grid's
            bins are then the fine classes, grouped into the fewest runs of
            at most ten (:func:`grid_groups`) for the coarse level, so the
            density is built on the grid itself and never rebinned onto it.
            A bin no context row falls in gets zero probability, so a true
            value there has zero density, and a context target outside the
            grid is clipped into an end bin, with a warning. Recorded,
            resolved, in ``provenance_["bins"]``.
        n_fine_bins: Classes within each coarse bin: an int from 2 to 10,
            ``"auto"`` or ``"grid"``, as for ``n_coarse_bins``. The density
            is built on ``n_coarse_bins * n_fine_bins`` equal-mass bins.
        n_dither: Repeats of the whole hierarchy with bin edges shifted by
            ``d/n_dither`` of a bin, mixed with equal weights. 1 disables
            dithering; 3 gives a better likelihood almost everywhere, at
            about three times the compute, and is what we recommend when
            accuracy matters. ``"grid"`` bins take only 1, since shifted
            copies of the grid's edges would no longer be its edges.
        prior_shift: ``"em"`` applies the label-shift correction of
            :func:`prior_shift_em` using the context's own bin fractions as
            the training prior, which is worth having when the context is a
            biased spectroscopic sample. ``None`` (default) leaves the
            posteriors alone.
        member_batch_size: Ensemble members processed together (prefilled
            together on the cached path; TabFM's own ``batch_size``), an int
            of at least 1. This and the next two are memory/throughput knobs
            that change the result only by the rounding ``kv_cache``
            describes.
        query_block_rows: Query rows whose member views are built at a time
            on the cached path, bounding host memory; an int of at least 1,
            rounded down to a whole number of ``chunk_size`` chunks (at least
            one), so it never changes the rows per forward pass.
        keep_cache_on_device: Whether the prefilled cache stays on the device
            rather than round-tripping through host memory.
        progress: A progress bar over the in-context stages: ``"auto"`` shows
            it on a terminal or in a notebook, ``True`` always, ``False``
            never. It counts stages, ``n_dither * (1 + n_coarse_bins)`` of
            them per member group, because each runs over every query row.
        verbose: Print per-level log messages to stdout.

    Attributes:
        grid_: The resolved default output grid.
        native_grid_: The union of every dither's equal-mass bin edges, a
            histogram-normalised grid set at fit; under a matched grid, that
            grid.
        inference_: The path actually used: ``"stream"`` or
            ``"predict_proba"``.
        support_: The range the equal-mass bins span, (low, high).
        provenance_: Which weights, code and ensemble answered, as a dict.
        checkpoint_: The pinned checkpoint, a :class:`pathlib.Path`. TabFM
            loads its backbone on the first prediction, so this appears then.
        bin_prior_: The context's bin fractions from the last prediction,
            shape (n_bins,).
        bins_: How the bins were chosen, for example
            ``"auto equal-mass 7x7"`` or ``"grid 50 bins in 5 groups"``.
        n_coarse_bins_: The resolved coarse classes, or a matched grid's
            groups.
        n_fine_bins_: The resolved fine classes, or the most bins in a
            matched grid's group.
        bin_grid_: The matched constructor grid, or None for equal-mass
            bins.
        n_context_: Context rows ``fit`` was given.

    Examples:
        >>> est = TabFMHistogram(n_estimators=2, n_dither=3)
        >>> est.n_dither
        3
    """

    backend = "tabfm"
    display_name = "TabFM"
    method_note = (
        "A hierarchy of TabFM in-context classifiers over "
        "equal-mass bins of the target."
    )
    extra = "tabfm"
    native_output = "histogram"
    # Only TabFM's norm methods that are lazy's transforms: its robust scales
    # to unit variance, its quantile subsamples 10,000 rows and its
    # quantile_rtdl adds different noise, so those three are scaffolded.
    native_transforms = {"none": "none", "power": "power"}
    auto_tokens = AUTO_NORM_METHODS
    supports_native_bagging = True
    native_outlier_clipping = True
    # Chunking changes the float rounding even on a CPU, and in bfloat16 on
    # CUDA it and the cache change it far more (see kv_cache above).
    exact_chunking = False
    # Float rounding in float32, measured at up to 4e-5 of the peak density
    # on CUDA.
    kv_cache_rtol = 1e-4
    chunks_queries = False

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v1.0",
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
        n_coarse_bins: int | str = AUTO_BINS,
        n_fine_bins: int | str = AUTO_BINS,
        n_dither: int = 1,
        prior_shift: str | None = None,
        member_batch_size: int = 1,
        query_block_rows: int = 262_144,
        keep_cache_on_device: bool = True,
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
        self.n_coarse_bins = n_coarse_bins
        self.n_fine_bins = n_fine_bins
        self.n_dither = n_dither
        self.prior_shift = prior_shift
        self.member_batch_size = member_batch_size
        self.query_block_rows = query_block_rows
        self.keep_cache_on_device = keep_cache_on_device
        self.progress = progress
        self.verbose = verbose

    # -- the per-backend interface -------------------------------------------

    def _import_backend(self) -> types.ModuleType:
        return _ensemble.import_extra(
            "tabfm", needed_by=type(self).__name__, extra=self.extra
        )

    @classmethod
    def setup(cls, *, dry_run: bool) -> bool:
        """Installs TabFM's repository build over the release, if need be.

        The extra brings the PyPI release, which has no KV-cache API, and
        PyPI refuses a git URL in an extra, so this installs
        :data:`REPOSITORY_BUILD` over it, with pip or uv. In a uv project it
        prints the ``uv add`` that keeps the build instead, since ``uv sync``
        would put the release back.

        Args:
            dry_run: Say what would be installed, and install nothing.

        Returns:
            Whether the repository build is installed, or would be after a
            dry run.
        """
        if util.find_spec("tabfm") is None:
            print("TabFM: not installed, skipped ('lazy-tfm[tabfm]' adds it).")
            return True
        if _icl_stream.streaming_available():
            print("TabFM: the repository build is installed.")
            return True
        if _in_uv_project():
            # uv sync and uv run would put the release back from uv.lock, so
            # the build has to be a requirement of the project itself.
            print(
                "TabFM: in a uv project, add the repository build to the "
                "project, so that uv sync keeps it:\n"
                f"  uv add {shlex.quote(REPOSITORY_BUILD)}"
            )
            return False
        command = _install_command(REPOSITORY_BUILD)
        if command is None:
            print(
                "TabFM: found neither pip nor uv to install the repository "
                "build with. Install it by hand:\n"
                "  pip install --force-reinstall --no-deps "
                f"{shlex.quote(REPOSITORY_BUILD)}"
            )
            return False
        print(
            f"TabFM: installing the repository build:\n  {shlex.join(command)}"
        )
        if dry_run:
            return True
        if subprocess.run(command, check=False).returncode != 0:
            print("TabFM: the install failed; see the output above.")
            return False
        check = subprocess.run(
            [sys.executable, "-c", _CHECK_BUILD], check=False
        )
        if check.returncode != 0:
            print("TabFM: installed, but the KV-cache API is still missing.")
            return False
        print("TabFM: the repository build is installed.")
        return True

    def _auto_softmax_temperature(self) -> float:
        return AUTO_SOFTMAX_TEMPERATURE

    def _auto_outlier_threshold(self) -> float:
        return AUTO_OUTLIER_THRESHOLD

    @classmethod
    def _pinned_recipe(cls, version: str) -> Mapping[str, Any]:
        if version not in _RECIPE_VERSIONS:
            return super()._pinned_recipe(version)
        return types.MappingProxyType(
            {
                "norm_methods": AUTO_NORM_METHODS,
                "outlier_threshold": AUTO_OUTLIER_THRESHOLD,
                "max_num_features": AUTO_MAX_NUM_FEATURES,
                "softmax_temperature": AUTO_SOFTMAX_TEMPERATURE,
                **_PINNED_CLASSIFIER_ARGS,
            }
        )

    def _check_backend_params(self) -> None:
        for name in ("n_coarse_bins", "n_fine_bins"):
            value = getattr(self, name)
            if _is_auto(value) or _is_grid(value):
                continue
            if not _is_count(value):
                raise ValueError(
                    f"{name} must be 'auto', 'grid' or an int from 2 to "
                    f"{MAX_CLASSES}: {name}={value!r}"
                )
            if value > MAX_CLASSES:
                raise ValueError(
                    f"TabFM classification supports at most {MAX_CLASSES}"
                    f" classes per level, got {name}={value}"
                )
            if value < 2:
                raise ValueError(
                    f"each hierarchy level needs at least two bins: "
                    f"{name}={value}"
                )
        if self.n_dither < 1:
            raise ValueError("n_dither must be at least 1")
        self._check_grid_bins()
        # _fit sets the clipped context targets before the base class
        # validates, so the layout is fixed here, before any group is fitted.
        self._resolve_bins(self.y_context_.size)
        if self.prior_shift not in (None, "em"):
            raise ValueError("prior_shift must be None or 'em'")
        for name in ("member_batch_size", "query_block_rows"):
            value = getattr(self, name)
            if not _is_count(value):
                raise ValueError(
                    f"{name} must be an int of at least 1: {name}={value!r}"
                )
        self.inference_ = self._resolve_inference()

    def _resolve_inference(self) -> str:
        """Picks the inference path, failing at fit time, not mid-prediction."""
        if self.kv_cache is False:
            return "predict_proba"  # asked for explicitly; say nothing
        if _icl_stream.streaming_available():
            return "stream"
        # Warn, and do not be shy about it. The fallback is correct but it
        # re-runs the context forward pass for every chunk of query rows,
        # which measures ~13.7 ms per member-row against ~0.53 ms on the
        # cached path -- around twenty-six times the work. On a large query
        # set that is the difference between one hour and a day, and because
        # both paths produce the same answer the only symptom is a run that
        # never seems to end. Nothing else reports it, so this does.
        warnings.warn(
            _SLOW_PATH_WARNING,
            TabFMPerformanceWarning,
            skip_file_prefixes=(_PACKAGE_PREFIX,),
        )
        return "predict_proba"

    def _load_checkpoint(self) -> None:
        # The backbone is several gigabytes and loads lazily on the first
        # prediction (see _backbone); fit only records what it will load.
        self.provenance_ = _hub.get_checkpoint(
            self.backend, self.version
        ).provenance(device=self.device_)

    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> Any:
        del y  # The clipped targets of the same rows are y_context_'s.
        if self.inference_ == "predict_proba":
            self.kv_cache_ = False
        rows = slice(None) if group.rows is None else group.rows
        return {"X": _frame(X), "y": self.y_context_[rows], "group": group}

    def _before_fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        del X  # Unused: the bins come from the targets alone.
        # One set of equal-mass bins for the whole ensemble, from the whole
        # context: natively bagged members share a classifier, so they must
        # share its classes, and every group then predicts on the same bins.
        self.support_ = self._support(y)
        self.y_context_ = _clip_to_support(y, self.support_)

    def _resolve_bins(self, n_context: int) -> None:
        """Fixes the bin layout: a matched grid, or equal-mass bin counts.

        Sets ``bin_grid_``, ``n_coarse_bins_``, ``n_fine_bins_`` and
        ``bins_``. Explicit counts need a context row per bin; ``"auto"``
        sizes them so that it has about five.

        Args:
            n_context: Context rows the bins are built from.

        Raises:
            ValueError: If explicit bin counts ask for more bins than there
                are context rows.
        """
        self.bin_grid_ = self._matched_grid()
        if self.bin_grid_ is not None:
            bounds = grid_groups(self.bin_grid_.n_bins)
            self.n_coarse_bins_ = int(bounds.size - 1)
            self.n_fine_bins_ = int(np.diff(bounds).max())
            self.bins_ = (
                f"grid {self.bin_grid_.n_bins} bins in "
                f"{self.n_coarse_bins_} groups"
            )
            return
        auto = auto_bin_count(n_context)
        counts = [
            auto if _is_auto(value) else int(value)
            for value in (self.n_coarse_bins, self.n_fine_bins)
        ]
        self.n_coarse_bins_, self.n_fine_bins_ = counts
        layout = f"equal-mass {counts[0]}x{counts[1]}"
        if _is_auto(self.n_coarse_bins) and _is_auto(self.n_fine_bins):
            self.bins_ = f"auto {layout}"
            return
        self.bins_ = layout
        if n_context < counts[0] * counts[1]:
            raise ValueError(
                f"context has {n_context} rows, fewer than the "
                f"{counts[0] * counts[1]} bins asked for; pass "
                "n_coarse_bins='auto' and n_fine_bins='auto' to size the "
                "bins from the context"
            )

    def _check_grid_bins(self) -> None:
        """Validates ``"grid"`` bins: both levels, a grid, and no dithers."""
        levels = (self.n_coarse_bins, self.n_fine_bins)
        if not any(_is_grid(value) for value in levels):
            return
        if not all(_is_grid(value) for value in levels):
            raise ValueError(
                "'grid' bins take the classes of both levels from y_grid, so "
                "n_coarse_bins and n_fine_bins must both be 'grid': "
                f"n_coarse_bins={self.n_coarse_bins!r}, "
                f"n_fine_bins={self.n_fine_bins!r}"
            )
        if self.y_grid is None or isinstance(self.y_grid, str):
            raise ValueError(
                "'grid' bins are the bins of the constructor's y_grid, so "
                f"they need a grid there: y_grid={self.y_grid!r}"
            )
        n_bins = grid_lib.as_grid(self.y_grid).n_bins
        if n_bins > MAX_GRID_BINS:
            raise ValueError(
                f"'grid' bins hold at most {MAX_GRID_BINS} bins (ten groups "
                f"of ten classes), but y_grid has {n_bins}; pass a coarser "
                "grid, or 'auto' bins, which work with a grid of any size"
            )
        if self.n_dither > 1:
            raise ValueError(
                "n_dither > 1 shifts the bin edges, but 'grid' bins are "
                f"y_grid's own and must keep its edges: n_dither="
                f"{self.n_dither}. Pass n_dither=1, or 'auto' or int bin "
                "counts for dithered equal-mass bins."
            )

    def _matched_grid(self) -> grid_lib.Grid | None:
        """The constructor grid under ``"grid"`` bins, else None."""
        if not _is_grid(self.n_coarse_bins):
            return None
        return grid_lib.as_grid(self.y_grid)

    def _recipe(self) -> dict[str, Any]:
        return {
            **super()._recipe(),
            "bins": self.bins_,
            "n_dither": int(self.n_dither),
        }

    def _support(self, y: _typing.FloatArray) -> tuple[float, float]:
        """The range the equal-mass bins span: the constructor grid's or z's."""
        if self.y_grid is not None and not isinstance(self.y_grid, str):
            fixed = grid_lib.as_grid(self.y_grid)
            return fixed.y_min, fixed.y_max
        return float(y.min()), float(y.max())

    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.Distribution:
        model = self._backbone()
        query = _frame(X)
        shifts = [d / self.n_dither for d in range(self.n_dither)]
        parts = []
        with _progress.bar(
            self.progress,
            total=self.n_dither * (1 + self.n_coarse_bins_),
            desc=f"TabFM {self.version} ({len(X):,} rows)",
            unit="stage",
        ) as progress:
            for i, shift in enumerate(shifts):
                self._log(
                    f"dither {i + 1}/{self.n_dither}"
                    f" (edge shift {shift:.2f} bins)"
                )
                probs, edges, prior = self._hierarchy(
                    model,
                    handle,
                    query,
                    shift,
                    progress=progress,
                    dither=f"{i + 1}/{self.n_dither}",
                )
                if self.prior_shift == "em":
                    probs, _ = prior_shift_em(probs, prior)
                parts.append(distributions.HistogramDistribution(edges, probs))
        self.bin_prior_ = prior
        if len(parts) == 1:
            return parts[0]
        return distributions.MixtureDistribution.equal(parts)

    def _native_grid(self) -> grid_lib.Grid:
        if self.bin_grid_ is not None:
            return self.bin_grid_
        shifts = [d / self.n_dither for d in range(self.n_dither)]
        edges = np.unique(
            np.concatenate(
                [self._edges(self.y_context_, shift)[0] for shift in shifts]
            )
        )
        if edges.size < 3:
            # A constant target has one bin; a grid needs two, and halving
            # it changes no density.
            edges = np.r_[edges[0], edges.mean(), edges[-1]]
        return grid_lib.Grid.from_edges(edges, normalization="histogram")

    # -- the hierarchy ------------------------------------------------------

    def _edges(
        self, y: _typing.FloatArray, shift: float
    ) -> tuple[
        _typing.FloatArray, _typing.FloatArray, list[_typing.FloatArray]
    ]:
        """Returns a tuple (all edges, coarse edges, fine edges per coarse bin).

        A matched grid's own edges, grouped by :func:`grid_groups`, or the
        equal-mass edges for one dither, spanning ``support_``; the top
        edge is nudged up so the largest value falls inside. Tied targets
        (discrete, zero-inflated or constant ones) collapse neighbouring
        quantiles, which leaves zero-width, empty bins: an empty coarse bin
        gets evenly spaced fine edges across itself, and every empty bin
        gets zero probability.
        """
        if self.bin_grid_ is not None:
            edges = np.array(self.bin_grid_.edges)
            bounds = grid_groups(self.bin_grid_.n_bins)
            fine = [
                edges[start : stop + 1]
                for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
            ]
            return edges, edges[bounds], fine
        low, high = self.support_
        # A constant target still gets a bin of non-zero width, on its scale.
        span = high - low if high > low else max(abs(high), 1.0)
        coarse = quantile_edges(
            y, self.n_coarse_bins_, low, high + 1e-6 * span, shift
        )
        labels = _bin_labels(coarse, y)
        fine = [
            quantile_edges(
                y[labels == j],
                self.n_fine_bins_,
                coarse[j],
                coarse[j + 1],
                shift,
            )
            for j in range(self.n_coarse_bins_)
        ]
        edges = np.r_[np.concatenate([f[:-1] for f in fine]), coarse[-1]]
        return edges, coarse, fine

    def _hierarchy(
        self,
        model: Any,
        handle: Any,
        X_query: pd.DataFrame,
        shift: float,
        *,
        progress: Any = None,
        dither: str = "1/1",
    ) -> tuple[_typing.FloatArray, _typing.FloatArray, _typing.FloatArray]:
        """Bin posteriors for one set of dithered edges.

        Args:
            model: The loaded TabFM backbone.
            handle: The fitted group: its context, targets and members.
            X_query: The query rows.
            shift: Edge shift in quantile space, in bins.
            progress: The bar :meth:`_predict_group` opened, advanced once per
                in-context stage; ``None`` draws nothing.
            dither: Which dither this is, for the bar's label.

        Returns:
            A tuple ``(probs, edges, context_bin_prior)``: posteriors, shape
            ``(n_query, n_bins)``; bin edges, shape ``(n_bins + 1,)``; and
            the context's bin fractions, shape ``(n_bins,)``.
        """
        y, X_context, group = handle["y"], handle["X"], handle["group"]
        edges, coarse_edges, fine_edges = self._edges(self.y_context_, shift)
        n_coarse = coarse_edges.size - 1
        coarse = _bin_labels(coarse_edges, y)
        # Each member's rows of this group's context, for natively bagged
        # members; None when every member sees all of them.
        member_rows = getattr(group, "member_rows", None)
        _stage(progress, dither, "coarse", y.size)
        p_coarse = self._class_probabilities(
            model,
            group,
            X_context,
            coarse,
            n_coarse,
            X_query,
            group.seed,
            member_rows,
        )
        _done(progress)
        prior: list[_typing.FloatArray] = []
        blocks: list[_typing.FloatArray] = []
        for j in range(n_coarse):
            rows = coarse == j
            n_fine = fine_edges[j].size - 1
            fine = _bin_labels(fine_edges[j], y[rows])
            level_rows = (
                None
                if member_rows is None
                else _level_rows(member_rows, np.flatnonzero(rows))
            )
            _stage(
                progress,
                dither,
                f"fine {j + 1}/{n_coarse}",
                int(rows.sum()),
            )
            p_fine = self._class_probabilities(
                model,
                group,
                X_context.iloc[rows],
                fine,
                n_fine,
                X_query,
                group.seed + 1 + j,
                level_rows,
            )
            _done(progress)
            prior.append(np.bincount(fine, minlength=n_fine) / max(y.size, 1))
            blocks.append(p_fine * p_coarse[:, [j]])
            self._log(
                f"  coarse bin {j + 1}/{n_coarse}"
                f" ({int(rows.sum())} context rows)"
            )
        return np.concatenate(blocks, axis=1), edges, np.concatenate(prior)

    def _class_probabilities(
        self,
        model: Any,
        group: _members.MemberGroup,
        X_context: pd.DataFrame,
        labels: _typing.IntArray,
        n_classes: int,
        X_query: pd.DataFrame,
        seed: int,
        member_rows: tuple[_typing.IntArray, ...] | None = None,
    ) -> _typing.FloatArray:
        """Member-averaged class posteriors, ``(n_query, n_classes)``.

        A level with no context rows gives every class zero probability, and
        one whose rows all share a class gives that class all of it; neither
        has anything for a classifier to learn, so none is run.

        Args:
            model: The loaded TabFM backbone.
            group: The members to run.
            X_context: The level's context rows.
            labels: Their classes, shape ``(n_rows,)``.
            n_classes: The level's class count.
            X_query: The query rows.
            seed: The classifier's seed.
            member_rows: Each member's rows of ``X_context``, as positions,
                or None for all of them. A member with none sits the level
                out, and the logits are averaged over the others.

        Returns:
            The posteriors.
        """
        present = np.unique(labels)
        if present.size < 2:
            full = np.zeros((len(X_query), n_classes))
            full[:, present.astype(int)] = 1.0
            return full
        classifier = self._classifier(model, group, seed)
        classifier.fit(X_context.reset_index(drop=True), labels)
        classes = np.asarray(classifier.classes_)

        if member_rows is not None:
            # TabFM stacks its members' contexts into one tensor, so members
            # of different lengths are run one by one, each in its place in
            # the classifier, and their logits averaged as TabFM would.
            members = [
                self._member_logits(model, view, X_query)
                for view in _member_views(classifier, member_rows)
            ]
            probs = (
                _icl_stream.softmax(
                    np.mean(members, axis=0), self._temperature()
                )
                if members
                else np.zeros((len(X_query), classes.size))
            )
        elif self.inference_ == "stream":
            logits = _icl_stream.classification_logits(
                classifier,
                model,
                {"query": X_query},
                member_batch_size=self.member_batch_size,
                query_block_rows=self.query_block_rows,
                chunk_size=self.chunk_size,
                keep_cache_on_device=self.keep_cache_on_device,
            )
            probs = _icl_stream.softmax(
                logits["query"]["mean_logits"], self._temperature()
            )
        else:
            probs = self._predict_proba_chunked(classifier, X_query)

        # A class with no context rows at all never appears in `classes_`; it
        # gets zero probability rather than shifting every later column.
        full = np.zeros((probs.shape[0], n_classes))
        full[:, classes.astype(int)] = probs
        return full

    def _member_logits(
        self, model: Any, view: Any, X_query: pd.DataFrame
    ) -> _typing.FloatArray:
        """One member's class-shift-corrected logits, ``(n_query, n_classes)``.

        Args:
            model: The loaded TabFM backbone.
            view: A one-member classifier from :func:`_member_views`.
            X_query: The query rows.

        Returns:
            The logits, before the temperature.
        """
        if self.inference_ == "stream":
            logits = _icl_stream.classification_logits(
                view,
                model,
                {"query": X_query},
                member_batch_size=1,
                query_block_rows=self.query_block_rows,
                chunk_size=self.chunk_size,
                keep_cache_on_device=self.keep_cache_on_device,
            )
            return logits["query"]["mean_logits"]
        size = self.chunk_size if self.chunk_size > 0 else len(X_query)
        frame = X_query.reset_index(drop=True)
        blocks = [
            np.asarray(
                view._predict_proba_internal(  # noqa: SLF001 - the raw logits.
                    frame.iloc[start : start + size]
                )[0],
                dtype=float,
            )
            for start in range(0, len(frame), size)
        ]
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _predict_proba_chunked(
        self, classifier: Any, X_query: pd.DataFrame
    ) -> _typing.FloatArray:
        """Upstream ``predict_proba``, a bounded number of query rows at a time.

        ``predict_proba`` materialises ``n_members x n_query x n_features`` in
        one go, which a survey-sized query set cannot afford. Feeding it chunks
        caps that at ``n_members x chunk_size x n_features``. The in-context
        stage builds its keys and values from the context rows alone, so
        query rows never influence one another; the result differs from a
        single pass only by float rounding, the kernels batching
        differently. The context forward pass is repeated per chunk, which
        is the price.
        """
        size = self.chunk_size if self.chunk_size > 0 else len(X_query)
        frame = X_query.reset_index(drop=True)
        blocks = [
            np.asarray(
                classifier.predict_proba(frame.iloc[start : start + size]),
                dtype=float,
            )
            for start in range(0, len(frame), size)
        ]
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _classifier(
        self,
        model: Any,
        group: _members.MemberGroup,
        seed: int,
    ) -> Any:
        """A ``TabFMClassifier`` with every argument set here.

        Nothing is left to upstream's defaults: the uniform parameters drive
        what they mean, and the rest of TabFM's recipe is pinned
        (:data:`AUTO_NORM_METHODS`, :data:`_PINNED_CLASSIFIER_ARGS`). The
        cache arguments exist on builds with the KV-cache API and not on the
        PyPI release; sending them unconditionally is a ``TypeError``.
        """
        tabfm = self._import_backend()
        kwargs: dict[str, Any] = {
            "model": model,
            "n_estimators": group.n_members,
            # Upstream cycles its norm methods over the members, so the
            # per-member list runs exactly as planned, repeats and all.
            "norm_methods": list(
                AUTO_NORM_METHODS
                if group.native_transforms is None
                else group.native_transforms
            ),
            "feat_shuffle_method": (
                "random" if group.feature_shuffle else "none"
            ),
            # TabFM always runs its clip; an infinite threshold makes it the
            # identity, bit for bit, which is how it is turned off.
            "outlier_threshold": (
                math.inf
                if self.outlier_threshold_ is None
                else self.outlier_threshold_
            ),
            # Upstream draws a member's column subset in shuffled order, so a
            # subsampling member always sees permuted columns: only the auto
            # recipe with shuffles subsamples, and otherwise every member sees
            # every column (in order, without feature_shuffle).
            "max_num_features": (
                AUTO_MAX_NUM_FEATURES
                if group.native_transforms is None and group.feature_shuffle
                else None
            ),
            # Never TabFM's own row draws: bagged members get lazy's bags
            # (_member_views).
            "max_num_rows": None,
            "softmax_temperature": self._temperature(),
            # Informational only in TabFM; the precision is the weights'.
            "use_amp": self.mixed_precision_,
            "batch_size": self.member_batch_size,
            "random_state": seed,
            **_PINNED_CLASSIFIER_ARGS,
        }
        accepted = inspect.signature(tabfm.TabFMClassifier.__init__).parameters
        for name, value in (
            ("cache_context", False),
            ("maybe_quantize_kv_cache", False),
            ("keep_cache_on_device", self.keep_cache_on_device),
        ):
            if name in accepted:
                kwargs[name] = value
        return tabfm.TabFMClassifier(**kwargs)

    # -- backbone -----------------------------------------------------------

    def _backbone(self) -> Any:
        """The loaded TabFM classification model, cached on the instance.

        The checkpoint is several gigabytes, so it is loaded once per process
        (:mod:`lazy.models._weights`) and shared by every estimator with the
        same version, device and precision: it holds nothing between calls,
        the classifiers and their caches being built per prediction. The
        weights are stored in float32 and TabFM is designed to compute in
        bfloat16, which it does here only under ``mixed_precision`` on CUDA;
        otherwise, the CPU included, the float32 weights are kept. The
        instance's reference is keyed by ``version``, device and precision so
        that changing one can never leave a prediction running on the
        previous model's weights, and it is dropped on pickling: an unpickled
        estimator reloads, or finds the shared model, on next use.
        """
        return self._memoized_network(
            (self.version, str(self.device_), self.mixed_precision_),
            self._shared_backbone,
        )

    def _shared_backbone(self) -> Any:
        """The checkpoint and the shared model; see :meth:`_backbone`."""
        from tabfm import (  # noqa: PLC0415 - optional backend, imported at use.
            tabfm_v1_0_0_pytorch as tabfm_v1,
        )
        import torch  # noqa: PLC0415 - optional backend, imported at use.

        self.checkpoint_ = _hub.get_checkpoint(
            self.backend, self.version
        ).download()
        # None keeps the stored float32 weights.
        dtype = torch.bfloat16 if self.mixed_precision_ else None
        load = tabfm_v1.load
        return _weights.network(
            _weights.key(
                self.backend,
                self.version,
                self.checkpoint_,
                self.device_,
                dtype or torch.float32,
                load,
            ),
            functools.partial(self._load_backbone, load, dtype),
        )

    def _load_backbone(self, load: Callable[..., Any], dtype: Any) -> Any:
        """Reads the backbone from the checkpoint; see :meth:`_backbone`.

        Upstream keeps a process-wide cache of its own, which
        :func:`lazy.clear_model_cache` could not empty; it is bypassed where
        the installed TabFM has the switch.
        """
        self._log(
            f"loading TabFM classification checkpoint from {self.checkpoint_}"
        )
        options: dict[str, Any] = {}
        if "use_cache" in inspect.signature(load).parameters:
            options["use_cache"] = False
        return load(
            model_type="classification",
            device=self.device_,
            checkpoint_path=str(self.checkpoint_),
            dtype=dtype,
            **options,
        )


def _install_command(requirement: str) -> list[str] | None:
    """The command replacing an installed package with ``requirement``.

    pip when this interpreter has it, else uv, whose environments have no
    pip; both reinstall even though the build and the release share a
    version number, and leave the dependencies alone.

    Args:
        requirement: A PEP 508 requirement, such as a ``name @ git+URL``.

    Returns:
        The command, or ``None`` when neither installer is available.
    """
    if util.find_spec("pip") is not None:
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            requirement,
        ]
    uv = shutil.which("uv")
    if uv is None:
        return None
    return [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall",
        "--no-deps",
        requirement,
    ]


def _in_uv_project() -> bool:
    """Whether the working directory is inside a project uv has locked."""
    here = pathlib.Path.cwd()
    return any((d / "uv.lock").is_file() for d in (here, *here.parents))


def _level_rows(
    member_rows: tuple[_typing.IntArray, ...], level: _typing.IntArray
) -> tuple[_typing.IntArray, ...]:
    """Each member's rows of a fine level, as positions in that level.

    Args:
        member_rows: Each member's sorted rows of the group's context.
        level: The sorted rows of the group's context in the coarse bin.

    Returns:
        For each member, the positions in ``level`` of its rows there.

    Examples:
        >>> bag, level = np.array([0, 2, 5]), np.array([2, 3, 5])
        >>> _level_rows((bag,), level)[0].tolist()
        [0, 2]
    """
    return tuple(np.flatnonzero(np.isin(level, rows)) for rows in member_rows)


def _member_views(
    classifier: Any, member_rows: tuple[_typing.IntArray, ...]
) -> list[Any]:
    """One-member copies of a fitted classifier, each on its member's rows.

    Member ``i`` of the classifier keeps what TabFM drew for it -- its norm
    method, column order and class shift, in the place TabFM's shuffle of its
    configurations put them -- and sees only ``member_rows[i]``, on which its
    standardisation, norm method and soft clip are refitted, exactly as
    ``EnsembleGenerator.fit`` fits them on a whole context. The labels keep
    the classifier's encoding, so every member answers over the same classes.

    Args:
        classifier: A fitted ``TabFMClassifier`` of ``len(member_rows)``
            members.
        member_rows: Each member's rows of the classifier's context, as
            positions, in member order.

    Returns:
        A classifier per member with any rows, in member order.
    """
    from tabfm.src import (  # noqa: PLC0415 - optional backend, imported at use.
        classifier_and_regressor as upstream,
    )

    generator = classifier.ensemble_generator_
    methods = list(generator.norm_methods_)
    n_members = len(member_rows)
    # Upstream deals the norm methods out to members cyclically, then groups
    # the members by method, in member order within each.
    cycle = (methods * -(-n_members // len(methods)))[:n_members]
    taken: collections.Counter[str] = collections.Counter()
    views = []
    for method, rows in zip(cycle, member_rows, strict=True):
        shuffle, shift, permutation, _ = generator.ensemble_configs_[method][
            taken[method]
        ]
        taken[method] += 1
        if not len(rows):
            continue
        member = copy.copy(generator)
        member.X_ = generator.X_[rows]
        member.y_ = generator.y_[rows]
        member.preprocessors_ = {
            method: upstream.PreprocessingPipeline(
                normalization_method=method,
                outlier_threshold=generator.outlier_threshold,
                random_state=generator.random_state,
            ).fit(member.X_)
        }

        def alone(value: Any, method: str = method) -> Any:
            return collections.OrderedDict([(method, [value])])

        member.ensemble_configs_ = alone((shuffle, shift, permutation, None))
        member.feature_shuffle_patterns_ = alone(shuffle)
        member.class_shift_offsets_ = alone(shift)
        member.cat_permutations_ = alone(permutation)
        member.row_subsample_patterns_ = alone(None)
        member.norm_methods_ = [method]
        member.n_estimators = 1
        view = copy.copy(classifier)
        view.ensemble_generator_ = member
        view.n_estimators = 1
        views.append(view)
    return views


def _is_auto(value: object) -> bool:
    """Whether a bin count is ``"auto"``."""
    return isinstance(value, str) and value == AUTO_BINS


def _is_grid(value: object) -> bool:
    """Whether a bin count is ``"grid"``."""
    return isinstance(value, str) and value == GRID_BINS


def _is_count(value: object) -> bool:
    """Whether ``value`` is an int (Python or NumPy, not a bool) >= 1."""
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool | np.bool_)
        and int(value) >= 1
    )


def _bin_labels(
    edges: _typing.FloatArray, values: _typing.FloatArray
) -> _typing.IntArray:
    """The bin each value falls in, ``edges[i] <= value < edges[i + 1]``.

    Of several zero-width bins at one value, a value there falls in the last,
    the one of non-zero width above them; values outside the edges go to the
    end bins.
    """
    n_bins = edges.size - 1
    return np.clip(
        np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1
    )


def _clip_to_support(
    y: _typing.FloatArray, support: tuple[float, float]
) -> _typing.FloatArray:
    """The targets clipped to the bins' range, warning if any lay outside it.

    Args:
        y: The context targets, shape ``(n,)``.
        support: The range the equal-mass bins span, (low, high).

    Returns:
        ``y`` clipped to ``support``, shape ``(n,)``.
    """
    low, high = support
    outside = int(np.count_nonzero((y < low) | (y > high)))
    if outside:
        warnings.warn(
            f"{outside} of {y.size} training targets lie outside the y_grid "
            f"range [{low:g}, {high:g}]; TabFMHistogram clips them to its "
            "ends, so their probability piles into the end bins. Pass a "
            "y_grid that covers the targets.",
            UserWarning,
            skip_file_prefixes=(_PACKAGE_PREFIX,),
        )
    return np.clip(y, low, high)


def _frame(features: _typing.FloatArray) -> pd.DataFrame:
    """Features as the DataFrame TabFM's preprocessing has always been given."""
    return pd.DataFrame(
        features, columns=[f"x{i}" for i in range(features.shape[1])]
    )


def _stage(progress: Any, dither: str, level: str, n_context: int) -> None:
    """Labels the progress bar with the stage about to run."""
    if progress is not None:
        progress.set_postfix(dither=dither, level=level, context=n_context)


def _done(progress: Any) -> None:
    """Advances the progress bar by one stage."""
    if progress is not None:
        progress.update(1)
