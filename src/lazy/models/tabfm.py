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

The uniform features map onto TabFM's own machinery: ``kv_cache`` onto the
streaming prefill/decode path (:mod:`lazy.models._icl_stream`),
``feature_shuffle`` and the transforms it has onto its classifier's own
shuffles and ``norm_methods``, and ``bag_size`` onto its per-member row cap.

Because the bins are equal-mass they are narrow where the targets are crowded,
so in the busy part of the distribution they are routinely *narrower* than
the output bin.
Mapping them onto the output grid is therefore done by exact, mass-conserving
integration (:meth:`lazy.grid.Grid.rebin`), not by sampling the density
at the output bin centres, which would drop whole bins and lose probability.

The ten-class limit constrains the hierarchy, never the output: ``z_grid`` can
have any number of bins, at any spacing, over any range.

There is deliberately no post-processing stage. Probability sharpening and
Gaussian smoothing were both tried and neither is used: TabFM's class
probabilities are already well calibrated in sharpness, and a Gaussian narrower
than the output bin cannot change a density tabulated on it.
"""

from __future__ import annotations

import inspect
import math
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

__all__ = [
    "TabFMHistogram",
    "TabFMPerformanceWarning",
    "prior_shift_em",
    "quantile_edges",
]


class TabFMPerformanceWarning(_ensemble.PerformanceWarning):
    """The prediction will be correct but far slower than it needs to be.

    Its own category so that it can be silenced on purpose --
    ``warnings.filterwarnings("ignore", category=TabFMPerformanceWarning)`` --
    without hiding anything else, and so that a test suite can assert it was
    raised.
    """


MAX_CLASSES = 10
"""TabFM's classifier ceiling, and hence the ceiling on each hierarchy level."""


_SLOW_PATH_WARNING = (
    "tabfm has no KV-cache API, so TabFMHistogram is falling back to "
    "its uncached inference path, which re-encodes the training "
    "context for every chunk of query rows -- roughly 26x the compute "
    "per query row (13.7 ms vs 0.53 ms per member-row). The answers "
    "agree up to rounding; only the runtime differs, so a large prediction "
    "will simply take far longer than expected. The KV-cache API "
    "ships in the repository build but not on PyPI:\n"
    "    pip install 'tabfm[pytorch] @ "
    "git+https://github.com/google-research/tabfm'\n"
    "Pass kv_cache=False to accept the slow path and silence this."
)


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

    Args:
        version: Which pinned TabFM checkpoint to load; see
            :func:`lazy.list_versions`. Recorded in ``provenance_``.
        n_coarse_bins: Classes at the first hierarchy level (at most ten).
        n_fine_bins: Classes within each coarse bin (at most ten). The
            density is built on ``n_coarse_bins * n_fine_bins`` equal-mass
            bins; the default 10 x 10 gives 100.
        n_estimators: Ensemble members per classifier. More members cost
            linearly more time and reduce member noise; 4 is enough for a
            smooth density.
        n_dither: Repeats of the whole hierarchy with bin edges shifted by
            ``d/n_dither`` of a bin, mixed with equal weights. 1 disables
            dithering; 3 is a good default when you can afford three times the
            compute.
        transforms: Per-member feature transforms: ``"auto"`` (TabFM's own
            ``none``/``power`` recipe), a recipe name, a transform name or a
            sequence of them; see :mod:`lazy.models._transforms`. TabFM's
            ``norm_methods`` implement ``none``, ``power``, ``quantile``,
            ``quantile_rtdl`` and ``robust``; the rest are scaffolded, each as
            a hierarchy of its own.
        feature_shuffle: Whether members see the columns in different orders
            (TabFM's own feature shuffles).
        bag_size: Context rows per member: an int count, a float fraction in
            (0, 1], or None for all of them. Native: each classifier's members
            subsample the same fraction of the rows it sees (TabFM's
            ``max_num_rows``).
        kv_cache: Prefill each member's context once and decode the queries
            against the cache (``True``; needs TabFM's repository build, and
            falls back with a :class:`TabFMPerformanceWarning` without it),
            or re-encode the context for every chunk of queries (``False``).
            TabFM computes in bfloat16, so the two paths agree to float
            rounding on the CPU but not on CUDA, whose kernels round
            differently for different batch shapes: there densities differ
            by up to a few per cent of their peak.
        z_grid: Default output grid: a :class:`lazy.grid.Grid`, an
            array of bin centres, ``"native"``, or None for the native grid
            (the union of every dither's bin edges). A constructor grid also
            sets the range the equal-mass bins span; without one they span the
            training values.
        prior_shift: ``"em"`` applies the label-shift correction of
            :func:`prior_shift_em` using the context's own bin fractions as
            the training prior, which is worth having when the context is a
            biased spectroscopic sample. ``None`` (default) leaves the
            posteriors alone.
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"cpu"``.
        random_state: Seed for TabFM's ensemble construction. None draws a
            fresh seed at fit, recorded as ``random_state_`` and in
            ``provenance_``.
        softmax_temperature: Temperature applied to the classifier logits. The
            upstream default of 0.9 is deliberately not 1.0 and should rarely
            be changed.
        chunk_size: Query rows handed to the upstream ``predict_proba`` at a
            time when ``kv_cache=False``; ``0`` does them in one pass. The
            in-context stage builds its keys and values from the context rows
            only, so a row's answer never depends on the other rows in its
            chunk; it is bit-identical on the CPU, and on CUDA changes by the
            bfloat16 rounding ``kv_cache`` describes. The cached path is
            bounded by ``query_block_rows`` and ``decode_chunk_rows``
            instead.
        member_batch_size: Ensemble members processed together on the cached
            path (and TabFM's own ``batch_size``). This and the next three are
            memory/throughput knobs that change the result only by the
            rounding ``kv_cache`` describes.
        decode_chunk_rows: Query rows decoded against the cache at a time.
        query_block_rows: Query rows whose member views are built at a time
            on the cached path, bounding host memory.
        keep_cache_on_device: Whether the prefilled cache stays on the device
            rather than round-tripping through host memory.
        progress: A progress bar over the in-context stages: ``"auto"`` shows
            it on a terminal or in a notebook, ``True`` always, ``False``
            never. It counts stages, ``n_dither * (1 + n_coarse_bins)`` of
            them, because each runs over every query row.
        verbose: Print per-level log messages to stdout.

    Attributes:
        grid_: The resolved default output grid.
        native_grid_: The union of every dither's equal-mass bin edges, a
            histogram-normalised grid set at fit.
        inference_: The path actually used: ``"stream"`` or
            ``"predict_proba"``.
        support_: The range the equal-mass bins span, (low, high).
        provenance_: Which weights, code and ensemble answered, as a dict.
        checkpoint_: The pinned checkpoint, a :class:`pathlib.Path`. TabFM
            loads its backbone on the first prediction, so this appears then.
        bin_prior_: The context's bin fractions from the last prediction,
            shape (n_coarse_bins * n_fine_bins,).
        n_context_: Context rows ``fit`` was given.

    Examples:
        >>> est = TabFMHistogram(n_estimators=2, n_dither=3)
        >>> est.n_dither
        3
    """

    backend = "tabfm"
    display_name = "TabFM"
    extra = "tabfm"
    native_output = "histogram"
    native_transforms = {
        "none": "none",
        "power": "power",
        "quantile": "quantile",
        "quantile_rtdl": "quantile_rtdl",
        "robust": "robust",
    }
    auto_tokens = ("none", "power")
    supports_native_bagging = True
    # bfloat16 on CUDA: chunking and the cache change the rounding.
    exact_chunking = False
    kv_cache_rtol = 5e-2
    chunks_queries = False

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v1.0",
        n_coarse_bins: int = 10,
        n_fine_bins: int = 10,
        n_estimators: int = 4,
        n_dither: int = 1,
        transforms: str | tuple[str, ...] = "auto",
        feature_shuffle: bool = True,
        bag_size: int | float | None = None,
        kv_cache: bool = True,
        z_grid: grid_lib.GridLike = None,
        prior_shift: str | None = None,
        device: str = "auto",
        random_state: int | None = 1,
        softmax_temperature: float = 0.9,
        chunk_size: int = 16_384,
        member_batch_size: int = 1,
        decode_chunk_rows: int = 16_384,
        query_block_rows: int = 262_144,
        keep_cache_on_device: bool = True,
        progress: _progress.Progress = "auto",
        verbose: bool = False,
    ):
        self.version = version
        self.n_coarse_bins = n_coarse_bins
        self.n_fine_bins = n_fine_bins
        self.n_estimators = n_estimators
        self.n_dither = n_dither
        self.transforms = transforms
        self.feature_shuffle = feature_shuffle
        self.bag_size = bag_size
        self.kv_cache = kv_cache
        self.z_grid = z_grid
        self.prior_shift = prior_shift
        self.device = device
        self.random_state = random_state
        self.softmax_temperature = softmax_temperature
        self.chunk_size = chunk_size
        self.member_batch_size = member_batch_size
        self.decode_chunk_rows = decode_chunk_rows
        self.query_block_rows = query_block_rows
        self.keep_cache_on_device = keep_cache_on_device
        self.progress = progress
        self.verbose = verbose

    # -- the per-backend interface -------------------------------------------

    def _import_backend(self) -> types.ModuleType:
        # A missing `tabfm` is the likeliest reason a first fit fails, and it
        # is reported before any parameter complaint.
        try:
            import tabfm  # noqa: PLC0415 - an optional, heavy extra.
        except ImportError as error:
            # Only the backend itself missing is a missing extra; anything
            # it fails to import in turn is reported as it is.
            if (error.name or "").partition(".")[0] != "tabfm":
                raise
            raise ImportError(
                "TabFMHistogram needs the tabfm backend: "
                "pip install 'lazy-tfm[tabfm]'"
            ) from error
        return tabfm

    def _check_backend_params(self) -> None:
        if max(self.n_coarse_bins, self.n_fine_bins) > MAX_CLASSES:
            raise ValueError(
                f"TabFM classification supports at most {MAX_CLASSES} classes"
                f" per level, got n_coarse_bins={self.n_coarse_bins},"
                f" n_fine_bins={self.n_fine_bins}"
            )
        if min(self.n_coarse_bins, self.n_fine_bins) < 2:
            raise ValueError("each hierarchy level needs at least two bins")
        if self.n_dither < 1:
            raise ValueError("n_dither must be at least 1")
        if self.prior_shift not in (None, "em"):
            raise ValueError("prior_shift must be None or 'em'")
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
        warnings.warn(_SLOW_PATH_WARNING, TabFMPerformanceWarning, stacklevel=5)
        return "predict_proba"

    def _load_checkpoint(self) -> None:
        # The backbone is several gigabytes and loads lazily on the first
        # prediction (see _backbone); fit only records what it will load.
        self.provenance_ = _hub.get_checkpoint(
            "tabfm", self.version
        ).provenance(device=self.device_)

    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> Any:
        if len(X) < self.n_coarse_bins * self.n_fine_bins:
            raise ValueError(
                f"context has {len(X)} rows, fewer than the "
                f"{self.n_coarse_bins * self.n_fine_bins} bins asked for"
            )
        if self.inference_ == "predict_proba":
            self.kv_cache_ = False
        self.support_ = self._support(y)
        y = _clip_to_support(y, self.support_)
        self.X_context_ = _frame(X)
        self.z_context_ = y
        return {
            "X": _frame(X),
            "z": y,
            "group": group,
            "bag_fraction": (
                None
                if group.member_rows is None
                else len(group.member_rows[0]) / len(X)
            ),
        }

    def _support(self, z: _typing.FloatArray) -> tuple[float, float]:
        """The range the equal-mass bins span: the constructor grid's or z's."""
        if self.z_grid is not None and not isinstance(self.z_grid, str):
            fixed = grid_lib.as_grid(self.z_grid)
            return fixed.z_min, fixed.z_max
        return float(z.min()), float(z.max())

    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.Distribution:
        model = self._backbone()
        query = _frame(X)
        shifts = [d / self.n_dither for d in range(self.n_dither)]
        parts = []
        with _progress.bar(
            self.progress,
            total=self.n_dither * (1 + self.n_coarse_bins),
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
        shifts = [d / self.n_dither for d in range(self.n_dither)]
        edges = np.unique(
            np.concatenate(
                [self._edges(self.z_context_, shift)[0] for shift in shifts]
            )
        )
        if edges.size < 3:
            # A constant target has one bin; a grid needs two, and halving
            # it changes no density.
            edges = np.r_[edges[0], edges.mean(), edges[-1]]
        return grid_lib.Grid.from_edges(edges, normalization="histogram")

    # -- the hierarchy ------------------------------------------------------

    def _edges(
        self, z: _typing.FloatArray, shift: float
    ) -> tuple[
        _typing.FloatArray, _typing.FloatArray, list[_typing.FloatArray]
    ]:
        """Returns a tuple (all edges, coarse edges, fine edges per coarse bin).

        The equal-mass edges for one dither, spanning ``support_``; the top
        edge is nudged up so the largest value falls inside. Tied targets
        (discrete, zero-inflated or constant ones) collapse neighbouring
        quantiles, which leaves zero-width, empty bins: an empty coarse bin
        gets evenly spaced fine edges across itself, and every empty bin
        gets zero probability.
        """
        low, high = self.support_
        # A constant target still gets a bin of non-zero width, on its scale.
        span = high - low if high > low else max(abs(high), 1.0)
        coarse = quantile_edges(
            z, self.n_coarse_bins, low, high + 1e-6 * span, shift
        )
        labels = _bin_labels(coarse, z)
        fine = [
            quantile_edges(
                z[labels == j],
                self.n_fine_bins,
                coarse[j],
                coarse[j + 1],
                shift,
            )
            for j in range(self.n_coarse_bins)
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
            handle: The fitted group: its context and members.
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
        z, X_context = handle["z"], handle["X"]
        edges, coarse_edges, fine_edges = self._edges(z, shift)
        coarse = _bin_labels(coarse_edges, z)
        _stage(progress, dither, "coarse", z.size)
        p_coarse = self._class_probabilities(
            model,
            handle,
            X_context,
            coarse,
            self.n_coarse_bins,
            X_query,
            handle["group"].seed,
        )
        _done(progress)
        prior: list[_typing.FloatArray] = []
        blocks: list[_typing.FloatArray] = []
        for j in range(self.n_coarse_bins):
            rows = coarse == j
            fine = _bin_labels(fine_edges[j], z[rows])
            _stage(
                progress,
                dither,
                f"fine {j + 1}/{self.n_coarse_bins}",
                int(rows.sum()),
            )
            p_fine = self._class_probabilities(
                model,
                handle,
                X_context.iloc[rows],
                fine,
                self.n_fine_bins,
                X_query,
                handle["group"].seed + 1 + j,
            )
            _done(progress)
            prior.append(
                np.bincount(fine, minlength=self.n_fine_bins) / max(z.size, 1)
            )
            blocks.append(p_fine * p_coarse[:, [j]])
            self._log(
                f"  coarse bin {j + 1}/{self.n_coarse_bins}"
                f" ({int(rows.sum())} context rows)"
            )
        return np.concatenate(blocks, axis=1), edges, np.concatenate(prior)

    def _class_probabilities(
        self,
        model: Any,
        handle: Any,
        X_context: pd.DataFrame,
        labels: _typing.IntArray,
        n_classes: int,
        X_query: pd.DataFrame,
        seed: int,
    ) -> _typing.FloatArray:
        """Member-averaged class posteriors, ``(n_query, n_classes)``.

        A level with no context rows gives every class zero probability, and
        one whose rows all share a class gives that class all of it; neither
        has anything for a classifier to learn, so none is run.
        """
        present = np.unique(labels)
        if present.size < 2:
            full = np.zeros((len(X_query), n_classes))
            full[:, present.astype(int)] = 1.0
            return full
        max_rows = (
            None
            if handle["bag_fraction"] is None
            else max(1, math.ceil(handle["bag_fraction"] * len(X_context)))
        )
        classifier = self._classifier(model, handle["group"], seed, max_rows)
        classifier.fit(X_context.reset_index(drop=True), labels)
        classes = np.asarray(classifier.classes_)

        if self.inference_ == "stream":
            logits = _icl_stream.classification_logits(
                classifier,
                model,
                {"query": X_query},
                member_batch_size=self.member_batch_size,
                query_block_rows=self.query_block_rows,
                decode_chunk_rows=self.decode_chunk_rows,
                keep_cache_on_device=self.keep_cache_on_device,
            )
            probs = _icl_stream.softmax(
                logits["query"]["mean_logits"], self.softmax_temperature
            )
        else:
            probs = self._predict_proba_chunked(classifier, X_query)

        # A class with no context rows at all never appears in `classes_`; it
        # gets zero probability rather than shifting every later column.
        full = np.zeros((probs.shape[0], n_classes))
        full[:, classes.astype(int)] = probs
        return full

    def _predict_proba_chunked(
        self, classifier: Any, X_query: pd.DataFrame
    ) -> _typing.FloatArray:
        """Upstream ``predict_proba``, a bounded number of query rows at a time.

        ``predict_proba`` materialises ``n_members x n_query x n_features`` in
        one go, which a survey-sized query set cannot afford. Feeding it chunks
        caps that at ``n_members x chunk_size x n_features``. The in-context
        stage builds its keys and values from the context rows alone, so
        query rows never influence one another: on the CPU the result is
        bit-identical to a single pass, and on CUDA it differs by bfloat16
        rounding, the kernels batching differently. The context forward pass
        is repeated per chunk, which is the price.
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
        max_rows: int | None,
    ) -> Any:
        """A ``TabFMClassifier``, passing only the knobs this build understands.

        The cache arguments exist on builds with the KV-cache API and not on
        the PyPI release; sending them unconditionally is a ``TypeError``. The
        uniform features are passed only when asked for, so the default
        recipe is upstream's own.
        """
        tabfm = self._import_backend()
        kwargs: dict[str, Any] = {
            "model": model,
            "n_estimators": group.n_members,
            "batch_size": self.member_batch_size,
            "random_state": seed,
            "softmax_temperature": self.softmax_temperature,
            "binary_calibration_method": None,
            "verbose": False,
        }
        if group.native_transforms is not None:
            # Upstream cycles its norm methods over the members, so the
            # per-member list runs exactly as planned, repeats and all.
            kwargs["norm_methods"] = list(group.native_transforms)
        if not group.feature_shuffle:
            kwargs["feat_shuffle_method"] = "none"
        if max_rows is not None:
            kwargs["max_num_rows"] = max_rows
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

        The checkpoint is several gigabytes, so it is loaded once per estimator
        and reused across calls. The cache is keyed by ``version`` so that
        changing it can never leave a prediction running on the previous
        model's weights, and it is dropped on pickling: an unpickled estimator
        reloads from the local cache on next use.
        """
        cached_version, model = getattr(self, "_backbone_cache", (None, None))
        if model is None or cached_version != self.version:
            from tabfm import (  # noqa: PLC0415 - optional backend, imported at use.
                tabfm_v1_0_0_pytorch as tabfm_v1,
            )

            self.checkpoint_ = _hub.get_checkpoint(
                "tabfm", self.version
            ).download()
            self._log(
                "loading TabFM classification checkpoint from"
                f" {self.checkpoint_}"
            )
            model = tabfm_v1.load(
                model_type="classification",
                device=self.device_,
                checkpoint_path=str(self.checkpoint_),
            )
            self._backbone_cache = (self.version, model)
        return model

    def __getstate__(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items() if k != "_backbone_cache"
        }


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
    z: _typing.FloatArray, support: tuple[float, float]
) -> _typing.FloatArray:
    """The targets clipped to the bins' range, warning if any lay outside it.

    Args:
        z: The context targets, shape ``(n,)``.
        support: The range the equal-mass bins span, (low, high).

    Returns:
        ``z`` clipped to ``support``, shape ``(n,)``.
    """
    low, high = support
    outside = int(np.count_nonzero((z < low) | (z > high)))
    if outside:
        warnings.warn(
            f"{outside} of {z.size} training targets lie outside the z_grid "
            f"range [{low:g}, {high:g}]; TabFMHistogram clips them to its "
            "ends, so their probability piles into the end bins. Pass a "
            "z_grid that covers the targets.",
            UserWarning,
            stacklevel=5,
        )
    return np.clip(z, low, high)


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
