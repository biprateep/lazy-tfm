# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Photo-z densities from TabFM's in-context classifier, as a bin hierarchy.

TabFM (Google, 2026) is an in-context tabular foundation model: it is never
trained on your data, it reads labelled rows as context and answers queries
against them. Its classifier handles at most ten classes, which is far short of
the resolution a photo-z PDF needs, so :class:`TabFMHistogram` builds the
density as a two-level hierarchy of *equal-mass* redshift bins:

* one classifier over ``n_coarse_bins`` quantile bins of the context redshifts;
* one classifier per coarse bin over ``n_fine_bins`` quantile bins inside it,
  with only that bin's context galaxies as context.

Then ``P(bin | x) = P(coarse | x) * P(fine | coarse, x)``, and the density is
``P(bin) / width``. With ``n_dither > 1`` the whole hierarchy is repeated with
the bin edges shifted in quantile space and the densities averaged -- an
averaged shifted histogram, which removes the edge artifacts a single binning
leaves behind.

Because the bins are equal-mass they are narrow where galaxies are crowded, so
in the busy part of N(z) they are routinely *narrower* than the output bin.
Mapping them onto the output grid is therefore done by exact, mass-conserving
integration (:meth:`lazy.grid.RedshiftGrid.rebin`), not by sampling the density
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
from typing import Any
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from lazy import _typing
from lazy import base
from lazy import grid as grid_lib
from lazy.models import _device
from lazy.models import _hub
from lazy.models import _icl_stream
from lazy.models import _progress

__all__ = [
    "TabFMHistogram",
    "TabFMPerformanceWarning",
    "prior_shift_em",
    "quantile_edges",
]


class TabFMPerformanceWarning(UserWarning):
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
    "are the same; only the runtime differs, so a large prediction "
    "will simply take far longer than expected. The KV-cache API "
    "ships in the repository build but not on PyPI:\n"
    "    pip install 'tabfm[pytorch] @ "
    "git+https://github.com/google-research/tabfm'\n"
    "Pass inference='predict_proba' to accept the slow path and "
    "silence this."
)


def quantile_edges(
    redshifts: npt.ArrayLike,
    n_bins: int,
    lo: float,
    hi: float,
    shift: float = 0.0,
) -> _typing.FloatArray:
    """Equal-mass bin edges over ``[lo, hi]``.

    Args:
        redshifts: The redshifts whose quantiles set the edges, shape
            ``(n,)``.
        n_bins: Number of bins.
        lo: The first edge, replacing the lowest quantile.
        hi: The last edge, replacing the highest quantile.
        shift: Moves the interior edges in quantile space, in bins, which is
            what dithering varies between repeats.

    Returns:
        Non-decreasing edges, shape ``(n_bins + 1,)``.

    Examples:
        >>> quantile_edges(np.linspace(0, 1, 101), 2, 0.0, 1.0).tolist()
        [0.0, 0.5, 1.0]
    """
    values = np.asarray(redshifts, dtype=float)
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

    A classifier's posteriors carry the redshift distribution of its context.
    When the context is a spectroscopic training set and the queries are a
    photometric sample, those distributions differ, and the posteriors inherit
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


class TabFMHistogram(base.BasePhotoZEstimator):
    """Conditional density from TabFM's classifier over a hierarchy of bins.

    Args:
        version: Which pinned TabFM checkpoint to load; see
            :func:`lazy.list_versions`. Recorded in ``provenance_``.
        n_coarse_bins: Classes at the coarse hierarchy level. The product
            ``n_coarse_bins * n_fine_bins`` is the number of equal-mass bins
            the density is built on. Each is capped at ten by the backbone.
            The default 10 x 10 gives 100 bins.
        n_fine_bins: Classes at the fine level, inside each coarse bin; see
            ``n_coarse_bins``.
        n_estimators: TabFM ensemble members per classifier. More members
            cost linearly more time and reduce member noise; 4 is enough for
            a smooth density.
        n_dither: Repeats of the whole hierarchy with bin edges shifted by
            ``d/n_dither`` of a bin, averaged. 1 disables dithering; 3 is a
            good default when you can afford three times the compute.
        z_grid: Output grid: a :class:`lazy.grid.RedshiftGrid`, an array of
            bin centres, or ``None`` for :data:`lazy.grid.DC1_GRID`.
        prior_shift: ``"em"`` applies the label-shift correction of
            :func:`prior_shift_em` using the context's own bin fractions as
            the training prior, which is worth having when the context is a
            biased spectroscopic sample. ``None`` (default) leaves the
            posteriors alone.
        inference: How query rows are pushed through the backbone.

            ``"auto"`` (default)
                Use the memory-bounded streaming decoder when the installed
                ``tabfm`` provides the KV-cache API, and the upstream
                ``predict_proba`` otherwise.
            ``"stream"``
                Force the streaming decoder; raise if it is unavailable.
                This prefills each ensemble member's context once and
                decodes queries in chunks, so peak memory does not grow with
                the size of the query set -- which is what makes a
                survey-sized run possible at all.
            ``"predict_proba"``
                Force the upstream API. Same estimator and same numbers (its
                defaults ``average_logits=True`` and
                ``softmax_temperature=0.9`` are exactly what the streaming
                path computes), but it materialises every member's view of
                every query row at once.

            The KV-cache API is not in the PyPI release of ``tabfm``; it
            only exists in later builds from the repository. Install one
            with
            ``pip install 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm'``
            if you need the streaming path.
        device: ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``,
            ``"cpu"``.
        random_state: Seed for TabFM's ensemble construction.
        softmax_temperature: Temperature applied to the classifier logits.
            The upstream default of 0.9 is deliberately not 1.0 and should
            rarely be changed.
        chunk_size: Query rows pushed through the backbone at a time, on
            either inference path. ``0`` does them in one pass. This is what
            bounds peak memory: the released ``tabfm`` builds every ensemble
            member's view of every query row up front, so a survey-sized
            query set in one pass is tens of gigabytes. Chunking is
            numerically exact -- the in-context stage builds its keys and
            values from the context rows only, so a query row's prediction
            never depends on which other query rows share its chunk -- and
            is verified bit-identical in the test suite.
        member_batch_size: Ensemble members processed together on the
            streaming path (and TabFM's own ``batch_size``). This and the
            next three are memory/throughput knobs: they trade host and
            device memory against the number of passes; none of them changes
            the result.
        decode_chunk_rows: Query rows decoded per forward pass on the
            streaming path.
        query_block_rows: Query rows whose member views are built at once on
            the streaming path.
        keep_cache_on_device: Keep each member batch's K/V cache on the
            device rather than round-tripping it through host memory.
        progress: A progress bar over the in-context stages: ``"auto"`` (the
            default) shows it on a terminal or in a notebook and not when
            output goes to a file, ``True`` always, ``False`` never. It
            counts stages rather than galaxies because every stage -- per
            dither, one coarse classification and then one fine one per
            coarse bin, ``n_dither * (1 + n_coarse_bins)`` in all -- runs
            over *every* query row, and shows which dither and level is
            running and how many context rows that stage has.
        verbose: Print per-level log messages to stdout.

    Attributes:
        grid_: The resolved output grid, a :class:`lazy.grid.RedshiftGrid`.
        inference_: The inference path actually chosen: ``"stream"`` or
            ``"predict_proba"``.
        provenance_: Which weights and which code answered: backend,
            version, repository, revision, package versions and device. See
            :meth:`lazy.models._hub.Checkpoint.provenance`.
        checkpoint_: The pinned checkpoint the weights were loaded from, a
            :class:`pathlib.Path`. Unlike the other two backends TabFM loads
            its backbone on the first prediction rather than at ``fit``, so
            this appears then.
        X_context_: The context rows, a :class:`pandas.DataFrame` kept as
            given -- "fitting" an in-context model stores the context rather
            than learning weights.
        z_context_: The context redshifts, shape ``(n_context,)``.
        bin_prior_: Context bin fractions from the last prediction, available
            after :meth:`predict_pdf`, shape
            ``(n_coarse_bins * n_fine_bins,)``. This is the prior the raw
            posteriors carry.

    Examples:
        >>> est = TabFMHistogram(n_estimators=4, n_dither=3)
        >>> est.n_coarse_bins * est.n_fine_bins
        100
    """

    backend = "tabfm"

    def __init__(
        self,
        *,
        version: str = "v1.0",
        n_coarse_bins: int = 10,
        n_fine_bins: int = 10,
        n_estimators: int = 4,
        n_dither: int = 1,
        z_grid: grid_lib.GridLike = None,
        prior_shift: str | None = None,
        inference: str = "auto",
        device: str = "auto",
        random_state: int = 1,
        softmax_temperature: float = 0.9,
        chunk_size: int = 16_384,
        member_batch_size: int = 1,
        decode_chunk_rows: int = 16_384,
        query_block_rows: int = 262_144,
        keep_cache_on_device: bool = True,
        progress: _progress.Progress = "auto",
        verbose: bool = False,
    ):
        """Stores the settings; see the class docstring for each one."""
        self.version = version
        self.n_coarse_bins = n_coarse_bins
        self.n_fine_bins = n_fine_bins
        self.n_estimators = n_estimators
        self.n_dither = n_dither
        self.z_grid = z_grid
        self.prior_shift = prior_shift
        self.inference = inference
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

    # -- estimator protocol -------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        # Resolve the backend first. A missing `tabfm` is the likeliest reason
        # a first fit fails, and telling someone their context is too small
        # for their bin count -- when the real problem is that the extra is
        # not installed -- sends them off fixing the wrong thing.
        self.inference_ = self._resolve_inference()
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
        if self.chunk_size < 0:
            raise ValueError(
                "chunk_size must be non-negative (0 means one pass)"
            )
        if self.prior_shift not in (None, "em"):
            raise ValueError("prior_shift must be None or 'em'")
        if len(X) < self.n_coarse_bins * self.n_fine_bins:
            raise ValueError(
                f"context has {len(X)} rows, fewer than the "
                f"{self.n_coarse_bins * self.n_fine_bins} bins asked for"
            )
        _progress.check_progress(self.progress)
        self.device_ = _device.resolve_device(self.device)
        self.provenance_ = _hub.get_checkpoint(
            "tabfm", self.version
        ).provenance(device=self.device_)
        self.X_context_ = X
        self.z_context_ = y

    def _resolve_inference(self) -> str:
        """Pick the inference path, failing at fit time, not mid-prediction."""
        if self.inference not in ("auto", "stream", "predict_proba"):
            raise ValueError(
                "inference must be 'auto', 'stream' or 'predict_proba'"
            )
        try:
            import tabfm  # noqa: F401, PLC0415 - probes the optional backend.
        except ImportError as error:
            raise ImportError(
                "TabFMHistogram needs the tabfm backend: "
                "pip install 'lazy-photoz[tabfm]'"
            ) from error

        available = _icl_stream.streaming_available()
        if self.inference == "stream" and not available:
            raise RuntimeError(
                "inference='stream' needs the KV-cache API, which the PyPI "
                "release of tabfm does not provide. Install a build from the "
                "repository (pip install 'tabfm[pytorch] @ "
                "git+https://github.com/google-research/tabfm') "
                "or use inference='auto'."
            )
        if self.inference == "predict_proba":
            return "predict_proba"  # asked for explicitly; say nothing
        if not available:
            # Warn, and do not be shy about it. The fallback is correct but it
            # re-runs the context forward pass for every chunk of query rows,
            # which measures ~13.7 ms per member-row against ~0.53 ms on the
            # cached path -- around twenty-six times the work. On a large
            # query set that is the difference between one hour and a day,
            # and because both paths produce the same answer the only symptom
            # is a run that never seems to end. Nothing else reports it, so
            # this does.
            warnings.warn(
                _SLOW_PATH_WARNING, TabFMPerformanceWarning, stacklevel=3
            )
            return "predict_proba"
        return "stream"

    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.RedshiftGrid
    ) -> _typing.FloatArray:
        model = self._backbone()
        shifts = [d / self.n_dither for d in range(self.n_dither)]
        densities = []
        stages = self.n_dither * (1 + self.n_coarse_bins)
        with _progress.bar(
            self.progress,
            total=stages,
            desc=f"TabFM {self.version} ({len(X):,} gal)",
            unit="stage",
        ) as progress:
            for i, shift in enumerate(shifts):
                self._log(
                    f"dither {i + 1}/{self.n_dither}"
                    f" (edge shift {shift:.2f} bins)"
                )
                probs, edges, prior = self._hierarchy(
                    model,
                    X,
                    shift,
                    grid,
                    progress=progress,
                    dither=f"{i + 1}/{self.n_dither}",
                )
                if self.prior_shift == "em":
                    probs, _ = prior_shift_em(probs, prior)
                densities.append(grid.rebin(probs, edges))
        self.bin_prior_ = prior
        return np.mean(densities, axis=0)

    # -- the hierarchy ------------------------------------------------------

    def _hierarchy(
        self,
        model: Any,
        X_query: pd.DataFrame,
        shift: float,
        grid: grid_lib.RedshiftGrid,
        *,
        progress: Any = None,
        dither: str = "1/1",
    ) -> tuple[_typing.FloatArray, _typing.FloatArray, _typing.FloatArray]:
        """Bin posteriors for one set of dithered edges.

        Args:
            model: The loaded TabFM backbone.
            X_query: The query rows.
            shift: Edge shift in quantile space, in bins.
            grid: The output grid, whose range the edges span.
            progress: The ``tqdm`` bar :meth:`_predict_pdf` opened, advanced
                once per in-context stage; ``None`` draws nothing.
            dither: Which dither this is, for the bar's label.

        Returns:
            A tuple ``(probs, edges, context_bin_prior)``: posteriors, shape
            ``(n_query, n_bins)``; bin edges, shape ``(n_bins + 1,)``; and
            the context's bin fractions, shape ``(n_bins,)``.
        """

        def stage(level: str, n_context: int) -> None:
            if progress is not None:
                progress.set_postfix(
                    dither=dither, level=level, context=n_context
                )

        def done() -> None:
            if progress is not None:
                progress.update(1)

        z = self.z_context_
        span = grid.z_max - grid.z_min
        coarse_edges = quantile_edges(
            z, self.n_coarse_bins, grid.z_min, grid.z_max + 1e-6 * span, shift
        )
        coarse = np.clip(
            np.searchsorted(coarse_edges, z, side="right") - 1,
            0,
            self.n_coarse_bins - 1,
        )
        stage("coarse", z.size)
        p_coarse = self._class_probabilities(
            model, self.X_context_, coarse, X_query, self.random_state
        )
        done()

        edges: list[_typing.FloatArray] = []
        prior: list[_typing.FloatArray] = []
        blocks: list[_typing.FloatArray] = []
        for j in range(self.n_coarse_bins):
            rows = coarse == j
            fine_edges = quantile_edges(
                z[rows],
                self.n_fine_bins,
                coarse_edges[j],
                coarse_edges[j + 1],
                shift,
            )
            fine = np.clip(
                np.searchsorted(fine_edges, z[rows], side="right") - 1,
                0,
                self.n_fine_bins - 1,
            )
            stage(f"fine {j + 1}/{self.n_coarse_bins}", int(rows.sum()))
            p_fine = self._class_probabilities(
                model,
                self.X_context_.iloc[rows],
                fine,
                X_query,
                self.random_state + 1 + j,
            )
            done()
            edges.append(fine_edges[:-1])
            prior.append(
                np.bincount(fine, minlength=self.n_fine_bins) / max(z.size, 1)
            )
            blocks.append(p_fine * p_coarse[:, [j]])
            self._log(
                f"  coarse bin {j + 1}/{self.n_coarse_bins}"
                f" ({int(rows.sum())} context rows)"
            )
        return (
            np.concatenate(blocks, axis=1),
            np.r_[np.concatenate(edges), coarse_edges[-1]],
            np.concatenate(prior),
        )

    def _class_probabilities(
        self,
        model: Any,
        X_context: pd.DataFrame,
        labels: _typing.IntArray,
        X_query: pd.DataFrame,
        seed: int,
    ) -> _typing.FloatArray:
        """Member-averaged class posteriors, ``(n_query, n_labels)``."""
        classifier = self._classifier(model, seed)
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
            probs = np.asarray(
                classifier.predict_proba(X_query.reset_index(drop=True)),
                dtype=float,
            )

        # A class with no context rows at all never appears in `classes_`; it
        # gets zero probability rather than shifting every later column.
        full = np.zeros((probs.shape[0], int(labels.max()) + 1))
        full[:, classes.astype(int)] = probs
        return full

    def _predict_proba_chunked(
        self, classifier: Any, X_query: pd.DataFrame
    ) -> _typing.FloatArray:
        """Upstream ``predict_proba``, a bounded number of query rows at a time.

        ``predict_proba`` materialises ``n_members x n_query x n_features`` in
        one go, which a survey-sized query set cannot afford. Feeding it chunks
        caps that at ``n_members x chunk_size x n_features``. The result is
        bit-identical to a single pass (``tests/lazy/test_backends.py``): the
        in-context stage builds its keys and values from the context rows
        alone, so query rows never influence one another. The context forward
        pass is repeated per chunk, which is the price.
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

    def _classifier(self, model: Any, seed: int) -> Any:
        """A ``TabFMClassifier``, passing only the knobs this build understands.

        The cache arguments exist on builds with the KV-cache API and not on
        the PyPI release; sending them unconditionally is a ``TypeError``.
        """
        import tabfm  # noqa: PLC0415 - optional backend, imported at use.

        kwargs = {
            "model": model,
            "n_estimators": self.n_estimators,
            "batch_size": self.member_batch_size,
            "random_state": seed,
            "softmax_temperature": self.softmax_temperature,
            "binary_calibration_method": None,
            "verbose": False,
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

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[TabFMHistogram] {message}", flush=True)
