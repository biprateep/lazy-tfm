"""Photo-z densities from TabPFN-3's bar distribution.

TabPFN-3 (Prior Labs, 2026) is an in-context tabular foundation model, and its
regressor does not answer with a number: internally it is a classifier over a
fixed set of *buckets* of the target, and its native output is the probability
mass in each one -- a ``FullSupportBarDistribution``, in the upstream name. The
bucket borders are placed by the distribution of the context targets, so they
are narrow where galaxies are crowded and wide in the tails.

That is already a conditional density estimate, and it is the same object
:class:`lazy.models.tabfm.TabFMHistogram` has to assemble by hand from a
hierarchy of ten-class classifiers -- here one forward pass produces it, and
with five thousand buckets rather than a hundred. So :class:`TabPFNBarDistribution`
does no post-processing: it takes the bucket masses and maps them onto the
output grid by exact, mass-conserving integration
(:meth:`lazy.grid.RedshiftGrid.rebin`).

The bucket masses, rather than the quantiles :class:`lazy.models.tabicl.TabICLQuantile`
is restricted to, because here the quantiles are the derived quantity: upstream
computes them by inverting this same piecewise-uniform bucket CDF, so going
through them would only re-encode the density at a coarser resolution than the
model actually has.

Mass the model places outside ``[z_min, z_max]`` is dropped and each row
renormalised, the same convention every backend here follows. The outermost two
buckets are the ones to know about: they carry the distribution's tails, and
upstream's own CDF -- the one its quantiles and its mean come from -- treats
them as uniform, which is what ``rebin`` reproduces.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lazy.base import BasePhotoZEstimator
from lazy.models._device import resolve_device
from lazy.models._hub import get_checkpoint
from lazy.models._progress import Progress, bar, check_progress

__all__ = [
    "FIT_MODES",
    "TabPFNBarDistribution",
    "bucket_masses",
    "path_for_tabpfn",
]

#: ``fit_mode`` values this backend accepts. Upstream also has ``"batched"``,
#: which belongs to ``predict_batched`` -- a different call than the one made
#: here -- so it is not among them.
FIT_MODES = ("low_memory", "fit_preprocessors", "fit_with_cache")


class TabPFNBarDistribution(BasePhotoZEstimator):
    """Conditional density from the bucket masses TabPFN-3's regressor predicts.

    Parameters
    ----------
    version
        Which TabPFN to run, in upstream's own vocabulary: ``"v2"``,
        ``"v2.5"``, ``"v2.6"``, ``"v3"`` (the default), ``"v3.5"`` or
        ``"v3.5-fast"``. Each is a separately pinned checkpoint
        (:func:`lazy.list_versions`), so sweeping this parameter compares model
        versions on equal terms, and ``provenance_`` records which one answered.
        They differ in more than accuracy: ``"v2"`` is the only one under a
        commercial-use licence, and it and ``"v2.5"`` declare much smaller
        context limits than v3 (see ``ignore_pretraining_limits``).
    n_estimators
        TabPFN ensemble members: repeats of the forward pass over differently
        preprocessed views of the same data, averaged. Costs scale linearly.
        ``"auto"`` defers to the count the checkpoint names.
    z_grid
        Output grid: a :class:`lazy.grid.RedshiftGrid`, an array of bin
        centres, or ``None`` for :data:`lazy.grid.DC1_GRID`.
    device
        ``"auto"`` (CUDA if available), ``"cuda"``, ``"cuda:1"``, ``"cpu"``.
    random_state
        Seed for TabPFN's ensemble construction.
    softmax_temperature
        Temperature on the bucket logits, which sets how sharp the densities
        are. ``"auto"`` takes the checkpoint's own value, which is the one it
        was evaluated with; lower sharpens, higher broadens.
    fit_mode
        What ``fit`` precomputes, trading memory for speed at predict time.
        ``"low_memory"`` keeps nothing, ``"fit_preprocessors"`` (the default,
        and upstream's) caches the fitted preprocessors and the transformed
        context, and ``"fit_with_cache"`` additionally caches the context's
        key/value tensors, so each chunk of query rows skips the context
        forward pass -- the analogue of ``inference="stream"`` on
        :class:`~lazy.models.tabfm.TabFMHistogram`, and worth its memory when
        the query set is much larger than the context.
    ignore_pretraining_limits
        Pass ``True`` to run a context larger than the row count the checkpoint
        declares it was pretrained for, which otherwise raises. Predictions
        beyond that limit are extrapolation, so this is opt-in rather than the
        default -- though TabPFN-3 declares a million rows, so a photo-z
        context is unlikely to reach it.
    chunk_size
        Query rows predicted at a time, to bound peak memory. ``0`` does them
        in one pass. Chunking is numerically exact -- TabPFN's attention builds
        its keys and values from the context rows alone, and its preprocessors
        are fitted on the context at ``fit`` time, so a query row's prediction
        never depends on which other query rows share its chunk -- and is
        verified bit-identical in the test suite.
    progress
        A progress bar over the query galaxies: ``"auto"`` (the default) shows
        it on a terminal or in a notebook and not when output goes to a file,
        ``True`` always, ``False`` never. It advances one chunk at a time and
        shows the context size and how many buckets the bar distribution has.
    verbose
        Print log messages to stdout.

    Attributes
    ----------
    grid_ : lazy.grid.RedshiftGrid
        The resolved output grid.
    checkpoint_ : pathlib.Path
        The pinned checkpoint file the weights were loaded from.
    provenance_ : dict
        Which weights and which code answered: backend, version, repository,
        revision, package versions and device. See
        :meth:`lazy.models._hub.Checkpoint.provenance`.
    regressor_ : object
        The fitted ``tabpfn.TabPFNRegressor``.
    borders_ : ndarray
        The bar distribution's bucket borders, in redshift. Set by the first
        prediction; fixed by the context redshifts, so every query row and
        every chunk shares them.
    n_buckets_ : int
        How many buckets the bar distribution has, ``borders_.size - 1``.
    n_context_ : int
        Context rows ``fit`` was given.

    Examples
    --------
    >>> est = TabPFNBarDistribution(n_estimators=8, chunk_size=50_000)
    >>> est.chunk_size
    50000
    >>> TabPFNBarDistribution(version="v2.5").name_
    'tabpfn:v2.5'
    """

    backend = "tabpfn"

    def __init__(
        self,
        *,
        version: str = "v3",
        n_estimators: int | str = 8,
        z_grid=None,
        device: str = "auto",
        random_state: int = 42,
        softmax_temperature: float | str = "auto",
        fit_mode: str = "fit_preprocessors",
        ignore_pretraining_limits: bool = False,
        chunk_size: int = 16_384,
        progress: Progress = "auto",
        verbose: bool = False,
    ):
        self.version = version
        self.n_estimators = n_estimators
        self.z_grid = z_grid
        self.device = device
        self.random_state = random_state
        self.softmax_temperature = softmax_temperature
        self.fit_mode = fit_mode
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.chunk_size = chunk_size
        self.progress = progress
        self.verbose = verbose

    # -- estimator protocol -------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: NDArray[np.float64]) -> None:
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as error:
            raise ImportError(
                "TabPFNBarDistribution needs the tabpfn backend: pip install 'lazy-photoz[tabpfn]'"
            ) from error

        if self.chunk_size < 0:
            raise ValueError(
                "chunk_size must be non-negative (0 means one pass)"
            )
        if self.fit_mode not in FIT_MODES:
            raise ValueError(
                f"fit_mode must be one of {FIT_MODES}, got {self.fit_mode!r}"
            )
        check_progress(self.progress)
        self.device_ = resolve_device(self.device)

        # Fetch the weights ourselves and hand over the path, rather than
        # letting TabPFN resolve its own default. Its default is the newest
        # model version it knows about, which changes with the package -- so
        # `version` would not mean anything, the checkpoint
        # `lazy.download_checkpoint("tabpfn")` pre-fetches would stop being the
        # one actually loaded, and an upgrade of `tabpfn` would silently swap
        # the model. This way `CHECKPOINTS` is the authority on which TabPFN
        # this is.
        spec = get_checkpoint("tabpfn", self.version)
        self.checkpoint_ = spec.download()
        self.provenance_ = spec.provenance(device=self.device_)
        self._log(
            f"fitting TabPFN {self.version} on {len(X)} context rows "
            f"({self.device_}), {self.checkpoint_.name}"
        )
        regressor = TabPFNRegressor(
            n_estimators=self.n_estimators,
            model_path=path_for_tabpfn(self.checkpoint_),
            device=self.device_,
            random_state=self.random_state,
            softmax_temperature=self.softmax_temperature,
            fit_mode=self.fit_mode,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            show_progress_bar=False,
        )
        regressor.fit(
            X.to_numpy(dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        self.regressor_ = regressor
        self.n_context_ = len(X)

    def _predict_pdf(self, X: pd.DataFrame, grid) -> NDArray[np.float64]:
        size = self.chunk_size if self.chunk_size > 0 else len(X)
        blocks = []
        with bar(
            self.progress,
            total=len(X),
            desc=f"TabPFN {self.version}",
            unit="gal",
        ) as progress:
            progress.set_postfix(context=self.n_context_)
            for start in range(0, len(X), size):
                stop = min(start + size, len(X))
                self._log(f"rows {start}:{stop} of {len(X)}")
                output = self.regressor_.predict(
                    X.iloc[start:stop].to_numpy(dtype=np.float64),
                    output_type="full",
                )
                borders, masses = bucket_masses(output)
                self.borders_, self.n_buckets_ = borders, int(borders.size - 1)
                blocks.append(grid.rebin(masses, borders))
                del output, masses
                progress.set_postfix(
                    context=self.n_context_,
                    buckets=self.n_buckets_,
                    refresh=False,
                )
                progress.update(stop - start)
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[TabPFNBarDistribution] {message}", flush=True)


def path_for_tabpfn(path: Path) -> Path:
    """The path to hand TabPFN, which is not always the one we downloaded.

    TabPFN decides a checkpoint's format from its *resolved* path
    (``model_loading.load_model`` does ``str(path.resolve())``, and
    ``checkpoint.Checkpoint`` then reads the suffix). In the Hugging Face cache
    a snapshot file is a symlink to a content-addressed blob with no extension,
    so resolving it throws the suffix away: a ``.safetensors`` checkpoint --
    which from ``v3.5`` on is how TabPFN ships -- is then handed to
    ``torch.load`` and dies with an ``UnpicklingError``. The ``.ckpt`` versions
    are unharmed, being torch archives already.

    A hardlink beside the blob gives the same bytes a name whose suffix does
    survive ``resolve()``. Nothing is copied -- it is one more directory entry
    for an inode that is already there -- and it lives inside the cache entry it
    belongs to, so deleting the model from the HF cache takes it too.

    The rewrite is conditional on the suffix actually being lost, so it stops
    happening by itself if upstream resolves the path differently, or when
    ``HF_HUB_DISABLE_SYMLINKS`` is set and the snapshot is a real file.
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
                f"cannot give {path.name} a name TabPFN can read its format from "
                f"({error}). Set HF_HUB_DISABLE_SYMLINKS=1 and re-download so the "
                f"cached file is a real one, or use a version whose checkpoint is a "
                f".ckpt, such as version='v3'."
            ) from error
    return link


def bucket_masses(output) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(borders, masses)`` from a TabPFN ``output_type="full"`` prediction.

    ``output["logits"]`` are log-probabilities over the buckets of
    ``output["criterion"]``, whose ``borders`` are in the raw target's units --
    redshift, here. A softmax turns the first into per-bucket probability mass,
    which is what every upstream reduction (the mean, the quantiles) integrates
    over too, so this is the model's own density and not a reading of it.
    """
    import torch

    criterion, logits = output["criterion"], output["logits"]
    borders = np.maximum.accumulate(
        np.asarray(criterion.borders.detach().cpu().numpy(), dtype=np.float64)
    )
    masses = torch.softmax(logits.detach().double(), dim=-1).cpu().numpy()
    if masses.ndim != 2 or masses.shape[1] != borders.size - 1:
        raise RuntimeError(
            f"TabPFN returned logits of shape {tuple(masses.shape)} for a bar distribution "
            f"with {borders.size - 1} buckets"
        )
    return borders, masses
