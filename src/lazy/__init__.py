"""LAZY -- Lazy but Accurate photo-Z for Yinz.

Photometric redshift PDFs from tabular foundation models. The models are
pretrained and never fine-tuned: you hand them labelled galaxies as *context*
and they answer queries in one forward pass, so there is no training loop, no
hyper-parameter search and no per-survey retraining -- hence lazy::

    from lazy import LazyModel, RedshiftGrid
    from lazy.datasets import fetch_dc1

    train, test = fetch_dc1(split=True)
    X_train, X_test = train.features("mag-color"), test.features("mag-color")

    model = LazyModel("tabfm", n_estimators=4, n_dither=3)
    model.fit(X_train, train.redshift)

    grid = RedshiftGrid.linear(0.0, 2.0, 200)
    pdfs = model.predict_proba(X_test, grid)           # (n, 200) densities
    z = model.predict(X_test, method="z_peak")         # point estimates
    model.evaluate(X_test, test.redshift)              # the full metric table

The API is scikit-learn's, with ``predict_proba`` returning a density on a
redshift grid rather than class probabilities, because that is the natural
output of a photo-z model. Three backends ship today -- ``tabfm`` builds the
density from a hierarchy of in-context classifiers, ``tabicl`` from a quantile
regression head, ``tabpfn`` from the bucket masses TabPFN-3 predicts natively
-- and all of them write onto whatever :class:`~lazy.grid.RedshiftGrid` you
pass, at prediction time, without refitting.

Layout
------
:mod:`lazy.models`
    :class:`~lazy.models.lazy_model.LazyModel`, the concrete backends, and the
    registry they are looked up in.
:mod:`lazy.grid`
    :class:`~lazy.grid.RedshiftGrid`: the output binning, normalisation and
    mass-conserving rebinning.
:mod:`lazy.metrics`
    LSST DESC PZ Data Challenge point and PDF metrics, and
    :func:`~lazy.metrics.summarize` for the comparison table.
:mod:`lazy.plotting`
    The publication style, and the standard diagnostic figures.
:mod:`lazy.datasets`
    The DC1 benchmark catalogue, and :class:`~lazy.datasets.Catalog`, which
    turns photometry -- DC1's or your own -- into the feature views a model
    sees. :func:`~lazy.datasets.fetch_dc1_biased` cuts the harder benchmark:
    a spectroscopically selected training set, and representative galaxies to
    calibrate and score with.
:mod:`lazy.selection`
    The spectroscopic selection function behind that split -- RAIL's HSC
    ``GridSelection``, ported -- for biasing a catalogue of your own.

Weights are fetched from the Hugging Face Hub on first use and cached
thereafter; :func:`download_checkpoint` warms that cache ahead of time.
"""

from importlib.metadata import PackageNotFoundError, version

from lazy.base import POINT_ESTIMATORS, BasePhotoZEstimator
from lazy.grid import DC1_GRID, RedshiftGrid, as_grid
from lazy.models import (
    CHECKPOINTS,
    DEFAULT_VERSIONS,
    ESTIMATORS,
    LazyModel,
    TabFMHistogram,
    TabICLQuantile,
    TabPFNBarDistribution,
    download_checkpoint,
    get_checkpoint,
    get_estimator,
    is_cached,
    list_estimators,
    list_versions,
)

try:
    __version__ = version("lazy-photoz")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "CHECKPOINTS",
    "DC1_GRID",
    "DEFAULT_VERSIONS",
    "ESTIMATORS",
    "POINT_ESTIMATORS",
    "BasePhotoZEstimator",
    "LazyModel",
    "RedshiftGrid",
    "TabFMHistogram",
    "TabICLQuantile",
    "TabPFNBarDistribution",
    "__version__",
    "as_grid",
    "download_checkpoint",
    "get_checkpoint",
    "get_estimator",
    "is_cached",
    "list_estimators",
    "list_versions",
]
