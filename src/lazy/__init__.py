# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LAZY -- Lazy but Accurate photo-Z for Yinz.

Photometric redshift PDFs from tabular foundation models. The models are
pretrained and never fine-tuned: you hand them labelled galaxies as *context*
and they answer queries in one forward pass, so there is no training loop, no
hyper-parameter search and no per-survey retraining -- hence lazy::

    from lazy import LazyModel, Grid
    from lazy.datasets import fetch_dc1

    train, test = fetch_dc1(split=True)
    X_train, X_test = train.features("mag-color"), test.features("mag-color")

    model = LazyModel("tabfm", n_estimators=4, n_dither=3)
    model.fit(X_train, train.redshift)

    grid = Grid.linear(0.0, 2.0, 200)
    pdfs = model.predict_proba(X_test, grid)           # (n, 200) densities
    z = model.predict(X_test, method="z_peak")         # point estimates
    model.evaluate(X_test, test.redshift)              # the full metric table

The API is scikit-learn's, with ``predict_proba`` returning a density on a
redshift grid rather than class probabilities, because that is the natural
output of a photo-z model. Three backends ship today -- ``tabfm`` builds the
density from a hierarchy of in-context classifiers, ``tabicl`` from a quantile
regression head, ``tabpfn`` from the bucket masses TabPFN-3 predicts natively
-- and all of them write onto whatever :class:`~lazy.grid.Grid` you
pass, at prediction time, without refitting.

The package is laid out as follows.

:mod:`lazy.models`
    :class:`~lazy.models.lazy_model.LazyModel`, the concrete backends, and the
    registry they are looked up in.
:mod:`lazy.grid`
    :class:`~lazy.grid.Grid`: the output binning, normalisation and
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

# This file's imports re-export the public API by name; that is its purpose.
# ruff: noqa: GS001

from importlib import metadata

from lazy import _namespace
from lazy.base import BaseDensityRegressor
from lazy.base import POINT_ESTIMATORS
from lazy.grid import as_grid
from lazy.grid import DC1_GRID
from lazy.grid import Grid
from lazy.models import CHECKPOINTS
from lazy.models import ContextSizeWarning
from lazy.models import DEFAULT_VERSIONS
from lazy.models import download_checkpoint
from lazy.models import ESTIMATORS
from lazy.models import get_checkpoint
from lazy.models import get_estimator
from lazy.models import is_cached
from lazy.models import LazyModel
from lazy.models import LimiXBarDistribution
from lazy.models import list_estimators
from lazy.models import list_versions
from lazy.models import PerformanceWarning
from lazy.models import TabFMHistogram
from lazy.models import TabICLQuantile
from lazy.models import TabPFNBarDistribution

try:
    __version__ = metadata.version("lazy-tfm")
except metadata.PackageNotFoundError:  # pragma: no cover - a source tree only.
    __version__ = "0.0.0.dev0"

_namespace.warn_if_shared()  # noqa: GS026 - the check must run at import.

__all__ = [
    "CHECKPOINTS",
    "DC1_GRID",
    "DEFAULT_VERSIONS",
    "ESTIMATORS",
    "POINT_ESTIMATORS",
    "BaseDensityRegressor",
    "ContextSizeWarning",
    "LazyModel",
    "LimiXBarDistribution",
    "PerformanceWarning",
    "Grid",
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
