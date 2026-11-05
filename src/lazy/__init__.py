# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LAZY -- Lazy but Accurate *z* for Yinz.

*z* is any continuous quantity you want to predict from tabular features
(the target ``y`` in the code below): a redshift, a metallicity, a mass, a
yield. LAZY predicts its full conditional distribution with pretrained
tabular foundation models. They are never fine-tuned: you hand them labeled
rows as *context* and they answer queries in one forward pass, so there is
no training loop, no hyper-parameter search and no per-dataset retraining --
hence lazy::

    from lazy import LazyModel

    model = LazyModel("tabpfn", version="v3.5")
    model.fit(X_train, y_train)

    pdfs = model.predict_proba(X_test)                 # densities, native grid
    lo, med, hi = model.predict_quantiles(X_test, [0.16, 0.5, 0.84]).T
    y_pred = model.predict(X_test, method="mode")      # point estimates
    model.score(X_test, y_test)                        # negative CDE loss

The API is scikit-learn's, with ``predict_proba`` returning a density on a
grid of the target rather than class probabilities. Five backends ship --
``tabpfn`` and ``limix`` predict bucket masses natively, ``tabicl`` and
``kumo`` quantiles, ``tabfm`` a hierarchy of in-context classifiers -- behind
one set of parameters, and all of them write onto whatever
:class:`~lazy.grid.Grid` you pass, at prediction time, without refitting. The
package grew out of photometric redshifts, and its benchmark tools (the DC1
catalog, the HSC selection function, the Data Challenge metrics) are
photo-z's.

The package is laid out as follows.

:mod:`lazy.models`
    :class:`~lazy.models.lazy_model.LazyModel`, the concrete backends, and the
    registry they are looked up in.
:mod:`lazy.grid`
    :class:`~lazy.grid.Grid`: the output binning, normalization and
    mass-conserving rebinning.
:mod:`lazy.distributions`
    Each model's native answer, with exact ``pdf``, ``cdf``, ``ppf``, moments
    and sampling, and ``to_qp()``.
:mod:`lazy.metrics`
    The LSST DESC PZ Data Challenge PDF and point metrics, and
    :func:`~lazy.metrics.summarize` for the comparison table.
:mod:`lazy.plotting`
    The publication style, and the standard diagnostic figures.
:mod:`lazy.datasets`
    The DC1 benchmark catalog, and :class:`~lazy.datasets.Catalog`, which
    turns photometry -- DC1's or your own -- into the feature views a model
    sees. :func:`~lazy.datasets.fetch_dc1_biased` cuts the harder benchmark:
    a spectroscopically selected training set, and representative galaxies to
    calibrate and score with.
:mod:`lazy.selection`
    The spectroscopic selection function behind that split -- RAIL's HSC
    ``GridSelection``, ported -- for biasing a catalog of your own.

Weights are fetched from the Hugging Face Hub on first use and cached
thereafter; :func:`~lazy.models.download_checkpoint` warms that cache ahead of
time.
"""

# This file's imports re-export the public API by name; that is its purpose.
# ruff: noqa: GS001

import importlib
from importlib import metadata
import types
import typing

from lazy import _namespace
from lazy._namespace import ImportNameWarning
from lazy._warnings import LazyWarning
from lazy.base import BaseDensityRegressor
from lazy.base import POINT_ESTIMATORS
from lazy.grid import as_grid
from lazy.grid import Grid
from lazy.models import Checkpoint
from lazy.models import CHECKPOINTS
from lazy.models import clear_model_cache
from lazy.models import ContextSizeWarning
from lazy.models import DEFAULT_VERSIONS
from lazy.models import download_checkpoint
from lazy.models import EnsembleSizeWarning
from lazy.models import ESTIMATORS
from lazy.models import get_checkpoint
from lazy.models import get_estimator
from lazy.models import GridCoverageWarning
from lazy.models import is_cached
from lazy.models import KumoQuantile
from lazy.models import LazyEnsembleModel
from lazy.models import LazyModel
from lazy.models import LicenseWarning
from lazy.models import LimiXBarDistribution
from lazy.models import LimiXSourceWarning
from lazy.models import list_estimators
from lazy.models import list_versions
from lazy.models import model_cache_enabled
from lazy.models import PerformanceWarning
from lazy.models import set_model_cache
from lazy.models import TabFMHistogram
from lazy.models import TabFMPerformanceWarning
from lazy.models import TabICLQuantile
from lazy.models import TabPFNBarDistribution
from lazy.models import UnknownNameError

try:
    __version__ = metadata.version("lazy-tfm")
except metadata.PackageNotFoundError:  # pragma: no cover - a source tree only.
    __version__ = "0.0.0.dev0"

_namespace.warn_if_shared()  # noqa: GS026 - the check must run at import.

__all__ = [
    "BaseDensityRegressor",
    "CHECKPOINTS",
    "Checkpoint",
    "ContextSizeWarning",
    "DEFAULT_VERSIONS",
    "ESTIMATORS",
    "EnsembleSizeWarning",
    "Grid",
    "GridCoverageWarning",
    "ImportNameWarning",
    "KumoQuantile",
    "LazyEnsembleModel",
    "LazyModel",
    "LazyWarning",
    "LicenseWarning",
    "LimiXBarDistribution",
    "LimiXSourceWarning",
    "POINT_ESTIMATORS",
    "PerformanceWarning",
    "TabFMHistogram",
    "TabFMPerformanceWarning",
    "TabICLQuantile",
    "TabPFNBarDistribution",
    "UnknownNameError",
    "__version__",
    "as_grid",
    "clear_model_cache",
    "download_checkpoint",
    "get_checkpoint",
    "get_estimator",
    "is_cached",
    "list_estimators",
    "list_versions",
    "model_cache_enabled",
    "set_model_cache",
]

if typing.TYPE_CHECKING:
    from lazy import datasets as datasets
    from lazy import selection as selection


def __getattr__(name: str) -> types.ModuleType:
    """Imports :mod:`lazy.datasets` or :mod:`lazy.selection` on first access.

    They import h5py and scikit-learn's dataset loaders, which most programs
    never use, so ``import lazy`` leaves them out.

    Args:
        name: The attribute asked for.

    Returns:
        The submodule.

    Raises:
        AttributeError: If ``name`` is not one of those submodules.
    """
    if name in ("datasets", "selection"):
        return importlib.import_module(f"lazy.{name}")
    raise AttributeError(f"module 'lazy' has no attribute {name!r}")
