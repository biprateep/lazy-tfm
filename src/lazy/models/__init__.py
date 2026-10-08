# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The backends, and the name-based entry point to them.

Two ways in, and they give you the same estimator::

    from lazy import LazyModel
    model = LazyModel("tabfm", n_estimators=4, n_dither=3)   # by name

    from lazy.models import TabFMHistogram
    model = TabFMHistogram(n_estimators=4, n_dither=3)       # by class

Use :class:`~lazy.models.lazy_model.LazyModel` when the backend is a
configuration value -- a CLI flag, a config file, a loop over methods -- and
the concrete class when you want its parameters documented at your fingertips.

Constructing either downloads nothing: the pretrained weights are fetched on
the first prediction and cached from then on (:func:`download_checkpoint`).
"""

# This file's imports re-export the public API by name; that is its purpose.
# ruff: noqa: GS001

from lazy.models._ensemble import ContextEnsembleEstimator
from lazy.models._ensemble import ContextSizeWarning
from lazy.models._ensemble import PerformanceWarning
from lazy.models._hub import CHECKPOINTS
from lazy.models._hub import DEFAULT_VERSIONS
from lazy.models._hub import download_checkpoint
from lazy.models._hub import get_checkpoint
from lazy.models._hub import is_cached
from lazy.models._hub import list_versions
from lazy.models._weights import clear_model_cache
from lazy.models._weights import model_cache_enabled
from lazy.models._weights import set_model_cache
from lazy.models.ensemble import EnsembleSizeWarning
from lazy.models.ensemble import LazyEnsembleModel
from lazy.models.lazy_model import LazyModel
from lazy.models.limix import LimiXBarDistribution
from lazy.models.registry import ESTIMATORS
from lazy.models.registry import get_estimator
from lazy.models.registry import list_estimators
from lazy.models.tabfm import TabFMHistogram
from lazy.models.tabicl import TabICLQuantile
from lazy.models.tabpfn import TabPFNBarDistribution

__all__ = [
    "CHECKPOINTS",
    "ContextEnsembleEstimator",
    "ContextSizeWarning",
    "DEFAULT_VERSIONS",
    "ESTIMATORS",
    "EnsembleSizeWarning",
    "LazyEnsembleModel",
    "LazyModel",
    "LimiXBarDistribution",
    "PerformanceWarning",
    "TabFMHistogram",
    "TabICLQuantile",
    "TabPFNBarDistribution",
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
