# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""TabICLRegressor without a backbone, with upstream's member plan.

The members are planned by upstream's own ``EnsembleGenerator``, which the
backend reads to check that the members it asked for ran, so this fake needs
the tabicl package. The rest follows upstream's contract where the backend
depends on it: the regressor loads its network from ``model_path`` in
``_load_model``, and the network keeps the cache of its last call on itself.
"""

import pathlib

import numpy as np
import pytest

import fakes
from lazy.models import _hub
from lazy.models import tabicl

#: Small enough for the behavioral suite.
SETTINGS = {"n_estimators": 2}

#: Where the fake's quantiles sit, in standardized units of the target.
SPREAD = np.linspace(-1.5, 1.5, 9)

#: Float64 weights in the fake network, so that a pickled one shows (4 MB).
N_WEIGHTS = 500_000


class Network:
    """Stands in for upstream's ``TabICL`` network.

    Like upstream's, it keeps the cache of its last call on itself
    (``_cache``) until ``clear_cache``.
    """

    def __init__(self):
        self._cache = None
        self.weights = np.zeros(N_WEIGHTS)

    def to(self, device):
        del device  # Unused: it stays where it is.
        return self

    def clear_cache(self):
        self._cache = None

    def cached_context(self):
        """The context a cached call answers from: the network's own."""
        return self._cache


class Regressor:
    """Stands in for ``tabicl.TabICLRegressor``."""

    #: Set by :func:`install`.
    recorder = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.random_state = kwargs["random_state"]
        self.model_path = kwargs["model_path"]
        self.device = kwargs["device"]
        self.use_amp = kwargs["use_amp"]
        self.kv_cache = kwargs["kv_cache"]

    def _resolve_device(self):
        self.device_ = str(self.device)

    def _build_inference_config(self):
        self.inference_config_ = {"device": self.device_, "amp": self.use_amp}

    def _move_cache_to_device(self):
        """Nothing to move: the fake's cache is a NumPy array."""

    def _load_model(self):
        """Loads the network from ``model_path``, which must exist.

        A checkpoint exists only where the hub puts it now.

        Raises:
            ValueError: If ``model_path`` is not there.
        """
        found = _hub.get_checkpoint("tabicl", "v2").download()
        if pathlib.Path(self.model_path) != found:
            raise ValueError(f"Checkpoint not found at '{self.model_path}'")
        self.recorder.loads.append(self.device)
        self.model_ = Network()
        self.model_config_ = {"embed_dim": 8}
        self.model_path_ = found

    def fit(self, X, y):
        from tabicl._sklearn import preprocessing  # noqa: PLC0415

        self.recorder.add(
            self.random_state, rows=fakes.row_ids(X), y=np.array(y)
        )
        self._resolve_device()
        self._build_inference_config()
        self._load_model()
        self.model_.to(self.device_)
        self.context_ = np.array(X, dtype=np.float64)
        self.ensemble_generator_ = preprocessing.EnsembleGenerator(
            classification=False,
            n_estimators=self.kwargs["n_estimators"],
            norm_methods=self.kwargs["norm_methods"],
            feat_shuffle_method=self.kwargs["feat_shuffle_method"],
            outlier_threshold=self.kwargs["outlier_threshold"],
            random_state=self.random_state,
        ).fit(X, y)
        self.model_kv_cache_ = None
        if self.kv_cache:
            # Upstream records the cache on the network, then takes it.
            self.model_._cache = self.context_
            self.model_kv_cache_ = self.model_.cached_context()
            self.model_.clear_cache()
        return self

    def predict(self, X, output_type):
        del output_type  # Unused: always the raw quantiles.
        if self.model_kv_cache_ is None:
            context = self.context_
        else:
            # Upstream hands the network the cache and leaves it there.
            self.model_._cache = self.model_kv_cache_
            context = self.model_.cached_context()
        signal = fakes.response(X, context)
        return (signal[:, None] + SPREAD).astype(np.float32)


def install(monkeypatch, recorder):
    """Installs the fake; see the module docstring.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        recorder: Records each regressor's rows and targets, and each
            network load's device.

    Returns:
        :class:`lazy.models.tabicl.TabICLQuantile`.
    """
    upstream = pytest.importorskip("tabicl")
    pytest.importorskip("tabicl._sklearn.preprocessing")
    monkeypatch.setattr(Regressor, "recorder", recorder)
    monkeypatch.setattr(upstream, "TabICLRegressor", Regressor)
    return tabicl.TabICLQuantile
