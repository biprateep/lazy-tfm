# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""TabICLRegressor without a backbone, with upstream's member plan.

The members are planned by upstream's own ``EnsembleGenerator``, which the
backend reads to check that the members it asked for ran, so this fake needs
the tabicl package.
"""

import numpy as np
import pytest

import fakes
from lazy.models import tabicl

#: Small enough for the behavioral suite.
SETTINGS = {"n_estimators": 2}

#: Where the fake's quantiles sit, in standardized units of the target.
SPREAD = np.linspace(-1.5, 1.5, 9)


class Regressor:
    """Stands in for ``tabicl.TabICLRegressor``."""

    #: Set by :func:`install`.
    recorder = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.random_state = kwargs["random_state"]

    def fit(self, X, y):
        from tabicl._sklearn import preprocessing  # noqa: PLC0415

        self.recorder.add(
            self.random_state, rows=fakes.row_ids(X), y=np.array(y)
        )
        self.context_ = np.array(X, dtype=np.float64)
        self.ensemble_generator_ = preprocessing.EnsembleGenerator(
            classification=False,
            n_estimators=self.kwargs["n_estimators"],
            norm_methods=self.kwargs["norm_methods"],
            feat_shuffle_method=self.kwargs["feat_shuffle_method"],
            outlier_threshold=self.kwargs["outlier_threshold"],
            random_state=self.random_state,
        ).fit(X, y)
        return self

    def predict(self, X, output_type):
        del output_type  # Unused: always the raw quantiles.
        signal = fakes.response(X, self.context_)
        return (signal[:, None] + SPREAD).astype(np.float32)


def install(monkeypatch, recorder):
    """Installs the fake; see the module docstring.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        recorder: Records each regressor's rows and targets.

    Returns:
        :class:`lazy.models.tabicl.TabICLQuantile`.
    """
    upstream = pytest.importorskip("tabicl")
    pytest.importorskip("tabicl._sklearn.preprocessing")
    monkeypatch.setattr(Regressor, "recorder", recorder)
    monkeypatch.setattr(upstream, "TabICLRegressor", Regressor)
    return tabicl.TabICLQuantile
