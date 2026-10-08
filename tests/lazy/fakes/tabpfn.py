# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""TabPFNRegressor without a network, validating sizes as upstream does.

Its bucket borders are fixed in standardized units and stretched by the
targets it is given, as upstream's are. It loads its "network" through
upstream's loader, as the real regressor does, and the loader counts the
networks it builds in ``recorder.loads``. Needs the tabpfn package.
"""

import types

import numpy as np
import pytest

import fakes
from lazy.models import tabpfn

#: Small enough for the behavioral suite.
SETTINGS = {"n_estimators": 2}

N_BUCKETS = 10


class Regressor:
    """Stands in for ``tabpfn.TabPFNRegressor``."""

    #: Set by :func:`install`: the recorder and the context-size limit.
    recorder = None
    limit = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        from tabpfn import base  # noqa: PLC0415 - optional extra.
        from tabpfn import validation  # noqa: PLC0415 - optional extra.
        import torch  # noqa: PLC0415 - optional extra.

        self.models_, *_ = base.initialize_tabpfn_model(
            model_path=self.kwargs["model_path"],
            which="regressor",
            fit_mode=self.kwargs["fit_mode"],
            softmax_temperature_override=None,
            n_estimators_override=self.kwargs["n_estimators"],
        )
        limit = self.limit or 10**9
        self.inference_config_ = types.SimpleNamespace(
            MAX_NUMBER_OF_SAMPLES=limit, MAX_CPU_SAMPLES=10**9
        )
        validation.validate_dataset_size(
            X,
            y,
            max_num_samples=limit,
            max_num_features=500,
            devices=(torch.device("cpu"),),
            ignore_pretraining_limits=self.kwargs["ignore_pretraining_limits"],
            max_cpu_samples=10**9,
        )
        self.recorder.add(
            self.kwargs["random_state"],
            rows=fakes.row_ids(X),
            y=np.array(y),
            config=self.kwargs["inference_config"],
        )
        self.context_ = np.array(X, dtype=np.float64)
        self.znorm_space_bardist_ = types.SimpleNamespace(
            borders=torch.linspace(
                -3.0, 3.0, N_BUCKETS + 1, dtype=torch.float64
            )
        )
        self.y_train_mean_ = float(np.mean(y))
        self.y_train_std_ = float(np.std(y))
        return self

    def predict(self, X, output_type):
        import torch  # noqa: PLC0415 - optional extra.

        del output_type  # Unused: always the full output.
        signal = fakes.response(X, self.context_)
        return {
            "criterion": self.znorm_space_bardist_,
            "logits": torch.from_numpy(fakes.bucket_logits(signal, N_BUCKETS)),
        }


def install(monkeypatch, recorder, *, limit=None):
    """Installs the fake; see the module docstring.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        recorder: Records each member's rows, targets and config.
        limit: The most context rows upstream accepts, or None for no
            limit.

    Returns:
        :class:`lazy.models.tabpfn.TabPFNBarDistribution`.
    """
    upstream = pytest.importorskip("tabpfn")
    from tabpfn import base  # noqa: PLC0415 - optional extra.
    import torch  # noqa: PLC0415 - optional extra.

    def load(**kwargs):
        recorder.loads.append(kwargs)
        return [object()], None, torch.nn.Module(), None

    monkeypatch.setattr(base, "initialize_tabpfn_model", load)
    monkeypatch.setattr(Regressor, "recorder", recorder)
    monkeypatch.setattr(Regressor, "limit", limit)
    monkeypatch.setattr(upstream, "TabPFNRegressor", Regressor)
    return tabpfn.TabPFNBarDistribution
