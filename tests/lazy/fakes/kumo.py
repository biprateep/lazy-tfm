# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Kumo Tabular with upstream's recipe machinery and a fake network.

The network is replaced by a subclass of upstream's ``ICLModel`` whose
forward pass answers from the processed features alone, so the backend's own
code and upstream's preprocessing, ensembling and context cache all run as
they do on the real model. It needs the sdm package, so it skips without it.
"""

import numpy as np
import pytest

import fakes
from lazy.models import kumo

#: Small enough for the behavioral suite; v1-small, the CPU-friendly size.
SETTINGS = {"version": "v1-small", "n_estimators": 2}

#: Where the fake's quantiles sit, in standardized units of the target.
SPREAD = np.linspace(-1.5, 1.5, 9)


def _network_class(sdm, torch):
    """A stand-in for ``sdm.models.KumoTabular``, built on upstream's base."""

    class Network(sdm.models.ICLModel):
        supported_feature_stypes = frozenset({sdm.Stype.numerical})
        supported_target_stypes = frozenset({sdm.Stype.numerical})
        supports_multi_target = False
        supports_related_tables = False

        #: Set by :func:`install`.
        recorder = None

        def __init__(self):
            super().__init__(task="regression")
            self.weight = torch.nn.Parameter(torch.zeros(1))

        @classmethod
        def default_recipe(cls):
            raise AssertionError("the backend passes its own recipe")

        def fit(self, x, y, **kwargs):
            self._record(x, y, kwargs["generator"])
            return super().fit(x, y, **kwargs)

        def forward(self, x_context, y_context, x_query, **kwargs):
            self._record(x_context, y_context, kwargs["generator"])
            return super().forward(x_context, y_context, x_query, **kwargs)

        def _record(self, x, y, generator):
            self.recorder.add(
                generator.initial_seed(),
                rows=fakes.row_ids(x.cpu().numpy()),
                y=y.cpu().numpy()[:, 0],
            )

        def _forward(
            self,
            x_context,
            y_context,
            x_query,
            related_context_tables,
            related_query_tables,
            cache,
            generator,
            *,
            categorical_mask,
            **kwargs,
        ):
            del y_context, related_context_tables, related_query_tables
            del generator, categorical_mask, kwargs  # Unused.
            if x_context is not None:
                context = x_context.numerical
                if cache is not None and cache.is_recording:
                    cache["context"] = context
            else:
                context = cache["context"]
            if x_query is None:
                return None  # A fit records its context and answers nothing.
            queries = x_query.numerical
            centre = context.nanmean(dim=(-2, -1), keepdim=True)
            spread = context.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            signal = torch.tanh(
                (queries.nanmean(-1, keepdim=True) - centre) / spread
            )
            out = signal + torch.as_tensor(SPREAD, dtype=queries.dtype)
            return sdm.TableTensor(
                columns={
                    sdm.Stype.numerical: [f"q{i}" for i in range(SPREAD.size)]
                },
                numerical=out,
            )

    return Network


def install(monkeypatch, recorder):
    """Installs the fake; see the module docstring.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        recorder: Records each group's rows and standardized targets, keyed
            by its seed.

    Returns:
        :class:`lazy.models.kumo.KumoQuantile`.
    """
    sdm = pytest.importorskip("sdm")
    torch = pytest.importorskip("torch")
    network = _network_class(sdm, torch)
    monkeypatch.setattr(network, "recorder", recorder)

    def load(sdm, path, size, device):
        del sdm, path, size  # Unused: nothing is read.
        recorder.loads.append(device)
        return network().to(device).eval()

    monkeypatch.setattr(kumo, "_load_network", load)
    return kumo.KumoQuantile
