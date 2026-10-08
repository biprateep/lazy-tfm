# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""TabFM's real classifier and member views, without the backbone.

Upstream's classifier fits its preprocessing as it would; only the forward
pass is faked, on both inference paths (the streamed one and
``predict_proba``). Records, for every classifier run, the rows of its view
and the scaling each of its standardizations was refitted with. Needs the
tabfm package.
"""

import types

import numpy as np
import pytest

import fakes
from lazy.models import _hub
from lazy.models import _icl_stream
from lazy.models import tabfm

#: Small enough for the behavioral suite: two levels of two bins.
SETTINGS = {
    "n_estimators": 2,
    "n_coarse_bins": 2,
    "n_fine_bins": 2,
    "n_dither": 1,
}


class _Logits:
    """The fake forward pass, recording what each classifier was handed."""

    def __init__(self, recorder):
        self.recorder = recorder

    def __call__(self, classifier, model, targets, **kwargs):
        del model, kwargs  # Unused: nothing runs.
        generator = classifier.ensemble_generator_
        # One pipeline per norm method; a bagged member's view has one.
        pipelines = list(generator.preprocessors_.values())
        self.recorder.add(
            "tabfm",
            rows=fakes.row_ids(generator.X_),
            n_members=classifier.n_estimators,
            mean=[p.standard_scaler_.mean_ for p in pipelines],
            features=np.asarray(generator.X_),
        )
        signal = fakes.response(targets["query"], generator.X_)
        n_classes = classifier.n_classes_
        return {
            "query": {
                "mean_logits": fakes.bucket_logits(signal, n_classes).astype(
                    np.float64
                )
            }
        }


def install(monkeypatch, recorder):
    """Installs the fake; see the module docstring.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        recorder: Records each classifier's rows and refitted scaling.

    Returns:
        :class:`lazy.models.tabfm.TabFMHistogram`.
    """
    upstream = pytest.importorskip("tabfm")
    logits = _Logits(recorder)

    def backbone(est):
        est.checkpoint_ = _hub.get_checkpoint(
            est.backend, est.version
        ).download()
        return types.SimpleNamespace(max_classes=10)

    def internal(classifier, X):
        frame = {"query": X}
        return logits(classifier, None, frame)["query"]["mean_logits"][None]

    monkeypatch.setattr(_icl_stream, "streaming_available", lambda: True)
    monkeypatch.setattr(tabfm.TabFMHistogram, "_shared_backbone", backbone)
    monkeypatch.setattr(_icl_stream, "classification_logits", logits)
    monkeypatch.setattr(
        upstream.TabFMClassifier, "_predict_proba_internal", internal
    )
    return tabfm.TabFMHistogram
