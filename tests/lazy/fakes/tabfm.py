# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""TabFM's real classifier and member views, on a small backbone.

Upstream's classifier fits its preprocessing and builds its members' views as
it would; only the backbone is faked (:class:`Network`), and both inference
paths (the streamed one and ``predict_proba``) run it through upstream's own
``_predict_proba_internal``. Records, for every classifier run, the rows of
its view, the scaling each of its standardizations was refitted with, the
columns it reads as categorical and its encoding of the queries, and in
``loads`` the checkpoint each backbone load was handed. Needs the tabfm
package and torch.

It replaces the private lazy names in ``LAZY_SEAMS`` and, upstream,
``tabfm.TabFMClassifier._predict_proba_internal`` (recording, then calling
upstream's) and ``tabfm.tabfm_v1_0_0_pytorch.load``.
"""

import numpy as np
import pytest

import fakes
from lazy.models import tabfm

LAZY_SEAMS = (
    "lazy.models._icl_stream.streaming_available",
    "lazy.models._icl_stream.classification_logits",
)
"""The private lazy names that :func:`install` replaces."""

SETTINGS = {
    "n_estimators": 2,
    "n_coarse_bins": 2,
    "n_fine_bins": 2,
    "n_dither": 1,
}
"""Small enough for the behavioral suite: two levels of two bins."""


def _network_class(torch):
    """The fake backbone's class, which needs torch to define.

    Args:
        torch: The torch module.

    Returns:
        :class:`Network`.
    """

    class Network(torch.nn.Module):
        """Stands in for upstream's ``TabFM``: a nearest-class-mean classifier.

        A row's signal is a mean of its features weighted by their place,
        and a class's logit is minus four times the squared distance from
        the row's signal to the mean signal of the context rows of that
        class. It answers on torch's graph, depends on the context, on each
        query alone and on the order of the columns, and relabels with the
        classes, so upstream's class shift undoes exactly.
        """

        max_classes = 10

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(4.0))

        def forward(self, x, y, train_size, cat_mask=None, d=None):
            """Logits for every row, as upstream's ``TabFM.forward``.

            Args:
                x: The members' rows, context first, shape (n_members,
                    n_rows, n_features).
                y: Their classes, the queries' padded, shape (n_members,
                    n_rows).
                train_size: Context rows per member, shape (n_members,).
                cat_mask: Unused: every column counts alike.
                d: Unused: no member is padded.

            Returns:
                The logits, shape (n_members, n_rows, max_classes).
            """
            del cat_mask, d  # Unused: see above.
            place = torch.linspace(1.0, 2.0, x.shape[-1], device=x.device)
            signal = (x * place).mean(dim=-1)
            # Upstream batches members of one context size. The sums run
            # over the context alone, so a query's answer rounds alike
            # whatever the other queries.
            n_context = int(train_size[0])
            member = (
                y[:, :n_context, None]
                == torch.arange(self.max_classes, device=x.device)
            ).to(signal.dtype)
            means = (member * signal[:, :n_context, None]).sum(
                dim=1
            ) / member.sum(dim=1).clamp(min=1)
            return -self.scale * (signal[..., None] - means[:, None]) ** 2

    return Network


class _Logits:
    """The streamed path, recording what each classifier was handed."""

    def __init__(self, recorder, internal):
        self.recorder = recorder
        self.internal = internal

    def record(self, classifier, query):
        """Records a classifier run on the rows ``query``."""
        generator = classifier.ensemble_generator_
        # One pipeline per norm method; a bagged member's view has one.
        pipelines = list(generator.preprocessors_.values())
        self.recorder.add(
            "tabfm",
            rows=fakes.row_ids(generator.X_),
            n_members=classifier.n_estimators,
            mean=[p.standard_scaler_.mean_ for p in pipelines],
            features=np.asarray(generator.X_),
            cat_features=generator.cat_features_,
            query=generator.unique_filter_.transform(
                classifier.X_encoder_.transform(query)
            ),
        )

    def __call__(self, classifier, model, targets, **kwargs):
        del model, kwargs  # Unused: the classifier holds the backbone.
        self.record(classifier, targets["query"])
        logits = self.internal(classifier, targets["query"])
        # Averaged in float64, as lazy's streamed path averages.
        return {"query": {"mean_logits": logits.astype(np.float64).mean(0)}}


def install(monkeypatch, recorder):
    """Installs the fake; see the module docstring.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        recorder: Records each classifier's rows and refitted scaling.

    Returns:
        :class:`lazy.models.tabfm.TabFMHistogram`.
    """
    upstream = pytest.importorskip("tabfm")
    torch = pytest.importorskip("torch")
    network = _network_class(torch)
    logits = _Logits(recorder, upstream.TabFMClassifier._predict_proba_internal)

    def load(
        model_type="classification",
        checkpoint_path=None,
        *,
        device=None,
        dtype=torch.bfloat16,
        use_cache=True,
    ):
        """Upstream's loader, recording the checkpoint it is handed."""
        del model_type, device, dtype  # Unused: nothing is read.
        if use_cache:
            raise AssertionError("lazy bypasses upstream's own cache")
        recorder.loads.append(checkpoint_path)
        return network()

    def internal(classifier, X):
        logits.record(classifier, X)
        return logits.internal(classifier, X)

    monkeypatch.setattr(upstream.tabfm_v1_0_0_pytorch, "load", load)
    fakes.patch(
        monkeypatch,
        LAZY_SEAMS,
        {
            "lazy.models._icl_stream.streaming_available": lambda: True,
            "lazy.models._icl_stream.classification_logits": logits,
        },
    )
    monkeypatch.setattr(
        upstream.TabFMClassifier, "_predict_proba_internal", internal
    )
    return tabfm.TabFMHistogram
