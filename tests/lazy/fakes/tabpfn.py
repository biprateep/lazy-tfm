# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""TabPFNRegressor without a network, validating sizes as upstream does.

It accepts only the arguments upstream's constructor accepts. Its bucket
borders are fixed in standardized units and stretched by the
targets it is given, as upstream's are. The network of a version in
``ranking_versions`` ranks each query value among the context's values of
its column, as upstream's ``tabpfn_v3_5`` does, through upstream's own
ranking functions; the mean rank moves the answer. It gets its "network"
through upstream's loader, as the real regressor does. The fake loader
builds one for a checkpoint path, counting them in ``recorder.loads``, and
hands a model spec to upstream's own loader, which answers with the spec's
network. Needs the tabpfn package.

It replaces no private lazy name; ``tabpfn.base.initialize_tabpfn_model`` and
``tabpfn.TabPFNRegressor`` are upstream's.
"""

import functools
import inspect
import os
import types

import numpy as np
import pytest

import fakes
from lazy.models import _hub
from lazy.models import tabpfn

LAZY_SEAMS = ()
"""The private lazy names that :func:`install` replaces: none."""

SETTINGS = {"n_estimators": 2}
"""Small enough for the behavioral suite."""

N_BUCKETS = 10

RANK_WEIGHT = 0.5
"""How far the mean rank of a query's values moves a v3.5 network's answer."""


def _ranks(network, queries, context):
    """The rank term of a network's answer for each query.

    A v3.5 network (one with upstream's ``ecdf_num_buckets``) ranks every
    cell among the context's values of its column through upstream's
    ``_build_ecdf_context`` and ``_in_context_ecdf``, the context rows
    first, as upstream's ``_preprocess_raw`` does, after filling the
    missing values; the term is the mean
    rank of the query's cells, centred. Other networks have none.

    Args:
        network: The fake network.
        queries: Query features, a torch tensor of shape (n, n_features).
        context: Context features, a torch tensor of shape
            (n_rows, n_features).

    Returns:
        The term, a torch tensor of shape (n,).
    """
    import tabpfn.architectures.tabpfn_v3_5  # noqa: PLC0415 - optional extra.
    import torch  # noqa: PLC0415 - optional extra.

    architecture = tabpfn.architectures.tabpfn_v3_5

    buckets = getattr(network, "ecdf_num_buckets", None)
    if buckets is None:
        return torch.zeros(len(queries), dtype=queries.dtype)

    # Upstream ranks after filling a missing value with the context's mean.
    fill = torch.nan_to_num(context.nanmean(dim=0))
    cells = torch.cat([context, queries])
    cells = torch.where(cells.isfinite(), cells, fill).to(torch.float32)[None]
    edges = architecture._build_ecdf_context(cells, len(context), buckets)
    ranks = architecture._in_context_ecdf(cells, edges)[0, len(context) :]
    return RANK_WEIGHT * (ranks.mean(dim=1).to(queries.dtype) - 0.5)


class Executor:
    """Stands in for upstream's inference engine, which holds the network."""

    def __init__(self, models):
        self.models = models

    def _set_models(self, models):
        """Hands the engine its networks, as upstream's unpickling does."""
        self.models = models


class Regressor:
    """Stands in for ``tabpfn.TabPFNRegressor``."""

    recorder = None
    """The recorder, set by :func:`install`."""
    limit = None
    """The context-size limit, set by :func:`install`."""
    upstream = None
    """Upstream's ``TabPFNRegressor``, set by :func:`install`."""

    def __init__(self, **kwargs):
        inspect.signature(self.upstream).bind(**kwargs)
        self.kwargs = kwargs
        self.model_path = kwargs["model_path"]  # upstream's attribute

    def fit(self, X, y):
        from tabpfn import base  # noqa: PLC0415 - optional extra.
        from tabpfn import validation  # noqa: PLC0415 - optional extra.
        import torch  # noqa: PLC0415 - optional extra.

        # Upstream passes the fit mode where its loader takes one (before
        # tabpfn 9.2).
        options = {"fit_mode": self.kwargs["fit_mode"]}
        accepted = inspect.signature(base.initialize_tabpfn_model).parameters
        self.models_, self.configs_, *_ = base.initialize_tabpfn_model(
            model_path=self.model_path,
            which="regressor",
            softmax_temperature_override=None,
            n_estimators_override=self.kwargs["n_estimators"],
            **{k: v for k, v in options.items() if k in accepted},
        )
        self.executor_ = Executor(list(self.models_))
        limit = self.limit or 10**9
        self.inference_config_ = types.SimpleNamespace(
            MAX_NUMBER_OF_SAMPLES=limit,
            MAX_CPU_SAMPLES=10**9,
            PREPROCESS_TRANSFORMS=self.kwargs["inference_config"][
                "PREPROCESS_TRANSFORMS"
            ],
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

    def get_params(self):
        """The constructor's arguments, as scikit-learn's ``get_params``."""
        return dict(self.kwargs)

    def fit_with_differentiable_input(self, X, y):
        """Fits on torch tensors, keeping them for :meth:`forward`."""
        assert self.kwargs["differentiable_input"]
        self.fit(X.detach().numpy(), y.detach().numpy())
        self.context_ = X
        return self

    def forward(self, X, *, use_inference_mode=False):
        """:meth:`predict`'s logits on torch's graph, one per member.

        Returns:
            As upstream: the averaged logits (unused here), the members'
            logits, each of shape (n_queries, n_buckets), and borders.
        """
        import torch  # noqa: PLC0415 - optional extra.

        assert use_inference_mode and self.kwargs["differentiable_input"]
        context = self.context_
        spread = context.std(correction=0)
        signal = torch.tanh((X.mean(dim=1) - context.mean()) / spread)
        signal = signal + _ranks(self.models_[0], X, context)
        centres = torch.linspace(-1.0, 1.0, N_BUCKETS, dtype=X.dtype)
        logits = -4.0 * (centres - signal[:, None]) ** 2
        members = [logits] * self.kwargs["n_estimators"]
        return None, members, None

    def to(self, device):
        """Records the device and, as upstream does, reads the precision.

        The fake's arithmetic runs on the CPU.
        """
        self.device = device
        self.use_autocast_ = self.inference_precision == "autocast"

    def predict(self, X, output_type):
        import torch  # noqa: PLC0415 - optional extra.

        del output_type  # Unused: always the full output.
        if None in (*self.models_, *self.executor_.models):
            raise RuntimeError("the regressor has no network")
        signal = fakes.response(X, self.context_)
        signal = (
            signal
            + _ranks(
                self.models_[0], torch.tensor(X), torch.tensor(self.context_)
            ).numpy()
        )
        return {
            "criterion": self.znorm_space_bardist_,
            "logits": torch.from_numpy(fakes.bucket_logits(signal, N_BUCKETS)),
        }


def _version(path):
    """The TabPFN version of a checkpoint path, or None if unknown.

    The path is a pinned checkpoint's file, or, in some tests, a directory
    named after the version.
    """
    name = os.path.basename(path)
    for checkpoint in _hub.CHECKPOINTS.values():
        if checkpoint.backend == "tabpfn" and name in (
            checkpoint.filename,
            checkpoint.version,
        ):
            return checkpoint.version
    return None


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
    from tabpfn import inference_config  # noqa: PLC0415 - optional extra.
    import torch  # noqa: PLC0415 - optional extra.

    real = base.initialize_tabpfn_model

    @functools.wraps(real)
    def load(**kwargs):
        inspect.signature(real).bind(**kwargs)
        if not isinstance(kwargs["model_path"], (str, os.PathLike)):
            return real(**kwargs)
        recorder.loads.append(kwargs)
        config = inference_config.InferenceConfig.get_default(
            task_type="regression", model_version="v2"
        )
        network = torch.nn.Module()
        if _version(kwargs["model_path"]) in (
            tabpfn.TabPFNBarDistribution.ranking_versions
        ):
            network.ecdf_num_buckets = 8192  # upstream's attribute
        return [network], [{}], torch.nn.Module(), config

    monkeypatch.setattr(base, "initialize_tabpfn_model", load)
    monkeypatch.setattr(Regressor, "recorder", recorder)
    monkeypatch.setattr(Regressor, "limit", limit)
    monkeypatch.setattr(Regressor, "upstream", upstream.TabPFNRegressor)
    # So that lazy reads upstream's arguments from the fake, as from the real.
    monkeypatch.setattr(
        Regressor,
        "__signature__",
        inspect.signature(upstream.TabPFNRegressor),
        raising=False,
    )
    monkeypatch.setattr(upstream, "TabPFNRegressor", Regressor)
    return tabpfn.TabPFNBarDistribution
