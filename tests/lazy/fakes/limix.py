# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LimiX without its source, network or torch.

The backend's own code runs, its key/value caches and pickling included;
what is faked is LimiX's source (not on PyPI), its preprocessing pipelines
(which come from that source), and the network's forward pass, prefill and
decode. Needs nothing beyond the package, so it runs everywhere.
"""

import types

import numpy as np

import fakes
from lazy.models import _limix_preprocess
from lazy.models import _limix_source
from lazy.models import _limix_stream
from lazy.models import limix

#: Small enough for the behavioral suite.
SETTINGS = {"n_estimators": 2}

#: The network's bucket borders, in standardized units of the target.
BORDERS = np.linspace(-3.0, 3.0, 11)


class Pipeline:
    """Stands in for a LimiX preprocessing pipeline: the identity."""

    def __init__(self, token, seeds, *, shuffle):
        self.token, self.seeds, self.shuffle = token, seeds, shuffle

    def fit(self, context):
        self.n_features_out_ = context.shape[1]
        return self

    def transform(self, features):
        return features


class Member(_limix_stream.Member):
    """A member's context, recorded as it is made."""

    #: Set by :func:`install`.
    recorder = None

    def __init__(self, x, y, seed):
        super().__init__(x, y, seed)
        self.recorder.add(seed, rows=fakes.row_ids(x), y=np.array(y))


class _Tensor:
    """Just enough of a tensor for ``.detach().cpu().numpy()``."""

    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Network:
    """Stands in for LimiX-2's network: only its bucket borders."""

    def __init__(self):
        self._reg_borders = _Tensor(BORDERS.copy())


def _logits(network, member, queries, **kwargs):
    """A member's fake logits, the same with a cache or without."""
    del kwargs  # Unused: the cache and the precision change nothing.
    signal = fakes.response(queries, member.x)
    return fakes.bucket_logits(signal, network._reg_borders.values.size - 1)


def _decode(network, member, cache, queries, **kwargs):
    assert cache == "cache", cache  # A decode always gets its prefill.
    return _logits(network, member, queries, **kwargs)


_SOURCE = types.SimpleNamespace(
    loading=types.ModuleType("limix_fake"),
    source=types.SimpleNamespace(
        commit="0" * 40, package="LimiX (fake)", origin="tests/lazy/fakes"
    ),
)


def install(monkeypatch, recorder):
    """Installs the fake; see the module docstring.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        recorder: Records each member's rows and standardized targets.

    Returns:
        :class:`lazy.models.limix.LimiXBarDistribution`.
    """
    monkeypatch.setattr(Member, "recorder", recorder)
    monkeypatch.setattr(_limix_preprocess, "MemberPipeline", Pipeline)
    monkeypatch.setattr(_limix_stream, "Member", Member)
    monkeypatch.setattr(_limix_source, "load", lambda: _SOURCE)
    monkeypatch.setattr(
        _limix_stream, "load_network", lambda path, device: Network()
    )
    monkeypatch.setattr(_limix_stream, "cache_supported", lambda net: True)
    monkeypatch.setattr(_limix_stream, "cache_bytes", lambda *args: 0)
    monkeypatch.setattr(
        _limix_stream, "prefill", lambda net, member, **kw: "cache"
    )
    monkeypatch.setattr(_limix_stream, "forward", _logits)
    monkeypatch.setattr(_limix_stream, "decode", _decode)
    monkeypatch.setattr(limix, "_cache_dtype", lambda net, **kw: None)
    monkeypatch.setattr(limix, "_free_device_memory", lambda net: None)
    monkeypatch.setattr(
        limix.LimiXBarDistribution,
        "_import_backend",
        lambda est: _SOURCE.loading,
    )
    return limix.LimiXBarDistribution
