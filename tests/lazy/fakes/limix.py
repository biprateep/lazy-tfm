# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LimiX without its source, network or torch.

The backend's own code runs, its key/value caches and pickling included;
what is faked is LimiX's source (not on PyPI), its preprocessing pipelines
(which come from that source), and the network's forward pass, prefill and
decode. Needs nothing beyond the package, so it runs everywhere.

It replaces the private lazy names in ``LAZY_SEAMS``; it provides no upstream
names, since LimiX's source is faked whole.
"""

import types

import numpy as np

import fakes
from lazy.models import _limix_stream
from lazy.models import limix

LAZY_SEAMS = (
    "lazy.models._limix_preprocess.MemberPipeline",
    "lazy.models._limix_source.load",
    "lazy.models._limix_stream.Member",
    "lazy.models._limix_stream.load_network",
    "lazy.models._limix_stream.cache_supported",
    "lazy.models._limix_stream.cache_bytes",
    "lazy.models._limix_stream.prefill",
    "lazy.models._limix_stream.forward",
    "lazy.models._limix_stream.decode",
    "lazy.models.limix._cache_dtype",
    "lazy.models.limix._free_device_memory",
    "lazy.models.limix._release_device_memory",
    "lazy.models.limix.LimiXBarDistribution._import_backend",
)
"""The private lazy names that :func:`install` replaces."""

SETTINGS = {"n_estimators": 2}
"""Small enough for the behavioral suite."""

BORDERS = np.linspace(-3.0, 3.0, 11)
"""The network's bucket borders, in standardized units of the target."""


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

    recorder = None
    """Set by :func:`install`."""

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
    fakes.patch(
        monkeypatch,
        LAZY_SEAMS,
        {
            "lazy.models._limix_preprocess.MemberPipeline": Pipeline,
            "lazy.models._limix_source.load": lambda: _SOURCE,
            "lazy.models._limix_stream.Member": Member,
            "lazy.models._limix_stream.load_network": (
                lambda path, device: Network()
            ),
            "lazy.models._limix_stream.cache_supported": lambda net: True,
            "lazy.models._limix_stream.cache_bytes": lambda *args: 0,
            "lazy.models._limix_stream.prefill": (
                lambda net, member, **kw: "cache"
            ),
            "lazy.models._limix_stream.forward": _logits,
            "lazy.models._limix_stream.decode": _decode,
            "lazy.models.limix._cache_dtype": lambda net, **kw: None,
            "lazy.models.limix._free_device_memory": lambda net: None,
            "lazy.models.limix._release_device_memory": lambda net: None,
            "lazy.models.limix.LimiXBarDistribution._import_backend": (
                lambda est: _SOURCE.loading
            ),
        },
    )
    return limix.LimiXBarDistribution
