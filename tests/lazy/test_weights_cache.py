# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Loaded networks are shared across fits, and can be let go of.

The stubbed tests replace each backend's loader with one that counts, so
they run on a CPU with nothing downloaded. The checkpoint tests check what
the stubs cannot: that a shared network answers bit for bit as a fresh one.
"""

import os
import pathlib
import pickle
import threading
import types
import weakref

import numpy as np
import pytest
from sklearn import base as sklearn_base

from lazy.models import _limix_source
from lazy.models import _limix_stream
from lazy.models import _weights
from lazy.models import limix
from lazy.models import tabfm
from lazy.models import tabicl
from lazy.models import tabpfn

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    """Every test starts with an empty cache and the switch at its default."""
    monkeypatch.delenv(_weights.ENV_VAR, raising=False)
    monkeypatch.setitem(_weights._SWITCH, "enabled", None)
    _weights.clear_model_cache()
    yield
    _weights.clear_model_cache()


class _Network:
    """A stand-in network: weak-referenceable, unlike ``object()``."""


class _Loader:
    """A loader that counts its calls and returns a new network each time."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _Network()


# -- the cache itself ---------------------------------------------------------


def test_a_key_loads_once():
    load = _Loader()
    key = _weights.key("tabicl", "v2", "/weights.ckpt", "cpu", "float32")
    first = _weights.network(key, load)
    assert _weights.network(key, load) is first
    assert len(load.calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"version": "v3"},
        {"checkpoint": "/other.ckpt"},
        {"device": "cuda"},
        {"precision": "torch.bfloat16"},
        {"extra": ("fit_preprocessors",)},
    ],
)
def test_anything_that_changes_the_weights_loads_again(change):
    settings = {
        "backend": "tabpfn",
        "version": "v3.5",
        "checkpoint": "/weights.ckpt",
        "device": "cpu",
        "precision": "torch.float32",
        "extra": ("fit_with_cache",),
    }
    load = _Loader()
    networks = []
    for params in (settings, {**settings, **change}):
        positional = [
            value for name, value in params.items() if name != "extra"
        ]
        key = _weights.key(*positional, *params["extra"])
        networks.append(_weights.network(key, load))
    assert networks[0] is not networks[1]
    assert len(load.calls) == 2


def test_the_key_carries_the_pinned_revision():
    key = _weights.key("tabpfn", "v3.5", "/w.ckpt", "cpu", "autocast")
    assert key[3] == "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
    assert _weights.key("custom", "v0", "/w", "cpu", "x")[3] is None


def test_clearing_frees_the_networks():
    key = _weights.key("limix", "v2", "/w.ckpt", "cpu", "float32")
    alive = weakref.ref(_weights.network(key, _Loader()))
    assert alive() is not None
    assert _weights.clear_model_cache() == 1
    assert alive() is None
    assert not _weights.cached_keys()
    assert _weights.clear_model_cache() == 0


def test_turned_off_every_call_loads(monkeypatch):
    key = _weights.key("tabicl", "v2", "/w.ckpt", "cpu", "float32")
    _weights.network(key, _Loader())
    _weights.set_model_cache(False)
    assert not _weights.model_cache_enabled()
    assert not _weights.cached_keys()  # turning it off lets go
    load = _Loader()
    assert _weights.network(key, load) is not _weights.network(key, load)
    assert len(load.calls) == 2
    _weights.set_model_cache(True)
    assert _weights.network(key, load) is _weights.network(key, load)


@pytest.mark.parametrize("value", ["0", "false", "Off", "no"])
def test_the_environment_variable_turns_it_off(monkeypatch, value):
    monkeypatch.setenv(_weights.ENV_VAR, value)
    assert not _weights.model_cache_enabled()
    monkeypatch.setenv(_weights.ENV_VAR, "1")
    assert _weights.model_cache_enabled()
    _weights.set_model_cache(False)  # the function wins over the variable
    assert not _weights.model_cache_enabled()


def test_threads_asking_at_once_load_once():
    calls = []
    started = threading.Barrier(4)

    def load():
        calls.append(None)
        return _Network()

    key = _weights.key("tabicl", "v2", "/w.ckpt", "cpu", "float32")
    results = []

    def ask():
        started.wait()
        results.append(_weights.network(key, load))

    threads = [threading.Thread(target=ask) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1
    assert all(result is results[0] for result in results)


# -- each backend, with its loader stubbed -------------------------------------


def _tabpfn(device="cpu", mixed_precision=False, version="v3.5"):
    """A TabPFN estimator in the state fit leaves before loading."""
    pytest.importorskip("torch")
    est = tabpfn.TabPFNBarDistribution(version=version)
    est.device_ = device
    est.mixed_precision_ = mixed_precision
    est.checkpoint_ = pathlib.Path(f"/hub/tabpfn-{version}.safetensors")
    return est


def _tabpfn_fit(est, load, fit_mode="fit_with_cache"):
    """What one regressor's fit asks upstream's loader for."""
    upstream = types.SimpleNamespace(
        base=types.SimpleNamespace(initialize_tabpfn_model=load)
    )
    with est._shared_network(upstream):
        models, *_ = upstream.base.initialize_tabpfn_model(
            model_path=est.checkpoint_,
            which="regressor",
            fit_mode=fit_mode,
            softmax_temperature_override=None,
            n_estimators_override=3,
        )
    assert upstream.base.initialize_tabpfn_model is load
    return models[0]


def _tabpfn_loader():
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return [_Network()], [{"emsize": 8}], {"borders": [0, 1]}, {"T": 1}

    return load, calls


def test_tabpfn_refits_reuse_the_network():
    load, calls = _tabpfn_loader()
    first = _tabpfn_fit(_tabpfn(), load)
    assert _tabpfn_fit(_tabpfn(), load) is first
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("estimator", "fit_mode"),
    [
        ({"device": "cuda"}, "fit_with_cache"),
        ({"mixed_precision": True}, "fit_with_cache"),
        ({"version": "v3.5-fast"}, "fit_with_cache"),
        ({}, "fit_preprocessors"),
    ],
)
def test_tabpfn_loads_again_for_other_weights(estimator, fit_mode):
    load, calls = _tabpfn_loader()
    first = _tabpfn_fit(_tabpfn(), load)
    assert _tabpfn_fit(_tabpfn(**estimator), load, fit_mode) is not first
    assert len(calls) == 2


def test_tabpfn_hands_out_copies_of_what_a_fit_may_change():
    upstream = types.SimpleNamespace(base=types.SimpleNamespace())
    load, _ = _tabpfn_loader()
    upstream.base.initialize_tabpfn_model = load
    est = _tabpfn()
    with est._shared_network(upstream):
        first = upstream.base.initialize_tabpfn_model(model_path="/w")
        second = upstream.base.initialize_tabpfn_model(model_path="/w")
    assert first[0] is not second[0]  # the lists, not the network
    assert first[0][0] is second[0][0]
    for left, right in zip(first[1:], second[1:], strict=True):
        assert left == right
        assert left is not right


class _TabICLRegressor:
    """Upstream's loader contract: it sets these three attributes."""

    loads = 0

    def _load_model(self):
        type(self).loads += 1
        self.model_ = _Network()
        self.model_config_ = {"embed_dim": 8}
        self.model_path_ = pathlib.Path("/hub/tabicl.ckpt")

    def fit(self):
        self._load_model()
        return self


def _tabicl(device="cpu"):
    est = tabicl.TabICLQuantile()
    est.device_ = device
    est.checkpoint_ = pathlib.Path("/hub/tabicl-regressor-v2.ckpt")
    return est


def _tabicl_fit(est):
    regressor = _TabICLRegressor()
    with est._shared_network(regressor):
        regressor.fit()
    assert "_load_model" not in vars(regressor)
    return regressor


def test_tabicl_refits_reuse_the_network(monkeypatch):
    monkeypatch.setattr(_TabICLRegressor, "loads", 0)
    first = _tabicl_fit(_tabicl())
    second = _tabicl_fit(_tabicl())
    assert second.model_ is first.model_
    assert second.model_config_ == first.model_config_
    assert second.model_config_ is not first.model_config_
    assert _TabICLRegressor.loads == 1
    third = _tabicl_fit(_tabicl(device="cuda"))
    assert third.model_ is not first.model_
    assert _TabICLRegressor.loads == 2


def test_a_tabicl_regressor_pickles_without_the_shared_loader(monkeypatch):
    monkeypatch.setattr(_TabICLRegressor, "loads", 0)
    restored = pickle.loads(pickle.dumps(_tabicl_fit(_tabicl())))
    assert "_load_model" not in vars(restored)
    restored.fit()  # upstream's own loader again
    assert _TabICLRegressor.loads == 2


def test_tabicl_drops_the_cache_a_predict_leaves_on_the_network():
    class Network:
        _cache = "a member's key/value cache"

        def clear_cache(self):
            self._cache = None

    regressor = types.SimpleNamespace(model_=Network())
    tabicl._release_network_cache(regressor)
    assert regressor.model_._cache is None
    tabicl._release_network_cache(types.SimpleNamespace())  # no network


def _limix(device="cpu"):
    est = limix.LimiXBarDistribution()
    est.device_ = device
    est.checkpoint_ = pathlib.Path("/hub/LimiX-2.ckpt")
    return est


def test_limix_refits_reuse_the_network(monkeypatch):
    load = _Loader()
    monkeypatch.setattr(_limix_stream, "load_network", load)
    first = _limix()._network()
    assert _limix()._network() is first
    assert len(load.calls) == 1
    assert _limix(device="cuda")._network() is not first
    assert load.calls[-1] == ((pathlib.Path("/hub/LimiX-2.ckpt"), "cuda"), {})


def _tabfm_backbone(mixed_precision=False):
    est = tabfm.TabFMHistogram()
    est.device_ = "cpu"
    est.mixed_precision_ = mixed_precision
    return est._backbone()


def test_tabfm_estimators_share_the_backbone(monkeypatch):
    upstream = pytest.importorskip("tabfm")
    pytest.importorskip("torch")
    calls = []

    def load(*, model_type, device, checkpoint_path, dtype, use_cache=True):
        calls.append({"dtype": dtype, "use_cache": use_cache})
        return _Network()

    monkeypatch.setattr(upstream.tabfm_v1_0_0_pytorch, "load", load)
    checkpoint = types.SimpleNamespace(download=lambda: "/hub/tabfm")
    monkeypatch.setattr(tabfm._hub, "get_checkpoint", lambda *args: checkpoint)

    first = _tabfm_backbone()
    assert _tabfm_backbone() is first
    assert _tabfm_backbone(mixed_precision=True) is not first
    # Upstream's own process-wide cache is bypassed for lazy's.
    assert [call["use_cache"] for call in calls] == [False, False]


# -- fitted estimators --------------------------------------------------------


def test_clone_takes_no_network():
    est = _limix()
    est._network_cache = (("v2", "cpu"), _Network())
    assert "_network_cache" not in vars(sklearn_base.clone(est))


# -- with the real weights -----------------------------------------------------


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(230, 3))
    z = X @ np.array([0.4, -0.2, 0.1]) + 0.05 * rng.normal(size=230)
    return X[:200], z[:200], X[200:]


def _has_limix():
    try:
        _limix_source.locate()
    except ImportError:
        return False
    return True


_REAL = [
    pytest.param(tabpfn.TabPFNBarDistribution, {"version": "v3.5-fast"}),
    pytest.param(tabicl.TabICLQuantile, {}),
    pytest.param(
        limix.LimiXBarDistribution,
        {},
        marks=pytest.mark.skipif(not _has_limix(), reason="needs LimiX"),
    ),
]


@needs_checkpoint
@pytest.mark.parametrize(("cls", "params"), _REAL)
def test_a_shared_network_answers_bit_for_bit(cls, params, data):
    X, z, X_test = data

    def model():
        return cls(n_estimators=2, device="cpu", progress=False, **params)

    _weights.set_model_cache(False)
    fresh = model().fit(X, z).predict_proba(X_test)
    _weights.set_model_cache(True)
    first = model().fit(X, z)
    assert len(_weights.cached_keys()) == 1
    # Another context in between must leave no trace on the network.
    model().fit(X[:120], np.sin(z[:120])).predict_proba(X_test)
    second = model().fit(X, z)
    assert len(_weights.cached_keys()) == 1
    np.testing.assert_array_equal(first.predict_proba(X_test), fresh)
    np.testing.assert_array_equal(second.predict_proba(X_test), fresh)
    restored = pickle.loads(pickle.dumps(second))
    np.testing.assert_array_equal(restored.predict_proba(X_test), fresh)
    refit = sklearn_base.clone(second).fit(X, z)
    np.testing.assert_array_equal(refit.predict_proba(X_test), fresh)
    assert len(_weights.cached_keys()) == 1
