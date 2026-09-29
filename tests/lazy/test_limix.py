# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LimiXBarDistribution: what the uniform conformance suite does not cover."""

import os
import pathlib
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest

import lazy
from lazy.models import _limix_source
from lazy.models import _limix_stream
from lazy.models import limix


def _has_limix() -> bool:
    try:
        _limix_source.locate()
    except ImportError:
        return False
    return True


needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1" or not _has_limix(),
    reason="needs LimiX's source and LAZY_RUN_CHECKPOINT_TESTS=1",
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    z = rng.uniform(0.1, 1.5, 230)
    X = np.column_stack(
        [np.sin(z * k) + rng.normal(0, 0.05, z.size) for k in (1, 2, 3)]
    )
    X[::9, 1] = np.nan
    return X[:200], z[:200], X[200:]


def _model(**params):
    settings = {
        "n_estimators": 2,
        "device": "cpu",
        "progress": False,
        "chunk_size": 11,
    }
    return limix.LimiXBarDistribution(**{**settings, **params})


def test_constructing_needs_nothing_installed():
    est = limix.LimiXBarDistribution()
    assert est.get_params()["random_state"] == 0
    assert est.recommended_max_context == 20_000


def _missing_source():
    raise ImportError("LimiX not found: pip install 'lazy-tfm[limix]'")


def test_fitting_without_the_source_says_how_to_install_it(monkeypatch, data):
    monkeypatch.setattr(_limix_source, "locate", _missing_source)
    monkeypatch.setattr(_limix_source, "_loaded", {})
    X, z, _ = data
    with pytest.raises(ImportError, match=r"lazy-tfm\[limix\]"):
        _model().fit(X, z)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"softmax_temperature": 0}, "softmax_temperature"),
        ({"softmax_temperature": "hot"}, "softmax_temperature"),
        ({"mixed_precision": "yes"}, "mixed_precision"),
    ],
)
def test_backend_parameters_are_checked(params, message):
    with pytest.raises(ValueError, match=message):
        _model(**params)._check_backend_params()


def test_standardisation_is_upstreams():
    y = np.array([1.0, 2.0, 4.0])
    assert limix._standardisation(y) == (y.mean(), y.std(ddof=1))
    assert limix._standardisation(np.ones(4)) == (1.0, 1.0)


@needs_checkpoint
def test_the_cache_matches_upstreams_forward_pass(data):
    X, z, X_test = data
    model = _model().fit(X, z)
    network = model._network()
    entry = model.handles_[0]["members"][0]
    queries = entry["pipeline"].transform(X_test)
    cached = _limix_stream.decode(
        network, entry["member"], entry["cache"], queries, mixed_precision=True
    )
    uncached = _limix_stream.forward(
        network, entry["member"], queries, mixed_precision=True
    )
    np.testing.assert_allclose(cached, uncached, atol=1e-4)


@needs_checkpoint
def test_upstreams_recipe_is_the_default(data):
    X, z, _ = data
    model = _model(n_estimators=8).fit(X, z)
    tokens = [entry["pipeline"].token for entry in model.handles_[0]["members"]]
    assert tokens == list(limix._limix_preprocess.AUTO_TOKENS)
    assert model.provenance_["attribution"] == "Built with StableAI LimiX"
    assert model.provenance_["source_commit"] is not None


@needs_checkpoint
def test_the_native_grid_uses_the_whole_context(data):
    X, z, _ = data
    model = _model(bag_size=0.5).fit(X, z)
    mean, std = z.mean(), z.std(ddof=1)
    edges = model.native_grid_.edges
    np.testing.assert_allclose(
        edges, np.unique(model.borders_ * std + mean), rtol=1e-12
    )
    assert model.n_buckets_ == 5000


@needs_checkpoint
def test_a_cache_that_does_not_fit_falls_back(monkeypatch, data):
    X, z, X_test = data
    monkeypatch.setattr(limix, "_free_device_memory", lambda network: 1)
    with pytest.warns(
        lazy.PerformanceWarning, match="2 of its 2 members without"
    ):
        fallback = _model().fit(X, z)
    assert fallback.kv_cache_ is False
    for handle in fallback.handles_:
        assert all(entry["cache"] is None for entry in handle["members"])
    np.testing.assert_array_equal(
        fallback.predict_proba(X_test),
        _model(kv_cache=False).fit(X, z).predict_proba(X_test),
    )


@needs_checkpoint
def test_a_pickled_model_predicts_the_same(data):
    X, z, X_test = data
    model = _model().fit(X, z)
    before = model.predict_proba(X_test)
    restored = pickle.loads(pickle.dumps(model))
    assert "_network_cache" not in vars(restored)
    np.testing.assert_array_equal(restored.predict_proba(X_test), before)


@needs_checkpoint
def test_a_large_unbagged_context_warns(monkeypatch, data):
    X, z, _ = data
    monkeypatch.setattr(
        limix.LimiXBarDistribution, "recommended_max_context", 150
    )
    with pytest.warns(lazy.ContextSizeWarning, match="bag_size=150"):
        _model().fit(X, z)
    with warnings.catch_warnings():
        warnings.simplefilter("error", lazy.ContextSizeWarning)
        _model(bag_size=150).fit(X, z)


def test_an_unfitted_model_pickles_without_the_source(monkeypatch):
    monkeypatch.setattr(_limix_source, "load", _missing_source)
    restored = pickle.loads(pickle.dumps(_model(n_estimators=3)))
    assert restored.get_params()["n_estimators"] == 3


_UNPICKLE_AND_PREDICT = """
import pickle, sys
import numpy as np
with open(sys.argv[1], "rb") as f:
    model = pickle.load(f)
np.save(sys.argv[3], model.predict_proba(np.load(sys.argv[2])))
"""


@needs_checkpoint
def test_a_pickled_model_predicts_the_same_in_a_fresh_process(data, tmp_path):
    X, z, X_test = data
    model = _model().fit(X, z)
    with open(tmp_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save(tmp_path / "X.npy", X_test)
    subprocess.run(
        [
            sys.executable,
            "-c",
            _UNPICKLE_AND_PREDICT,
            str(tmp_path / "model.pkl"),
            str(tmp_path / "X.npy"),
            str(tmp_path / "out.npy"),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "out.npy"), model.predict_proba(X_test)
    )


@needs_checkpoint
def test_an_unpickled_model_finds_its_device_and_checkpoint_afresh(
    monkeypatch, data
):
    X, z, X_test = data
    model = _model().fit(X, z)
    before = model.predict_proba(X_test)
    checkpoint = model.checkpoint_
    # As if fitted with device="auto" on another machine's GPU and cache.
    model.device = "auto"
    model.device_ = "cuda:7"
    model.checkpoint_ = pathlib.Path("/elsewhere/LimiX-2.ckpt")
    pickled = pickle.dumps(model)
    monkeypatch.setattr(limix._device, "resolve_device", lambda device: "cpu")
    restored = pickle.loads(pickled)
    assert restored.device_ == "cpu"
    assert restored.provenance_["device"] == "cpu"
    np.testing.assert_array_equal(restored.predict_proba(X_test), before)
    assert restored.checkpoint_ == checkpoint


def _simulated_gpu(monkeypatch, free_bytes, cache_bytes):
    """Fakes a GPU whose free memory shrinks as caches are made."""
    held = [0]
    calls = []

    def free(network):
        calls.append(held[0])
        return free_bytes - held[0]

    prefill = _limix_stream.prefill

    def counted_prefill(*args, **kwargs):
        held[0] += cache_bytes
        return prefill(*args, **kwargs)

    monkeypatch.setattr(limix, "_free_device_memory", free)
    monkeypatch.setattr(
        _limix_stream, "cache_bytes", lambda *args, **kwargs: cache_bytes
    )
    monkeypatch.setattr(_limix_stream, "prefill", counted_prefill)
    return held, calls


@needs_checkpoint
def test_the_caches_that_fit_are_kept(monkeypatch, data):
    X, z, X_test = data
    # 0.6 of 10 units free leaves room for 6 of the 8 members' caches.
    held, calls = _simulated_gpu(monkeypatch, 10, 1)
    with pytest.warns(lazy.PerformanceWarning) as record:
        model = _model(n_estimators=8).fit(X, z)
    assert len(record) == 1
    assert "2 of its 8 members without" in str(record[0].message)
    assert record[0].filename == __file__
    assert calls == [0]  # measured once, before the first cache
    assert held[0] == 6
    assert model.kv_cache_ is True
    uses = [entry["use_cache"] for entry in model.handles_[0]["members"]]
    assert uses == [True] * 6 + [False] * 2
    uncached = _model(n_estimators=8, kv_cache=False).fit(X, z)
    np.testing.assert_allclose(
        model.predict_proba(X_test),
        uncached.predict_proba(X_test),
        rtol=limix.LimiXBarDistribution.kv_cache_rtol,
        atol=1e-7,
    )


@needs_checkpoint
def test_an_unpickled_model_rebuilds_only_the_caches_that_fit(
    monkeypatch, data
):
    X, z, X_test = data
    model = _model(n_estimators=4).fit(X, z)
    before = model.predict_proba(X_test)
    restored = pickle.loads(pickle.dumps(model))
    held, calls = _simulated_gpu(monkeypatch, 5, 1)  # room for 3 of 4
    with pytest.warns(lazy.PerformanceWarning, match="1 of its 4 members"):
        after = restored.predict_proba(X_test)
    assert calls == [0]
    assert held[0] == 3
    np.testing.assert_allclose(after, before, rtol=1e-4, atol=1e-7)


def test_fitting_without_torch_says_how_to_install_limix(monkeypatch, data):
    monkeypatch.setitem(sys.modules, "torch", None)  # as if not installed
    X, z, _ = data
    with pytest.raises(ImportError, match=r"lazy-tfm\[limix\]") as caught:
        _model().fit(X, z)
    assert "LimiX @ git+" in str(caught.value)
