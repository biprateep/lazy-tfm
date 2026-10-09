# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""TabICL: the network fits share, and pickles that leave it out.

The behavioral contract runs on the fake upstream (tests/lazy/fakes/tabicl.py)
in the conformance and bagging suites. Here are the consequences of sharing
one network between models, which upstream does not expect: it keeps a
model's cache on the network itself, and its pickles reload a network of
their own, from the pickled path, as soon as they are unpickled.
"""

import io
import os
import pathlib
import pickle
import threading
import time

import fakes
from fakes import tabicl as fake_tabicl
import numpy as np
import pytest

import lazy
from lazy.models import _device
from lazy.models import _hub

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)


@pytest.fixture
def data():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(80, 3))
    return X[:60], X[:60, 0] + 0.1 * rng.normal(size=60), X[60:]


def _in_threads(function, n):
    """Runs ``function(i)`` for each i in range(n), each in its own thread."""
    answers = [None] * n
    errors = []

    def run(i):
        try:
            answers[i] = function(i)
        except BaseException as error:  # noqa: BLE001 - reported below.
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    return answers


@pytest.mark.parametrize("kv_cache", [True, "repr"])
def test_the_shared_network_serves_one_context_at_a_time(
    monkeypatch, data, kv_cache
):
    """Two models on one network, predicting from two threads."""
    X, y, X_test = data
    backend = fakes.Backend("tabicl", monkeypatch)
    models = [
        backend(kv_cache=kv_cache).fit(X, y),
        backend(kv_cache=kv_cache).fit(X + 1.0, -y),
    ]
    expected = [m.predict_distribution(X_test).locs for m in models]
    network = models[0].regressor_.model_
    assert models[1].regressor_.model_ is network
    cached_context = type(network).cached_context
    active = []

    def slow_cached_context(self):
        active.append(1)
        assert len(active) == 1, "two contexts on the network at once"
        time.sleep(0.01)
        try:
            return cached_context(self)
        finally:
            active.pop()

    monkeypatch.setattr(type(network), "cached_context", slow_cached_context)
    answers = _in_threads(
        lambda i: models[i].predict_distribution(X_test).locs, 2
    )
    for answer, reference in zip(answers, expected, strict=True):
        np.testing.assert_array_equal(answer, reference)
    assert network._cache is None


def test_fits_in_threads_record_their_own_caches(monkeypatch, data):
    X, y, X_test = data
    backend = fakes.Backend("tabicl", monkeypatch)
    contexts = [(X, y), (X + 1.0, -y)]
    expected = [
        backend().fit(*context).predict_distribution(X_test).locs
        for context in contexts
    ]
    network = backend().fit(X, y).regressor_.model_
    cached_context = type(network).cached_context

    def slow_cached_context(self):
        time.sleep(0.01)
        return cached_context(self)

    monkeypatch.setattr(type(network), "cached_context", slow_cached_context)
    models = _in_threads(lambda i: backend().fit(*contexts[i]), 2)
    for model, reference in zip(models, expected, strict=True):
        assert model.regressor_.model_ is network
        np.testing.assert_array_equal(
            model.predict_distribution(X_test).locs, reference
        )


class _Classes(pickle.Unpickler):
    """Unpickles, recording every class the pickle refers to."""

    def __init__(self, blob):
        super().__init__(io.BytesIO(blob))
        self.found = set()

    def find_class(self, module, name):
        found = super().find_class(module, name)
        self.found.add(found)
        return found


def _regressors(model):
    regressors = []
    for handle in model.handles_:
        regressors.extend(handle.regressors)
    return regressors


@pytest.mark.parametrize("kv_cache", [True, False])
def test_a_pickle_holds_no_network_and_shares_one_again(
    monkeypatch, data, kv_cache
):
    X, y, X_test = data
    backend = fakes.Backend("tabicl", monkeypatch)
    model = backend(n_estimators=4, bag_size=40, kv_cache=kv_cache).fit(X, y)
    expected = model.predict_distribution(X_test).locs
    loads = len(fake_tabicl.Regressor.recorder.loads)
    blob = pickle.dumps(model)
    assert len(blob) < fake_tabicl.N_WEIGHTS  # bytes: an eighth of one
    unpickler = _Classes(blob)
    restored = unpickler.load()
    assert fake_tabicl.Network not in unpickler.found
    assert not any(hasattr(r, "model_") for r in _regressors(restored))
    np.testing.assert_array_equal(
        restored.predict_distribution(X_test).locs, expected
    )
    networks = {id(r.model_) for r in _regressors(restored)}
    assert networks == {id(model.handles_[0].regressors[0].model_)}
    assert len(fake_tabicl.Regressor.recorder.loads) == loads


def test_a_pickle_finds_the_checkpoint_where_it_is_now(monkeypatch, data):
    X, y, X_test = data
    backend = fakes.Backend("tabicl", monkeypatch)
    model = backend(n_estimators=4, bag_size=40).fit(X, y)
    expected = model.predict_distribution(X_test).locs
    blob = pickle.dumps(model)
    elsewhere = pathlib.Path("/nonexistent/elsewhere")

    def moved(checkpoint, *, local_files_only=False):
        del local_files_only  # Unused: nothing is fetched.
        return elsewhere / checkpoint.filename

    monkeypatch.setattr(_hub.Checkpoint, "download", moved)
    loads = len(fake_tabicl.Regressor.recorder.loads)
    restored = pickle.loads(blob)
    np.testing.assert_array_equal(
        restored.predict_distribution(X_test).locs, expected
    )
    assert restored.checkpoint_.parent == elsewhere
    assert {r.model_path for r in _regressors(restored)} == {
        str(restored.checkpoint_)
    }
    # One network for every regressor of the bagged model.
    assert len(fake_tabicl.Regressor.recorder.loads) == loads + 1


def test_an_auto_device_is_resolved_again_on_unpickling(monkeypatch, data):
    X, y, X_test = data
    model = fakes.Backend("tabicl", monkeypatch)(device="auto").fit(X, y)
    expected = model.predict_distribution(X_test).locs
    model.device_ = "cuda:7"  # as if fitted there
    for regressor in _regressors(model):
        regressor.device = regressor.device_ = "cuda:7"
    restored = pickle.loads(pickle.dumps(model))
    device = _device.resolve_device("auto")
    assert restored.device_ == device
    assert restored.provenance_["device"] == device
    assert restored.mixed_precision_ == device.startswith("cuda")
    assert restored.provenance_["tabicl"]["use_amp"] == (
        restored.mixed_precision_
    )
    for regressor in _regressors(restored):
        assert regressor.device_ == device
        assert regressor.use_amp == restored.mixed_precision_
    np.testing.assert_array_equal(
        restored.predict_distribution(X_test).locs, expected
    )


@needs_checkpoint
def test_a_pickle_predicts_bit_for_bit_on_a_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabicl")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(140, 3))
    y = 2.0 + X[:, 0] + 0.1 * rng.normal(size=140)
    model = lazy.TabICLQuantile(
        n_estimators=4, bag_size=60, device="cpu", progress=False
    ).fit(X[:100], y[:100])
    expected = model.predict_distribution(X[100:]).locs
    blob = pickle.dumps(model)
    # The key/value caches stay in, but no copy of the weights.
    assert len(blob) < model.checkpoint_.stat().st_size / 2
    unpickler = _Classes(blob)
    restored = unpickler.load()
    assert not any(
        issubclass(found, torch.nn.Module)
        for found in unpickler.found
        if isinstance(found, type)
    )
    np.testing.assert_array_equal(
        restored.predict_distribution(X[100:]).locs, expected
    )
    networks = {id(r.model_) for r in _regressors(restored)}
    assert networks == {id(model.handles_[0].regressors[0].model_)}
