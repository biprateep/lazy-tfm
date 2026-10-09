# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""TabICL: the network fits share, used by one model at a time.

The behavioral contract runs on the fake upstream (tests/lazy/fakes/tabicl.py)
in the conformance and bagging suites. Here are the consequences of sharing
one network between models, which upstream does not expect: it keeps a
model's cache on the network itself.
"""

import threading
import time

import fakes
import numpy as np
import pytest


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
