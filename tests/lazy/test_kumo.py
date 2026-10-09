# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Kumo Tabular: the recipe pinned here is upstream's, and so is the answer.

The behavioral contract runs on the fake upstream (tests/lazy/fakes/kumo.py)
in the conformance and bagging suites. Here the pinned recipe is compared
with the one upstream ships, and, with the checkpoint, the backend's answer
with upstream's own default path.
"""

import os
import pickle
import re
import sys
import threading
import time
import types

import fakes
import numpy as np
import pytest

import lazy
from lazy.models import _device
from lazy.models import kumo

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)

# The categorical branch of upstream's recipe: lazy hands Kumo numeric
# columns only, so its recipe has none.
_CATEGORICAL = re.compile(r"\n( +)categorical=.*?\n\1\),", flags=re.DOTALL)


def _upstream_recipe():
    sdm = pytest.importorskip("sdm")
    return sdm.models.KumoTabular.default_recipe()


def _auto_recipe():
    sp = pytest.importorskip("sdm.processing")
    return kumo._recipe(
        sp,
        kumo.AUTO_TRANSFORMS,
        feature_shuffle=True,
        outlier_threshold=kumo.AUTO_OUTLIER_THRESHOLD,
    )


def _find(module, kind):
    return [m for m in module.modules() if type(m).__name__ == kind]


def test_the_pinned_features_and_target_are_upstreams_numeric_branch():
    upstream, pinned = _upstream_recipe(), _auto_recipe()
    for role in ("features", "target"):
        expected = _CATEGORICAL.sub("", repr(getattr(upstream, role)))
        assert repr(getattr(pinned, role)) == expected
    ((upstream_select,), (pinned_select,)) = (
        _find(upstream.features, "SelectColumns"),
        _find(pinned.features, "SelectColumns"),
    )
    assert vars(pinned_select) == vars(upstream_select)


def test_the_pinned_output_keeps_the_members_upstream_trims():
    upstream, pinned = _upstream_recipe(), _auto_recipe()
    trims = {
        m.trim_fraction for m in _find(upstream.output, "AverageEstimators")
    }
    assert kumo.AUTO_TRIM_FRACTION in trims
    assert kumo.KumoQuantile.member_trim_fraction == kumo.AUTO_TRIM_FRACTION
    assert _find(pinned.output, "SortQuantiles")
    assert not _find(pinned.output, "AverageEstimators")


def test_an_explicit_recipe_drops_the_clip_and_the_shuffle():
    sp = pytest.importorskip("sdm.processing")
    recipe = kumo._recipe(
        sp, ("identity",), feature_shuffle=False, outlier_threshold=None
    )
    assert not _find(recipe.features, "ClipSigma")
    assert not _find(recipe.features, "ShuffleColumns")
    assert _find(recipe.features, "FlipSign")


def test_members_cycle_through_their_transforms():
    assert kumo._cycle(("identity", "power", "identity")) == (
        "identity",
        "power",
    )
    with pytest.raises(ValueError, match="cycles its transforms"):
        kumo._cycle(("power", "power", "identity"))
    with pytest.raises(ValueError, match="no transform 'yeo'"):
        kumo._transform(None, "yeo")


def test_every_version_is_a_size_with_a_pinned_checkpoint():
    assert sorted(kumo.SIZES) == lazy.list_versions("kumo")
    for version, size in kumo.SIZES.items():
        checkpoint = lazy.get_checkpoint("kumo", version)
        assert checkpoint.filename == f"{size}/regressor.pt"
        assert kumo.KumoQuantile._pinned_recipe(version)["size"] == size


@pytest.fixture
def data():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(80, 3))
    return X[:60], X[:60, 0] + 0.1 * rng.normal(size=60), X[60:]


def test_a_pickle_leaves_the_caches_out_and_records_them_again(
    monkeypatch, data
):
    X, y, X_test = data
    model = fakes.Backend("kumo", monkeypatch)(n_estimators=3).fit(X, y)
    expected = model.predict_distribution(X_test).locs
    restored = pickle.loads(pickle.dumps(model))
    assert all(handle.cache is None for handle in restored.handles_)
    assert restored.regressor_ is restored.handles_[0]
    assert all(handle.cache is not None for handle in model.handles_)
    np.testing.assert_array_equal(
        restored.predict_distribution(X_test).locs, expected
    )
    assert restored.handles_[0].cache is not None


def test_an_auto_device_is_resolved_again_on_unpickling(monkeypatch, data):
    X, y, _ = data
    model = fakes.Backend("kumo", monkeypatch)().fit(X, y)
    model.device, model.device_ = "auto", "cuda:7"  # as if fitted there
    restored = pickle.loads(pickle.dumps(model))
    assert restored.device_ == _device.resolve_device("auto")
    assert restored.provenance_["device"] == restored.device_
    assert restored.mixed_precision_ == restored.device_.startswith("cuda")


def test_the_shared_network_serves_one_context_at_a_time(monkeypatch, data):
    """Two models on one network, predicting from two threads."""
    X, y, X_test = data
    backend = fakes.Backend("kumo", monkeypatch)
    models = [backend().fit(X, y), backend().fit(X, -y)]
    expected = [m.predict_distribution(X_test).locs for m in models]
    network = models[0]._network()
    assert models[1]._network() is network
    predict = type(network).predict
    active = []

    def slow_predict(self, x, **kwargs):
        active.append(1)
        assert len(active) == 1, "two contexts on the network at once"
        time.sleep(0.01)
        try:
            return predict(self, x, **kwargs)
        finally:
            active.pop()

    monkeypatch.setattr(type(network), "predict", slow_predict)
    answers = [None, None]
    errors = []

    def run(i):
        try:
            answers[i] = models[i].predict_distribution(X_test).locs
        except BaseException as error:  # noqa: BLE001 - reported below.
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for answer, reference in zip(answers, expected, strict=True):
        np.testing.assert_array_equal(answer, reference)


def test_a_missing_sdm_names_lazy_setup(monkeypatch):
    monkeypatch.setitem(sys.modules, "sdm", None)
    with pytest.raises(ImportError, match="then lazy setup kumo"):
        kumo.KumoQuantile()._import_backend()


def test_another_package_called_sdm_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "sdm", types.ModuleType("sdm"))
    with pytest.raises(ImportError, match="has no models.KumoTabular"):
        kumo.KumoQuantile()._import_backend()


def test_the_trim_follows_the_versions_recipe(monkeypatch, data):
    X, y, X_test = data
    cls = fakes.install("kumo", monkeypatch)
    pinned = cls._pinned_recipe

    def recipe(version):
        return {**pinned(version), "trim_fraction": 0.4}

    monkeypatch.setattr(cls, "_pinned_recipe", staticmethod(recipe))
    settings = {**fakes.settings("kumo"), "n_estimators": 5}
    model = cls(**settings, device="cpu", progress=False).fit(X, y)
    (handle,) = model.handles_
    members = model._predict_members(handle, X_test)
    np.testing.assert_array_equal(
        model.predict_distribution(X_test).locs,
        lazy.distributions.QuantileDistribution.trimmed_average(
            members, 0.4
        ).locs,
    )
    assert model.provenance_["kumo"]["trim_fraction"] == 0.4


@needs_checkpoint
def test_the_default_path_is_upstreams_on_a_cpu():
    """The trim is lazy's, in float64; upstream's runs in float32."""
    sdm = pytest.importorskip("sdm")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(260, 4))
    y = 2.0 + np.sin(2.0 * X[:, 0]) + 0.1 * rng.normal(size=260)
    model = lazy.KumoQuantile(
        version="v1-small", n_estimators=5, device="cpu", progress=False
    ).fit(X[:200], y[:200])
    ours = model.predict_distribution(X[200:]).locs
    network = sdm.models.KumoTabular(
        task="regression", size="small", device="cpu"
    )
    offset, scale = y[:200].mean(), y[:200].std()
    (group,) = model.member_groups_
    theirs = (
        network(
            x_context=torch.tensor(X[:200]),
            y_context=torch.tensor(
                ((y[:200] - offset) / scale)[:, None]
            ).float(),
            x_query=torch.tensor(X[200:]),
            num_estimators=5,
            generator=torch.Generator().manual_seed(group.seed),
        )
        .numerical.double()
        .numpy()
    )
    np.testing.assert_allclose(ours, theirs * scale + offset, atol=1e-5)


@needs_checkpoint
@pytest.mark.parametrize("kv_cache", [True, False])
def test_one_row_chunks_change_the_answer_by_rounding_alone(kv_cache):
    """A one-row chunk takes another matrix-multiply path: not bit for bit."""
    pytest.importorskip("sdm")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(112, 3))
    y = 2.0 + X[:, 0] + 0.1 * rng.normal(size=112)
    model = lazy.KumoQuantile(
        version="v1-small",
        n_estimators=4,
        device="cpu",
        progress=False,
        chunk_size=0,
        kv_cache=kv_cache,
    ).fit(X[:100], y[:100])
    whole = model.predict_distribution(X[100:]).locs
    model.chunk_size = 1
    single = model.predict_distribution(X[100:]).locs
    if model.exact_chunking:
        np.testing.assert_array_equal(single, whole)
    spread = (whole[:, -1] - whole[:, 0]).min()
    np.testing.assert_allclose(single, whole, rtol=0, atol=1e-4 * spread)
