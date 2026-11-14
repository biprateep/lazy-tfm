# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Every backend supports the uniform features, the same way (tier A).

The behavioral tests (the grid, quantiles, seeds, chunking, the cache,
transforms, bagging, pickling) run on the backend-free stand-ins, which
implement exactly the per-backend interface, and on every registered backend
with its upstream model replaced by its fake (tests/lazy/fakes): the
backend's own code runs, on a CPU, with no checkpoint. A fake that needs its
backend's upstream package skips without it. The registry's static contract
is checked on the real classes. Nothing here needs a GPU or a checkpoint.
"""

import builtins
import copy
import inspect
import pathlib
import pickle
import re
import sys
import threading
import warnings

import fakes
import numpy as np
import pandas as pd
import pytest
from sklearn import base as sklearn_base
from sklearn import exceptions
import standins
import threads

import lazy
from lazy.models import _conformance
from lazy.models import _device
from lazy.models import _ensemble
from lazy.models import _members
from lazy.models import _progress
from lazy.models import _transforms
from lazy.models import _weights
from lazy.models import registry


@pytest.fixture
def data():
    rng = np.random.default_rng(5)
    z = rng.uniform(0.1, 1.6, 90)
    X = np.column_stack(
        [np.sin(z * k) + rng.normal(0, 0.05, z.size) for k in (1, 2, 3)]
    )
    return X[:70], z[:70], X[70:]


@pytest.fixture(
    params=[*standins.STANDINS, *sorted(lazy.ESTIMATORS)],
    ids=lambda model: model if isinstance(model, str) else model.__name__,
)
def cls(request, monkeypatch):
    """A stand-in class, or a registered backend on its fake upstream.

    Either is called like a class; the backend is built small, on a CPU.
    """
    if isinstance(request.param, str):
        return fakes.Backend(request.param, monkeypatch)
    return request.param


# -- the parameter contract --------------------------------------------------


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_the_uniform_parameters_are_keyword_only_with_fixed_defaults(cls):
    parameters = inspect.signature(cls.__init__).parameters
    assert not any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    for name in _ensemble.UNIFORM_PARAMS:
        assert name in parameters, name
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    for name, default in _ensemble.UNIFORM_DEFAULTS.items():
        assert parameters[name].default == default, name


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_clone_reproduces_every_parameter(cls):
    model = cls(n_estimators=3, transforms="limix", bag_size=0.5)
    assert sklearn_base.clone(model).get_params() == model.get_params()


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"kv_cache": "yes"}, "kv_cache must be one of"),
        ({"kv_cache": 1}, "kv_cache must be one of"),
        ({"transforms": "banana"}, "unknown transform"),
        ({"transforms": ()}, "empty sequence"),
        ({"bag_size": 0}, "must be positive"),
        ({"bag_size": 1.5}, r"in \(0, 1\]"),
        ({"chunk_size": -1}, "chunk_size"),
        ({"n_estimators": 0}, "positive int"),
        ({"feature_shuffle": "yes"}, "feature_shuffle"),
        ({"random_state": "seven"}, "random_state"),
        ({"random_state": 1.5}, "random_state"),
    ],
)
def test_bad_uniform_parameters_are_rejected_at_fit(data, params, match):
    X, z, _ = data
    model = standins.HistogramStandIn(**params)  # construction checks nothing
    with pytest.raises(ValueError, match=match):
        model.fit(X, z)


# -- the large-context warning -----------------------------------------------


class _Limited(standins.HistogramStandIn):
    recommended_max_context = 50


def test_a_large_unbagged_context_warns(data):
    X, z, _ = data
    with pytest.warns(lazy.ContextSizeWarning, match="bag_size=50") as record:
        _Limited().fit(X, z)
    assert "degrades on contexts larger than about 50 rows" in str(
        record[0].message
    )


class _Pretrained(_Limited):
    context_limit_note = "was pretrained on tables of up to {limit:,} rows"


def test_the_warning_says_what_is_known_of_the_limit(data):
    X, z, _ = data
    with pytest.warns(lazy.ContextSizeWarning, match="bag_size=50") as record:
        _Pretrained().fit(X, z)
    message = str(record[0].message)
    assert "was pretrained on tables of up to 50 rows, and this one" in message
    assert "degrades" not in message


def test_bagging_silences_the_warning(data):
    X, z, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("error", lazy.ContextSizeWarning)
        _Limited(bag_size=50).fit(X, z)
        _Limited().fit(X[:50], z[:50])


def test_a_bag_of_a_handful_of_rows_warns(data):
    X, z, _ = data
    with pytest.warns(UserWarning, match="bag_size=1.0 all of them"):
        standins.HistogramStandIn(bag_size=1).fit(X, z)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        standins.HistogramStandIn(bag_size=1.0).fit(X, z)
        standins.HistogramStandIn(bag_size=10).fit(X, z)
        standins.HistogramStandIn().fit(X[:5], z[:5])


def test_bags_larger_than_the_limit_still_warn(data):
    X, z, _ = data
    with pytest.warns(lazy.ContextSizeWarning, match="each bag has 60"):
        _Limited(bag_size=60).fit(X, z)


# -- the CPU warning -----------------------------------------------------------


class _SlowOnCPU(standins.HistogramStandIn):
    cpu_friendly = False


@pytest.fixture
def no_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_a_gpu_model_falling_back_to_the_cpu_warns(data, no_gpu):
    X, z, _ = data
    with pytest.warns(lazy.PerformanceWarning, match="sees no GPU") as info:
        _SlowOnCPU().fit(X, z)
    message = str(info[0].message)
    assert "LazyModel('tabicl')" in message
    assert "LazyModel('tabpfn', version='v3.5-fast')" in message
    assert "device='cpu'" in message
    assert "Supported models" in message


def test_an_explicit_cpu_or_a_cpu_friendly_model_does_not_warn(data, no_gpu):
    X, z, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("error", lazy.PerformanceWarning)
        _SlowOnCPU(device="cpu").fit(X, z)
        _SlowOnCPU(device="CPU").fit(X, z)
        standins.HistogramStandIn().fit(X, z)


def test_the_cpu_warning_suggests_every_cpu_friendly_declaration(data, no_gpu):
    expected = set()
    for name, cls in lazy.ESTIMATORS.items():
        if cls.cpu_friendly:
            expected.add(f"LazyModel({name!r})")
        expected.update(
            f"LazyModel({name!r}, version={version!r})"
            for version in cls.cpu_friendly_versions
        )
    X, z, _ = data
    with pytest.warns(lazy.PerformanceWarning, match="sees no GPU") as info:
        _SlowOnCPU().fit(X, z)
    suggested = re.findall(r"LazyModel\([^)]*\)", str(info[0].message))
    assert sorted(suggested) == sorted(expected)
    # The shipped ones: TabICL, and TabPFN's fast checkpoint.
    assert {
        "LazyModel('tabicl')",
        "LazyModel('tabpfn', version='v3.5-fast')",
    } <= expected


def test_a_cpu_friendly_version_does_not_warn():
    model = lazy.TabPFNBarDistribution(version="v3.5-fast")
    model.device_ = "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("error", lazy.PerformanceWarning)
        model._warn_if_slow_on_cpu()
    model = lazy.TabPFNBarDistribution(version="v3.5")
    model.device_ = "cpu"
    with pytest.warns(lazy.PerformanceWarning, match="v3.5-fast"):
        model._warn_if_slow_on_cpu()


def test_cpu_friendly_versions_must_be_pinned(monkeypatch):
    monkeypatch.setattr(
        lazy.TabPFNBarDistribution, "cpu_friendly_versions", ("v9",)
    )
    problems = _conformance.problems(lazy.TabPFNBarDistribution, "tabpfn")
    assert any("cpu_friendly_versions" in p for p in problems)


def test_the_tabpfn_cpu_warning_names_its_context_limit():
    model = lazy.TabPFNBarDistribution()
    model.device_ = "cpu"
    with pytest.warns(lazy.PerformanceWarning, match="5,000 context rows"):
        model._warn_if_slow_on_cpu()


# -- planning ----------------------------------------------------------------


def test_transforms_are_assigned_round_robin():
    groups = _members.plan(
        n_estimators=5,
        transforms=_transforms.parse(("none", "quantile", "power")),
        native_transforms={"none": "none", "power": "power"},
        feature_shuffle=True,
        bag_rows=10,
        n_rows=10,
        n_features=3,
        random_state=7,
        supports_native_bagging=True,
    )
    assert [g.index for g in groups] == [0, 1]
    native, scaffolded = groups
    assert [m.index for m in native.members] == [0, 2, 3]
    assert native.native_transforms == ("none", "power", "none")
    assert native.seed == _members.member_seed(7, 0)
    assert native.scaffold is None
    assert [m.index for m in scaffolded.members] == [1, 4]
    assert scaffolded.scaffold.name == "quantile"
    assert scaffolded.native_transforms == ("none", "none")
    # A group is seeded by its first member.
    assert scaffolded.seed == _members.member_seed(7, 1)


def test_native_bagging_gives_scaffolded_members_groups_of_their_own():
    """So that a scaffold is fitted on its member's bag, as everywhere."""
    groups = _members.plan(
        n_estimators=5,
        transforms=_transforms.parse(("none", "quantile", "power")),
        native_transforms={"none": "none", "power": "power"},
        feature_shuffle=True,
        bag_rows=6,
        n_rows=10,
        n_features=3,
        random_state=7,
        supports_native_bagging=True,
    )
    bags = _members.draw_bags(5, 6, 10, 7)
    permutations = _members.feature_permutations(5, 3, 7)
    assert [[m.index for m in g.members] for g in groups] == [
        [0, 2, 3],
        [1],
        [4],
    ]
    assert [g.index for g in groups] == [0, 1, 2]
    native, *alone = groups
    assert native.rows is None and native.feature_shuffle
    for member, rows in zip(native.members, native.member_rows, strict=True):
        np.testing.assert_array_equal(rows, bags[member.index])
    for group in alone:
        (member,) = group.members
        assert group.scaffold.name == "quantile"
        assert group.native_transforms == ("none",)
        np.testing.assert_array_equal(group.rows, bags[member.index])
        np.testing.assert_array_equal(
            group.permutation, permutations[member.index]
        )
        assert group.seed == _members.member_seed(7, member.index)


def test_scaffolded_bagging_gives_each_member_its_own_group():
    groups = _members.plan(
        n_estimators=6,
        transforms=None,
        native_transforms={"none": "none"},
        feature_shuffle=True,
        bag_rows=4,
        n_rows=10,
        n_features=3,
        random_state=3,
        supports_native_bagging=False,
        auto_tokens=("none", "power"),
    )
    assert len(groups) == 6
    assert [g.seed for g in groups] == [
        _members.member_seed(3, i) for i in range(6)
    ]
    assert [g.native_transforms for g in groups] == [("none",), ("power",)] * 3
    pairs = {(tuple(g.permutation), g.native_transforms) for g in groups}
    assert len(pairs) == 6  # no two members share (permutation, transform)
    assert all(len(g.rows) == 4 and not g.feature_shuffle for g in groups)


def test_bags_are_drawn_as_the_paper_drew_them():
    rng = np.random.default_rng(0)
    expected = [np.sort(rng.choice(20, 5, replace=False)) for _ in range(3)]
    for ours, theirs in zip(
        _members.draw_bags(3, 5, 20, 0), expected, strict=True
    ):
        np.testing.assert_array_equal(ours, theirs)
    # A golden value: the paper's bag50 split depends on this exact stream.
    assert _members.draw_bags(1, 5, 20, 0)[0].tolist() == expected[0].tolist()


def test_bag_sizes_resolve_as_counts_or_fractions():
    assert _members.resolve_bag_size(None, 100) == 100
    assert _members.resolve_bag_size(30, 100) == 30
    assert _members.resolve_bag_size(300, 100) == 100
    assert _members.resolve_bag_size(0.25, 100) == 25


def test_numpy_numbers_are_accepted_as_bag_sizes():
    assert _members.resolve_bag_size(np.int64(30), 100) == 30
    assert _members.resolve_bag_size(np.float32(0.25), 100) == 25
    with pytest.raises(ValueError, match="bag_size"):
        _members.resolve_bag_size(np.True_, 100)


def test_numpy_integers_are_accepted_as_counts_and_seeds(cls, data):
    """Values from np.arange or a parameter grid are NumPy integers."""
    X, z, X_test = data
    python = {"n_estimators": 3, "bag_size": 40, "random_state": 7}
    numpy = {name: np.int64(value) for name, value in python.items()}
    expected = cls(**python, chunk_size=9).fit(X, z).predict_proba(X_test)
    model = cls(**numpy, chunk_size=np.int64(9)).fit(X, z)
    np.testing.assert_array_equal(model.predict_proba(X_test), expected)
    assert type(model.provenance_["n_estimators"]) is int


# -- behavior, on every stand-in and every backend's fake ---------------------


def test_the_default_grid_is_native_and_integrates_to_one(cls, data):
    X, z, X_test = data
    model = cls().fit(X, z)
    assert model.grid_ == model.native_grid_
    assert model.grid_.normalization == "histogram"
    pdfs = model.predict_proba(X_test)
    assert pdfs.shape == (len(X_test), model.native_grid_.n_bins)
    np.testing.assert_allclose(pdfs @ model.grid_.widths, 1.0)


def test_a_constructor_grid_overrides_the_native_one(cls, data):
    X, z, X_test = data
    model = cls(y_grid=lazy.datasets.DC1_GRID).fit(X, z)
    assert model.grid_ == lazy.datasets.DC1_GRID
    assert model.predict_proba(X_test).shape == (len(X_test), 200)
    assert model.predict_proba(X_test, "native").shape[1] == (
        model.native_grid_.n_bins
    )


def test_quantiles_invert_the_distribution(cls, data):
    X, z, X_test = data
    model = cls().fit(X, z)
    levels = np.array([0.2, 0.5, 0.8])
    quantiles = model.predict_quantiles(X_test, levels)
    dist = model.predict_distribution(X_test)
    for row in range(len(X_test)):
        np.testing.assert_allclose(
            dist[row].cdf(quantiles[row])[0], levels, atol=1e-9
        )


def _refuse(*args, **kwargs):
    """A network load or group fit that a constant target must not reach."""
    raise AssertionError("a constant target ran the model")


def _assert_a_point_mass(model, value, X_test):
    """Every answer of ``model`` is a point mass at ``value``."""
    quantiles = model.predict_quantiles(X_test, [0.0, 0.16, 0.5, 0.84, 1.0])
    np.testing.assert_array_equal(quantiles, value)
    np.testing.assert_array_equal(model.predict_interval(X_test), value)
    for method in lazy.POINT_ESTIMATORS:
        np.testing.assert_allclose(
            model.predict(X_test, method), value, rtol=1e-15
        )
    grid = lazy.Grid.linear(
        value - 1.0, value + 1.5, 25, normalization="histogram"
    )
    masses = model.predict_proba(X_test, grid) * grid.widths
    expected = np.zeros(grid.n_bins)
    expected[grid.bin_index(value)] = 1.0
    np.testing.assert_allclose(masses, np.tile(expected, (len(X_test), 1)))
    np.testing.assert_array_equal(
        model.predict_pit(X_test, np.full(len(X_test), value)), 1.0
    )
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_quantiles(X_test), value)


# The std of 70 copies rounds to 4.2e-17 for 0.1, 5.6e-17 for 0.3, 0 for 7.0.
@pytest.mark.parametrize("value", [0.1, 0.3, 7.0])
def test_a_constant_target_gets_a_point_mass_without_the_model(
    cls, value, data, monkeypatch
):
    X, _, X_test = data
    monkeypatch.setattr(getattr(cls, "cls", cls), "_fit_group", _refuse)
    monkeypatch.setattr(lazy.Checkpoint, "download", _refuse)
    model = cls(n_estimators=3).fit(X, np.full(len(X), value))
    assert model.provenance_["constant_target"] == value
    _assert_a_point_mass(model, value, X_test)


def test_a_constant_target_answers_every_block_of_a_long_query(
    cls, data, monkeypatch
):
    X, _, X_test = data
    model = cls(n_estimators=1, chunk_size=4).fit(X, np.full(len(X), 0.3))
    monkeypatch.setattr(lazy.base, "_BLOCK_BYTES", 7 * 8 * model.grid_.n_bins)
    np.testing.assert_allclose(model.predict(X_test), 0.3, rtol=1e-15)
    assert np.isfinite(model.score(X_test, np.full(len(X_test), 0.3)))


def test_a_constant_targets_point_mass_converts_to_qp(cls, data):
    pytest.importorskip("qp")
    X, _, X_test = data
    model = cls(n_estimators=1).fit(X, np.full(len(X), 0.3))
    ensemble = model.predict_distribution(X_test).to_qp()
    assert ensemble.npdf == len(X_test)
    np.testing.assert_allclose(
        ensemble.cdf(np.array([0.29, 0.31])), [[0, 1]] * len(X_test), atol=1e-12
    )


@pytest.mark.parametrize("pooling", lazy.models.ensemble.POOLINGS)
def test_a_pool_of_a_constant_target_is_its_members_point_mass(
    cls, pooling, data
):
    X, _, X_test = data
    model = lazy.LazyEnsembleModel(
        [cls(n_estimators=2), standins.QuantileStandIn(n_estimators=2)],
        pooling=pooling,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", lazy.EnsembleSizeWarning)
        model.fit(X, np.full(len(X), 0.3))
    _assert_a_point_mass(model, 0.3, X_test)


@pytest.mark.parametrize("value", [0.0, 0.5, -1e4])
def test_a_constant_targets_grid_is_centred_on_it_at_its_own_scale(value, data):
    X, _, _ = data
    model = standins.HistogramStandIn().fit(X, np.full(len(X), value))
    grid = model.native_grid_
    pad = 0.01 * max(abs(value), 1.0)
    np.testing.assert_allclose(
        grid.edges, value + pad * np.array([-1, -1 / 3, 1 / 3, 1])
    )
    assert grid.centers[1] == value
    assert grid.normalization == "histogram"


def test_predict_cdf_at_a_quantile_returns_its_level(cls, data):
    X, z, X_test = data
    model = cls().fit(X, z)
    levels = np.array([0.2, 0.5, 0.8])
    quantiles = model.predict_quantiles(X_test[:4], levels)
    for row, values in enumerate(quantiles):
        for level, value in zip(levels, values, strict=True):
            # The middle bin's centre is the quantile, up to one ulp.
            edges = value + np.array([-3e-6, -1e-6, 1e-6, 3e-6])
            cdf = model.predict_cdf(
                X_test[row : row + 1], lazy.Grid.from_edges(edges)
            )
            # ppf and cdf invert each other up to float rounding, as in
            # test_quantiles_invert_the_distribution.
            np.testing.assert_allclose(cdf[0, 1], level, atol=1e-9)


@pytest.mark.parametrize("bag_size", [None, 40], ids=["unbagged", "bagged"])
@pytest.mark.parametrize(
    "y_grid",
    [lazy.datasets.DC1_GRID, lazy.Grid.linear(0.5, 1.2, 40), "native"],
    ids=["dc1", "cut", "native"],
)
def test_predict_proba_is_the_distribution_on_the_grid(
    cls, bag_size, y_grid, data
):
    X, z, X_test = data
    model = cls(n_estimators=3, bag_size=bag_size).fit(X, z)
    grid = model.native_grid if y_grid == "native" else y_grid
    np.testing.assert_allclose(
        model.predict_proba(X_test, y_grid),
        model.predict_distribution(X_test).on_grid(grid),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("bag_size", [None, 40], ids=["unbagged", "bagged"])
def test_predict_pit_is_exact_and_agrees_with_a_fine_grid(cls, bag_size, data):
    X, z, _ = data
    model = cls(n_estimators=3, bag_size=bag_size).fit(X, z)
    pit = model.predict_pit(X, z)
    assert pit.shape == (len(X),)
    assert ((pit >= 0.0) & (pit <= 1.0)).all()
    dist = model.predict_distribution(X)
    np.testing.assert_allclose(pit, np.diag(dist.cdf(z)), atol=1e-15)
    low, high = dist.ppf([0.0, 1.0]).T
    grid = lazy.Grid.linear(
        low.min() - 0.01, high.max() + 0.01, 20_000, normalization="histogram"
    )
    _, grid_pit = lazy.metrics.per_object_scores(
        z, grid, model.predict_proba(X, grid)
    )
    np.testing.assert_allclose(pit, grid_pit, atol=1e-4)


def test_interval_is_the_central_quantiles(cls, data):
    X, z, X_test = data
    model = cls().fit(X, z)
    np.testing.assert_allclose(
        model.predict_interval(X_test, coverage=0.9),
        model.predict_quantiles(X_test, [0.05, 0.95]),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="coverage"):
        model.predict_interval(X_test, coverage=1.0)


def test_chunking_changes_nothing_but_rounding(cls, data):
    """A row's answer does not depend on the rows chunked with it.

    Equal up to the last bit: on CI's x86 runners the quantile stand-in's
    answers for a 7-row chunk and for the whole set differ by one ulp in
    a few bins (bit-identical on the Arm machines the suite is developed
    on), as blocked scoring did. The checkpoint tests check each real
    backend on a CPU: bit for bit where it claims ``exact_chunking`` (none
    does), and to float rounding otherwise.
    """
    X, z, X_test = data
    whole = cls(chunk_size=0).fit(X, z).predict_proba(X_test)
    chunked = cls(chunk_size=7).fit(X, z).predict_proba(X_test)
    np.testing.assert_allclose(chunked, whole, rtol=1e-12, atol=0)


def test_the_cache_changes_nothing(cls, data):
    X, z, X_test = data
    cached = cls(kv_cache=True, bag_size=40).fit(X, z)
    uncached = cls(kv_cache=False, bag_size=40).fit(X, z)
    expected = uncached.predict_proba(X_test)
    # Tighter than a backend's kv_cache_rtol, which is its checkpoint's
    # agreement on a CPU: the fakes' cached path is the same arithmetic.
    rtol = 1e-5
    np.testing.assert_allclose(
        cached.predict_proba(X_test),
        expected,
        rtol=rtol,
        atol=rtol * expected.max(),
    )


@pytest.mark.parametrize("bag_size", [None, 40], ids=["unbagged", "bagged"])
@pytest.mark.parametrize("kv_cache", [True, False], ids=["cached", "uncached"])
def test_a_pickled_model_predicts_the_same(cls, bag_size, kv_cache, data):
    X, z, X_test = data
    model = cls(n_estimators=3, bag_size=bag_size, kv_cache=kv_cache)
    model.fit(X, z)
    expected = model.predict_proba(X_test)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(X_test), expected)
    np.testing.assert_array_equal(model.predict_proba(X_test), expected)
    assert restored.provenance_ == model.provenance_


def _limix_caches(model):
    caches = []
    for handle in model.handles_:
        caches.extend(entry["cache"] for entry in handle["members"])
    return caches


def _tabicl_caches(model):
    caches = []
    for handle in model.handles_:
        caches.extend(r.model_kv_cache_ for r in handle.regressors)
    return caches


_CACHES = {
    "kumo": lambda model: [handle.cache for handle in model.handles_],
    "limix": _limix_caches,
    "tabicl": _tabicl_caches,
}
"""Each backend's key/value caches in a fitted model, by name. TabPFN's
upstream regressors keep theirs, which its pickle still holds."""


@pytest.mark.parametrize("bag_size", [None, 40], ids=["unbagged", "bagged"])
@pytest.mark.parametrize("name", sorted(_CACHES))
def test_a_pickle_leaves_the_caches_out_and_a_prediction_rebuilds_them(
    name, bag_size, monkeypatch, data
):
    X, z, X_test = data
    caches = _CACHES[name]
    model = fakes.Backend(name, monkeypatch)(n_estimators=3, bag_size=bag_size)
    model.fit(X, z)
    expected = model.predict_proba(X_test)
    restored = pickle.loads(pickle.dumps(model))
    assert all(cache is not None for cache in caches(model))
    assert all(cache is None for cache in caches(restored))
    if getattr(model, "regressor_", None) is model.handles_[0]:
        assert restored.regressor_ is restored.handles_[0]
    np.testing.assert_array_equal(restored.predict_proba(X_test), expected)
    assert all(cache is not None for cache in caches(restored))


def test_a_failed_cache_rebuild_is_tried_again_by_the_next_prediction(
    cls, data, monkeypatch
):
    X, z, X_test = data
    model = cls(n_estimators=3).fit(X, z)
    expected = model.predict_proba(X_test)
    restored = pickle.loads(pickle.dumps(model))
    rebuild = restored._rebuild_caches
    calls = []

    def fails_once():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("out of memory")
        rebuild()

    monkeypatch.setattr(restored, "_rebuild_caches", fails_once)
    with pytest.raises(RuntimeError, match="out of memory"):
        restored.predict_proba(X_test)
    assert restored._caches_pending
    np.testing.assert_array_equal(restored.predict_proba(X_test), expected)
    restored.predict_proba(X_test)
    assert len(calls) == 2


@pytest.mark.filterwarnings("ignore::lazy.PerformanceWarning")
def test_an_auto_device_is_resolved_again_when_a_gpu_pickle_loads(
    cls, data, monkeypatch
):
    X, z, X_test = data
    monkeypatch.setattr(_device, "resolve_device", lambda device: "cpu")
    # "auto" in any case, as resolve_device reads it.
    model = cls(device="Auto", mixed_precision=True).fit(X, z)
    expected = model.predict_proba(X_test)
    provenance = copy.deepcopy(model.provenance_)
    # As if fitted on a GPU, under mixed precision.
    # The recipe, not the hook under test, writes the GPU's provenance.
    model.device_, model.mixed_precision_ = "cuda:7", True
    model.provenance_ = {**model.provenance_, **model._recipe()}
    assert model.provenance_ != provenance
    restored = pickle.loads(pickle.dumps(model))
    assert restored.device_ == "cpu"
    assert restored.mixed_precision_ is False
    assert restored.provenance_ == provenance
    np.testing.assert_array_equal(restored.predict_proba(X_test), expected)


def test_an_explicit_device_is_kept_when_a_pickle_loads(data):
    X, z, _ = data
    model = standins.HistogramStandIn(n_estimators=1, progress=False)
    model.fit(X, z)
    # As if fitted on a GPU that was named.
    model.device = model.device_ = "cuda:7"
    model.mixed_precision_ = True
    model.provenance_ = {**model.provenance_, "device": "cuda:7"}
    restored = pickle.loads(pickle.dumps(model))
    assert restored.device_ == "cuda:7"
    assert restored.mixed_precision_ is True
    assert restored.provenance_ == model.provenance_


def test_a_bag_as_large_as_the_context_is_no_bag(cls, data):
    X, z, X_test = data
    plain = cls().fit(X, z)
    bagged = cls(bag_size=10_000).fit(X, z)
    assert not bagged.bagging_
    np.testing.assert_array_equal(
        plain.predict_proba(X_test), bagged.predict_proba(X_test)
    )


@pytest.mark.parametrize(
    "transforms",
    [
        *_transforms.BASE_TRANSFORMS,
        "quantile_uniform+original",
        "limix",
        ("none", "power", "robust"),
    ],
)
def test_every_transform_runs(cls, transforms, data):
    X, z, X_test = data
    model = cls(transforms=transforms, bag_size=0.6).fit(X, z)
    pdfs = model.predict_proba(X_test, lazy.datasets.DC1_GRID)
    assert np.isfinite(pdfs).all() and pdfs.shape == (len(X_test), 200)


def test_the_same_seed_repeats_and_another_does_not(cls, data):
    X, z, X_test = data
    first = (
        cls(bag_size=0.5)
        .fit(X, z)
        .predict_proba(X_test, lazy.datasets.DC1_GRID)
    )
    again = (
        cls(bag_size=0.5)
        .fit(X, z)
        .predict_proba(X_test, lazy.datasets.DC1_GRID)
    )
    other = (
        cls(bag_size=0.5, random_state=1)
        .fit(X, z)
        .predict_proba(X_test, lazy.datasets.DC1_GRID)
    )
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_no_seed_draws_one_and_records_it(cls, data):
    X, z, X_test = data
    params = {"n_estimators": 3, "bag_size": 40, "random_state": None}
    first = cls(**params).fit(X, z)
    second = cls(**params).fit(X, z)
    assert first.random_state_ != second.random_state_
    assert first.provenance_["random_state"] == first.random_state_
    assert 0 <= first.random_state_ < 2**31
    replay = cls(**{**params, "random_state": first.random_state_})
    np.testing.assert_array_equal(
        replay.fit(X, z).predict_proba(X_test), first.predict_proba(X_test)
    )


def test_bagged_bar_members_with_their_own_buckets_form_a_mixture(data):
    X, z, X_test = data
    model = standins.ScaffoldedHistogramStandIn(
        n_estimators=4, bag_size=0.5
    ).fit(X, z)
    dist = model.predict_distribution(X_test)
    assert isinstance(dist, lazy.distributions.MixtureDistribution)
    assert len(dist.components) == 4


def test_bagged_quantile_members_average_their_quantiles(data):
    X, z, X_test = data
    model = standins.QuantileStandIn(n_estimators=4, bag_size=0.5).fit(X, z)
    assert len(model.member_groups_) == 4
    assert isinstance(
        model.predict_distribution(X_test),
        lazy.distributions.QuantileDistribution,
    )


@pytest.mark.parametrize(
    "params",
    [{"bag_size": 0.5}, {"transforms": ("none", "power")}],
    ids=["bagged", "two-groups"],
)
def test_a_trimmed_average_trims_every_member_at_once(data, params):
    """Not each group on its own: the ensemble's trim is the members'."""
    X, z, X_test = data
    model = standins.TrimmedQuantileStandIn(n_estimators=5, **params)
    model.fit(X, z)
    assert len(model.member_groups_) > 1
    members = []
    for handle, group, fitted in zip(
        model.handles_,
        model.member_groups_,
        model.transformers_,
        strict=True,
    ):
        prepared = model._prepare(X_test, group, fitted)
        members.extend(model._predict_members(handle, prepared))
    expected = lazy.distributions.QuantileDistribution.trimmed_average(
        members, 0.2
    )
    np.testing.assert_array_equal(
        model.predict_distribution(X_test).locs, expected.locs
    )


def test_a_trimmed_average_needs_every_member_of_a_group(data, monkeypatch):
    X, z, X_test = data
    model = standins.TrimmedQuantileStandIn(n_estimators=3).fit(X, z)
    monkeypatch.setattr(
        model, "_predict_members", lambda handle, X: [], raising=False
    )
    with pytest.raises(RuntimeError, match="0 members for a group of 3"):
        model.predict_distribution(X_test)


def test_a_trimmed_combination_needs_its_hook_and_quantiles(monkeypatch):
    cls = standins.TrimmedQuantileStandIn
    monkeypatch.setattr(
        cls,
        "_predict_members",
        vars(_ensemble.ContextEnsembleEstimator)["_predict_members"],
    )
    monkeypatch.setattr(cls, "native_output", "histogram")
    monkeypatch.setattr(cls, "member_trim_fraction", 0.5)
    problems = "\n".join(_conformance.problems(cls, "trimmed"))
    assert "needs _predict_members" in problems
    assert "needs native_output='quantiles'" in problems
    assert "member_trim_fraction must be in [0, 0.5)" in problems


def test_a_trim_fraction_needs_the_trimmed_combination(monkeypatch):
    monkeypatch.setattr(standins.QuantileStandIn, "member_trim_fraction", 0.2)
    problems = _conformance.problems(standins.QuantileStandIn, "quantile")
    assert any("member_trim_fraction needs" in p for p in problems)


def test_the_recipe_is_recorded_in_provenance(data):
    X, z, _ = data
    model = standins.HistogramStandIn(
        n_estimators=3, transforms="limix", bag_size=30
    ).fit(X, z)
    assert model.provenance_["transforms"] == [
        "quantile_uniform+original",
        "power",
        "quantile_uniform+original",
    ]
    assert model.provenance_["bag_rows"] == 30
    assert model.provenance_["kv_cache"] is True


def test_provenance_records_every_resolved_setting(data):
    X, z, _ = data
    model = standins.HistogramStandIn(device="cpu", outlier_threshold=4).fit(
        X, z
    )
    recorded = model.provenance_
    assert recorded["n_estimators"] == 8
    assert recorded["random_state"] == 0
    assert recorded["chunk_size"] == 8_192
    assert recorded["softmax_temperature"] is None  # "auto", no softmax here
    assert recorded["mixed_precision"] is False  # never on a CPU
    assert recorded["outlier_threshold"] == 4.0


# -- the runtime contract ------------------------------------------------------


class _AnswersInQuantiles(standins.HistogramStandIn):
    """Declares histograms but answers with quantiles."""

    def _predict_group(self, handle, X):
        locs = np.tile([0.0, 1.0], (len(X), 1))
        return lazy.distributions.QuantileDistribution(
            np.array([0.25, 0.75]), locs
        )


class _DropsARow(standins.HistogramStandIn):
    """Answers one row short."""

    def _predict_group(self, handle, X):
        return super()._predict_group(handle, X)[1:]


# One group serves the whole ensemble, or a scaffolded transform splits it.
GROUPINGS = [
    pytest.param({}, 1, id="one-group"),
    pytest.param({"transforms": ("robust", "none")}, 2, id="two-groups"),
]


@pytest.mark.parametrize(("params", "n_groups"), GROUPINGS)
def test_a_group_answering_in_the_wrong_kind_is_refused(params, n_groups, data):
    X, z, X_test = data
    model = _AnswersInQuantiles(n_estimators=2, **params).fit(X, z)
    assert len(model.member_groups_) == n_groups
    with pytest.raises(TypeError, match="needs a HistogramDistribution"):
        model.predict_distribution(X_test)


@pytest.mark.parametrize(("params", "n_groups"), GROUPINGS)
def test_a_group_answering_the_wrong_rows_is_refused(params, n_groups, data):
    X, z, X_test = data
    model = _DropsARow(n_estimators=2, **params).fit(X, z)
    assert len(model.member_groups_) == n_groups
    with pytest.raises(RuntimeError, match="19 rows for 20 query rows"):
        model.predict_distribution(X_test)


def test_a_fitted_provenance_holds_every_required_key(cls, data):
    X, z, _ = data
    model = cls().fit(X, z)
    assert set(_ensemble.REQUIRED_PROVENANCE) <= set(model.provenance_)


class _Hooked(standins.HistogramStandIn):
    """Records when its fit hooks run, and loads a network of its own."""

    def _before_fit(self, X, y):
        self.calls_ = [("before", hasattr(self, "random_state_"), len(y))]

    def _check_backend_params(self):
        self.calls_.append(("check",))

    def _after_fit(self):
        self.calls_.append(("after", "groups" in self.provenance_))

    def _checkpoint(self):
        return pathlib.Path("/hub/hooked.ckpt")

    def _network_loader(self):
        return "float32", _read

    def _read_network(self, load, checkpoint):
        self.loads_ = getattr(self, "loads_", 0) + 1
        return load(checkpoint, self.device_)


def _read(checkpoint, device):
    """The stand-in loader: a new network on every call."""
    del checkpoint, device  # Unused: nothing is read.
    return object()


def test_the_fit_hooks_bracket_the_fit(data):
    X, z, _ = data
    model = _Hooked().fit(X, z)
    assert model.calls_ == [
        ("before", False, len(z)),
        ("check",),
        ("after", True),
    ]


def test_a_memoized_network_is_kept_until_its_key_changes(monkeypatch, data):
    # Off, so that only the instance's memo keeps a network.
    monkeypatch.setattr(_weights, "_switch", False)
    X, z, _ = data
    model = _Hooked().fit(X, z)
    first = model._network()
    assert model._network() is first
    assert model.loads_ == 1
    model.version = "v1"
    assert model._network() is not first
    assert model.loads_ == 2


def test_a_memoized_network_is_not_pickled(monkeypatch, data):
    # Off, so that only the instance's memo keeps a network.
    monkeypatch.setattr(_weights, "_switch", False)
    X, z, X_test = data
    model = _Hooked().fit(X, z)
    model._network()
    restored = pickle.loads(pickle.dumps(model))
    assert "_network_cache" not in vars(restored)
    assert "_network_cache" in vars(model)  # the original keeps it
    np.testing.assert_array_equal(
        restored.predict_pdf(X_test), model.predict_pdf(X_test)
    )
    restored._network()
    assert restored.loads_ == 2


def test_a_missing_extra_is_named(monkeypatch):
    monkeypatch.setitem(sys.modules, "mymodel", None)
    with pytest.raises(ImportError, match=r"pip install 'lazy-tfm\[mine\]'"):
        _ensemble.import_extra("mymodel", needed_by="Mine", extra="mine")


def test_a_broken_extra_reports_its_own_error(monkeypatch):
    real_import = builtins.__import__

    def importing(module, *args, **kwargs):
        if module == "mymodel":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ModuleNotFoundError, match="torch") as raised:
        _ensemble.import_extra("mymodel", needed_by="Mine", extra="mine")
    assert "lazy-tfm" not in str(raised.value)


# -- softmax_temperature, mixed_precision, outlier_threshold -----------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"softmax_temperature": 0}, "softmax_temperature"),
        ({"softmax_temperature": -1.0}, "softmax_temperature"),
        ({"softmax_temperature": "hot"}, "softmax_temperature"),
        ({"softmax_temperature": True}, "softmax_temperature"),
        ({"mixed_precision": "yes"}, "mixed_precision"),
        ({"mixed_precision": 1}, "mixed_precision"),
        ({"outlier_threshold": 0}, "outlier_threshold"),
        ({"outlier_threshold": "off"}, "outlier_threshold"),
        ({"outlier_threshold": float("inf")}, "outlier_threshold"),
    ],
)
def test_the_shared_settings_are_validated(params, message, data):
    X, z, _ = data
    with pytest.raises(ValueError, match=message):
        standins.HistogramStandIn(**params).fit(X, z)


def test_a_model_without_a_softmax_refuses_a_temperature(data, monkeypatch):
    X, z, _ = data
    monkeypatch.setattr(standins.QuantileStandIn, "has_softmax", False)
    with pytest.raises(ValueError, match="no softmax"):
        standins.QuantileStandIn(softmax_temperature=0.9).fit(X, z)
    standins.QuantileStandIn(softmax_temperature="auto").fit(X, z)


def test_auto_temperature_is_the_models_own(data, monkeypatch):
    X, z, _ = data
    monkeypatch.setattr(
        standins.HistogramStandIn,
        "_auto_softmax_temperature",
        lambda self: 0.9,
    )
    model = standins.HistogramStandIn().fit(X, z)
    assert model.softmax_temperature_ == 0.9
    assert standins.HistogramStandIn(softmax_temperature=2).fit(
        X, z
    ).softmax_temperature_ == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("transforms", "threshold", "expected"),
    [
        ("auto", "auto", 7.0),  # the model's own recipe, its own clip
        ("none", "auto", None),  # an explicit recipe: nothing optional
        ("power", "auto", None),
        ("auto", None, None),  # off, even in the model's own recipe
        ("auto", 3, 3.0),
        ("none", 2.5, 2.5),
    ],
)
def test_auto_outlier_threshold_follows_the_transforms(
    transforms, threshold, expected, data, monkeypatch
):
    X, z, _ = data
    monkeypatch.setattr(
        standins.HistogramStandIn, "_auto_outlier_threshold", lambda self: 7.0
    )
    model = standins.HistogramStandIn(
        transforms=transforms, outlier_threshold=threshold
    ).fit(X, z)
    assert model.outlier_threshold_ == expected


def test_the_clip_is_scaffolded_for_a_model_without_one(data, monkeypatch):
    X, z, X_test = data
    X = X.copy()
    X[0, 0] = 40.0  # far outside the others
    seen = []
    original = standins.HistogramStandIn._fit_group

    def recording(self, features, y, group):
        seen.append(features[:, :].max())
        return original(self, features, y, group)

    monkeypatch.setattr(standins.HistogramStandIn, "_fit_group", recording)
    standins.HistogramStandIn(
        feature_shuffle=False, outlier_threshold=None
    ).fit(X, z)
    model = standins.HistogramStandIn(
        feature_shuffle=False, outlier_threshold=4
    ).fit(X, z)
    assert seen[0] == 40.0
    assert seen[1] < 10.0
    # The queries are clipped by the bounds fitted on the context.
    clipper = model.clippers_[0]
    query = np.array([[40.0, 0.0, 0.0]])
    assert clipper.transform(query)[0, 0] == pytest.approx(seen[1])


def test_a_native_clip_is_left_to_the_model(data, monkeypatch):
    X, z, _ = data
    monkeypatch.setattr(
        standins.HistogramStandIn, "native_outlier_clipping", True
    )
    model = standins.HistogramStandIn(outlier_threshold=4).fit(X, z)
    assert model.outlier_threshold_ == 4.0
    assert model.clippers_ == [None]


def test_soft_clip_keeps_order_and_ignores_missing_values():
    rng = np.random.default_rng(0)
    context = rng.normal(size=(500, 2))
    context[3, 1] = np.nan
    clip = _transforms.SoftClip(3.0).fit(context)
    values = np.array([[-50.0, np.nan], [0.0, 1.0], [5.0, 2.0], [50.0, 3.0]])
    out = clip.transform(values)
    assert np.isnan(out[0, 1])
    assert out[1, 0] == 0.0  # inside the bounds: untouched
    assert np.all(np.diff(out[:, 0]) > 0)  # order kept
    assert out[3, 0] < 3.5 + np.log1p(50.0)  # pulled back to the bound


def test_soft_clip_passes_a_column_without_spread_through():
    rng = np.random.default_rng(0)
    context = np.full((50, 4), np.nan)
    context[:, 0] = rng.normal(size=50)  # an ordinary column
    context[7, 1] = 2.0  # one observed value
    # Three values, of which the second pass keeps only the middle one.
    context[:3, 3] = [-1.0, 0.01, 1.0]
    clip = _transforms.SoftClip(0.1).fit(context)
    queries = np.array([[0.0, -40.0, 3.0, 25.0], [np.nan, np.nan, np.nan, 0.5]])
    for values in (context, queries):
        out = clip.transform(values)
        np.testing.assert_array_equal(out[:, 1:], values[:, 1:])
        np.testing.assert_array_equal(
            out[:, 0],
            _transforms.SoftClip(0.1)
            .fit(context[:, :1])
            .transform(values[:, :1])[:, 0],
        )


def test_soft_clip_is_the_upstream_models_clip():
    preprocessing = pytest.importorskip("tabicl._sklearn.preprocessing")
    rng = np.random.default_rng(1)
    # Upstream imputes before it clips, so its clip never sees a NaN.
    context = rng.standard_t(2, size=(400, 4))
    queries = rng.standard_t(1, size=(60, 4)) * 5
    ours = _transforms.SoftClip(4.0).fit(context).transform(queries)
    theirs = (
        preprocessing.OutlierRemover(threshold=4.0)
        .fit(context)
        .transform(queries)
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-12)


def _table(sdm, torch, features):
    """Features as an all-numerical sdm.TableTensor."""
    names = [f"c{i}" for i in range(features.shape[1])]
    return sdm.TableTensor(
        columns={sdm.Stype.numerical: names},
        numerical=torch.tensor(features),
    )


def test_soft_clip_is_kumos_clip():
    """Kumo applies its own; the uniform clip must be the same one."""
    sdm = pytest.importorskip("sdm")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    context = rng.standard_t(2, size=(400, 4))
    queries = rng.standard_t(1, size=(60, 4)) * 5
    # Kumo has no imputation: missing values reach the clip.
    context[::7, 1] = np.nan
    queries[::5, 2] = np.nan
    ours = _transforms.SoftClip(4.0).fit(context).transform(queries)
    clip = sdm.processing.ClipSigma(threshold=4.0)
    clip.fit(_table(sdm, torch, context))
    theirs = clip.transform(_table(sdm, torch, queries)).numerical.numpy()
    np.testing.assert_allclose(ours, theirs, rtol=1e-12)
    # A column with one observed value: no spread for ours to bound it by,
    # and ClipSigma's floor of 1e-6 standard deviations.
    context[1:, 3] = np.nan
    ours = _transforms.SoftClip(4.0).fit(context).transform(queries)
    clip.fit(_table(sdm, torch, context))
    theirs = clip.transform(_table(sdm, torch, queries)).numerical.numpy()
    np.testing.assert_array_equal(ours[:, 3], queries[:, 3])
    bound = clip.upper_bound.numpy()[0, 3] - context[0, 3]
    np.testing.assert_allclose(bound, 4e-6, rtol=1e-6)
    assert not np.allclose(theirs[:, 3], queries[:, 3])


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason="TabFM loads its backbone, setting checkpoint_, on "
                "the first prediction",
            ),
        )
        if name == "tabfm"
        else name
        for name in sorted(lazy.ESTIMATORS)
    ],
)
def test_a_prediction_sets_no_attribute(name, data, monkeypatch):
    """scikit-learn's rule: predict leaves the estimator's __dict__ alone."""
    X, z, X_test = data
    model = fakes.Backend(name, monkeypatch)(n_estimators=3).fit(X, z)
    if model.native_output == "quantiles":
        assert model.n_quantiles_ > 0
    before = dict(vars(model))
    model.predict_distribution(X_test)
    model.predict_proba(X_test)
    assert vars(model).keys() == before.keys()
    changed = [name for name in before if vars(model)[name] is not before[name]]
    assert not changed


def test_a_refit_into_several_groups_drops_the_single_regressor(data):
    X, z, _ = data
    model = standins.ScaffoldedHistogramStandIn(n_estimators=3).fit(X, z)
    assert model.regressor_ is model.handles_[0]
    model.set_params(bag_size=40).fit(X, z)
    assert len(model.handles_) == 3
    assert not hasattr(model, "regressor_")


class _FailsInSecondGroup(standins.ScaffoldedHistogramStandIn):
    """A stand-in whose fit fails partway through its groups."""

    def _fit_group(self, X, y, group):
        if group.index == 1 and X.shape[1] > 3:
            raise RuntimeError("the backend failed")
        return super()._fit_group(X, y, group)


def test_a_refit_failing_partway_leaves_the_model_unfitted(data):
    X, z, _ = data
    model = _FailsInSecondGroup(n_estimators=3, bag_size=40).fit(X, z)
    assert len(model.handles_) == 3
    wider = np.column_stack([X, X[:, :2]])
    with pytest.raises(RuntimeError, match="the backend failed"):
        model.fit(wider, z)
    assert not [name for name in vars(model) if name.endswith("_")]
    for features in (X, wider):
        with pytest.raises(exceptions.NotFittedError):
            model.predict(features)


def test_a_scaffold_is_fitted_on_its_members_bag(cls, data):
    """As the _transforms docstring says, with native bagging or without."""
    X, z, _ = data
    params = {"n_estimators": 2, "transforms": "robust", "bag_size": 40}
    model = cls(**params).fit(X, z)
    bags = _members.draw_bags(2, 40, len(X), 0)
    assert len(model.transformers_) == 2
    for group, fitted in zip(
        model.member_groups_, model.transformers_, strict=True
    ):
        (member,) = group.members
        np.testing.assert_array_equal(group.rows, bags[member.index])
        np.testing.assert_allclose(
            fitted._transformer.center_,
            np.nanmedian(X[group.rows], axis=0),
        )


@pytest.mark.parametrize("fill", [np.nan, 2.0], ids=["all_nan", "constant"])
def test_the_power_scaffold_passes_a_column_without_spread_through(fill):
    X = np.random.default_rng(0).normal(size=(50, 3))
    X[:, 1] = fill
    spec = _transforms.TransformSpec("power", original=True)
    transform = _transforms.ScaffoldTransform(spec, seed=0).fit(X)
    out = transform.transform(X)
    assert out.shape == (50, 6)
    np.testing.assert_array_equal(out[:, 1], X[:, 1])
    np.testing.assert_allclose(out[:, [0, 2]].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_array_equal(out[:, 3:], X)


def test_missing_values_reach_the_model(data):
    X, z, X_test = data
    X = X.copy()
    X[::4, 1] = np.nan
    model = standins.HistogramStandIn(transforms="limix").fit(X, z)
    assert np.isfinite(model.predict_proba(X_test)).all()


# -- the registry enforces the contract --------------------------------------

REGISTERED = [lazy.ESTIMATORS[name] for name in sorted(lazy.ESTIMATORS)]


@pytest.mark.parametrize("name", sorted(lazy.ESTIMATORS))
def test_every_registered_backend_meets_the_static_contract(name):
    assert _conformance.problems(lazy.ESTIMATORS[name], name) == []


@pytest.mark.parametrize("cls", REGISTERED, ids=lambda c: c.__name__)
def test_registered_backends_take_the_uniform_parameters(cls):
    test_the_uniform_parameters_are_keyword_only_with_fixed_defaults(cls)
    test_clone_reproduces_every_parameter(cls)


@pytest.mark.parametrize("cls", REGISTERED, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"kv_cache": "maybe"}, "kv_cache must be one of"),
        ({"transforms": "banana"}, "unknown transform"),
        ({"bag_size": -3}, "must be positive"),
        ({"n_estimators": 0}, "positive int"),
    ],
)
def test_registered_backends_reject_bad_parameters_before_loading(
    cls, params, match, data, monkeypatch
):
    X, z, _ = data
    monkeypatch.setattr(cls, "_import_backend", lambda self: None)
    with pytest.raises(ValueError, match=match):
        cls(**params).fit(X, z)


@pytest.mark.parametrize(
    ("cls", "params", "match"),
    [
        pytest.param(
            lazy.TabFMHistogram,
            {"n_dither": value},
            "n_dither must be an int",
            id=f"n_dither={value!r}",
        )
        for value in (1.5, True, "2")
    ]
    + [
        pytest.param(
            lazy.TabFMHistogram,
            {"keep_cache_on_device": "yes"},
            "keep_cache_on_device must be a bool",
            id="keep_cache_on_device",
        ),
        pytest.param(
            lazy.TabPFNBarDistribution,
            {"ignore_pretraining_limits": 1},
            "ignore_pretraining_limits must be a bool",
            id="ignore_pretraining_limits",
        ),
    ],
)
def test_backend_parameters_are_checked_before_loading(
    cls, params, match, data, monkeypatch
):
    X, z, _ = data
    monkeypatch.setattr(cls, "_import_backend", lambda self: None)
    with pytest.raises(ValueError, match=match):
        cls(**params).fit(X, z)


def _missing_backend(self):
    raise ImportError("backend not installed")


@pytest.mark.parametrize("cls", REGISTERED, ids=lambda c: c.__name__)
def test_a_missing_backend_is_reported_before_bad_parameters(
    cls, data, monkeypatch
):
    X, z, _ = data
    monkeypatch.setattr(cls, "_import_backend", _missing_backend)
    with pytest.raises(ImportError, match="not installed"):
        cls(n_estimators=0).fit(X, z)


@pytest.mark.parametrize("cls", REGISTERED, ids=lambda c: c.__name__)
def test_an_unknown_version_is_reported_before_a_missing_backend(
    cls, data, monkeypatch
):
    X, z, _ = data
    monkeypatch.setattr(cls, "_import_backend", _missing_backend)
    with pytest.raises(lazy.UnknownNameError, match="unknown version 'v99'"):
        cls(version="v99").fit(X, z)


def test_a_backend_without_the_uniform_features_cannot_register():
    class Incomplete(standins.HistogramStandIn):
        backend = "incomplete"

        def __init__(self, *, n_estimators=4, y_grid=None, device="cpu"):
            self.n_estimators = n_estimators
            self.y_grid = y_grid
            self.device = device

    with pytest.raises(TypeError, match="cannot be registered") as error:
        registry.register("incomplete", Incomplete)
    message = str(error.value)
    assert "no pinned checkpoint" in message
    assert "lacks the uniform parameter 'kv_cache'" in message
    assert "'device' defaults to 'cpu'" in message
    assert "incomplete" not in lazy.ESTIMATORS


def test_only_uniform_layer_backends_can_register():
    class Plain(lazy.BaseDensityRegressor):
        def _fit(self, X, y):
            pass

        def _predict_pdf(self, X, grid):
            return None

    with pytest.raises(TypeError, match="ContextEnsembleEstimator"):
        registry.register("plain", Plain)


@pytest.mark.parametrize(
    ("call", "shape"),
    [
        (lambda m, X, z: m.predict(X), (0,)),
        (lambda m, X, z: m.predict_proba(X), "bins"),
        (lambda m, X, z: m.predict_pdf(X), "bins"),
        (lambda m, X, z: m.predict_cdf(X), "bins"),
        (lambda m, X, z: m.predict_quantiles(X), (0, 3)),
        (lambda m, X, z: m.predict_interval(X), (0, 2)),
        (lambda m, X, z: m.predict_pit(X, z), (0,)),
        (lambda m, X, z: m.predict_distribution(X), "distribution"),
    ],
    ids=[
        "predict",
        "predict_proba",
        "predict_pdf",
        "predict_cdf",
        "predict_quantiles",
        "predict_interval",
        "predict_pit",
        "predict_distribution",
    ],
)
def test_predicting_no_query_rows_warns_and_returns_an_empty_result(
    cls, data, call, shape
):
    X, z, X_test = data
    model = cls(n_estimators=2, progress=False).fit(X, z)
    with pytest.warns(FutureWarning, match="zero rows of X") as record:
        out = call(model, X_test[:0], z[:0])
    assert len(record) == 1
    if shape == "distribution":
        assert len(out) == 0
    elif shape == "bins":
        assert out.shape == (0, model.grid_.n_bins)
    else:
        assert out.shape == shape


@pytest.mark.parametrize("method", ["score", "evaluate"])
def test_scoring_no_query_rows_is_refused(cls, data, method):
    X, z, X_test = data
    model = cls(n_estimators=2, progress=False).fit(X, z)
    with pytest.raises(ValueError, match=r"Found array with 0 sample\(s\)"):
        getattr(model, method)(X_test[:0], z[:0])


def test_fitting_no_rows_is_refused(cls, data):
    X, z, _ = data
    with pytest.raises(ValueError, match=r"Found array with 0 sample\(s\)"):
        cls(n_estimators=2, progress=False).fit(X[:0], z[:0])


@pytest.mark.parametrize("chunk_size", [0, 4])
def test_blocked_scoring_matches_the_whole_array(
    cls, chunk_size, data, monkeypatch
):
    """predict, score and evaluate never build all rows' densities at once."""
    X, z, X_test = data
    z_test = np.linspace(0.2, 1.5, len(X_test))
    model = cls(n_estimators=2, progress=False, chunk_size=chunk_size)
    model.fit(X, z)
    grid = model.grid_
    pdfs = model.predict_proba(X_test)
    whole = {
        "predict": lazy.metrics.grid_point_estimates(grid, pdfs)["peak_mean"],
        "score": -lazy.metrics.cde_loss(z_test, grid, pdfs),
        "evaluate": lazy.metrics.summarize(
            z_test, grid, pdfs, point="peak_mean", label=model.name_
        ),
    }
    calls = []
    original = type(model)._tabulate

    def counting(self, chunks, grid):
        pdfs = original(self, chunks, grid)
        calls.append(len(pdfs))
        return pdfs

    monkeypatch.setattr(type(model), "_tabulate", counting)
    # About seven rows of densities per block.
    monkeypatch.setattr(lazy.base, "_BLOCK_BYTES", 7 * 8 * grid.n_bins)
    # Equal up to rounding: a block of rows goes through a smaller matrix
    # product than the whole array, which some BLAS builds round differently
    # in the last bit.
    np.testing.assert_allclose(
        model.predict(X_test, method="peak_mean"), whole["predict"], rtol=1e-12
    )
    assert max(calls) == (4 if chunk_size else 7) and len(calls) > 1
    assert model.score(X_test, z_test) == pytest.approx(
        whole["score"], rel=1e-12
    )
    pd.testing.assert_frame_equal(
        model.evaluate(X_test, z_test, method="peak_mean"),
        whole["evaluate"],
        check_exact=False,
        rtol=1e-12,
    )


# A NumPy integer, as a parameter grid supplies, must align blocks as an int.
@pytest.mark.parametrize(
    "chunk_size", [0, 4, np.int64(4)], ids=["0", "4", "int64-4"]
)
def test_predict_score_and_evaluate_show_one_progress_bar_over_all_blocks(
    cls, chunk_size, data, monkeypatch
):
    X, z, X_test = data
    z_test = np.linspace(0.2, 1.5, len(X_test))
    model = cls(n_estimators=2, progress=False, chunk_size=chunk_size)
    model.fit(X, z)
    if not model.chunks_queries:
        pytest.skip("a model that does not chunk its queries draws its bars")
    totals = []
    original = _progress.bar

    def counting(*args, **kwargs):
        totals.append(kwargs["total"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_progress, "bar", counting)
    # About seven rows of densities per block, so three blocks.
    monkeypatch.setattr(lazy.base, "_BLOCK_BYTES", 7 * 8 * model.grid_.n_bins)
    model.predict(X_test)
    model.score(X_test, z_test)
    model.evaluate(X_test, z_test)
    assert totals == [len(X_test)] * 3


@pytest.mark.parametrize("chunk_size", [0, 4])
def test_quantiles_intervals_and_pits_reduce_one_chunk_at_a_time(
    cls, chunk_size, data, monkeypatch
):
    """They never hold all rows' distributions at once, nor change them."""
    X, z, X_test = data
    z_test = np.linspace(0.2, 1.5, len(X_test))
    model = cls(n_estimators=2, progress=False, chunk_size=chunk_size)
    model.fit(X, z)
    dist = model.predict_distribution(X_test)
    seen = []
    original = type(model)._distribution_chunks

    def recording(self, features, *args):
        for chunk in original(self, features, *args):
            seen.append(len(chunk))
            yield chunk

    monkeypatch.setattr(type(model), "_distribution_chunks", recording)
    monkeypatch.setattr(
        type(model), "_predict_distribution", standins.refuse_every_row
    )
    # The same passes as predict_distribution, so the same answers, exactly.
    np.testing.assert_array_equal(
        model.predict_quantiles(X_test, [0.1, 0.5, 0.9]),
        dist.ppf([0.1, 0.5, 0.9]),
    )
    np.testing.assert_array_equal(
        model.predict_interval(X_test, 0.8), dist.interval(0.8)
    )
    np.testing.assert_array_equal(
        model.predict_pit(X_test, z_test), dist.pit(z_test)
    )
    # A model that chunks its queries hands over each chunk as it goes.
    n = len(X_test)
    step = 4 if chunk_size and model.chunks_queries else n
    chunks = [min(step, n - start) for start in range(0, n, step)]
    assert seen == chunks * 3


def test_threads_relocating_a_pickles_checkpoint_never_see_the_old_path(
    monkeypatch, data
):
    X, z = data[0], data[1]
    model = standins.HistogramStandIn(n_estimators=1, progress=False)
    model.fit(X, z)
    model.checkpoint_ = "/the/machine/that/pickled/it.ckpt"
    restored = pickle.loads(pickle.dumps(model))
    start = threading.Barrier(2, timeout=threads.TIMEOUT)
    # Each download waits for the other thread's. If the flag were cleared
    # before the path is assigned, the second thread would not download, and
    # this barrier would break.
    downloading = threading.Barrier(2, timeout=threads.TIMEOUT)

    class Spec:
        def download(self):
            downloading.wait()
            return "/here/it.ckpt"

    monkeypatch.setattr(type(restored), "_checkpoint_spec", lambda self: Spec())
    seen = []

    def relocate():
        start.wait()
        seen.append(restored._checkpoint())

    assert threads.run(relocate, relocate)
    assert seen == ["/here/it.ckpt", "/here/it.ckpt"]
    assert "_relocate_checkpoint" not in vars(restored)
