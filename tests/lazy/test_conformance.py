# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Every backend supports the uniform features, the same way (tier A).

Parametrised over the backend-free stand-ins, which implement exactly the
per-backend interface; the real backends join through the registry. Nothing
here needs a GPU or a checkpoint.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn import base as sklearn_base
import standins

import lazy
from lazy.models import _conformance
from lazy.models import _ensemble
from lazy.models import _members
from lazy.models import _transforms
from lazy.models import registry


@pytest.fixture
def data():
    rng = np.random.default_rng(5)
    z = rng.uniform(0.1, 1.6, 90)
    X = np.column_stack(
        [np.sin(z * k) + rng.normal(0, 0.05, z.size) for k in (1, 2, 3)]
    )
    return X[:70], z[:70], X[70:]


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
    with pytest.warns(lazy.ContextSizeWarning, match="bag_size=50"):
        _Limited().fit(X, z)


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


def test_only_tabicl_and_tabpfns_fast_checkpoint_are_cpu_friendly():
    friendly = [n for n, c in lazy.ESTIMATORS.items() if c.cpu_friendly]
    assert friendly == ["tabicl"]
    versions = {
        n: c.cpu_friendly_versions
        for n, c in lazy.ESTIMATORS.items()
        if c.cpu_friendly_versions
    }
    assert versions == {"tabpfn": ("v3.5-fast",)}


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


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_numpy_integers_are_accepted_as_counts_and_seeds(cls, data):
    """Values from np.arange or a parameter grid are NumPy integers."""
    X, z, X_test = data
    python = {"n_estimators": 3, "bag_size": 40, "random_state": 7}
    numpy = {name: np.int64(value) for name, value in python.items()}
    expected = cls(**python, chunk_size=9).fit(X, z).predict_proba(X_test)
    model = cls(**numpy, chunk_size=np.int64(9)).fit(X, z)
    np.testing.assert_array_equal(model.predict_proba(X_test), expected)
    assert type(model.provenance_["n_estimators"]) is int


# -- behaviour, on every stand-in -------------------------------------------


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_the_default_grid_is_native_and_integrates_to_one(cls, data):
    X, z, X_test = data
    model = cls().fit(X, z)
    assert model.grid_ == model.native_grid_
    assert model.grid_.normalization == "histogram"
    pdfs = model.predict_proba(X_test)
    assert pdfs.shape == (len(X_test), model.native_grid_.n_bins)
    np.testing.assert_allclose(pdfs @ model.grid_.widths, 1.0)


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_a_constructor_grid_overrides_the_native_one(cls, data):
    X, z, X_test = data
    model = cls(y_grid=lazy.datasets.DC1_GRID).fit(X, z)
    assert model.grid_ == lazy.datasets.DC1_GRID
    assert model.predict_proba(X_test).shape == (len(X_test), 200)
    assert model.predict_proba(X_test, "native").shape[1] == (
        model.native_grid_.n_bins
    )


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
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


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
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


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
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


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
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


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_chunking_changes_nothing_but_rounding(cls, data):
    """A row's answer does not depend on the rows chunked with it.

    Equal up to the last bit: on CI's x86 runners the quantile stand-in's
    answers for a 7-row chunk and for the whole set differ by one ulp in
    a few bins (bit-identical on the Arm machines the suite is developed
    on), as blocked scoring did. The real backends' exactness on a CPU is
    checked bit for bit by the checkpoint tests.
    """
    X, z, X_test = data
    whole = cls(chunk_size=0).fit(X, z).predict_proba(X_test)
    chunked = cls(chunk_size=7).fit(X, z).predict_proba(X_test)
    np.testing.assert_allclose(chunked, whole, rtol=1e-12, atol=0)


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_a_bag_as_large_as_the_context_is_no_bag(cls, data):
    X, z, X_test = data
    plain = cls().fit(X, z)
    bagged = cls(bag_size=10_000).fit(X, z)
    assert not bagged.bagging_
    np.testing.assert_array_equal(
        plain.predict_proba(X_test), bagged.predict_proba(X_test)
    )


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
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


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
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


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
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


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_no_query_rows_give_an_empty_answer(cls, data):
    X, z, X_test = data
    model = cls(n_estimators=3, bag_size=40).fit(X, z)
    empty = pd.DataFrame(X_test[:0])
    pdfs = model._predict_pdf(empty, model.grid_)
    assert pdfs.shape == (0, model.grid_.n_bins)
    dist = model._predict_distribution(empty)
    full = model._predict_distribution(pd.DataFrame(X_test))
    assert type(dist) is type(full)
    assert dist.on_grid(model.grid_).shape == (0, model.grid_.n_bins)


def test_a_refit_into_several_groups_drops_the_single_regressor(data):
    X, z, _ = data
    model = standins.ScaffoldedHistogramStandIn(n_estimators=3).fit(X, z)
    assert model.regressor_ is model.handles_[0]
    model.set_params(bag_size=40).fit(X, z)
    assert len(model.handles_) == 3
    assert not hasattr(model, "regressor_")


@pytest.mark.parametrize("cls", standins.STANDINS)
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
        ({"version": "v99"}, r"unknown version 'v99'.*known: \['"),
    ],
)
def test_registered_backends_reject_bad_parameters_before_loading(
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
        cls(version="v99", n_estimators=0).fit(X, z)


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


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_no_query_rows_give_empty_answers_of_the_right_shape(cls, data):
    X, z, X_test = data
    model = cls(n_estimators=2, progress=False).fit(X, z)
    empty = X_test[:0]
    assert model.predict_proba(empty).shape == (0, model.grid_.n_bins)
    assert model.predict_cdf(empty).shape == (0, model.grid_.n_bins)
    assert model.predict(empty).shape == (0,)
    assert model.predict_quantiles(empty).shape == (0, 3)
    assert len(model.predict_distribution(empty)) == 0


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_fitting_no_rows_is_refused(cls, data):
    X, z, _ = data
    with pytest.raises(ValueError, match=r"Found array with 0 sample\(s\)"):
        cls(n_estimators=2, progress=False).fit(X[:0], z[:0])


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
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
    original = type(model)._predict_pdf

    def counting(self, features, grid):
        calls.append(len(features))
        return original(self, features, grid)

    monkeypatch.setattr(type(model), "_predict_pdf", counting)
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
