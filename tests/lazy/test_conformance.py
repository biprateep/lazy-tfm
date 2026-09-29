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
        ({"random_state": None}, "random_state"),
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
    assert native.seed == 7 and native.scaffold is None
    assert [m.index for m in scaffolded.members] == [1, 4]
    assert scaffolded.scaffold.name == "quantile"
    assert scaffolded.native_transforms == ("none", "none")
    assert scaffolded.seed == 7 + _members.GROUP_SEED_STRIDE


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
    assert [g.seed for g in groups] == [3, 4, 5, 6, 7, 8]
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
    model = cls(z_grid=lazy.DC1_GRID).fit(X, z)
    assert model.grid_ == lazy.DC1_GRID
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
def test_chunking_is_bit_identical(cls, data):
    X, z, X_test = data
    whole = cls(chunk_size=0).fit(X, z).predict_proba(X_test)
    chunked = cls(chunk_size=7).fit(X, z).predict_proba(X_test)
    np.testing.assert_array_equal(whole, chunked)


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
    pdfs = model.predict_proba(X_test, lazy.DC1_GRID)
    assert np.isfinite(pdfs).all() and pdfs.shape == (len(X_test), 200)


@pytest.mark.parametrize("cls", standins.STANDINS, ids=lambda c: c.__name__)
def test_the_same_seed_repeats_and_another_does_not(cls, data):
    X, z, X_test = data
    first = cls(bag_size=0.5).fit(X, z).predict_proba(X_test, lazy.DC1_GRID)
    again = cls(bag_size=0.5).fit(X, z).predict_proba(X_test, lazy.DC1_GRID)
    other = (
        cls(bag_size=0.5, random_state=1)
        .fit(X, z)
        .predict_proba(X_test, lazy.DC1_GRID)
    )
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_bagged_bar_members_with_their_own_buckets_form_a_mixture(data):
    X, z, X_test = data
    model = standins.ScaffoldedHistogramStandIn(bag_size=0.5).fit(X, z)
    dist = model.predict_distribution(X_test)
    assert isinstance(dist, lazy.distributions.MixtureDistribution)
    assert len(dist.components) == 4


def test_bagged_quantile_members_average_their_quantiles(data):
    X, z, X_test = data
    model = standins.QuantileStandIn(bag_size=0.5).fit(X, z)
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
def test_an_unknown_version_is_reported_before_the_backend_import(
    cls, data, monkeypatch
):
    X, z, _ = data
    monkeypatch.setattr(cls, "_import_backend", _missing_backend)
    with pytest.raises(ValueError, match="unknown version"):
        cls(version="v99").fit(X, z)


def test_a_backend_without_the_uniform_features_cannot_register():
    class Incomplete(standins.HistogramStandIn):
        backend = "incomplete"

        def __init__(self, *, n_estimators=4, z_grid=None, device="cpu"):
            self.n_estimators = n_estimators
            self.z_grid = z_grid
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
