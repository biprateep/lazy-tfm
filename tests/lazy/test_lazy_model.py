# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LazyModel: the name-addressed wrapper, its delegation, its sklearn contract.

None of this constructs a backbone or touches the hub, so it runs anywhere.
"""

import pickle

import numpy as np
import pytest
from sklearn import base as sklearn_base

import lazy
from lazy.models import registry


def _explode(*args, **kwargs):
    raise AssertionError("constructing a model must not touch the hub")


def test_the_default_model_is_tabpfn():
    model = lazy.LazyModel()
    assert model.model == "tabpfn"
    assert isinstance(model._build(), lazy.TabPFNBarDistribution)


def test_the_default_model_is_tabpfn_3_5():
    assert lazy.LazyModel().name_ == "tabpfn:v3.5"


@pytest.mark.parametrize("name", sorted(lazy.ESTIMATORS))
def test_every_registered_name_builds(name):
    model = lazy.LazyModel(name)
    assert isinstance(model, lazy.BaseDensityRegressor)
    assert model.model == name
    assert model.name_ == f"{name}:{lazy.DEFAULT_VERSIONS[name]}"


def test_the_backend_is_the_concrete_class():
    assert isinstance(
        lazy.LazyModel("limix")._build(), lazy.LimiXBarDistribution
    )
    assert isinstance(lazy.LazyModel("tabfm")._build(), lazy.TabFMHistogram)
    assert isinstance(lazy.LazyModel("tabicl")._build(), lazy.TabICLQuantile)
    assert isinstance(
        lazy.LazyModel("tabpfn")._build(), lazy.TabPFNBarDistribution
    )


def test_parameters_reach_the_backend():
    backend = lazy.LazyModel("tabfm", n_estimators=4, n_dither=3)._build()
    assert backend.n_estimators == 4
    assert backend.n_dither == 3


def test_backend_attributes_are_reachable_through_the_wrapper():
    model = lazy.LazyModel("tabicl", n_estimators=16)
    assert model.n_estimators == 16


def test_a_missing_attribute_names_both_classes():
    model = lazy.LazyModel("tabicl")
    missing = "no_such_attribute"
    with pytest.raises(AttributeError, match="TabICLQuantile"):
        getattr(model, missing)


def test_unknown_model_name_names_the_alternatives_at_fit():
    model = lazy.LazyModel("tabdpt")  # construction validates nothing
    with pytest.raises(ValueError, match="tabicl"):
        model.fit(np.zeros((3, 2)), np.ones(3))


def test_the_version_reaches_the_backend_and_the_label():
    """A results row has to say which model version produced it."""
    model = lazy.LazyModel("tabpfn", version="v2.5")
    assert model.version == "v2.5"
    assert model.name_ == "tabpfn:v2.5"
    assert model.get_params()["version"] == "v2.5"


def test_a_parameter_the_backend_does_not_take_is_rejected_at_fit():
    model = lazy.LazyModel("tabicl", n_dither=3)  # that is a TabFM parameter
    with pytest.raises(
        TypeError, match="TabICLQuantile does not take 'n_dither'"
    ):
        model.fit(np.zeros((3, 2)), np.ones(3))


def test_init_stores_its_arguments_and_nothing_else():
    model = lazy.LazyModel("tabfm", z_grid=None, n_dither=3)
    public = {name for name in vars(model) if not name.startswith("_")}
    assert public == {"model", "z_grid"}
    assert not hasattr(model, "estimator_")


def test_constructing_touches_neither_hub_nor_backend(monkeypatch):
    monkeypatch.setattr("lazy.models._hub.Checkpoint.download", _explode)
    for name in lazy.list_estimators():
        lazy.LazyModel(name)


# -- scikit-learn parameter protocol ----------------------------------------


def test_get_params_flattens_the_backend_parameters():
    params = lazy.LazyModel("tabfm", n_dither=3).get_params()
    assert params["model"] == "tabfm"
    assert params["n_dither"] == 3
    assert "z_grid" in params


def test_clone_reproduces_the_model():
    model = lazy.LazyModel("tabfm", n_estimators=4, n_dither=3)
    copy = sklearn_base.clone(model)
    assert copy.get_params() == model.get_params()
    assert isinstance(copy._build(), lazy.TabFMHistogram)
    assert copy is not model


def test_set_params_updates_the_backend_in_place():
    model = lazy.LazyModel("tabfm", n_dither=1)
    returned = model.set_params(n_dither=4)
    assert returned is model
    assert model.n_dither == 4
    assert model._build().n_dither == 4


def test_set_params_rejects_a_parameter_the_backend_does_not_take():
    with pytest.raises(ValueError, match="n_dither"):
        lazy.LazyModel("tabicl").set_params(n_dither=3)


def test_set_params_can_switch_backend():
    model = lazy.LazyModel("tabfm", n_dither=3)
    model.set_params(model="tabicl", n_estimators=16)
    assert model.name_ == "tabicl:v2"
    assert isinstance(model._build(), lazy.TabICLQuantile)
    assert model.n_estimators == 16


def test_switching_backend_keeps_the_parameters_both_take():
    model = lazy.LazyModel("tabfm", n_estimators=3, device="cpu", n_dither=5)
    model.set_params(model="tabicl")
    assert model.n_estimators == 3
    assert model.device == "cpu"
    assert "n_dither" not in model.get_params()
    assert model.version == lazy.DEFAULT_VERSIONS["tabicl"]
    model.set_params(model="tabfm", n_estimators=2)
    assert model.n_estimators == 2 and model.device == "cpu"


def test_a_search_over_backends_keeps_its_fixed_settings():
    """GridSearchCV clones, which spells every default out, then sets."""
    base = lazy.LazyModel("tabfm", n_estimators=3, device="cpu")
    for name in ["tabicl", "tabpfn", "limix"]:
        model = sklearn_base.clone(base).set_params(model=name)
        backend = model._build()
        assert backend.n_estimators == 3
        assert backend.device == "cpu"
        defaults = registry.ESTIMATORS[name]().get_params()
        assert backend.version == defaults["version"]
        assert backend.random_state == defaults["random_state"]


def test_a_switch_rejects_a_parameter_the_new_backend_does_not_take():
    with pytest.raises(ValueError, match="n_dither"):
        lazy.LazyModel("tabfm").set_params(model="tabicl", n_dither=3)


def test_an_unknown_backend_set_by_set_params_is_rejected_at_fit():
    model = lazy.LazyModel("tabfm").set_params(model="nope")
    with pytest.raises(ValueError, match="tabfm"):
        model.fit(np.zeros((3, 2)), np.ones(3))


def test_repr_names_the_backend_and_only_non_default_parameters():
    assert repr(lazy.LazyModel("tabfm")) == "LazyModel('tabfm')"
    with_dither = lazy.LazyModel("tabfm", n_dither=3)
    assert repr(with_dither) == "LazyModel('tabfm', n_dither=3)"


def test_repr_shows_a_grid_other_than_the_native_one():
    grid = lazy.Grid.linear(0.0, 3.0, 37)
    model = lazy.LazyModel("tabfm", z_grid=grid, n_dither=3)
    assert repr(model) == f"LazyModel('tabfm', z_grid={grid!r}, n_dither=3)"


def test_the_grid_default_is_carried_down_to_the_backend():
    grid = lazy.Grid.linear(0.0, 3.0, 37)
    model = lazy.LazyModel("tabicl", z_grid=grid)
    assert model.z_grid is grid
    assert model._build().z_grid is grid
    assert model.get_params()["z_grid"] is grid


# -- the full protocol, through a stand-in backend --------------------------


class _Uniform(lazy.BaseDensityRegressor):
    """A backend that answers with a flat density, for wiring tests only."""

    def __init__(self, *, width=1.0, z_grid=None):
        self.width = width
        self.z_grid = z_grid

    def _fit(self, X, y):
        self.n_context_ = len(X)

    def _predict_pdf(self, X, grid):
        return np.full((len(X), grid.n_bins), self.width)


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setitem(registry.ESTIMATORS, "uniform", _Uniform)
    return "uniform"


def test_fit_predict_round_trip_through_the_wrapper(registered):
    X = np.random.default_rng(0).normal(size=(20, 3))
    z = np.random.default_rng(1).uniform(0.2, 1.8, 20)
    model = lazy.LazyModel(registered).fit(X, z)

    assert model.n_context_ == 20  # a fitted backend attribute, via delegation
    pdfs = model.predict_proba(X)
    assert pdfs.shape == (20, lazy.DC1_GRID.n_bins)
    assert np.allclose(np.trapezoid(pdfs, lazy.DC1_GRID.centers, axis=1), 1.0)
    assert model.predict(X, method="z_median").shape == (20,)


def test_a_call_time_grid_overrides_the_default(registered):
    X = np.random.default_rng(0).normal(size=(5, 2))
    z = np.random.default_rng(1).uniform(0.2, 1.8, 5)
    model = lazy.LazyModel(
        registered, z_grid=lazy.Grid.linear(0.0, 2.0, 50)
    ).fit(X, z)
    assert model.predict_proba(X).shape == (5, 50)
    assert model.predict_proba(X, lazy.Grid.linear(0.0, 3.0, 11)).shape == (
        5,
        11,
    )


def test_evaluate_labels_the_row_with_the_backend_name(registered):
    X = np.random.default_rng(0).normal(size=(30, 2))
    z = np.random.default_rng(1).uniform(0.2, 1.8, 30)
    table = lazy.LazyModel(registered).fit(X, z).evaluate(X, z)
    assert table["model"].iloc[0] == "uniform"


def test_a_fitted_model_survives_pickling(registered):
    X = np.random.default_rng(0).normal(size=(10, 2))
    z = np.random.default_rng(1).uniform(0.2, 1.8, 10)
    model = lazy.LazyModel(registered, width=2.0).fit(X, z)
    copy = pickle.loads(pickle.dumps(model))
    assert copy.width == 2.0 and copy.n_context_ == 10
    assert np.array_equal(copy.predict_proba(X), model.predict_proba(X))


def test_a_parameter_set_after_fit_reads_back_as_set(registered):
    X = np.random.default_rng(0).normal(size=(10, 2))
    z = np.random.default_rng(1).uniform(0.2, 1.8, 10)
    model = lazy.LazyModel(registered, width=2.0).fit(X, z)
    model.set_params(width=3.0)
    assert model.width == 3.0
    assert model.get_params()["width"] == 3.0
    assert model.estimator_.width == 2.0  # until the next fit
    assert model.n_context_ == 10


def test_the_wrapper_forwards_distributions_and_quantiles(registered):
    X = np.random.default_rng(0).normal(size=(12, 2))
    z = np.random.default_rng(1).uniform(0.2, 1.8, 12)
    model = lazy.LazyModel(registered).fit(X, z)
    dist = model.predict_distribution(X)
    assert len(dist) == 12
    np.testing.assert_allclose(
        model.predict_quantiles(X, [0.5])[:, 0], 1.0, atol=0.02
    )
