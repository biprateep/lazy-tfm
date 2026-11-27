# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""scikit-learn's own checks of the estimator contract.

The backends get only the checks that need no fitting, since fitting would
load a checkpoint. Between them they hold ``__init__`` to storing its
arguments and nothing else, and ``get_params``/``set_params``/``clone`` to
agreeing with one another. The stand-ins, which run the shared layer with no
checkpoint, and a pool of them get scikit-learn's whole suite. The
meta-estimators users run on a model (a pipeline, parallel cross-validation,
a search over backends) get one test each.
"""

import numpy as np
import pytest
import sklearn
from sklearn import base as sklearn_base
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import utils as sklearn_utils
from sklearn.utils import estimator_checks
import standins

import lazy
from lazy.models import registry

ESTIMATORS = [
    lazy.LazyModel(),
    lazy.LazyModel("tabfm", n_estimators=4, n_dither=3),
    lazy.TabFMHistogram(),
    lazy.TabICLQuantile(),
    lazy.TabPFNBarDistribution(),
    lazy.LimiXBarDistribution(),
    lazy.KumoQuantile(),
]
CHECKS = [
    estimator_checks.check_no_attributes_set_in_init,
    estimator_checks.check_parameters_default_constructible,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_do_not_raise_errors_in_init_or_set_params,
]


@pytest.mark.filterwarnings("error::UserWarning")
@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("estimator", ESTIMATORS, ids=repr)
def test_sklearn_contract(estimator, check):
    check(type(estimator).__name__, estimator)


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=repr)
def test_every_estimator_is_a_regressor_that_accepts_nan(estimator):
    tags = sklearn_utils.get_tags(estimator)
    assert tags.estimator_type == "regressor"
    assert tags.input_tags.allow_nan


@pytest.mark.parametrize(
    "estimator",
    [lazy.LazyModel(), lazy.LazyEnsembleModel(["tabpfn", "tabicl"])],
    ids=repr,
)
def test_y_grid_is_not_routable_metadata(estimator):
    routing = estimator.get_metadata_routing()
    for method in ("predict", "predict_proba", "score"):
        assert not routing.consumes(method, ["y_grid"])
    assert not hasattr(estimator, "set_predict_proba_request")
    assert not hasattr(estimator, "set_score_request")
    assert hasattr(estimator, "set_predict_request")  # for ``method``


def test_routing_refuses_a_grid_rather_than_split_it_by_fold(regression):
    """A grid as long as the data would otherwise be cut per fold."""
    X, y = regression
    model = standins.HistogramStandIn(n_estimators=2, progress=False)
    with sklearn.config_context(enable_metadata_routing=True):
        with pytest.raises(TypeError, match="y_grid"):
            model_selection.cross_validate(
                model, X, y, cv=2, params={"y_grid": np.linspace(-6, 6, 60)}
            )


# The checks every model fails on purpose, and why.
_DELIBERATE = {
    "check_regressors_no_decision_function": (
        "predict_proba returns densities on a grid; whether a regressor keeps "
        "it waits for the design note (SK-02)"
    ),
    "check_dtype_object": (
        "a non-numeric cell raises ValueError, not the TypeError the check "
        "expects; the error type is part of the API"
    ),
}


def _expected_failures(estimator):
    """What scikit-learn's suite is expected to fail, and why."""
    if isinstance(estimator, standins.GaussianDummy):
        return {
            **_DELIBERATE,
            "check_regressors_train": (
                "the dummy echoes the first feature, so it cannot fit"
            ),
        }
    return _DELIBERATE


@estimator_checks.parametrize_with_checks(
    [
        standins.GaussianDummy(),
        standins.HistogramStandIn(),
        standins.QuantileStandIn(),
        lazy.LazyEnsembleModel(
            [standins.HistogramStandIn(), standins.QuantileStandIn()]
        ),
    ],
    expected_failed_checks=_expected_failures,
)
def test_scikit_learns_whole_suite(estimator, check):
    check(estimator)


# -- meta-estimators -----------------------------------------------------------


@pytest.fixture
def regression():
    """Features, shape (60, 3), and a target that follows the first."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3))
    return X, 2.0 * X[:, 0] + rng.normal(0.0, 0.1, 60)


def test_a_pipeline_predicts_as_its_model_on_the_scaled_features(regression):
    X, y = regression
    model = standins.HistogramStandIn(n_estimators=2, progress=False)
    piped = pipeline.make_pipeline(preprocessing.StandardScaler(), model)
    piped = sklearn_base.clone(piped).fit(X, y)
    scaled = preprocessing.StandardScaler().fit_transform(X)
    np.testing.assert_array_equal(
        piped.predict_proba(X), model.fit(scaled, y).predict_proba(scaled)
    )


def test_cross_validation_in_worker_processes_scores_as_in_this_one(
    regression,
):
    """Each fold's model is pickled into a worker process and back."""
    X, y = regression
    model = standins.QuantileStandIn(n_estimators=2, progress=False)
    np.testing.assert_allclose(
        model_selection.cross_val_score(model, X, y, cv=2, n_jobs=2),
        model_selection.cross_val_score(model, X, y, cv=2),
        rtol=1e-12,
    )


def test_a_grid_search_over_backends_refits_the_best(
    regression, standin_backends
):
    X, y = regression
    search = model_selection.GridSearchCV(
        lazy.LazyModel("tabpfn", n_estimators=2, progress=False),
        {"model": ["tabpfn", "tabicl"]},
        cv=2,
    ).fit(X, y)
    # Two backends, two different scores: each candidate ran its own.
    assert len(set(search.cv_results_["mean_test_score"])) == 2
    best = search.best_estimator_
    assert isinstance(best.estimator_, registry.ESTIMATORS[best.model])
    assert best.estimator_.n_estimators == 2
    assert search.predict_proba(X).shape == (len(X), best.grid_.n_bins)
