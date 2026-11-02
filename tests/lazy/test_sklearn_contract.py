# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""scikit-learn's own checks of the estimator contract.

The backends get only the checks that need no fitting, since fitting would
load a checkpoint. Between them they hold ``__init__`` to storing its
arguments and nothing else, and ``get_params``/``set_params``/``clone`` to
agreeing with one another. The stand-ins, which run the shared layer with no
checkpoint, and a pool of them get scikit-learn's whole suite.
"""

import pytest
from sklearn import utils as sklearn_utils
from sklearn.utils import estimator_checks
import standins

import lazy

ESTIMATORS = [
    *(lazy.LazyModel(name) for name in lazy.list_estimators()),
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
