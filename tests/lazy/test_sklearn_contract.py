# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""scikit-learn's own checks of the estimator contract, on every estimator.

Only the checks that need no fitting: those would load a checkpoint. Between
them they hold ``__init__`` to storing its arguments and nothing else, and
``get_params``/``set_params``/``clone`` to agreeing with one another.
"""

import pytest
from sklearn.utils import estimator_checks

import lazy

ESTIMATORS = [
    *(lazy.LazyModel(name) for name in lazy.list_estimators()),
    lazy.LazyModel("tabfm", n_estimators=4, n_dither=3),
    lazy.TabFMHistogram(),
    lazy.TabICLQuantile(),
    lazy.TabPFNBarDistribution(),
]
CHECKS = [
    estimator_checks.check_no_attributes_set_in_init,
    estimator_checks.check_parameters_default_constructible,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
]


@pytest.mark.filterwarnings("error::UserWarning")
@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("estimator", ESTIMATORS, ids=repr)
def test_sklearn_contract(estimator, check):
    check(type(estimator).__name__, estimator)
