# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""scikit-learn's own checks of the estimator contract, on every estimator.

Only the checks that need no fitting: those would load a checkpoint. Between
them they hold ``__init__`` to storing its arguments and nothing else, and
``get_params``/``set_params``/``clone`` to agreeing with one another.
"""

import pytest
from sklearn.utils.estimator_checks import check_get_params_invariance
from sklearn.utils.estimator_checks import check_no_attributes_set_in_init
from sklearn.utils.estimator_checks import (
    check_parameters_default_constructible,
)
from sklearn.utils.estimator_checks import check_set_params

from lazy import LazyModel
from lazy import list_estimators
from lazy.models import TabFMHistogram
from lazy.models import TabICLQuantile
from lazy.models import TabPFNBarDistribution

ESTIMATORS = [
    *(LazyModel(name) for name in list_estimators()),
    LazyModel("tabfm", n_estimators=4, n_dither=3),
    TabFMHistogram(),
    TabICLQuantile(),
    TabPFNBarDistribution(),
]
CHECKS = [
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_get_params_invariance,
    check_set_params,
]


@pytest.mark.filterwarnings("error::UserWarning")
@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("estimator", ESTIMATORS, ids=repr)
def test_sklearn_contract(estimator, check):
    check(type(estimator).__name__, estimator)
