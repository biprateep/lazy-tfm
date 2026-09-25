# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Refactored backends reproduce the densities recorded before the refactor.

Needs the checkpoints (LAZY_RUN_CHECKPOINT_TESTS=1). The tolerance is exact
equality for the backends whose path is unchanged, and float rounding where
the reduction order legitimately changed.
"""

import os

import golden_data
import numpy as np
import pytest

import lazy

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)

#: Relative tolerance per backend; 0 means bit-identical.
RTOL = {"tabpfn": 0.0, "tabicl": 0.0, "tabfm": 1e-12}


@needs_checkpoint
@pytest.mark.parametrize("name", sorted(golden_data.RECORDED_PARAMS))
def test_the_default_path_reproduces_the_golden_densities(name):
    pytest.importorskip(name)
    reference = np.load(golden_data.GOLDEN_DIR / f"{name}.npz")
    X_train, z, X_test = golden_data.problem()
    model = lazy.LazyModel(name, **golden_data.CURRENT_PARAMS[name])
    pdfs = model.fit(X_train, z).predict_proba(X_test, golden_data.GRID)
    np.testing.assert_array_equal(reference["edges"], golden_data.GRID.edges)
    if RTOL[name] == 0.0:
        np.testing.assert_array_equal(pdfs, reference["pdfs"])
    else:
        np.testing.assert_allclose(pdfs, reference["pdfs"], rtol=RTOL[name])
