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

#: Relative tolerance per backend; 0 means bit-identical. TabPFN's bucket
#: borders are now built in float64 where the recorded run had upstream's
#: float32 ones, and TabICL's targets are standardised in float64 before its
#: float32 sees them: both moved the densities by float32 rounding, which
#: the quantiles' narrow spacing amplifies to ~5e-4 for TabICL.
RTOL = {"tabpfn": 1e-4, "tabicl": 1e-3, "tabfm": 1e-12}

#: Absolute tolerance per backend, as a fraction of the peak density.
ATOL = {"tabpfn": 1e-6}

#: Backends recorded in a reduced precision that lazy now uses only on CUDA
#: (``mixed_precision``; a CPU runs float32). TabFM's recording ran it in
#: bfloat16, as every GPU run does; the test reruns that path on the CPU, so
#: the golden still guards the precision the GPU results are computed in.
#: TabFM chooses its precision at prediction, so setting the resolved flag
#: after ``fit`` is enough.
RECORDED_IN_MIXED_PRECISION = {"tabfm"}


@needs_checkpoint
@pytest.mark.parametrize("name", sorted(golden_data.RECORDED_PARAMS))
def test_the_default_path_reproduces_the_golden_densities(name):
    pytest.importorskip(name)
    reference = np.load(golden_data.GOLDEN_DIR / f"{name}.npz")
    X_train, z, X_test = golden_data.problem()
    model = lazy.LazyModel(name, **golden_data.CURRENT_PARAMS[name])
    model.fit(X_train, z)
    if name in RECORDED_IN_MIXED_PRECISION:
        model.estimator_.mixed_precision_ = True
    pdfs = model.predict_proba(X_test, golden_data.GRID)
    np.testing.assert_array_equal(reference["edges"], golden_data.GRID.edges)
    if RTOL[name] == 0.0:
        np.testing.assert_array_equal(pdfs, reference["pdfs"])
    else:
        np.testing.assert_allclose(
            pdfs,
            reference["pdfs"],
            rtol=RTOL[name],
            atol=ATOL.get(name, 0.0) * reference["pdfs"].max(),
        )
