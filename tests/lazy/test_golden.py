# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Every backend reproduces the densities recorded in tests/lazy/golden.

Needs the checkpoints (LAZY_RUN_CHECKPOINT_TESTS=1). The tolerances allow
only float rounding, so a refactor that moves a prediction fails here.
"""

import os

import golden_data
import numpy as np
import pytest

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)

#: Relative tolerance per backend; 0 means bit-identical. The files were
#: recorded by this code on one machine's CPU; another machine's BLAS may
#: round float32 differently, which TabICL's narrowly spaced quantiles
#: amplify most.
RTOL = {"tabpfn": 1e-4, "tabicl": 1e-3, "tabfm": 1e-12}

#: Absolute tolerance per backend, as a fraction of the peak density.
#: TabFM's is float64 rounding: rebinning onto the grid takes differences
#: of cumulative masses, whose rounding is absolute (about 1e-16 of the
#: peak), so a far-tail density of 1e-9 can move by 1e-6 of itself when
#: the masses are scaled first (as on_grid has done since ee67572).
ATOL = {"tabpfn": 1e-6, "tabfm": 1e-12}


@needs_checkpoint
@pytest.mark.parametrize("name", sorted(golden_data.RECORDED_PARAMS))
def test_the_default_path_reproduces_the_golden_densities(name):
    pytest.importorskip(name)
    reference = np.load(golden_data.GOLDEN_DIR / f"{name}.npz")
    _, _, X_test = golden_data.problem()
    pdfs = golden_data.fit(name).predict_proba(X_test, golden_data.GRID)
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
