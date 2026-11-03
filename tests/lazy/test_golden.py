# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Every backend reproduces the densities recorded in tests/lazy/golden.

Needs the checkpoints (LAZY_RUN_CHECKPOINT_TESTS=1). The tolerances allow
only float rounding, so a refactor that moves a prediction fails here.
"""

import golden_data
import numpy as np
import pytest

import lazy

RTOL = {
    "kumo": 1e-3,
    "tabpfn": 1e-4,
    "tabicl": 1e-3,
    "tabfm": 1e-12,
    "limix": 1e-4,
}
"""Relative tolerance per backend. The files were
recorded by this code on one machine's CPU; another machine's BLAS may round
float32 differently, which the narrowly spaced quantiles of TabICL and Kumo
amplify most. LimiX moves by 4e-6 with the thread count alone.
"""

ATOL = {
    "kumo": 3e-6,
    "tabpfn": 1e-6,
    "tabicl": 3e-6,
    "tabfm": 1e-12,
    "limix": 1e-6,
}
"""Absolute tolerance per backend, as a fraction of the peak density. TabFM's is
float64 rounding: rebinning onto the grid takes differences of cumulative
masses, whose rounding is absolute (about 1e-16 of the peak), so a far-tail
density of 1e-9 can move by 1e-6 of itself when the masses are scaled first (as
on_grid has done since ee67572). Kumo's and TabICL's far-tail bins move by more
than their relative tolerance on another CPU: by up to 7.1e-7 and 2.4e-7 of
the peak on a 4-core x86 machine (TabICL's at one thread and at four), so
3e-6 leaves a margin of four.
"""


@pytest.mark.checkpoint
@pytest.mark.parametrize("name", sorted(golden_data.RECORDED_PARAMS))
def test_the_default_path_reproduces_the_golden_densities(name):
    try:
        lazy.ESTIMATORS[name]()._import_backend()
    except ImportError as error:
        pytest.skip(f"{name} is not installed: {error}")
    reference = np.load(golden_data.GOLDEN_DIR / f"{name}.npz")
    _, _, X_test = golden_data.problem()
    pdfs = golden_data.fit(name).predict_proba(X_test, golden_data.GRID)
    np.testing.assert_array_equal(reference["edges"], golden_data.GRID.edges)
    np.testing.assert_allclose(
        pdfs,
        reference["pdfs"],
        rtol=RTOL[name],
        atol=ATOL[name] * reference["pdfs"].max(),
    )
