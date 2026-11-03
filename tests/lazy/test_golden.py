# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Every backend reproduces the densities recorded in tests/lazy/golden.

Needs the checkpoints (LAZY_RUN_CHECKPOINT_TESTS=1). The tolerances allow
only float rounding, so a refactor that moves a prediction fails here.
"""

import json

import golden_data
import numpy as np
import pytest

import lazy

RTOL = {
    "kumo": 1e-3,
    "tabpfn": 1e-4,
    "tabicl": 1e-3,
    "tabfm": 1e-4,
    "limix": 1e-4,
}
"""Relative tolerance per backend. Another machine's BLAS may round float32
differently, which the narrowly spaced quantiles of TabICL and Kumo amplify
most. LimiX moves by 4e-6 with the thread count alone.

TabPFN's and TabFM's files hold on a GitHub Linux runner (AVX2), where the
checkpoint workflow recorded them: two runners gave the same densities to the
bit. Another processor moves them by 3% to 5% of the peak, so their golden
test passes only on that platform. TabPFN hashes each row of its float32
preprocessed data into a fingerprint feature, so a last-bit difference gives a
different feature; TabFM's golden runs in bfloat16. See "Across machines" in
the reproducibility guide.
"""

ATOL = {
    "kumo": 3e-6,
    "tabpfn": 1e-5,
    "tabicl": 3e-6,
    "tabfm": 1e-5,
    "limix": 1e-6,
}
"""Absolute tolerance per backend, as a fraction of the peak density. Two
runners agreed to the bit on TabPFN and TabFM, so their 1e-5 (and their 1e-4
relative) are a margin chosen, not measured, for a change of runner image on
the same processor. Kumo's and TabICL's far-tail bins move by more
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
    with golden_data.pinned_threads():
        pdfs = golden_data.fit(name).predict_proba(X_test, golden_data.GRID)
    np.testing.assert_array_equal(reference["edges"], golden_data.GRID.edges)
    np.testing.assert_allclose(
        pdfs,
        reference["pdfs"],
        rtol=RTOL[name],
        atol=ATOL[name] * reference["pdfs"].max(),
        err_msg=_platforms(reference),
    )


def _platforms(reference: np.lib.npyio.NpzFile) -> str:
    """The CPU capability a file was recorded on, and this machine's."""
    import torch  # noqa: PLC0415 - collected without torch installed.

    if "provenance" not in reference:
        return ""
    provenance = json.loads(str(reference["provenance"]))
    return (
        f"recorded on {provenance.get('cpu_capability', 'an unrecorded CPU')}; "
        f"this machine is {torch.backends.cpu.get_cpu_capability()}"
    )
