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

TabPFN has a file per CPU vendor (``golden_data.PER_VENDOR``), each recorded
by the checkpoint workflow on a GitHub runner of that vendor.
"""

ATOL = {
    "kumo": 3e-6,
    "tabpfn": 1e-5,
    "tabicl": 3e-6,
    "tabfm": 2.5e-5,
    "limix": 1e-6,
}
"""Absolute tolerance per backend, as a fraction of the peak density. TabPFN's
1e-5 is twice the largest difference measured between runners of one vendor:
4.8e-6, between AMD's EPYC 7763 and its 9V74 and 9V45. TabFM's file, in
float32, differed by up to 6.1e-6 of the peak between the runner (AVX2) that
recorded it and a 4-core AVX-512 machine, so 2.5e-5 leaves a margin of four.
Kumo's and TabICL's far-tail bins move by more than their relative tolerance
on another CPU: by up to 7.1e-7 and 2.4e-7 of the peak on a 4-core x86 machine
(TabICL's at one thread and at four), so 3e-6 leaves a margin of four.
"""


@pytest.mark.checkpoint
@pytest.mark.parametrize("name", sorted(golden_data.RECORDED_PARAMS))
def test_the_default_path_reproduces_the_golden_densities(name):
    try:
        lazy.ESTIMATORS[name]()._import_backend()
    except ImportError as error:
        pytest.skip(f"{name} is not installed: {error}")
    vendor = golden_data.cpu_vendor()
    if name in golden_data.PER_VENDOR and vendor not in golden_data.VENDORS:
        pytest.skip(
            f"{name} has no golden file for this processor (vendor {vendor});"
            f" it has one for each of {', '.join(golden_data.VENDORS)}"
        )
    reference = np.load(golden_data.path(name, vendor))
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
    recorded = " ".join(
        str(provenance.get(key, "unrecorded"))
        for key in ("cpu_vendor", "cpu_capability")
    )
    here = (
        f"{golden_data.cpu_vendor()} {torch.backends.cpu.get_cpu_capability()}"
    )
    return f"recorded on {recorded}; this machine is {here}"


MPS_GAP = {"tabpfn": 0.1}
"""Largest change of a row's 5%, 50% and 95% quantiles from the CPU to MPS,
as a fraction of that row's 5% to 95% range, per backend; 1e-3 for the others.
On an Apple M3 Ultra, with these settings, LimiX moved by 3.5e-6, TabICL by
1.5e-5 and TabFM by 1.8e-5 (float32 rounding), and TabPFN by 0.039, since its
fingerprint feature moves it across machines (see ``RTOL``). Each limit leaves
a margin of over two; MPS's wrong strided products moved Kumo Tabular by 3.9.
"""


@pytest.mark.checkpoint
@pytest.mark.parametrize("name", sorted(golden_data.RECORDED_PARAMS))
def test_auto_on_an_apple_gpu_answers_as_the_cpu_does(name):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no Apple GPU (MPS) here")
    try:
        lazy.ESTIMATORS[name]()._import_backend()
    except ImportError as error:
        pytest.skip(f"{name} is not installed: {error}")
    X_train, z, X_test = golden_data.problem()
    levels = [0.05, 0.5, 0.95]
    answers = {}
    for device in ("cpu", "auto"):
        params = {**golden_data.RECORDED_PARAMS[name], "device": device}
        model = lazy.LazyModel(name, **params).fit(X_train, z)
        answers[device] = model.predict_quantiles(X_test, quantiles=levels)
    cpu = answers["cpu"]
    gap = np.abs(answers["auto"] - cpu).max(axis=1) / (cpu[:, 2] - cpu[:, 0])
    assert gap.max() < MPS_GAP.get(name, 1e-3)
