# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""A fixed small problem and the backend settings the golden outputs use.

The golden files in ``tests/lazy/golden/`` hold what each backend predicted
on this problem with the settings below. Refactors must reproduce them: they
are the guard on the paper's numbers. Regenerate them only on purpose, with
``python tests/lazy/record_golden.py``, as was done when member seeds moved
to ``SeedSequence([random_state, i])``.

``RECORDED_PARAMS`` is what produced each file, in today's API; when the API
changes, update it to the same model in the new API, never the files.
"""

from collections.abc import Iterator
import contextlib
import pathlib
from typing import Any

import numpy as np

import lazy
from lazy import datasets

GOLDEN_DIR = pathlib.Path(__file__).with_name("golden")

PER_VENDOR = frozenset({"tabpfn"})
"""Backends whose densities differ between Intel and AMD processors beyond the
golden tolerance (see "Across machines" in the reproducibility guide), so
they have a golden file per vendor (see :func:`path`).
"""

VENDORS = ("AuthenticAMD", "GenuineIntel")
"""The CPU vendors a backend in :data:`PER_VENDOR` has a golden file for."""

GRID = datasets.DC1_GRID
"""The grid every golden density is tabulated on."""

RECORDED_PARAMS: dict[str, dict[str, Any]] = {
    "kumo": {
        "version": "v1-small",
        "n_estimators": 5,
        "random_state": 42,
        "device": "cpu",
        "progress": False,
    },
    "tabpfn": {
        "version": "v3",
        "n_estimators": 2,
        "kv_cache": False,
        "random_state": 42,
        "device": "cpu",
        "progress": False,
    },
    "tabicl": {
        "n_estimators": 2,
        "kv_cache": False,
        "random_state": 42,
        "device": "cpu",
        "progress": False,
    },
    "tabfm": {
        # Recorded at 10 x 10, before "auto" sized the bins from the context
        # (it gives 300 rows 7 x 7).
        "n_coarse_bins": 10,
        "n_fine_bins": 10,
        "n_estimators": 1,
        "n_dither": 2,
        "kv_cache": True,
        "y_grid": GRID,
        "random_state": 1,
        "device": "cpu",
        "progress": False,
    },
    "limix": {
        "n_estimators": 2,
        "random_state": 42,
        "device": "cpu",
        "progress": False,
    },
}
"""Settings the files were recorded with, per backend. They keep the seeds and
paths of the first recording (no key/value cache on TabPFN and TabICL, seed 42
there and 1 on TabFM, whose bins span the DC1 grid). LimiX joined later, with
its defaults but for two members (one of each of its recipe's pipelines), and
Kumo Tabular later still, at its small size with five members, the fewest its
trimmed mean drops one from each tail of.
"""

THREADS = 4
"""Torch threads the golden densities are recorded and checked at, as many as a
GitHub Linux runner has CPUs. TabPFN's CPU densities move with the count by
up to 8% of the peak (audit finding TC-31), through its fingerprint feature
(see ``RTOL`` in ``test_golden.py``).
"""


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """Runs torch at :data:`THREADS` threads, then restores the count."""
    import torch  # noqa: PLC0415 - collected without torch installed.

    before = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def fit(name: str) -> Any:
    """The backend ``name``, fitted as its golden file was recorded.

    Args:
        name: A key of :data:`RECORDED_PARAMS`.

    Returns:
        The fitted :class:`lazy.LazyModel`, ready to predict on the problem.
    """
    X_train, z, _ = problem()
    return lazy.LazyModel(name, **RECORDED_PARAMS[name]).fit(X_train, z)


def path(name: str, vendor: str | None = None) -> pathlib.Path:
    """The golden file of the backend ``name`` for a CPU vendor.

    Args:
        name: A key of :data:`RECORDED_PARAMS`.
        vendor: A ``vendor_id`` such as ``"GenuineIntel"``; None for this
            machine's (:func:`cpu_vendor`).

    Returns:
        ``golden/<name>.npz``, or ``golden/<name>_intel.npz`` for a backend
        in :data:`PER_VENDOR` on an Intel processor.
    """
    vendor = vendor or cpu_vendor()
    if name in PER_VENDOR and vendor == "GenuineIntel":
        return GOLDEN_DIR / f"{name}_intel.npz"
    return GOLDEN_DIR / f"{name}.npz"


def cpu_vendor() -> str | None:
    """The CPU vendor in ``/proc/cpuinfo``, such as ``"AuthenticAMD"``.

    Returns:
        The first ``vendor_id``, or None where the file or the field is
        missing (off Linux, or on a processor that does not state one).
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("vendor_id"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a small photometry-like problem as a tuple (X_train, z, X_test).

    Five colour-like features that depend smoothly on redshift, with noise:
    300 labelled rows and 50 queries, float64, fully deterministic.
    """
    rng = np.random.default_rng(20260924)
    n_train, n_test = 300, 50
    z = rng.uniform(0.05, 1.6, n_train + n_test)
    features = np.column_stack(
        [
            np.sin(2.0 * z + k) + 0.3 * z * k + rng.normal(0.0, 0.05, z.size)
            for k in range(5)
        ]
    )
    return features[:n_train], z[:n_train], features[n_train:]
