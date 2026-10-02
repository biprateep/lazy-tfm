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

import pathlib
from typing import Any

import numpy as np

import lazy
from lazy import datasets

GOLDEN_DIR = pathlib.Path(__file__).with_name("golden")

#: The grid every golden density is tabulated on.
GRID = datasets.DC1_GRID

#: Settings the files were recorded with, per backend. They keep the seeds
#: and paths of the first recording (no key/value cache on TabPFN and
#: TabICL, seed 42 there and 1 on TabFM, whose bins span the DC1 grid).
RECORDED_PARAMS: dict[str, dict[str, Any]] = {
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
        "n_estimators": 1,
        "n_dither": 2,
        "kv_cache": True,
        "y_grid": GRID,
        "random_state": 1,
        "device": "cpu",
        "progress": False,
    },
}

#: Backends recorded in a reduced precision that lazy now uses only on CUDA
#: (``mixed_precision``; a CPU runs float32). TabFM's recording ran it in
#: bfloat16, as every GPU run does, so the golden still guards the precision
#: the GPU results are computed in. TabFM chooses its precision at
#: prediction, so setting the resolved flag after ``fit`` is enough.
RECORDED_IN_MIXED_PRECISION = {"tabfm"}


def fit(name: str) -> Any:
    """The backend ``name``, fitted as its golden file was recorded.

    Args:
        name: A key of :data:`RECORDED_PARAMS`.

    Returns:
        The fitted :class:`lazy.LazyModel`, ready to predict on the problem.
    """
    X_train, z, _ = problem()
    model = lazy.LazyModel(name, **RECORDED_PARAMS[name]).fit(X_train, z)
    if name in RECORDED_IN_MIXED_PRECISION:
        model.estimator_.mixed_precision_ = True
    return model


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
