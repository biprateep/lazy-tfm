# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""A fixed small problem and the backend settings the golden outputs use.

The golden files in ``tests/lazy/golden/`` hold what each backend predicted
on this problem before the uniform feature layer was introduced. Refactors
must reproduce them: they are the guard on the paper's numbers. Regenerate
them only on purpose, with ``python tests/lazy/record_golden.py``.

``RECORDED_PARAMS`` is what produced each file, in the API of the day.
``CURRENT_PARAMS`` is the same model in today's API, and is what the golden
test constructs; update it, never the files, when the API changes.
"""

import pathlib
from typing import Any

import numpy as np

import lazy

GOLDEN_DIR = pathlib.Path(__file__).with_name("golden")

#: The grid every golden density is tabulated on.
GRID = lazy.DC1_GRID

#: Settings when the files were recorded (commit 5144b23), per backend.
RECORDED_PARAMS: dict[str, dict[str, Any]] = {
    "tabpfn": {
        "version": "v3",
        "n_estimators": 2,
        "fit_mode": "fit_preprocessors",
        "device": "cpu",
        "progress": False,
    },
    "tabicl": {"n_estimators": 2, "device": "cpu", "progress": False},
    "tabfm": {
        "n_estimators": 1,
        "n_dither": 2,
        "inference": "stream",
        "device": "cpu",
        "progress": False,
    },
}

#: The same models in the current API.
CURRENT_PARAMS: dict[str, dict[str, Any]] = {
    **RECORDED_PARAMS,
    # The recorded runs had no key/value cache.
    "tabicl": {**RECORDED_PARAMS["tabicl"], "kv_cache": False},
    "tabpfn": {
        **{
            k: v
            for k, v in RECORDED_PARAMS["tabpfn"].items()
            if k != "fit_mode"
        },
        "kv_cache": False,
    },
    # The recorded run pinned its bins to the DC1 grid it predicted on; the
    # bins now follow the constructor's grid, and inference='stream' is the
    # cache.
    "tabfm": {
        **{
            k: v
            for k, v in RECORDED_PARAMS["tabfm"].items()
            if k != "inference"
        },
        "kv_cache": True,
        "z_grid": GRID,
    },
}


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
