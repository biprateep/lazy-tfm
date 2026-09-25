# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Small CPU settings for every registered backend's checkpoint tests.

A backend the conformance suite cannot find here fails that suite, so a new
backend arrives with its settings.
"""

#: Per backend: parameters small enough for CPU, as its own tests use.
BACKEND_SETTINGS = {
    "limix": {"n_estimators": 2, "chunk_size": 7},
    "tabpfn": {"n_estimators": 2, "chunk_size": 7},
    "tabicl": {"n_estimators": 2, "chunk_size": 7},
    "tabfm": {
        "n_coarse_bins": 2,
        "n_fine_bins": 2,
        "n_estimators": 1,
        "n_dither": 2,
    },
}
