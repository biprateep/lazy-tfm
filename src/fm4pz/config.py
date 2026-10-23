# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Constants every experiment shares: sizes, seeds, the selection, the grid.

The training sets are those of the submitted paper, regenerated from the same
seeds (:mod:`fm4pz.splits`). The redshift grid is the common grid every method
is scored on, the classical baselines now and the foundation models later.
"""

from __future__ import annotations

import os
import pathlib

__all__ = [
    "BIASED_SIZES",
    "BIASED_TEST",
    "BIASED_TRAIN",
    "CALIBRATION",
    "CALIBRATION_METHODS",
    "CALIBRATION_SEEDS",
    "CALIBRATION_SIZES",
    "CV_FOLDS",
    "CV_SEED",
    "FINE_BINS",
    "FULL_TRAIN",
    "N_TEST",
    "REGIMES",
    "REPRESENTATIVE_SIZES",
    "SEEDS",
    "SELECTION",
    "Z_MAX",
    "Z_MIN",
    "results_dir",
]

#: Rows of the DC1 training and test files (Zenodo 10975874).
FULL_TRAIN = 43_486
N_TEST = 390_990

#: Training sizes of the representative scaling curve; the last is the file.
REPRESENTATIVE_SIZES: tuple[int, ...] = (
    1_000,
    5_000,
    10_000,
    15_000,
    20_000,
    30_000,
    FULL_TRAIN,
)

#: The HSC-like spectroscopic selection of the biased regime, as the
#: submitted paper drew it with ``lazy.datasets.fetch_dc1_biased``.
SELECTION: dict[str, int | bool] = {
    "n_train": 35_000,
    "n_calibration": 10_000,
    "control": True,
    "seed": 1,
    "selection_seed": 12345,
}

#: What that selection yields: the biased pool and the representative test.
BIASED_TRAIN = 35_011
BIASED_TEST = 19_383

#: Training sizes of the biased scaling curve; the last is the whole pool.
BIASED_SIZES: tuple[int, ...] = (
    1_000,
    5_000,
    10_000,
    15_000,
    20_000,
    30_000,
    BIASED_TRAIN,
)

#: The representative calibration sample the selection sets aside.
CALIBRATION = 10_000

#: Calibration galaxies added to a biased training set in the calibration
#: experiment, in steps of 500 up to the whole sample.
CALIBRATION_SIZES: tuple[int, ...] = tuple(range(0, CALIBRATION + 1, 500))

#: The methods and seeds of the calibration experiment so far.
CALIBRATION_METHODS: tuple[str, ...] = ("tabpfn",)
CALIBRATION_SEEDS: tuple[int, ...] = (0,)

REGIMES: tuple[str, ...] = ("representative", "biased")

#: The random subsamples each final number is averaged over. Seed 0 is the
#: subset of the submitted paper; at the full size every seed draws the same
#: rows, and only the methods' own randomness differs.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Cross-validation used to choose every hyperparameter, on training rows only.
CV_FOLDS = 5
CV_SEED = 2026

#: The common redshift grid: 4,000 bins of width 0.0005 on [0, 2].
Z_MIN = 0.0
Z_MAX = 2.0
FINE_BINS = 4_000


def results_dir() -> pathlib.Path:
    """Where intermediate products go: ``$FM4PZ_RESULTS`` or ``./results``.

    Returns:
        The directory, created if needed.
    """
    path = pathlib.Path(os.environ.get("FM4PZ_RESULTS", "results"))
    path.mkdir(parents=True, exist_ok=True)
    return path
