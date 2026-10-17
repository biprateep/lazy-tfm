# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""The feature sets the classical baselines choose between.

Which inputs a baseline sees is one of its hyperparameters: each searches over
the common set the foundation models see and the set it used in the Data
Challenge (Schmidt et al. 2020) or in RAIL. The DC1 files hold six magnitudes
and their errors and no non-detections, so no placeholder values are needed.

Typical usage example:

  X = features.build(catalog, "common12")
"""

from __future__ import annotations

from lazy import datasets
import numpy as np
import numpy.typing as npt

__all__ = ["BANDS", "FEATURE_SETS", "build", "colors"]

BANDS: tuple[str, ...] = ("U", "G", "R", "I", "Z", "Y")

#: What each feature set holds.
FEATURE_SETS: dict[str, str] = {
    "common12": (
        "the foundation models' set: i, the five adjacent colours and the "
        "errors of all six (lazy's 'mag-color')"
    ),
    "i_colors": "i and the five adjacent colours, no errors (RAIL FlexZBoost)",
    "mags_errs": "the six magnitudes and their errors (DC1 FlexZBoost)",
    "mags_logerrs": (
        "the six magnitudes and the logarithms of their errors (DC1 and RAIL "
        "GPz)"
    ),
    "common12_logerrs": (
        "i, the five adjacent colours and the logarithms of the six errors"
    ),
}


def colors(
    catalog: datasets.Catalog,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The five adjacent colours and their errors, in quadrature.

    Args:
        catalog: A DC1 catalogue.

    Returns:
        A tuple (colours, errors), each of shape (n_rows, 5), in mag.
    """
    raw = catalog.raw
    mags = raw[list(BANDS)].to_numpy(np.float64)
    errs = raw[[f"{band}ERR" for band in BANDS]].to_numpy(np.float64)
    color = mags[:, :-1] - mags[:, 1:]
    color_err = np.sqrt(errs[:, :-1] ** 2 + errs[:, 1:] ** 2)
    return color, color_err


def build(catalog: datasets.Catalog, name: str) -> npt.NDArray[np.float64]:
    """One feature set of a catalogue, as a float64 array.

    Args:
        catalog: A DC1 catalogue.
        name: One of :data:`FEATURE_SETS`.

    Returns:
        The features, shape (n_rows, n_features).

    Raises:
        ValueError: If ``name`` is not a known feature set.
    """
    raw = catalog.raw
    mags = raw[list(BANDS)].to_numpy(np.float64)
    errs = raw[[f"{band}ERR" for band in BANDS]].to_numpy(np.float64)
    i_band = mags[:, [BANDS.index("I")]]
    color, _ = colors(catalog)
    if name == "common12":
        return catalog.features("mag-color").to_numpy(np.float64)
    if name == "i_colors":
        return np.hstack([i_band, color])
    if name == "mags_errs":
        return np.hstack([mags, errs])
    if name == "mags_logerrs":
        return np.hstack([mags, np.log10(errs)])
    if name == "common12_logerrs":
        return np.hstack([i_band, color, np.log10(errs)])
    raise ValueError(f"unknown feature set: {name=}")
