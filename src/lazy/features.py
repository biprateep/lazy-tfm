"""Turning raw photometry into the tabular view an estimator sees.

Nothing in :mod:`lazy` requires these recipes -- an estimator takes any
DataFrame you give it -- but which columns you feed a tabular foundation model
matters more than most hyper-parameters, and these are the views the benchmarks
compare. Colours beat magnitudes, and propagating the magnitude errors into
colour errors beats leaving them as magnitude errors::

    from lazy.datasets import fetch_dc1
    from lazy.features import build_features

    train = fetch_dc1("train")
    X = build_features(train.raw, recipe="adjcolors_cerr")

The recipes assume ``ugrizy``-style columns: one magnitude per band named after
the band, and one error per band named ``<BAND>ERR``. Pass ``bands`` to use a
different filter set.
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["FEATURE_RECIPES", "build_features"]

FeatureRecipe = Literal["raw12", "adjcolors", "adjcolors_cerr", "allcolors", "colors5"]

#: Recipe name -> one-line description, for help text and documentation.
FEATURE_RECIPES: dict[str, str] = {
    "raw12": "six magnitudes + six magnitude errors (no engineering)",
    "adjcolors": "i magnitude + five adjacent colours + six magnitude errors",
    "adjcolors_cerr": "i magnitude + five adjacent colours + the matching colour errors",
    "allcolors": "six magnitudes + all fifteen colours + six magnitude errors",
    "colors5": "the five adjacent colours only (magnitude-agnostic)",
}

DEFAULT_BANDS = ("U", "G", "R", "I", "Z", "Y")


def build_features(
    raw: pd.DataFrame,
    recipe: FeatureRecipe = "adjcolors_cerr",
    *,
    bands: tuple[str, ...] = DEFAULT_BANDS,
    reference_band: str = "I",
) -> pd.DataFrame:
    """Construct the feature view named by ``recipe``.

    Parameters
    ----------
    raw
        Photometry with one column per band and one ``<BAND>ERR`` column per
        band.
    recipe
        One of :data:`FEATURE_RECIPES`.

        ``raw12``
            The magnitudes and their errors, untouched. The honest baseline:
            whatever the model achieves here, it achieved without help.
        ``adjcolors``
            The reference magnitude plus the five adjacent colours, keeping the
            six *magnitude* errors. Colours carry the spectral shape that sets
            the redshift, so this is a strictly better view of the same twelve
            numbers.
        ``adjcolors_cerr``
            The same twelve columns, but with the errors propagated to match:
            the reference-band error and the five adjacent *colour* errors,
            each the quadrature sum of the two magnitude errors forming the
            colour. Same dimensionality as ``adjcolors``, so a comparison
            between the two isolates the error representation alone.
        ``allcolors``
            Magnitudes, all fifteen pairwise colours, and the magnitude errors.
            Redundant by construction -- every colour is a difference of two
            columns already present -- and a test of whether the model would
            rather be handed the differences than infer them.
        ``colors5``
            The five adjacent colours and nothing else: no magnitude, no
            errors, so no way to use apparent brightness as a distance prior.
    bands
        Band names in wavelength order.
    reference_band
        The single magnitude the colour recipes keep.

    Returns
    -------
    pandas.DataFrame
        One row per input row, float32 columns, names chosen to be readable in
        feature-importance output (``"G-R"``, ``"G-RERR"``, ...).

    Examples
    --------
    >>> raw = pd.DataFrame(
    ...     {b: np.full(3, 20.0) for b in DEFAULT_BANDS}
    ...     | {f"{b}ERR": np.full(3, 0.1) for b in DEFAULT_BANDS}
    ... )
    >>> list(build_features(raw, "adjcolors_cerr").columns)
    ['I', 'U-G', 'G-R', 'R-I', 'I-Z', 'Z-Y', 'IERR', 'U-GERR', 'G-RERR', 'R-IERR', 'I-ZERR', 'Z-YERR']
    """
    if recipe not in FEATURE_RECIPES:
        raise ValueError(f"unknown feature recipe {recipe!r}; known: {sorted(FEATURE_RECIPES)}")
    err_columns = tuple(f"{band}ERR" for band in bands)
    missing = [c for c in (*bands, *err_columns) if c not in raw.columns]
    if missing:
        raise KeyError(f"raw photometry is missing columns: {missing}")
    if reference_band not in bands:
        raise ValueError(f"reference_band {reference_band!r} is not one of {bands}")

    mags = raw[list(bands)].to_numpy(dtype=np.float32)
    errs = raw[list(err_columns)].to_numpy(dtype=np.float32)
    index = {band: i for i, band in enumerate(bands)}
    out: dict[str, np.ndarray] = {}

    if recipe == "raw12":
        for band in bands:
            out[band] = mags[:, index[band]]
    elif recipe in ("adjcolors", "adjcolors_cerr"):
        out[reference_band] = mags[:, index[reference_band]]
        for first, second in itertools.pairwise(bands):
            out[f"{first}-{second}"] = mags[:, index[first]] - mags[:, index[second]]
        if recipe == "adjcolors_cerr":
            out[f"{reference_band}ERR"] = errs[:, index[reference_band]]
            for first, second in itertools.pairwise(bands):
                out[f"{first}-{second}ERR"] = np.hypot(errs[:, index[first]], errs[:, index[second]])
            return pd.DataFrame(out)
    elif recipe == "allcolors":
        for band in bands:
            out[band] = mags[:, index[band]]
        for i, first in enumerate(bands):
            for second in bands[i + 1 :]:
                out[f"{first}-{second}"] = mags[:, index[first]] - mags[:, index[second]]
    elif recipe == "colors5":
        for first, second in itertools.pairwise(bands):
            out[f"{first}-{second}"] = mags[:, index[first]] - mags[:, index[second]]
        return pd.DataFrame(out)

    for i, name in enumerate(err_columns):
        out[name] = errs[:, i]
    return pd.DataFrame(out)
