# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Every plotting test runs headless; set this before pyplot is first imported.
matplotlib.use("Agg")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def photometry():
    """A small ugrizy-shaped frame: six magnitudes and six magnitude errors."""
    generator = np.random.default_rng(1)
    bands = ("U", "G", "R", "I", "Z", "Y")
    n = 32
    return pd.DataFrame(
        {b: generator.normal(23.0, 1.0, n).astype(np.float32) for b in bands}
        | {
            f"{b}ERR": generator.uniform(0.01, 0.2, n).astype(np.float32)
            for b in bands
        }
    )


@pytest.fixture
def hsc_grid():
    """A toy stand-in for the HSC grid: four pixels, two of them targetable.

    Magnitude bins [20, 22) and [22, 24); colour bins [0, 1) and [1, 2). The
    bright colour-0 pixel is spectroscopically easy (half its galaxies get a
    redshift, out to z = 1), the faint one is hard (a tenth, out to z = 0.5),
    and the colour-1 row is off limits -- no HSC spectra, so nothing survives.
    """
    from lazy.selection import HSCGrid

    return HSCGrid(
        ratios=np.array([[0.5, 0.1], [0.0, 0.0]]),
        x_edges=np.array([20.0, 22.0, 24.0]),
        y_edges=np.array([0.0, 1.0, 2.0]),
        spec=pd.DataFrame(
            {
                "mag": [21.0, 21.0, 23.0, 23.0],
                "color": [0.5, 0.5, 0.5, 0.5],
                "specz": [0.2, 1.0, 0.1, 0.5],
            }
        ),
    )


@pytest.fixture
def biasable_catalog():
    """2,000 galaxies in the toy grid's bright pixel, with a DC1-like `source` column.

    Every row has colour 0.5 and magnitude 21, so all of them sit in the pixel
    of ratio 0.5; the first hundred are marked as coming from the "train file".
    """
    from lazy.datasets import BANDS
    from lazy.datasets import Catalog

    n = 2_000
    generator = np.random.default_rng(3)
    frame = pd.DataFrame(
        {b: np.full(n, 21.0, dtype=np.float32) for b in BANDS}
        | {f"{b}ERR": np.full(n, 0.1, dtype=np.float32) for b in BANDS}
    )
    frame["G"] = np.float32(21.5)  # G - Z = 0.5, the grid's colour-0 row
    return Catalog(
        split="combined",
        raw=frame,
        redshift=generator.uniform(0.05, 0.95, n),
        object_id=np.arange(n),
        source=np.where(np.arange(n) < 100, 0, 1).astype(np.int8),
    )
