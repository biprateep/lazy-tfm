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
        | {f"{b}ERR": generator.uniform(0.01, 0.2, n).astype(np.float32) for b in bands}
    )
