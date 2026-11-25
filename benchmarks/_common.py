# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""What the benchmarks of a backend share.

:func:`setup` records what a run ran on, as "Measurements" in CLAUDE.md
asks; :func:`sample` and :func:`days` prepare the rows, and :func:`score`
scores a fitted model from one prediction. The benchmarks import
this module from their own folder::

    import _common

    print(_common.setup("tabpfn"))
"""

from importlib import metadata
import os
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd

import lazy
from lazy import metrics

_PACKAGES = (
    "lazy-tfm",
    "torch",
    "transformers",
    "chronos-forecasting",
    "numpy",
    "pandas",
    "scikit-learn",
)
_THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def setup(backend: str) -> dict[str, str]:
    """The commit, machine, GPU, threads and versions of a run.

    Args:
        backend: The registered backend the run fits, whose upstream
            package's version is recorded too.

    Returns:
        Each fact as text, by name.
    """
    import torch  # noqa: PLC0415 - every backend's extra installs it.

    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
    upstream = lazy.ESTIMATORS[backend].upstream_package
    return {
        "commit": commit,
        "machine": _cpu(),
        "gpu": gpu,
        "cuda": str(torch.version.cuda),
        "cpus": str(os.cpu_count()),
        "torch_threads": str(torch.get_num_threads()),
        **{name: os.environ.get(name, "unset") for name in _THREADS},
        "python": platform.python_version(),
        **{name: _version(name) for name in (*_PACKAGES, upstream)},
    }


def sample(
    X: pd.DataFrame, y: np.ndarray, n: int | None, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """At most ``n`` rows, drawn with ``seed``, in their original order.

    Args:
        X: The features.
        y: The targets.
        n: The most rows to keep; ``None`` keeps them all.
        seed: The seed of the draw.

    Returns:
        The rows kept, and their targets as float64.
    """
    y = np.asarray(y, dtype=float)
    if n is None or n >= len(X):
        return X, y
    rows = np.sort(np.random.default_rng(seed).permutation(len(X))[:n])
    return X.iloc[rows], y[rows]


def score(model: Any, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """The CRPS, NLL and RMSE of the mean, from one prediction of ``X``.

    Args:
        model: A fitted model, or a pipeline that ends in one.
        X: The test features.
        y: Their targets, shape (n,).

    Returns:
        The mean CRPS and NLL over the rows, and the RMSE of the mean of
        each row's distribution, all on the model's default grid.
    """
    grid = getattr(model, "_final_estimator", model).grid_
    pdfs = model.predict_proba(X)
    _, _, crps, nll = metrics.per_object_metrics(y, grid, pdfs)
    mean = metrics.grid_point_estimate(grid, pdfs, "mean")
    return {
        "crps": float(np.mean(crps)),
        "nll": float(np.mean(nll)),
        "rmse": float(np.sqrt(np.mean((mean - y) ** 2))),
    }


def days(X: pd.DataFrame) -> pd.DataFrame:
    """The table with each date column as days since 1970."""
    X = X.copy()
    for column in X:
        if pd.api.types.is_datetime64_any_dtype(X[column]):
            X[column] = (X[column] - pd.Timestamp(0)).dt.total_seconds() / 86400
    return X


def _cpu() -> str:
    """The processor's model name, where Linux gives one."""
    try:
        with open("/proc/cpuinfo", encoding="ascii") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _version(name: str) -> str:
    """A package's installed version, or ``"absent"``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"
