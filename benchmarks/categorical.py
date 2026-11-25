# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
r"""Native categorical columns against their codes read as numbers.

For each dataset with an unordered categorical column, fits one backend
twice on the same rows: "codes" gives the model each category's integer
code with ``categorical_features=None``, and "native" keeps the category
type, which the default ``"from_dtype"`` marks. Text and image columns are
dropped, and dates become days since 1970. Each fit takes ``--n-train``
rows of the training part and scores ``--n-test`` rows of the test part:
MulTaBench's own split (fold 0), or a seeded 70/30 split for the others.

Run it from the repository root, with the backend's extra and pyarrow::

    uv run --extra tabfm --extra multabench \
        python benchmarks/categorical.py --backend tabfm --out tabfm.csv

It writes one row per dataset and arm, with the CRPS, the NLL, the RMSE of
the mean and the seconds taken, and prints the setup first.
"""

import argparse
from importlib import metadata
import os
import pathlib
import platform
import subprocess
import time

import numpy as np
import pandas as pd

import lazy
from lazy import _multabench
from lazy import datasets

DATASETS = (
    "insurance",
    "ames_housing",
    "bike_sharing",
    "kings_county",
    "wine_quality",
    *(f"multabench_{name}" for name in sorted(_multabench.DATASETS)),
)
"""lazy's datasets with categorical columns. diamonds is left out: its
categorical columns are ordered, so every arm reads them as codes."""


def _setup(backend: str) -> dict[str, str]:
    """The commit, machine, threads and versions the run records."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    packages = ("lazy-tfm", "torch", "numpy", "pandas", "scikit-learn")
    versions = {name: _version(name) for name in packages}
    upstream = lazy.ESTIMATORS[backend].upstream_package
    versions[upstream] = _version(upstream)
    import torch  # noqa: PLC0415 - every backend's extra installs it.

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
    return {
        "commit": commit,
        "machine": _cpu(),
        "gpu": gpu,
        "cpus": str(os.cpu_count()),
        "threads": os.environ.get("OMP_NUM_THREADS", "default"),
        **versions,
    }


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


def _tabular(X: pd.DataFrame, modalities: dict[str, str]) -> pd.DataFrame:
    """The columns of numbers, categories and dates, dates as days."""
    keep = [
        column
        for column in X
        if column not in modalities
        and (
            isinstance(X[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_numeric_dtype(X[column])
            or pd.api.types.is_datetime64_any_dtype(X[column])
        )
    ]
    X = X[keep].copy()
    for column in X:
        if pd.api.types.is_datetime64_any_dtype(X[column]):
            X[column] = (X[column] - pd.Timestamp(0)).dt.total_seconds() / 86400
    return X


def _codes(X: pd.DataFrame) -> pd.DataFrame:
    """Each categorical column as its integer codes, missing as NaN."""
    X = X.copy()
    for column in X:
        if isinstance(X[column].dtype, pd.CategoricalDtype):
            codes = X[column].cat.codes.astype(float)
            X[column] = codes.where(X[column].notna())
    return X


def _parts(name: str, data_home: str | None) -> tuple:
    """The training and test features and targets of one dataset."""
    if name.startswith("multabench_"):
        train, test = datasets.load_dataset(
            name, split=True, data_home=data_home, images=False
        )
        return (
            _tabular(train.X, train.modalities),
            train.y,
            _tabular(test.X, test.modalities),
            test.y,
        )
    dataset = datasets.load_dataset(name, data_home=data_home)
    X = _tabular(pd.DataFrame(dataset.X), dataset.modalities)
    rows = np.random.default_rng(42).permutation(len(X))
    cut = int(0.7 * len(X))
    y = np.asarray(dataset.y)
    return X.iloc[rows[:cut]], y[rows[:cut]], X.iloc[rows[cut:]], y[rows[cut:]]


def _sample(X: pd.DataFrame, y: np.ndarray, n: int, seed: int) -> tuple:
    """At most ``n`` rows, drawn with ``seed``, in their original order."""
    rows = np.sort(np.random.default_rng(seed).permutation(len(X))[:n])
    return X.iloc[rows], np.asarray(y, dtype=float)[rows]


def main() -> None:
    """Runs every dataset and arm, appending a row to ``--out`` for each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--data-home", default=None)
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()
    print(_setup(args.backend), flush=True)
    params = {"random_state": 0}
    if args.version:
        params["version"] = args.version
    for name in args.datasets:
        X_train, y_train, X_test, y_test = _parts(name, args.data_home)
        X_train, y_train = _sample(X_train, y_train, args.n_train, 0)
        X_test, y_test = _sample(X_test, y_test, args.n_test, 1)
        unordered = [
            column
            for column in X_train
            if isinstance(X_train[column].dtype, pd.CategoricalDtype)
            and not X_train[column].dtype.ordered
        ]
        if not unordered:
            continue
        for arm in ("codes", "native"):
            fit_X, test_X = (
                (_codes(X_train), _codes(X_test))
                if arm == "codes"
                else (X_train, X_test)
            )
            extra = {"categorical_features": None} if arm == "codes" else {}
            start = time.perf_counter()
            model = lazy.LazyModel(args.backend, **params, **extra)
            model.fit(fit_X, y_train)
            scores = model.evaluate(test_X, y_test, method="mean").iloc[0]
            mean = model.predict(test_X, method="mean")
            row = {
                "dataset": name,
                "backend": args.backend,
                "version": args.version,
                "arm": arm,
                "n_categorical": len(unordered),
                "n_features": X_train.shape[1],
                "crps": scores["crps"],
                "nll": scores["nll"],
                "rmse": float(np.sqrt(np.mean((mean - y_test) ** 2))),
                "seconds": time.perf_counter() - start,
            }
            print(row, flush=True)
            pd.DataFrame([row]).to_csv(
                args.out, mode="a", header=not args.out.exists(), index=False
            )


if __name__ == "__main__":
    main()
