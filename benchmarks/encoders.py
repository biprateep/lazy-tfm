# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
r"""Tabular columns against classical and pretrained encoders (TE-08).

For each dataset and fold, fits one model on three sets of features and
scores it on the fold's test rows:

* ``tabular``: the columns of numbers, categories and dates; the columns of
  text, images and time series are dropped;
* ``classical``: those columns too, each item column encoded by the
  classical encoder of its modality and reduced by :class:`lazy.Tabularizer`;
* ``pretrained``: the same with each modality's pretrained encoder.

The datasets are MulTaBench's 40 regression tables and PLAsTiCC's Deep
Drilling Field objects (``"plasticc_ddf"``), each split by MulTaBench's
split (``size="10k"``: three folds repeated twice with seed 42, at most
10,000 training and 5,000 test rows). Columns of other types are dropped,
and dates become days since 1970. ``--n-train`` and ``--n-test`` cut the
rows further, for a quick run.

Run it from the repository root on a GPU, with the extras::

    uv sync --extra tabpfn --extra encoders --extra multabench
    uv run python benchmarks/encoders.py --out encoders.csv

It prints the setup, then appends one row per dataset, fold and arm to
``--out``: the CRPS, the NLL and the RMSE of the mean on the test rows, all
from one prediction, the seconds to fit and to score, and the columns the
model saw. A dataset or a row that fails records its error, and the run goes
on. The pretrained encoders cache
their embeddings, so a second fold or arm encodes no item twice.
"""

import argparse
import pathlib
import time
import traceback
from typing import Any

import _common
import numpy as np
import pandas as pd

import lazy
from lazy import _multabench
from lazy import datasets

DATASETS = (
    *(f"multabench_{name}" for name in sorted(_multabench.DATASETS)),
    "plasticc_ddf",
)
"""Every dataset of the benchmark."""

ARMS = ("tabular", "classical", "pretrained")
"""The sets of features each dataset is fitted on."""

CLASSICAL = {
    "text": "text_hashing",
    "image": "image_statistics",
    "time": "time_statistics",
}
"""The classical encoder of each modality."""

COLUMNS = (
    "dataset",
    "fold",
    "arm",
    "backend",
    "version",
    "n_train",
    "n_test",
    "crps",
    "nll",
    "rmse",
    "fit_seconds",
    "score_seconds",
    "n_features",
    "error",
)
"""The columns of the results, in order."""


def _load(
    name: str, data_home: str | None, images: bool
) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    """A whole dataset: the columns a model can read, its target, modalities.

    Args:
        name: The dataset.
        data_home: The ``data_home`` argument of ``load_dataset``.
        images: Whether to load MulTaBench's images.

    Returns:
        The item columns and the columns of numbers, categories and dates
        (as days), the target, and the modalities of the item columns.
    """
    options = {"images": images} if name.startswith("multabench_") else {}
    dataset = datasets.load_dataset(name, data_home=data_home, **options)
    X = dataset.X
    keep = [
        column
        for column in X
        if column in dataset.modalities
        or isinstance(X[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_numeric_dtype(X[column])
        or pd.api.types.is_datetime64_any_dtype(X[column])
    ]
    return _common.days(X[keep]), dataset.y, dataset.modalities


def _model(
    arm: str, modalities: dict[str, str], args: argparse.Namespace
) -> Any:
    """The model an arm fits: the backend alone, or behind a Tabularizer."""
    params = {"random_state": 0}
    if args.version:
        params["version"] = args.version
    model = lazy.LazyModel(args.backend, **params)
    if arm == "tabular":
        return model
    columns = (
        {column: CLASSICAL[kind] for column, kind in modalities.items()}
        if arm == "classical"
        else modalities
    )
    tabularizer = lazy.Tabularizer(columns, n_columns=args.n_columns)
    return lazy.make_pipeline(tabularizer, model)


def _run(
    arm: str,
    train: tuple[pd.DataFrame, np.ndarray],
    test: tuple[pd.DataFrame, np.ndarray],
    modalities: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fits one arm on a fold's training rows and scores its test rows."""
    (X_train, y_train), (X_test, y_test) = train, test
    if arm == "tabular":
        X_train = X_train.drop(columns=list(modalities))
        X_test = X_test.drop(columns=list(modalities))
    model = _model(arm, modalities, args)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fitted = time.perf_counter()
    scores = _common.score(model, X_test, y_test)
    estimator = getattr(model, "_final_estimator", model)
    return {
        **scores,
        "fit_seconds": fitted - start,
        "score_seconds": time.perf_counter() - fitted,
        "n_features": estimator.n_features_in_,
    }


def _write(row: dict[str, Any], out: pathlib.Path) -> None:
    """Prints a row and appends it to ``out``, under :data:`COLUMNS`."""
    print(row, flush=True)
    pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
        out, mode="a", header=not out.exists(), index=False
    )


def main() -> None:
    """Runs every dataset, fold and arm, appending a row to ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--backend", default="tabpfn")
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--folds", type=int, nargs="+", choices=range(6), default=[0]
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--n-columns", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--data-home", default=None)
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()
    print(_common.setup(args.backend), flush=True)
    for name in args.datasets:
        base = {"dataset": name, "backend": args.backend}
        base["version"] = args.version
        try:
            X, y, modalities = _load(name, args.data_home, not args.no_images)
        except Exception as error:  # noqa: BLE001 - a benchmark records a failure and goes on.
            traceback.print_exc()
            _write({**base, "error": _message(error)}, args.out)
            continue
        for fold in args.folds:
            train_rows, test_rows = _multabench.split(len(y), fold, "10k")
            train = _common.sample(
                X.iloc[train_rows], y[train_rows], args.n_train, 0
            )
            test = _common.sample(
                X.iloc[test_rows], y[test_rows], args.n_test, 1
            )
            for arm in args.arms:
                row = {**base, "fold": fold, "arm": arm}
                row |= {"n_train": len(train[1]), "n_test": len(test[1])}
                try:
                    row |= _run(arm, train, test, modalities, args)
                except Exception as error:  # noqa: BLE001 - a benchmark records a failure and goes on.
                    traceback.print_exc()
                    row["error"] = _message(error)
                _write(row, args.out)


def _message(error: Exception) -> str:
    """An error as one short line for the results."""
    return f"{type(error).__name__}: {error}"[:300]


if __name__ == "__main__":
    main()
