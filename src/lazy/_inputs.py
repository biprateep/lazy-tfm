# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Coercion of the feature tables and targets the estimators accept.

Every estimator takes its features in whatever tabular form the user has,
following scikit-learn's conventions:

* a 2-D NumPy array, or a nested list;
* a NumPy structured array or record array, one field per column;
* a pandas DataFrame;
* an astropy ``Table`` or ``QTable`` (read by duck typing, so astropy is not
  a dependency): quantities give their values, masked entries become NaN;
* any other object with a ``to_pandas()`` method (polars, for example).

They all become one float64 DataFrame. Missing values are NaN and pass
through untouched, for each model to handle in its own way; infinities are
rejected. Column names are kept when the input has them (all strings), and
then enforced at predict time, as ``feature_names_in_`` is in scikit-learn.

Typical usage example:

  frame, named = _inputs.as_feature_frame(table)
  target = _inputs.as_target(table["z"])
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from lazy import _typing

__all__ = ["as_feature_frame", "as_target"]


def as_feature_frame(features: Any) -> tuple[pd.DataFrame, bool]:
    """Converts any supported feature table to a float64 DataFrame.

    Args:
        features: The features, shape (n_samples, n_features), in any of the
            forms the module docstring lists.

    Returns:
        A tuple (frame, named): the features as a float64 DataFrame with a
        fresh index, and whether its column names came from the input (they
        are ``x0, x1, ...`` otherwise).

    Raises:
        ValueError: If the table is not two-dimensional, has no columns, has
            a column that is not numeric, or holds an infinity.
    """
    frame, named = _to_frame(features)
    if frame.shape[1] == 0:
        raise ValueError("X has no feature columns")
    try:
        values = frame.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as error:
        bad = [
            str(name)
            for name in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[name])
        ]
        raise ValueError(
            f"every feature column must be numeric; not numeric: {bad}"
        ) from error
    infinite = np.isinf(values).any(axis=0)
    if infinite.any():
        columns = [str(name) for name in frame.columns[infinite]]
        raise ValueError(
            "X holds infinite values (mark missing values with NaN) in "
            f"columns {columns}"
        )
    columns = list(frame.columns) if named else _default_names(values.shape[1])
    return pd.DataFrame(values, columns=columns), named


def as_target(target: Any) -> _typing.FloatArray:
    """Converts target values to a flat float64 array; masked become NaN.

    Args:
        target: Target values, shape (n_samples,): an array, list, pandas
            Series, astropy Column or Quantity, or masked array.

    Returns:
        The values, shape (n_samples,).
    """
    values = getattr(target, "value", target)  # astropy Quantity
    if np.ma.isMaskedArray(values):
        return np.ma.filled(np.ma.asarray(values, dtype=np.float64), np.nan)
    return np.asarray(values, dtype=np.float64).ravel()


def _to_frame(features: Any) -> tuple[pd.DataFrame, bool]:
    """Dispatches on the input's form; see :func:`as_feature_frame`."""
    if isinstance(features, pd.DataFrame):
        frame = features.reset_index(drop=True)
        return frame, _all_strings(frame.columns)
    if hasattr(features, "colnames") and hasattr(features, "columns"):
        return _from_astropy(features), True
    if hasattr(features, "to_pandas"):
        frame = features.to_pandas().reset_index(drop=True)
        return frame, _all_strings(frame.columns)
    if isinstance(features, np.ndarray) and features.dtype.names is not None:
        return _from_structured(features), True
    array = features
    if np.ma.isMaskedArray(array):
        array = np.ma.filled(np.ma.asarray(array, dtype=np.float64), np.nan)
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {array.shape}")
    return pd.DataFrame(array), False


def _from_astropy(table: Any) -> pd.DataFrame:
    """An astropy Table's columns as floats, masked entries as NaN."""
    columns: dict[str, npt.NDArray[np.float64]] = {}
    for name in table.colnames:
        column = table[name]
        if getattr(column, "ndim", 1) != 1:
            raise ValueError(
                f"column {name!r} is multidimensional; give one column per "
                "feature"
            )
        columns[str(name)] = _column_values(column)
    return pd.DataFrame(columns)


def _from_structured(array: npt.NDArray[Any]) -> pd.DataFrame:
    """A structured or record array's fields as columns."""
    names = array.dtype.names or ()
    columns: dict[str, npt.NDArray[Any]] = {}
    for name in names:
        field = array[name]
        if field.ndim != 1:
            raise ValueError(
                f"field {name!r} is multidimensional; give one field per "
                "feature"
            )
        columns[str(name)] = _column_values(field)
    return pd.DataFrame(columns)


def _column_values(column: Any) -> npt.NDArray[Any]:
    """One column's values, units dropped and masked entries as NaN."""
    values = getattr(column, "value", column)  # astropy Quantity
    if np.ma.isMaskedArray(values) or hasattr(column, "mask"):
        masked = np.ma.asarray(values)
        if masked.dtype.kind in "biuf":
            return np.ma.filled(masked.astype(np.float64), np.nan)
        return np.asarray(masked)
    return np.asarray(values)


def _all_strings(columns: pd.Index) -> bool:
    """Whether every column name is a string, as scikit-learn requires."""
    return not columns.empty and all(isinstance(name, str) for name in columns)


def _default_names(n_columns: int) -> list[str]:
    """The internal column names of an unnamed table: x0, x1, ..."""
    return [f"x{i}" for i in range(n_columns)]
