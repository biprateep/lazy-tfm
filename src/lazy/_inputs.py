# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Coercion of the feature tables and targets the estimators accept.

Every estimator takes its features in whatever tabular form the user has,
following scikit-learn's conventions:

* a 2-D NumPy array, or a nested list;
* a NumPy structured array or record array, one field per column;
* a pandas DataFrame;
* an astropy ``Table`` or ``QTable`` (read by duck typing, so astropy is not
  a dependency): quantities give their values;
* any other object with a ``to_pandas()`` method (polars, for example).

They all become one float64 DataFrame. Anything with a ``mask`` -- a NumPy
masked array, an astropy ``MaskedColumn``, an astropy ``Masked`` array or
quantity (a masked ``QTable`` column) -- has its masked entries read as NaN,
in features and targets alike, never as the values underneath. Missing
values are NaN and pass through untouched, for each model to handle in its
own way; infinities are rejected. Column names are kept when the input has
them (all strings), and then enforced at predict time, as
``feature_names_in_`` is in scikit-learn. Two lenient readings are
deprecated, each with a ``FutureWarning``, and become errors in 0.7.0:
an array or list of strings that hold numbers, and column names that mix
strings with other types.

Typical usage example:

  frame, named = _inputs.to_frame(table)
  mask = _inputs.categorical_mask("from_dtype", frame, named)
  features = _inputs.numeric_frame(
      frame, named, _inputs.fit_categories(frame, mask)
  )
  target = _inputs.as_target(table["z"])
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from sklearn import exceptions
from sklearn.utils import validation

from lazy import _typing
from lazy import _warnings

__all__ = [
    "as_target",
    "categorical_mask",
    "check_target",
    "fit_categories",
    "numeric_frame",
    "to_frame",
]

_COMPLEX = "Complex data not supported"
_NUMERIC_STRINGS = (
    "X is an array or list of strings that hold numbers. It is converted to "
    "float64 now. From 0.7.0 this raises ValueError, as "
    "scikit-learn does; convert it first, for example with X.astype(float)."
)
_MIXED_NAMES = (
    "X has column names of both str and non-str types: {types}. They are "
    "read as no feature names now. From 0.7.0 this raises TypeError, as "
    "scikit-learn does; make every name a string."
)
_COLUMN_VECTOR = (
    "A column-vector y was passed when a 1d array was expected. Please "
    "change the shape of y to (n_samples, ), for example using ravel()."
)


def to_frame(features: Any) -> tuple[pd.DataFrame, bool]:
    """Reads any supported feature table as a DataFrame of its own types.

    Args:
        features: The features, shape (n_samples, n_features), in any of the
            forms the module docstring lists.

    Returns:
        A tuple (frame, named) -- the features with a fresh index and their
        columns' types, and whether the column names came from the input.

    Raises:
        TypeError: If the features are a sparse matrix or array.
        ValueError: If the table is not two-dimensional, has no columns, or
            has a complex column.
    """
    if sparse.issparse(features):
        raise TypeError(
            "sparse X is not supported; pass dense features, e.g. with "
            "OneHotEncoder(sparse_output=False) or "
            "ColumnTransformer(sparse_threshold=0)"
        )
    frame, named = _to_frame(features)
    if frame.shape[1] == 0:
        raise ValueError(
            f"Found array with 0 feature(s) (shape={frame.shape}) while a "
            "minimum of 1 is required."
        )
    if frame.dtypes.map(pd.api.types.is_complex_dtype).any():
        raise ValueError(_COMPLEX)
    return frame, named


def categorical_mask(
    categorical_features: Any, frame: pd.DataFrame, named: bool
) -> npt.NDArray[np.bool_]:
    """Which columns are categorical, as ``categorical_features`` says.

    Args:
        categorical_features: ``"from_dtype"``, for the columns of pandas'
            ``category`` type that are not ordered; ``None``, for none; or
            the columns as names
            (of a table with names), as integer positions, or as a boolean
            mask with one entry per column.
        frame: The features, as :func:`to_frame` returns them.
        named: Whether the column names came from the input.

    Returns:
        One boolean per column, shape (n_features,).

    Raises:
        ValueError: If ``categorical_features`` names a column that the
            table does not have, has a position out of range, or is a mask
            of the wrong length.
        TypeError: If ``categorical_features`` is of none of these forms.
    """
    n_columns = frame.shape[1]
    if categorical_features is None:
        return np.zeros(n_columns, dtype=bool)
    if isinstance(categorical_features, str):
        if categorical_features != "from_dtype":
            raise ValueError(
                "categorical_features must be 'from_dtype', None, or column"
                f" names, positions or a mask; got {categorical_features!r}"
            )
        return np.array(
            [
                isinstance(t, pd.CategoricalDtype) and not t.ordered
                for t in frame.dtypes
            ]
        )
    spec = np.asarray(categorical_features)
    if spec.ndim != 1:
        raise TypeError(
            "categorical_features must be 'from_dtype', None, or a list of"
            " column names, positions or booleans; got"
            f" {categorical_features!r}"
        )
    mask = np.zeros(n_columns, dtype=bool)
    if spec.size == 0:
        return mask
    if spec.dtype == bool:
        if spec.size != n_columns:
            raise ValueError(
                f"categorical_features is a mask of {spec.size} entries, but"
                f" X has {n_columns} columns"
            )
        return spec.copy()
    if spec.dtype.kind in "iu":
        if spec.size and (spec.min() < -n_columns or spec.max() >= n_columns):
            raise ValueError(
                f"categorical_features has positions {spec.tolist()}, but X"
                f" has {n_columns} columns"
            )
        mask[spec] = True
        return mask
    if spec.dtype.kind in "UO" and named:
        missing = [str(name) for name in spec if name not in frame.columns]
        if missing:
            raise ValueError(
                f"categorical_features names {missing}, which are not"
                " columns of X"
            )
        mask[frame.columns.get_indexer(pd.Index(spec))] = True
        return mask
    raise TypeError(
        "categorical_features must be 'from_dtype', None, or column names"
        " (of X with column names), positions or a boolean mask; got"
        f" {categorical_features!r}"
    )


def fit_categories(
    frame: pd.DataFrame, mask: npt.NDArray[np.bool_]
) -> list[pd.Index | None]:
    """The categories of each column read as codes, in the order of its codes.

    A column is read as codes when it is categorical, or when it is of
    pandas' ordered ``category`` type, whose codes are then numbers in the
    order of its categories. A column of pandas' ``category`` type keeps the
    categories it holds, in the order of its type; any other categorical
    column takes its distinct values that are not missing, sorted. Codes are
    therefore 0 to k - 1 for the k categories present, and encoding the
    codes again changes nothing.

    Args:
        frame: The features, as :func:`to_frame` returns them.
        mask: Which columns are categorical, shape (n_features,).

    Returns:
        For each column, its categories, or ``None`` if it is read as it is.

    Raises:
        ValueError: If a categorical column has no value that is not
            missing.
    """
    categories: list[pd.Index | None] = []
    for position, is_categorical in enumerate(mask):
        column = frame.iloc[:, position]
        ordered = getattr(column.dtype, "ordered", False) is True
        if not (is_categorical or ordered):
            categories.append(None)
            continue
        if isinstance(column.dtype, pd.CategoricalDtype):
            found = column.cat.remove_unused_categories().cat.categories
        else:
            try:
                found = pd.Index(column.dropna().unique()).sort_values()
            except TypeError:
                raise ValueError(
                    f"categorical column {frame.columns[position]!r} holds"
                    " values that cannot be ordered, such as text and"
                    " numbers; give it one type, or pandas' category type"
                ) from None
        if found.empty:
            raise ValueError(
                f"categorical column {frame.columns[position]!r} has no value"
                " that is not missing"
            )
        categories.append(found)
    return categories


def numeric_frame(
    frame: pd.DataFrame, named: bool, categories: list[pd.Index | None]
) -> pd.DataFrame:
    """The features as float64, each categorical column as its codes.

    A categorical column holds the position of each value in its
    categories, 0 to n_categories - 1, and NaN for a missing value or one
    that is not among the categories.

    Args:
        frame: The features, as :func:`to_frame` returns them.
        named: Whether the column names came from the input.
        categories: For each column, its categories, or ``None`` if it is
            not categorical (:func:`fit_categories`).

    Returns:
        A float64 DataFrame with a fresh index, named by the input's names
        or ``x0, x1, ...``.

    Raises:
        ValueError: If a column that is not categorical is not numeric, or
            holds an infinity.
    """
    values = None
    if all(found is None for found in categories):
        try:  # One conversion, column-major, as pandas holds a frame.
            values = frame.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError):
            pass  # The loop below names the columns that fail.
    if values is None:
        values = np.empty(frame.shape, dtype=np.float64, order="F")
        problems = []
        for position, found in enumerate(categories):
            column = frame.iloc[:, position]
            if found is not None:
                codes = found.get_indexer(pd.Index(column))
                values[:, position] = np.where(codes < 0, np.nan, codes)
                continue
            try:
                values[:, position] = column.to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
            except (TypeError, ValueError):
                problems.append(
                    f"'{frame.columns[position]}' ({_kind(column)})"
                )
        if problems:
            raise ValueError(
                "every feature column must be numeric or categorical; not"
                f" numeric: {', '.join(problems)}. Mark a column of"
                " categories with .astype('category') or"
                " categorical_features; encode a column of text, images or"
                " time series with lazy.Tabularizer; convert a date to"
                " numbers, such as days since a reference date"
            )
    infinite = np.isinf(values).any(axis=0)
    if infinite.any():
        names = [str(name) for name in frame.columns[infinite]]
        raise ValueError(
            "X holds infinite values (mark missing values with NaN) in "
            f"columns {names}"
        )
    names = list(frame.columns) if named else _default_names(values.shape[1])
    return pd.DataFrame(values, columns=names)


def as_target(target: Any) -> _typing.FloatArray:
    """Converts target values to a flat float64 array; masked become NaN.

    A column vector, shape (n_samples, 1), is flattened with a
    ``DataConversionWarning``, as scikit-learn's regressors do.

    Args:
        target: Target values, shape (n_samples,): an array, list, pandas
            Series, astropy Column or Quantity, or masked array (NumPy's or
            astropy's).

    Returns:
        The values, shape (n_samples,).

    Raises:
        ValueError: If the target is None, a scalar, complex, or has more
            than one column.

    Warns:
        sklearn.exceptions.DataConversionWarning: If the target is a column
            vector.
    """
    if target is None:
        raise ValueError(
            "This estimator requires y to be passed, but the target y is None"
        )
    data, mask = _unmask(target)
    _reject_complex(data)
    values = np.asarray(data, dtype=np.float64)
    if mask is not None:
        values = np.where(mask, np.nan, values)
    if values.ndim == 2 and values.shape[1] == 1:
        _warnings.warn(_COLUMN_VECTOR, exceptions.DataConversionWarning)
    return validation.column_or_1d(values)


def check_target(
    target: Any, n_rows: int, names: tuple[str, str] = ("X", "y")
) -> _typing.FloatArray:
    """Validates target values against the feature rows they label.

    Args:
        target: Target values, shape (n_rows,), in any form
            :func:`as_target` accepts.
        n_rows: The number of feature rows.
        names: What the errors call the features and the target.

    Returns:
        The values as a float64 array, shape (n_rows,).

    Raises:
        ValueError: If the number of values is not ``n_rows``, or a value
            is not finite.
    """
    features, name = names
    values = as_target(target)
    if values.size != n_rows:
        raise ValueError(
            f"{features} has {n_rows} rows but {name} has {values.size} values"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def check_levels(levels: npt.ArrayLike, *, name: str) -> _typing.FloatArray:
    """Validates cumulative probabilities, as a 1-D array.

    Args:
        levels: The probabilities, a scalar or shape (k,).
        name: The argument's name, for the errors.

    Returns:
        The probabilities as float64, shape (k,); ``levels`` is not modified.

    Raises:
        ValueError: If they are not numbers, are not a scalar or 1-D, or one
            is not finite or lies outside [0, 1].
    """
    message = f"{name} must be a scalar or 1-D array within [0, 1]"
    try:
        out = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if out.ndim != 1 or not np.all((out >= 0.0) & (out <= 1.0)):
        raise ValueError(message)
    return out


def check_coverage(coverage: float, *, name: str) -> float:
    """Validates the probability inside a central interval.

    Args:
        coverage: The probability, in (0, 1).
        name: The argument's name, for the errors.

    Returns:
        ``coverage``, unchanged.

    Raises:
        ValueError: If it does not lie in (0, 1).
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"{name} must lie in (0, 1): {name}={coverage!r}")
    return coverage


def check_rows(rows: npt.ArrayLike, n: int, *, name: str) -> _typing.IntArray:
    """Validates positions of rows of a table, flattened.

    Args:
        rows: The positions, of any shape; none at all is allowed.
        n: The number of rows of the table.
        name: The argument's name, for the errors.

    Returns:
        The positions as int64, flattened, in their order.

    Raises:
        ValueError: If they are not integers, are out of ``[0, n)``, or
            repeat.
    """
    rows = np.asarray(rows).ravel()
    if not rows.size:
        return rows.astype(np.int64)
    if not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(
            f"{name} must be integer row positions, not {rows.dtype}"
        )
    if rows.min() < 0 or rows.max() >= n:
        raise ValueError(
            f"{name} must be row positions in [0, {n:,}); got "
            f"{rows.min():,} to {rows.max():,}"
        )
    if np.unique(rows).size != rows.size:
        raise ValueError(f"{name} repeats rows; give each row once")
    return rows.astype(np.int64)


def _to_frame(features: Any) -> tuple[pd.DataFrame, bool]:
    """Dispatches on the input's form; see :func:`to_frame`."""
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
    array, mask = _unmask(features)
    if array.dtype.kind in "SU":  # scikit-learn refuses these; see above.
        try:
            array = array.astype(np.float64)
        except ValueError:
            pass  # numeric_frame names the columns that fail.
        else:
            _warnings.warn(_NUMERIC_STRINGS, FutureWarning)
    if mask is not None:
        _reject_complex(array)
        array = _masked(array, mask)
    if array.ndim == 1:
        raise ValueError(
            f"X must be 2D, got a 1D array of shape {array.shape}. Reshape "
            "your data either using array.reshape(-1, 1) if your data has a "
            "single feature or array.reshape(1, -1) if it contains a single "
            "sample."
        )
    if array.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {array.shape}")
    return pd.DataFrame(array), False


def _reject_complex(array: Any) -> None:
    """Raises before a cast to float would drop an imaginary part."""
    if np.iscomplexobj(array):
        raise ValueError(_COMPLEX)


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
    values, mask = _unmask(column)
    return values if mask is None else _masked(values, mask)


def _masked(
    values: npt.NDArray[Any], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[Any]:
    """Values with their masked entries missing: NaN, or None for objects.

    Numbers become float64 and anything else, such as text, objects.
    """
    if values.dtype.kind in "biuf":
        filled = values.astype(np.float64)
        filled[mask] = np.nan
        return filled
    filled = values.astype(object)
    filled[mask] = None
    return filled


def _unmask(
    values: Any,
) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_] | None]:
    """Splits values into their data, units dropped, and their mask.

    Anything with an array-valued ``mask`` attribute counts as masked: a
    NumPy masked array, an astropy ``MaskedColumn``, and an astropy
    ``Masked`` array or quantity, whose ``value`` is itself masked.

    Args:
        values: An array-like of any shape.

    Returns:
        A tuple (data, mask) -- the values underneath as an array, and the
        mask as a boolean array of the same shape, or ``None`` if the
        values carry no mask.
    """
    mask = getattr(values, "mask", None)
    if callable(mask):  # pandas' Series.mask, a method.
        mask = None
    data = getattr(values, "unmasked", values)  # astropy Masked
    data = getattr(data, "value", data)  # astropy Quantity
    if isinstance(data, np.ma.MaskedArray):
        data = data.data
    data = np.asarray(data)
    if mask is None:
        return data, None
    return data, np.broadcast_to(np.asarray(mask, dtype=bool), data.shape)


def _kind(column: pd.Series) -> str:
    """What a column that is not numeric holds, for an error message."""
    if isinstance(column.dtype, pd.CategoricalDtype):
        return "categories, not marked as categorical"
    if pd.api.types.is_datetime64_any_dtype(column) or isinstance(
        column.dtype, pd.PeriodDtype
    ):
        return "dates"
    return "text or objects"


def _all_strings(columns: pd.Index) -> bool:
    """Whether every column name is a string, as scikit-learn requires.

    Warns:
        FutureWarning: If the names mix strings with other types.
    """
    kinds = {isinstance(name, str) for name in columns}
    if kinds == {True, False}:
        types = sorted({type(name).__name__ for name in columns})
        _warnings.warn(_MIXED_NAMES.format(types=types), FutureWarning)
    return kinds == {True}


def _default_names(n_columns: int) -> list[str]:
    """The internal column names of an unnamed table: x0, x1, ..."""
    return [f"x{i}" for i in range(n_columns)]
