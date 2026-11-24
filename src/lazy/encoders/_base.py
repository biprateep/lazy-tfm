# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The contract every encoder meets.

An *encoder* turns one item -- an image, a text, a time series -- into a
vector of fixed length, its *embedding*, so that a column of items becomes
numeric columns a tabular foundation model can take as features.
:class:`BaseEncoder` is a scikit-learn transformer that holds what encoders
share: reading the column, batching, missing items, checking each answer,
the embedding cache and the names of the output columns. A subclass
implements two small methods.

Typical usage example:

  class Length(BaseEncoder):
      modality = "text"

      def __init__(self, *, batch_size=64, cache=False):
          self.batch_size = batch_size
          self.cache = cache

      def _width(self):
          return 1

      def _encode(self, items):
          return np.array([[len(item)] for item in items], dtype=float)
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
import json
import os
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base as sklearn_base
from sklearn.utils import validation

import lazy
from lazy import _typing
from lazy.encoders import _cache
from lazy.models import _ensemble

__all__ = ["BASE_DEFAULTS", "BASE_PARAMS", "MODALITIES", "BaseEncoder"]

MODALITIES: tuple[str, ...] = ("image", "text", "time")
"""The kinds of item an encoder can declare that it reads."""

BASE_PARAMS: tuple[str, ...] = ("batch_size", "cache")
"""The parameters every encoder's constructor must take, keyword-only."""

BASE_DEFAULTS: dict[str, Any] = {"batch_size": 64}
"""The defaults that are the same on every encoder. ``cache`` is ``False`` on
an encoder that is cheap to run, and may be ``True`` on one that is not.
"""


class BaseEncoder(
    sklearn_base.TransformerMixin,
    sklearn_base.BaseEstimator,
    abc.ABC,
):
    """A transformer from one column of items to their embeddings.

    ``fit`` and ``transform`` take the items as a sequence, a pandas Series,
    a DataFrame, or an array whose first axis indexes the items: shape (n,)
    of objects, or (n, ...) of numbers, such as n time series of equal
    length. A DataFrame is one column that is not numeric, or numeric
    columns, which are read as the array of their values, one item per
    row. ``transform``
    returns float64 embeddings, shape (n, n_features_out). A missing item
    (a scalar that :func:`pandas.isna` knows, such as ``None``, ``NaN`` or
    ``pd.NA``) gets a row of ``NaN``, which the models read as missing
    features, and is never passed to a subclass.

    An encoder is frozen: an item's embedding does not depend on the items
    that the encoder was fitted on, and ``fit`` only validates.

    The contract a subclass meets is the following. Its constructor takes
    every name in ``BASE_PARAMS`` as a keyword, stores each verbatim, gives
    ``batch_size`` the default in ``BASE_DEFAULTS``, gives every parameter
    a default and takes no ``**kwargs``, so that ``cls()`` builds a working
    encoder (:func:`lazy.encoders.register` checks this). It declares
    ``modality`` and implements :meth:`_width` and :meth:`_encode`; it may
    name its columns with :meth:`_names`. Each
    row of an answer of :meth:`_encode` depends on its own item alone,
    whatever the batch holds, and an encoding changes neither the
    encoder's parameters nor its later answers. Every answer is checked to
    be finite, with one row per item and :meth:`_width` columns.

    With ``cache`` set, the embedding of each item is filed on disk under
    the content of the item (:meth:`_item_bytes`) and the identity of the
    encoder (:meth:`_cache_identity`), and an item that is filed is not
    encoded again. :func:`lazy.encoders.clear_embedding_cache` empties the
    default cache.

    Attributes:
        modality: What the encoder reads, one of
            :data:`~lazy.encoders.MODALITIES`.
        batch_size: The most items one call of :meth:`_encode` gets.
        cache: ``False`` for no cache, ``True`` for the default one, under
            :func:`lazy.datasets.data_home`, or the directory to use.
    """

    modality: ClassVar[str] = ""

    # The parameters that do not change an embedding, left out of the
    # identity of the cache.
    _UNKEYED: ClassVar[tuple[str, ...]] = BASE_PARAMS

    # Set by each encoder's __init__; declared for the type checker only.
    batch_size: int
    cache: bool | str | os.PathLike[str]

    # -- the per-encoder interface -------------------------------------------

    @abc.abstractmethod
    def _width(self) -> int:
        """The length of an embedding.

        ``fit`` calls it, so it is also where an encoder checks its own
        parameters.

        Raises:
            ValueError: If a parameter of the encoder is not valid.
        """

    def _names(self) -> Sequence[str] | None:
        """A name for each column of an embedding, or ``None`` for numbers.

        Returns:
            :meth:`_width` names, which :meth:`get_feature_names_out`
            prefixes with the modality and the name of the class; ``None``
            for the columns to be numbered.
        """
        return None

    @abc.abstractmethod
    def _encode(self, items: list[Any]) -> _typing.FloatArray:
        """The embeddings of a batch of items.

        Args:
            items: At most ``batch_size`` items, none of them missing.

        Returns:
            Finite embeddings, shape (len(items), :meth:`_width`).
        """

    def _length(self, item: Any) -> int:
        """How long an item is; items of similar length share a batch.

        An encoder that pads each batch to its longest item returns the
        length it pads, such as the number of characters or of values, so
        that ``transform`` batches items in order of length. By default
        every item has the length 0, and the items keep their order.

        Args:
            item: One item, not missing.

        Returns:
            Its length, in any unit that grows with its padded size.
        """
        del item  # Unused: by default, the order of the column.
        return 0

    def _item_bytes(self, item: Any) -> bytes:
        """The content of an item, which keys its embedding in the cache.

        Text, bytes and plain numeric arrays (or lists and tuples of
        numbers) are covered here. An encoder of
        other items returns the bytes that set its embedding. One that
        reads files overrides this for a path given as text too, which is
        keyed here as the text and not as the content of the file.

        Args:
            item: One item, not missing.

        Returns:
            Bytes that two items share only if their embeddings are equal.

        Raises:
            TypeError: If the item is of no kind covered here.
        """
        if isinstance(item, str):
            return b"str:" + item.encode()
        if isinstance(item, bytes):
            return b"bytes:" + item
        plain = isinstance(item, (np.ndarray, list, tuple)) and not isinstance(
            item, np.ma.MaskedArray
        )
        array = np.asarray(item) if plain else np.empty(0, dtype=object)
        if array.dtype.kind not in "biufc":
            raise TypeError(
                f"{type(self).__name__} cannot cache an item of type"
                f" {type(item).__name__}: implement _item_bytes"
            )
        header = f"array:{array.dtype.str}:{array.shape}:".encode()
        return header + np.ascontiguousarray(array).tobytes()

    def _cache_identity(self) -> str:
        """What sets the encoder's output, which names its cache folder.

        It holds :meth:`_identity_head`, the class and every parameter but
        those in :attr:`_UNKEYED`.

        Returns:
            Text that two encoders share only if their outputs are equal.

        Raises:
            TypeError: If a parameter is not a number, text, ``None``, or a
                list, tuple or dict of those, or the class is defined in a
                script (``__main__``) or inside a function.
        """
        cls = type(self)
        if cls.__module__ == "__main__" or "<locals>" in cls.__qualname__:
            raise TypeError(
                f"{cls.__name__} is defined in a script or a function, so"
                " its name does not tell it from another class of that"
                " name: override _identity_head to use the cache"
            )
        params = {
            name: value
            for name, value in self.get_params(deep=False).items()
            if name not in self._UNKEYED
        }
        return json.dumps(
            [*self._identity_head(), cls.__module__, cls.__qualname__, params],
            sort_keys=True,
            default=self._unkeyed_parameter,
        )

    def _identity_head(self) -> list[str]:
        """What sets the output beyond the class and its parameters.

        By default the version of lazy, so that a release discards the
        embeddings of the last one. An encoder whose output depends on
        more, such as the revision of its weights, overrides it.

        Returns:
            Text that leads the identity.
        """
        return [lazy.__version__]

    def _unkeyed_parameter(self, value: object) -> Any:
        """A NumPy scalar as Python's; refuses any other parameter.

        Returns:
            The Python number or bool that a NumPy scalar holds.

        Raises:
            TypeError: If ``value`` is not a number, text, ``None``, a
                list or a dict of those, so its text may not tell two
                values apart.
        """
        if isinstance(value, np.generic) and value.dtype.kind in "biuf":
            return value.item()
        raise TypeError(
            f"{type(self).__name__} cannot cache with a parameter of type"
            f" {type(value).__name__}: override _identity_head, and name it"
            " in _UNKEYED"
        )

    # -- scikit-learn's interface --------------------------------------------

    def fit(self, X: Any, y: Any = None) -> BaseEncoder:  # noqa: GS030 - scikit-learn's X, y.
        """Validates the parameters and the column; learns nothing.

        Args:
            X: One column of items, n of them; see the class docstring.
            y: Ignored; scikit-learn passes it.

        Returns:
            The encoder itself.

        Raises:
            TypeError: If ``X`` is not a column of items.
            ValueError: If ``batch_size`` is not a positive integer,
                ``cache`` is not a bool or a path, or ``X`` is a table that
                is not one column of items.
        """
        del y  # Unused: an encoder is fitted without the target.
        positive_integer("batch_size", self.batch_size)
        if not isinstance(self.cache, (bool, str, os.PathLike)) or (
            isinstance(self.cache, str) and not self.cache
        ):
            raise ValueError(
                "cache must be False, True or the path of a directory:"
                f" {self.cache=}"
            )
        _as_items(X)
        self._n_features_out = int(self._width())
        return self

    def transform(self, X: Any) -> _typing.FloatArray:  # noqa: GS030 - scikit-learn's X, y.
        """The embedding of each item.

        Items are encoded ``batch_size`` at a time, in order of
        :meth:`_length`. With ``cache`` set, each embedding is read from, or
        written to, its own file.

        Args:
            X: One column of items, n of them; see the class docstring.

        Returns:
            The embeddings, shape (n, n_features_out), with a row of NaN
            for each missing item.

        Raises:
            sklearn.exceptions.NotFittedError: If the encoder is not fitted.
            TypeError: If ``X`` is not a column of items.
            ValueError: If ``X`` has more than one column, or
                :meth:`_encode` breaks its contract.
        """
        validation.check_is_fitted(self, "_n_features_out")
        items = _as_items(X)
        width = self._n_features_out
        embeddings = np.full((len(items), width), np.nan)
        todo = np.array(
            [i for i, item in enumerate(items) if not _is_missing(item)],
            dtype=np.intp,
        )
        keys = np.empty(len(todo), dtype=object)
        if self.cache is not False:
            store = _cache.Store(self.cache, self._cache_identity(), width)
            keys[:] = [_cache.digest(self._item_bytes(items[i])) for i in todo]
            embeddings[todo], found = store.load(keys)
            todo, keys = todo[~found], keys[~found]
        order = np.argsort(
            [self._length(items[i]) for i in todo], kind="stable"
        )
        todo, keys = todo[order], keys[order]
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start : start + self.batch_size]
            rows = self._checked(
                self._encode([items[i] for i in batch]), len(batch)
            )
            embeddings[batch] = rows
            if self.cache is not False:
                store.save(keys[start : start + self.batch_size], rows)
        return embeddings

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.object_]:
        """The names of the output columns.

        Args:
            input_features: Ignored; scikit-learn passes it.

        Returns:
            ``<modality>_<class>_`` before each column's own name, as
            ``time_statistics_mean``, or before its number, as
            ``image_pixels_0``, for an encoder that names no columns. The
            class is in lower case. Shape (n_features_out,).

        Raises:
            sklearn.exceptions.NotFittedError: If the encoder is not fitted.
            ValueError: If :meth:`_names` does not give one name per
                column.
        """
        del input_features  # Unused: the names do not depend on the input.
        validation.check_is_fitted(self, "_n_features_out")
        names = self._names()
        if names is None:
            names = [str(i) for i in range(self._n_features_out)]
        if len(names) != self._n_features_out:
            raise ValueError(
                f"{type(self).__name__}._names gave {len(names)} names for"
                f" {self._n_features_out} columns"
            )
        prefix = f"{self.modality}_{type(self).__name__.lower()}"
        return np.array([f"{prefix}_{name}" for name in names], dtype=object)

    def _checked(self, rows: Any, n_items: int) -> _typing.FloatArray:
        """An answer of :meth:`_encode`, checked against its contract.

        Args:
            rows: What :meth:`_encode` returned.
            n_items: How many items it was given.

        Returns:
            The embeddings as float64, shape (n_items, n_features_out).

        Raises:
            ValueError: If the shape is wrong or a value is not finite.
        """
        rows = np.asarray(rows, dtype=float)
        expected = (n_items, self._n_features_out)
        if rows.shape != expected:
            raise ValueError(
                f"{type(self).__name__}._encode returned shape {rows.shape}"
                f" for {n_items} items; expected {expected}"
            )
        if not np.isfinite(rows).all():
            raise ValueError(
                f"{type(self).__name__}._encode returned values that are"
                " not finite"
            )
        return rows


def _as_items(X: Any) -> Sequence[Any] | npt.NDArray[Any]:  # noqa: GS030 - scikit-learn's X, y.
    """One column of items, as a sequence that indexes them.

    Args:
        X: A sequence, a pandas Series, a DataFrame, or an array whose
            first axis indexes the items.

    Returns:
        The items; ``len`` counts them and an integer indexes one.

    Raises:
        TypeError: If ``X`` is one string, or has no length.
        ValueError: If ``X`` is a table of several columns that are not
            all numeric.
    """
    if isinstance(X, pd.DataFrame):
        numeric = [pd.api.types.is_numeric_dtype(t) for t in X.dtypes]
        if X.shape[1] == 1 and not numeric[0]:
            return X.iloc[:, 0].tolist()
        if not numeric or not all(numeric):
            raise ValueError(
                "X must be one column of items, or numeric columns that"
                f" hold one item per row; got the columns {list(X.columns)}"
            )
        X = X.to_numpy()
    if isinstance(X, pd.Series):
        return X.tolist()
    if isinstance(X, np.ndarray):
        if X.ndim == 0:
            raise TypeError("X must be a column of items, not a scalar")
        if X.dtype.kind in "OUS" and X.ndim > 1:
            # scikit-learn hands a column of objects over as shape (n, 1).
            if X.shape[1:] != (1,):
                raise ValueError(
                    "X must be one column of items; got an array of"
                    f" objects or text of shape {X.shape}"
                )
            return X[:, 0]
        return X
    if isinstance(X, (str, bytes)) or not hasattr(X, "__len__"):
        raise TypeError(
            "X must be a column of items (a sequence, a Series, a"
            f" DataFrame or an array), not {type(X).__name__}"
        )
    return list(X)


def positive_integer(name: str, value: object) -> int:
    """A parameter that must be a positive integer, checked.

    Args:
        name: The name of the parameter, for the message.
        value: Its value.

    Returns:
        The value as an ``int``.

    Raises:
        ValueError: If it is not a positive integer.
    """
    if not _ensemble.is_integer(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer: {name}={value!r}")
    return int(value)


def _is_missing(item: Any) -> bool:
    """Whether an item is a scalar that pandas reads as missing."""
    return pd.api.types.is_scalar(item) and bool(pd.isna(item))
