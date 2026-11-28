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

    An encoder whose embedding is differentiable in its items declares
    ``gradient_kind = "differentiable"`` (:mod:`lazy._steps`), and
    implements :meth:`_array`, one item as an array, and ``forward``, so
    that a gradient can be taken in the items themselves. A subclass that
    changes the embedding (:meth:`_encode`, or how an item is read) must
    give its own ``forward``, or set ``gradient_kind = "barrier"``: it
    inherits both otherwise, and they would describe the parent.
    ``forward(items)`` takes what :meth:`_stacked` returns, as a float
    torch tensor of shape (n, ...), and gives what :meth:`transform` gives
    for the same items, to float rounding: a tensor of shape (n,
    n_features_out), of the type and on the device of ``items``, that
    keeps their gradient. An encoder that cannot, such as one of text,
    may say why in ``barrier_reason``.

    An encoder whose items differ in length, or may lack some of their
    elements, sets ``pads_items`` to the side on which its items are
    aligned, ``"left"`` or ``"right"``. Its :meth:`_array` then gives
    ``NaN`` for a missing element, and :meth:`_stacked` pads each item's
    array with ``NaN`` on that side to the longest along the first axis.
    Its ``forward`` reads a ``NaN`` as a missing element: it gives what
    :meth:`transform` gives for the item without the padding, and the
    gradient in a missing element is 0, with no ``NaN`` on the autograd
    graph.

    An attribution measures each element of an item against a baseline,
    by default the mean of the context's items. An encoder that knows the
    item of no content, such as a text of padding tokens, gives the
    baselines of stacked items in :meth:`_baseline`. An encoder whose
    element is a vector, such as a token's embedding, sets
    ``element_ndim`` to the number of trailing axes that hold it, and an
    element's attribution is the sum over them.

    With ``cache`` set, the embedding of each item is filed on disk under
    the content of the item (:meth:`_item_bytes`) and the identity of the
    encoder (:meth:`_cache_identity`), and an item that is filed is not
    encoded again. :func:`lazy.encoders.clear_embedding_cache` empties the
    default cache.

    Attributes:
        modality: What the encoder reads, one of
            :data:`~lazy.encoders.MODALITIES`.
        pads_items: ``""`` if :meth:`_stacked` stacks only items of one
            shape with every element; else the side, ``"left"`` or
            ``"right"``, on which it pads them with ``NaN``.
        element_ndim: The number of trailing axes of an item's array that
            hold one element, such as the vector of a token's embedding; 0
            when each number is an element.
        batch_size: The most items one call of :meth:`_encode` gets.
        cache: ``False`` for no cache, ``True`` for the default one, under
            :func:`lazy.datasets.data_home`, or the directory to use.
    """

    modality: ClassVar[str] = ""
    gradient_kind: ClassVar[str] = "barrier"
    barrier_reason: ClassVar[str] = ""
    pads_items: ClassVar[str] = ""
    element_ndim: ClassVar[int] = 0

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

    def _read_item(self, item: Any) -> Any:
        """An item as a step that transforms items before this encoder takes it.

        By default the item itself. An encoder that reads files, or
        objects that are not arrays, returns the array it would encode, so
        that an augmentation gets it in the encoder's own reading.

        Args:
            item: One item, not missing.

        Returns:
            The item, or the array the encoder reads of it.
        """
        return item

    def _array(self, item: Any) -> _typing.FloatArray | None:
        """One item as the float array that :meth:`forward` takes.

        Args:
            item: One item, not missing.

        Returns:
            The array, or ``None`` if the encoder has no forward, or the
            item has none: a series with a gap, say, unless the encoder
            pads its items, whose array holds ``NaN`` for a missing
            element.
        """
        del item  # Unused: by default, an encoder has no forward.
        return None

    def _keeps_values(self, item: Any) -> bool:
        """Whether :meth:`_array` keeps the values of an item as it is read.

        An encoder that maps an item through a function of all its values,
        such as a stretch by its percentiles, says no for such an item. An
        augmentation's forward, which runs on the arrays, then does not
        give what the augmentation gives before the encoder, so a column
        of such items behind augmentations is a barrier.

        Args:
            item: One item, not missing, as the encoder takes it.

        Returns:
            True by default.
        """
        del item  # Unused: by default, the array holds the item's values.
        return True

    def _baseline(  # noqa: GS030 - scikit-learn's X.
        self, X: Any, stacked: _typing.FloatArray
    ) -> _typing.FloatArray | None:
        """The items of no content that stand in for stacked items.

        Args:
            X: One column of items, none missing.
            stacked: What :meth:`_stacked` gives for them.

        Returns:
            The baseline of each item for an attribution, of the shape and
            type of ``stacked``; ``None``, by default, for the mean of the
            context's items.
        """
        del X, stacked  # Unused: by default, the context's mean.
        return None

    def _stacked(self, X: Any) -> _typing.FloatArray | None:  # noqa: GS030 - scikit-learn's X.
        """The items of a column as one array, for :meth:`forward`.

        The caller chooses the rows: the array holds every item given, in
        the type of the items' arrays, so a column of large items is
        stacked in parts.

        Args:
            X: One column of items, n of them.

        Returns:
            The items' arrays (:meth:`_array`), shape (n, ...), or ``None``
            if an item is missing or has no array, or they do not all have
            one shape: with ``pads_items``, the arrays padded with ``NaN``
            to the longest, which then need one shape beyond the first
            axis. It stops reading at the first such item.
        """
        items = _as_items(X)
        if self.pads_items:
            return self._padded(items)
        stacked = None
        for index, item in enumerate(items):
            array = None if _is_missing(item) else self._array(item)
            if stacked is None and array is not None:
                stacked = np.empty((len(items), *array.shape), array.dtype)
            if (
                array is None
                or stacked is None
                or array.shape != stacked.shape[1:]
            ):
                return None
            stacked[index] = array
        return stacked

    def _padded(self, items: Any) -> _typing.FloatArray | None:
        """Items' arrays padded with ``NaN`` to the longest, and stacked.

        Args:
            items: The items, as :func:`_as_items` gives them.

        Returns:
            The arrays, aligned on the side ``pads_items`` names, shape (n,
            longest, ...); ``None`` as :meth:`_stacked` says.
        """
        arrays: list[_typing.FloatArray] = []
        for item in items:
            array = None if _is_missing(item) else self._array(item)
            if array is None or (
                arrays and array.shape[1:] != arrays[0].shape[1:]
            ):
                return None
            arrays.append(array)
        if not arrays:
            return None
        longest = max(len(array) for array in arrays)
        shape = (len(arrays), longest, *arrays[0].shape[1:])
        stacked = np.full(shape, np.nan, arrays[0].dtype)
        for row, array in zip(stacked, arrays, strict=True):
            if self.pads_items == "left":
                row[longest - len(array) :] = array
            else:
                row[: len(array)] = array
        return stacked

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
