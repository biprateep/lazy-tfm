# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Augmentations: steps that transform the inputs, with no averaging mode.

An augmentation is a scikit-learn transformer that learns nothing. It
applies one transform to a table of numbers, or to each item of a column
(an image, a time series, a text), the same way to the context and to the
queries. A random augmentation draws from its ``random_state``, so one
pipeline sees one view of the data. Averaging over views is an ensemble of
pipelines that differ only in their seed, whose predictions pool:
:class:`~lazy.models.ensemble.LazyEnsembleModel` seeds step ``j`` of
member ``i`` from ``SeedSequence([random_state, i, j])``::

    tab = lazy.Tabularizer({"cutout": [augment.Dihedral(), "image"]})
    member = lazy.make_pipeline(tab, lazy.LazyModel())
    model = lazy.LazyEnsembleModel([member] * 8, pooling="linear")

The augmentations, by the data they suit and their gradient kind:

* numbers, in any array: :class:`Rescale` and :class:`Noise`
  (differentiable);
* images: :class:`Dihedral` and :class:`Crop` (index maps), and
  :class:`Mask` (a mask);
* time series: :class:`Crop`, :class:`Mask`, :class:`Noise` and
  :class:`Rescale`;
* text: :class:`TokenDropout` and :class:`Truncate` (barriers);
* tables: :class:`ColumnDropout` (an index map).

Each declares its gradient kind (:mod:`lazy._steps`) and, unless it is a
barrier, a ``forward`` on torch tensors, so that a gradient method of
:mod:`lazy.interp` passes it. A draw depends on the seed and on the shape
or the content of what it is applied to, never on the other rows, so a
row's augmentation does not depend on the rows that come with it.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
import numbers
import os
from typing import Any, ClassVar
import zlib

import array_api_compat
import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image
from sklearn import base as sklearn_base
from sklearn.utils import validation

from lazy import _arrays
from lazy.encoders import _base

__all__ = [
    "FUNCTIONS",
    "VIEWS",
    "ColumnDropout",
    "Crop",
    "Dihedral",
    "Mask",
    "Noise",
    "Rescale",
    "TokenDropout",
    "Truncate",
]

FUNCTIONS: tuple[str, ...] = ("identity", "asinh", "log", "log1p")
"""The functions that :class:`Rescale` applies."""

# Each view as a transpose or not, then the axes flipped; axes 1 and 2
# are an image's rows and columns. A counterclockwise quarter turn is a
# transpose, then a flip of the rows; a clockwise one flips the columns.
_VIEWS: dict[str, tuple[bool, tuple[int, ...]]] = {
    "identity": (False, ()),
    "rotate_90": (True, (1,)),
    "rotate_180": (False, (1, 2)),
    "rotate_270": (True, (2,)),
    "flip_left_right": (False, (2,)),
    "flip_top_bottom": (False, (1,)),
    "transpose": (True, ()),
    "transverse": (True, (1, 2)),
}

VIEWS: tuple[str, ...] = tuple(_VIEWS)
"""The views of an image that :class:`Dihedral` gives: the eight rotations
and flips of a square (the dihedral group D4), the rotations
counterclockwise, by Pillow's names."""


def _fraction(name: str, value: object, closed: bool = False) -> None:
    """Checks a share in (0, 1), or in (0, 1] if ``closed``."""
    validation.check_scalar(
        value,
        name,
        numbers.Real,
        min_val=0.0,
        max_val=1.0,
        include_boundaries="right" if closed else "neither",
    )


def _seed(random_state: object) -> None:
    """Checks ``random_state``, a seed: a non-negative integer."""
    validation.check_scalar(
        random_state, "random_state", numbers.Integral, min_val=0
    )


class _Augmentation(
    sklearn_base.TransformerMixin, sklearn_base.BaseEstimator, abc.ABC
):
    """What augmentations share: ``fit`` learns nothing, rows stand alone.

    A subclass implements :meth:`_check` and :meth:`_batch`, one function
    of an array whose first axis indexes rows or items, for numpy arrays
    and torch tensors alike; one that draws from the table it is fitted
    on implements :meth:`_learn` too.
    """

    gradient_kind: ClassVar[str] = "differentiable"

    # Whether the values keep their type, as when they are only moved; if
    # not, they are read as float64.
    _keeps_type: ClassVar[bool] = False

    @abc.abstractmethod
    def _check(self) -> None:
        """Refuses a parameter that is not as the class documents it.

        Raises:
            ValueError: If a parameter is out of its range.
            TypeError: If a parameter is of the wrong type.
        """

    @abc.abstractmethod
    def _batch(self, values: Any) -> Any:
        """The augmentation of rows or of stacked items.

        Args:
            values: A float numpy array or torch tensor of shape (n, ...).

        Returns:
            The augmented values, of that kind, shape (n, ...); a tensor
            keeps its gradient.
        """

    def fit(self, X: Any, y: Any = None) -> _Augmentation:  # noqa: GS030 - scikit-learn's X, y.
        """Checks the parameters; learns nothing but a table's columns.

        Args:
            X: A table of numbers or one column of items; only a table's
                column names are read.
            y: Ignored; scikit-learn passes it.

        Returns:
            The augmentation itself.

        Raises:
            ValueError: If a parameter is out of its range.
            TypeError: If a parameter is of the wrong type.
        """
        del y  # Unused: an augmentation learns nothing.
        # A failed fit leaves the augmentation unfitted.
        for name in ("fitted_", "n_features_in_", "feature_names_in_"):
            self.__dict__.pop(name, None)
        self._check()
        self._learn(X)
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            self.n_features_in_ = X.shape[1]
        self.fitted_ = True
        return self

    def _learn(self, X: Any) -> None:  # noqa: GS030 - scikit-learn's X.
        """Draws what depends on the table of ``fit``; nothing by default."""

    def transform(self, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
        """The augmented table, or the augmented items.

        Args:
            X: A table (a DataFrame, whose numeric columns are augmented
                and whose other columns pass unchanged, or a
                numeric array of shape (n_rows, n_columns)), or one column
                of items, each an array, a Pillow image or the path of an
                image file; a missing item stays as it is.

        Returns:
            A DataFrame with the index and the columns of ``X`` for a
            DataFrame, an array for an array, and a list of arrays for
            a column of items. An augmentation that only moves or zeroes
            values keeps their type, so that an 8-bit image stays one;
            the others give float64.

        Raises:
            sklearn.exceptions.NotFittedError: If it is not fitted.
        """
        validation.check_is_fitted(self, "fitted_")
        numeric = _numeric(X) if isinstance(X, pd.DataFrame) else None
        if numeric is not None and numeric.any():
            values = self._batch(self._read(X.loc[:, numeric].to_numpy()))
            out = X.copy()
            out[X.columns[numeric]] = values
            return out
        if isinstance(X, np.ndarray) and X.dtype.kind in "biuf":
            return self._batch(self._read(X))
        return [
            item
            if _base._is_missing(item)  # noqa: SLF001 - the encoders' reading.
            else self._batch(self._read(item)[np.newaxis])[0]
            for item in _base._as_items(X)  # noqa: SLF001 - as above.
        ]

    def _read(self, item: Any) -> npt.NDArray[Any]:
        """An item, or a table's values, as the array to augment."""
        if isinstance(item, (str, os.PathLike)):
            with Image.open(item) as image:
                item = np.asarray(image)
        array = np.asarray(item)
        return array if self._keeps_type else array.astype(np.float64)

    def forward(self, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
        """:meth:`transform` on a torch tensor, on the autograd graph.

        Args:
            X: Rows of a table or stacked items, a float torch tensor of
                shape (n, ...).

        Returns:
            The augmented tensor, which keeps the gradient of ``X``.
        """
        return self._batch(X)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.object_]:
        """The names of a table's columns, which an augmentation keeps.

        Args:
            input_features: The names of the input columns; ``None`` for
                those of the table given to ``fit``.

        Returns:
            Those names, shape (n_columns,).

        Raises:
            ValueError: If the names are not those of ``fit``, or ``fit``
                was given no table.
        """
        if not hasattr(self, "n_features_in_"):
            raise ValueError(
                f"{type(self).__name__} was fitted on no table, so it has"
                " no column names"
            )
        return validation._check_feature_names_in(self, input_features)  # noqa: SLF001 - scikit-learn's own rule for these names.


def _numeric(frame: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Which columns of a frame are numeric, shape (n_columns,)."""
    # As select_dtypes("number") reads them: a bool column is not one.
    return np.array(
        [
            pd.api.types.is_numeric_dtype(kind)
            and not pd.api.types.is_bool_dtype(kind)
            for kind in frame.dtypes
        ],
        dtype=bool,
    )


class Rescale(_Augmentation):
    r"""A fixed function of every value: :math:`f(x / s)`.

    ``function="asinh"`` compresses a wide range and keeps the sign;
    ``"log"`` and ``"log1p"`` need values above 0 and above -1;
    ``"identity"`` only scales. It has no randomness.

    Attributes:
        function: One of :data:`FUNCTIONS`.
        scale: The positive number :math:`s` that divides every value.
    """

    def __init__(self, function: str = "asinh", *, scale: float = 1.0) -> None:  # noqa: D107 - arguments documented on the class.
        self.function = function
        self.scale = scale

    def _check(self) -> None:
        if self.function not in FUNCTIONS:
            raise ValueError(
                f"function must be one of {FUNCTIONS}: {self.function=}"
            )
        validation.check_scalar(
            self.scale,
            "scale",
            numbers.Real,
            min_val=0.0,
            include_boundaries="neither",
        )

    def _batch(self, values: Any) -> Any:
        scaled = values / self.scale
        if self.function == "identity":
            return scaled
        xp = array_api_compat.array_namespace(values)
        return getattr(xp, self.function)(scaled)


class Noise(_Augmentation):
    r"""Gaussian noise added to every value: :math:`x + s\,\epsilon`.

    The noise is reparameterized: :math:`\epsilon` is a standard normal
    draw that is held fixed, so the output is differentiable in the
    input, with :math:`s` the scale, or the scale times the row's own
    standard deviation if ``relative``.

    Each row, or item, draws :math:`\epsilon` from the seed and its own
    values, rounded to float32, so a row's noise does not depend on the
    rows that come with it, the same row always gets the same noise, and
    a change in a value's last bits does not change the draw. Two rows
    that differ by more draw separately, so along a path of integrated
    gradients each point has a draw of its own, and the attributions sum
    to the gain only roughly; gradient times input is exact.

    Attributes:
        scale: The standard deviation of the noise: in units of each row's
            own standard deviation if ``relative``, else in the values'
            units.
        relative: Whether ``scale`` multiplies each row's own standard
            deviation, which suits an image or a time series; a table's
            rows mix units, so a table is first standardized by a step
            before this one, or takes an absolute ``scale``.
        random_state: The seed, a non-negative integer.
    """

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        scale: float = 0.1,
        *,
        relative: bool = False,
        random_state: int = 0,
    ) -> None:
        self.scale = scale
        self.relative = relative
        self.random_state = random_state

    def _check(self) -> None:
        validation.check_scalar(self.scale, "scale", numbers.Real, min_val=0.0)
        _seed(self.random_state)

    def _batch(self, values: Any) -> Any:
        rows = _arrays.detached(values)
        draws = np.empty_like(rows)
        for index, row in enumerate(rows):
            # Keyed on float32: the last bits of a float64 do not count.
            key = zlib.crc32(np.ascontiguousarray(row, np.float32).tobytes())
            rng = np.random.default_rng([self.random_state, key])
            draws[index] = rng.standard_normal(row.shape)
        xp = array_api_compat.array_namespace(values)
        noise = self.scale * _arrays.constant(draws, values)
        if self.relative:
            axes = tuple(range(1, values.ndim))
            centred = values - xp.mean(values, axis=axes, keepdims=True)
            variance = xp.mean(centred**2, axis=axes, keepdims=True)
            # A row of one value has no spread, and no gradient of it.
            spread = xp.where(
                variance > 0,
                xp.sqrt(xp.where(variance > 0, variance, 1.0)),
                0.0,
            )
            noise = noise * spread
        return values + noise


class Dihedral(_Augmentation):
    """One rotation or flip of every image, the same for all of them.

    An image is an array of shape (height, width) or (height, width,
    channels). ``view`` names the view, or ``None`` draws one of the eight
    of :data:`VIEWS` from ``random_state``. An ensemble of eight pipelines
    with seeds of their own samples the views.

    Attributes:
        view: One of :data:`VIEWS`, or ``None`` to draw one.
        random_state: The seed of the draw, a non-negative integer.
    """

    gradient_kind = "index_map"
    _keeps_type = True

    def __init__(  # noqa: D107 - arguments documented on the class.
        self, view: str | None = None, *, random_state: int = 0
    ) -> None:
        self.view = view
        self.random_state = random_state

    def _check(self) -> None:
        _seed(self.random_state)
        if self.view is not None and self.view not in VIEWS:
            raise ValueError(
                f"view must be None or one of {VIEWS}: {self.view=}"
            )

    def _view(self) -> str:
        """The view applied: the one named, or the seed's draw."""
        if self.view is not None:
            return self.view
        rng = np.random.default_rng(self.random_state)
        return VIEWS[int(rng.integers(len(VIEWS)))]

    def _batch(self, values: Any) -> Any:
        xp = array_api_compat.array_namespace(values)
        swap, flipped = _VIEWS[self._view()]
        if swap:
            values = xp.permute_dims(values, (0, 2, 1, *range(3, values.ndim)))
        return xp.flip(values, axis=flipped) if flipped else values


def _window(size: int, fraction: float, offset: float) -> slice:
    """A window of ``fraction`` of an axis, placed by ``offset`` in [0, 1)."""
    length = max(1, round(fraction * size))
    start = int(offset * (size - length + 1))
    return slice(start, start + length)


class Crop(_Augmentation):
    """A window of every item, at the same relative place in all of them.

    An image keeps ``fraction`` of its rows and of its columns, and a time
    series ``fraction`` of its values. The place of the window is drawn
    from ``random_state``.

    Attributes:
        fraction: The share of each axis that the window keeps, in (0, 1].
        random_state: The seed of the place, a non-negative integer.
    """

    gradient_kind = "index_map"
    _keeps_type = True

    def __init__(  # noqa: D107 - arguments documented on the class.
        self, fraction: float = 0.8, *, random_state: int = 0
    ) -> None:
        self.fraction = fraction
        self.random_state = random_state

    def _check(self) -> None:
        _fraction("fraction", self.fraction, closed=True)
        _seed(self.random_state)

    def _batch(self, values: Any) -> Any:
        offsets = np.random.default_rng(self.random_state).random(2)
        # An image has two axes to crop, a series one.
        n_axes = min(2, values.ndim - 1)
        windows = tuple(
            _window(values.shape[axis + 1], self.fraction, offsets[axis])
            for axis in range(n_axes)
        )
        return values[(slice(None), *windows)]


class Mask(_Augmentation):
    """Patches of every item set to zero, the same for items of one shape.

    An image is cut into squares of ``patch`` pixels, and a time series
    into runs of ``patch`` values; ``fraction`` of them, drawn from
    ``random_state`` and the item's shape, are set to zero. A masked value
    gets no gradient.

    Attributes:
        fraction: The share of the patches that are masked, in (0, 1).
        patch: The side of a patch, in pixels or values.
        random_state: The seed of the patches, a non-negative integer.
    """

    gradient_kind = "mask"
    _keeps_type = True

    def __init__(  # noqa: D107 - arguments documented on the class.
        self, fraction: float = 0.2, *, patch: int = 4, random_state: int = 0
    ) -> None:
        self.fraction = fraction
        self.patch = patch
        self.random_state = random_state

    def _check(self) -> None:
        _fraction("fraction", self.fraction)
        _base.positive_integer("patch", self.patch)
        _seed(self.random_state)

    def _kept(self, shape: tuple[int, ...]) -> npt.NDArray[np.bool_]:
        """Which values of an item of ``shape`` stay, of that shape."""
        n_axes = min(2, len(shape))
        cells = tuple(-(-size // self.patch) for size in shape[:n_axes])
        # Keyed on the masked axes alone: a grey image masks alike with
        # and without its axis of one channel.
        rng = np.random.default_rng([self.random_state, *shape[:n_axes]])
        kept = rng.random(cells) >= self.fraction
        for axis in range(n_axes):
            kept = np.repeat(kept, self.patch, axis=axis)
        kept = kept[tuple(slice(size) for size in shape[:n_axes])]
        return kept.reshape(*kept.shape, *[1] * (len(shape) - n_axes))

    def _batch(self, values: Any) -> Any:
        xp = array_api_compat.array_namespace(values)
        kept = xp.asarray(
            self._kept(tuple(values.shape[1:])),
            device=array_api_compat.device(values),
        )
        return xp.where(kept, values, xp.zeros_like(values))


class ColumnDropout(_Augmentation):
    """A table without some of its columns, drawn once from the seed.

    ``by="column"`` drops ``fraction`` of the columns. ``by="prefix"``
    treats the columns that share the text before the first ``"__"`` of
    their names as one unit, and drops ``fraction`` of the units: after a
    :class:`~lazy.encoders.Tabularizer` whose ``reducer`` is no torch
    module, a unit is one column of items, so this drops a whole data
    type; the columns that pass through are one more unit. At least one
    unit is always kept, and of two or more at least one is dropped.

    Attributes:
        fraction: The share of the units that are dropped, in (0, 1).
        by: ``"column"`` or ``"prefix"``.
        random_state: The seed of the draw, a non-negative integer.
        kept_: The positions of the columns kept, in order, shape
            (n_columns_kept,).
    """

    gradient_kind = "index_map"
    _keeps_type = True

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        fraction: float = 0.2,
        *,
        by: str = "column",
        random_state: int = 0,
    ) -> None:
        self.fraction = fraction
        self.by = by
        self.random_state = random_state

    def _check(self) -> None:
        _fraction("fraction", self.fraction)
        _seed(self.random_state)
        if self.by not in ("column", "prefix"):
            raise ValueError(f"by must be 'column' or 'prefix': {self.by=}")

    def _learn(self, X: Any) -> None:  # noqa: GS030 - scikit-learn's X.
        """Draws the columns to keep of a table, shape (n_rows, n_columns).

        Raises:
            ValueError: If ``by="prefix"`` meets a table without column
                names.
        """
        self.__dict__.pop("kept_", None)
        names = list(getattr(X, "columns", range(np.shape(X)[1])))
        if self.by == "prefix":
            if not isinstance(X, pd.DataFrame):
                raise ValueError("by='prefix' needs a table with column names")
            units = [str(name).split("__", 1)[0] for name in names]
        else:
            units = [str(position) for position in range(len(names))]
        distinct = list(dict.fromkeys(units))
        rng = np.random.default_rng(self.random_state)
        n_dropped = min(
            max(1, round(self.fraction * len(distinct))), len(distinct) - 1
        )
        dropped = {
            distinct[i]
            for i in rng.choice(len(distinct), n_dropped, replace=False)
        }
        self.kept_ = np.flatnonzero([unit not in dropped for unit in units])
        self._names = np.asarray(names, dtype=object)

    def transform(self, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
        """The table's kept columns.

        Args:
            X: A table with the columns of ``fit``.

        Returns:
            A DataFrame for a DataFrame and an array for an array, of the
            kept columns in their order.

        Raises:
            sklearn.exceptions.NotFittedError: If it is not fitted.
        """
        validation.check_is_fitted(self, "fitted_")
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.kept_]
        return np.asarray(X)[:, self.kept_]

    def _batch(self, values: Any) -> Any:
        return values[:, self.kept_]

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.object_]:
        """The names of the kept columns.

        Args:
            input_features: Ignored; the fitted table names the columns.

        Returns:
            Their names, shape (n_columns_kept,).
        """
        del input_features  # Unused: the names come from the fit.
        validation.check_is_fitted(self, "fitted_")
        return np.asarray(self._names[self.kept_], dtype=object)


class _TextAugmentation(_Augmentation):
    """An augmentation of texts, word by word; a barrier to gradients."""

    gradient_kind = "barrier"
    barrier_reason = "a text is a sequence of words, which have no gradient"

    def _batch(self, values: Any) -> Any:
        del values  # Unused: a text is no array.
        raise TypeError(f"{type(self).__name__} augments texts, not numbers")

    @abc.abstractmethod
    def _words(self, text: str) -> Sequence[str]:
        """The words kept of one text.

        Args:
            text: The text, whose words white space splits.

        Returns:
            The words of the augmented text, in order.
        """

    def transform(self, X: Any) -> list[Any]:  # noqa: GS030 - scikit-learn's X.
        """The augmented texts.

        Args:
            X: One column of texts; a missing one stays as it is.

        Returns:
            The texts, their kept words joined by single spaces.

        Raises:
            sklearn.exceptions.NotFittedError: If it is not fitted.
        """
        validation.check_is_fitted(self, "fitted_")
        return [
            item
            if _base._is_missing(item)  # noqa: SLF001 - the encoders' reading.
            else " ".join(self._words(str(item)))
            for item in _base._as_items(X)  # noqa: SLF001 - as above.
        ]


class TokenDropout(_TextAugmentation):
    """Each text without some of its words.

    Each word is dropped with probability ``fraction``, by a draw from the
    seed and the text itself, so a text's augmentation does not depend on
    the texts that come with it. A text never loses its last word.

    Attributes:
        fraction: The probability that a word is dropped, in (0, 1).
        random_state: The seed, a non-negative integer.
    """

    def __init__(  # noqa: D107 - arguments documented on the class.
        self, fraction: float = 0.1, *, random_state: int = 0
    ) -> None:
        self.fraction = fraction
        self.random_state = random_state

    def _check(self) -> None:
        _fraction("fraction", self.fraction)
        _seed(self.random_state)

    def _words(self, text: str) -> Sequence[str]:
        words = text.split()
        key = zlib.crc32(text.encode())
        rng = np.random.default_rng([self.random_state, key])
        kept = rng.random(len(words)) >= self.fraction
        if words and not kept.any():
            kept[int(rng.integers(len(words)))] = True
        return [word for word, keep in zip(words, kept, strict=True) if keep]


class Truncate(_TextAugmentation):
    """Each text cut to its first words. It has no randomness.

    Attributes:
        max_words: The most words a text keeps.
    """

    def __init__(self, max_words: int = 256) -> None:  # noqa: D107 - arguments documented on the class.
        self.max_words = max_words

    def _check(self) -> None:
        _base.positive_integer("max_words", self.max_words)

    def _words(self, text: str) -> Sequence[str]:
        return text.split()[: self.max_words]
