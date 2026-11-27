# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""From a table with columns of items to a table of numbers.

:class:`Tabularizer` encodes each column of a table that holds items
(images, texts, time series) with an encoder, reduces all the embeddings
together to a few columns, and passes the other columns through. It is the
step before the model in a pipeline::

    tab = lazy.Tabularizer({"cutout": "image", "notes": "text"})
    model = lazy.make_pipeline(tab, lazy.LazyModel())
    model.fit(table, y)
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import array_api_compat
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base as sklearn_base
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.utils import validation

from lazy import _typing
from lazy import _warnings
from lazy.encoders import _base
from lazy.encoders import registry

__all__ = ["WEIGHTS", "Tabularizer"]

WEIGHTS: tuple[str, ...] = ("equal",)
"""The named values of ``weights``."""

_FITTED = (
    "encoders_",
    "remainder_",
    "mean_",
    "scale_",
    "components_",
    "offset_",
    "weights_",
    "_positions",
    "n_features_in_",
    "feature_names_in_",
)


class Tabularizer(sklearn_base.TransformerMixin, sklearn_base.BaseEstimator):
    r"""Encodes the columns of items, reduces them together, keeps the others.

    Each column named in ``columns`` is encoded by its encoder, which a
    modality (``"image"``, ``"text"`` or ``"time"``, through
    :data:`~lazy.encoders.DEFAULT_ENCODERS`), the name of a registered
    encoder, or an encoder object names. The embeddings of all the columns
    are then reduced together to ``n_columns`` columns, whatever the number
    of columns of items: each column of each embedding is standardized,
    each embedding is weighted as ``weights`` says, and the weighted
    embeddings, side by side, are projected on their first ``n_columns``
    principal components. :doc:`/guide/encoders` gives the equations.
    Embeddings that have no more than ``n_columns`` columns between them
    pass unchanged, with their encoders' names. The other columns of the
    table pass through unreduced, or are dropped, as ``remainder`` says.

    A missing item gives its embedding the mean of the context, so that a
    row keeps what its other items say; a row whose items are all missing
    is ``NaN`` in every reduced column, and is left out of the fit. Without
    a reduction, a missing item stays a row of ``NaN`` in its embedding.

    The output is a DataFrame with the index of ``X`` and the names of
    :meth:`get_feature_names_out`. The encoded columns are float64, and a
    column that passes through keeps its type. ``set_output``, for example
    with ``transform="polars"``, converts it to another container.

    The encoders are frozen. The standardization and the principal
    components are fitted on the rows given to ``fit``, the context, and
    a row's output depends on that row alone.

    Attributes:
        columns: Column name (or position, for an array) -> what encodes
            it: a modality, an encoder name or an encoder.
        n_columns: The most columns that all the embeddings keep between
            them; ``None`` keeps every column of every embedding.
        weights: The share of each embedding in the reduction: ``"equal"``
            gives every embedding the same total variance, 1, whatever its
            width; ``None`` leaves every standardized column at variance
            1, so that a wider embedding weighs more; a list gives each
            embedding its total variance, in the order of ``columns``.
        remainder: What becomes of the columns not in ``columns``:
            ``"passthrough"`` keeps them, ``"drop"`` drops them, and a
            transformer transforms them, as in
            :class:`~sklearn.compose.ColumnTransformer`.
        encoders_: The fitted encoder of each column of items, by the
            column's name, in the order of ``columns``.
        mean_: The mean of each column of the embeddings, side by side,
            shape (n_embedding_columns,).
        scale_: What each of those columns is divided by, its standard
            deviation over its embedding's weight, of that shape.
        weights_: The factor that multiplies each standardized embedding,
            shape (n_item_columns,).
        components_: The principal components, shape (n_columns_kept,
            n_embedding_columns), or ``None`` when the embeddings pass
            unchanged.
        offset_: What is subtracted from the projection, shape
            (n_columns_kept,), or ``None``.
        remainder_: The fitted transformer of the other columns, or
            ``remainder`` itself for ``"passthrough"`` and ``"drop"``.
    """

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        columns: Mapping[Hashable, str | _base.BaseEncoder],
        *,
        n_columns: int | None = 30,
        weights: str | Sequence[float] | None = "equal",
        remainder: Any = "passthrough",
    ) -> None:
        self.columns = columns
        self.n_columns = n_columns
        self.weights = weights
        self.remainder = remainder

    def fit(self, X: Any, y: Any = None) -> Tabularizer:  # noqa: GS030 - scikit-learn's X, y.
        """Builds the encoders, and fits the reduction on the context.

        Each item is encoded once; :meth:`fit_transform` does the same work
        and returns the table.

        Args:
            X: The table: a DataFrame, or an array whose columns are named
                by position.
            y: Ignored; scikit-learn passes it.

        Returns:
            The tabularizer itself.

        Raises:
            TypeError: If ``columns`` is not a mapping, or a value of it is
                neither text nor an encoder.
            UnknownNameError: If a text value is neither a modality nor a
                registered encoder.
            ValueError: If ``columns`` is empty or names a column that
                ``X`` does not have, ``n_columns`` is not a positive
                integer or ``None``, ``weights`` or ``remainder`` is not
                valid, or the embeddings to reduce have fewer than two rows
                with an item that is not missing.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: Any, y: Any = None) -> pd.DataFrame:  # noqa: GS030 - scikit-learn's X, y.
        """Fits as :meth:`fit` does, and returns the table it encoded.

        Args:
            X: The table: a DataFrame, or an array whose columns are named
                by position.
            y: Ignored; scikit-learn passes it.

        Returns:
            The table as :meth:`transform` returns it.

        Raises:
            TypeError: As :meth:`fit` raises it.
            UnknownNameError: As :meth:`fit` raises it.
            ValueError: As :meth:`fit` raises it.
        """
        del y  # Unused: the steps are fitted without the target.
        # A failed fit leaves the tabularizer unfitted.
        for name in _FITTED:
            self.__dict__.pop(name, None)
        if not isinstance(self.columns, Mapping):
            raise TypeError(
                "columns must map a column to a modality, an encoder name or"
                f" an encoder; got {type(self.columns).__name__}"
            )
        if not self.columns:
            raise ValueError("columns must name at least one column")
        if self.n_columns is not None:
            _base.positive_integer("n_columns", self.n_columns)
        plain = isinstance(self.remainder, str) and self.remainder in (
            "passthrough",
            "drop",
        )
        if not plain and not hasattr(self.remainder, "fit_transform"):
            raise ValueError(
                "remainder must be 'passthrough', 'drop' or a transformer:"
                f" {self.remainder=}"
            )
        frame = _frame(X)
        positions = [_position(frame, column) for column in self.columns]
        encoders = {
            column: _encoder(column, spec)
            for column, spec in self.columns.items()
        }
        blocks = [
            encoder.fit_transform(frame.iloc[:, position])
            for encoder, position in zip(
                encoders.values(), positions, strict=True
            )
        ]
        fitted = self._fit_reduction(blocks)
        rest = frame.drop(columns=frame.columns[positions])
        remainder = self.remainder
        if not plain:
            remainder = sklearn_base.clone(self.remainder).fit(rest)
        validation.validate_data(self, X, skip_check_array=True)
        self.encoders_ = encoders
        self.remainder_ = remainder
        self._positions = positions
        self.__dict__.update(fitted)
        return self._table(blocks, rest)

    def _fit_reduction(
        self, blocks: list[_typing.FloatArray]
    ) -> dict[str, Any]:
        """The fitted standardization, weights and components.

        Args:
            blocks: Each column's embeddings of the context, shape
                (n_rows, width), a missing item's row all ``NaN``.

        Returns:
            The fitted attributes ``mean_``, ``scale_``, ``weights_``,
            ``components_`` and ``offset_``, by name.

        Raises:
            ValueError: If ``weights`` is not valid, or the embeddings must
                be reduced and fewer than two rows have an item.
        """
        scalers = [
            preprocessing.StandardScaler().fit(
                block[~np.isnan(block).any(axis=1)]
            )
            if not np.isnan(block).all()
            else None
            for block in blocks
        ]
        widths = [block.shape[1] for block in blocks]
        mean = np.concatenate(
            [
                np.zeros(width) if scaler is None else scaler.mean_
                for scaler, width in zip(scalers, widths, strict=True)
            ]
        )
        spread = [
            np.ones(width) if scaler is None else scaler.scale_
            for scaler, width in zip(scalers, widths, strict=True)
        ]
        # A column that is constant has no variance to share.
        varying = [
            0 if scaler is None else int((scaler.var_ > 0).sum())
            for scaler in scalers
        ]
        factors = _factors(self.weights, varying)
        scale = np.concatenate(
            [s / f for s, f in zip(spread, factors, strict=True)]
        )
        fitted: dict[str, Any] = {
            "mean_": mean,
            "scale_": scale,
            "weights_": factors,
            "components_": None,
            "offset_": None,
        }
        if self.n_columns is None or sum(widths) <= self.n_columns:
            return fitted
        wide = np.hstack(blocks)
        present = ~np.isnan(wide).all(axis=1)
        if present.sum() < 2:
            raise ValueError(
                f"the embeddings to reduce have {present.sum()} rows with an"
                " item that is not missing; they need at least 2"
            )
        # The centred rows span at most one direction fewer than they are.
        n_kept = min(self.n_columns, int(present.sum()) - 1)
        if n_kept < self.n_columns:
            _warnings.warn(
                f"the embeddings reduced to n_columns={self.n_columns} have"
                f" {present.sum()} rows with an item that is not missing,"
                f" so they keep {n_kept} columns",
                _warnings.LazyWarning,
            )
        standardized = np.nan_to_num((wide[present] - mean) / scale, nan=0.0)
        pca = decomposition.PCA(n_kept, svd_solver="full").fit(standardized)
        fitted["components_"] = pca.components_
        fitted["offset_"] = pca.mean_ @ pca.components_.T
        return fitted

    def transform(self, X: Any) -> pd.DataFrame:  # noqa: GS030 - scikit-learn's X, y.
        """The reduced embeddings, then the columns kept.

        Args:
            X: A table with the columns of the one given to ``fit``.

        Returns:
            The table, shape (n_rows, n_features_out), with the index of
            ``X`` and the columns of :meth:`get_feature_names_out`.

        Raises:
            sklearn.exceptions.NotFittedError: If it is not fitted.
            ValueError: If the columns of ``X`` are not those of ``fit``.
        """
        validation.check_is_fitted(self, "encoders_")
        validation.validate_data(self, X, reset=False, skip_check_array=True)
        blocks, rest = self._encoded(X)
        return self._table(blocks, rest)

    def _encoded(self, X: Any) -> tuple[list[_typing.FloatArray], pd.DataFrame]:  # noqa: GS030 - scikit-learn's X.
        """Each column's embeddings of a table, and its other columns.

        Args:
            X: A table with the columns of the one given to ``fit``.

        Returns:
            A tuple (blocks, rest): the embeddings of each column of items,
            in the order of ``columns``, each of shape (n_rows, width); and
            the frame of the other columns, as given.
        """
        frame = _frame(X)
        positions = [_position(frame, column) for column in self.columns]
        blocks = [
            encoder.transform(frame.iloc[:, position])
            for encoder, position in zip(
                self.encoders_.values(), positions, strict=True
            )
        ]
        return blocks, frame.drop(columns=frame.columns[positions])

    def _reduced(self, blocks: Sequence[Any]) -> Any:
        """The reduced columns of some rows' embeddings.

        One implementation for numpy arrays and for torch tensors, which
        keep their gradients.

        Args:
            blocks: Each column's embeddings, in the order of ``columns``,
                arrays of one kind and of shape (n_rows, width).

        Returns:
            The embeddings side by side if they pass unchanged, else their
            components, of that kind and shape (n_rows, n_columns_kept).
        """
        xp = array_api_compat.array_namespace(*blocks)
        wide = xp.concat(list(blocks), axis=1)
        if self.components_ is None:
            return wide

        def constant(values: _typing.FloatArray) -> Any:
            """A fitted array, of the kind of the embeddings."""
            return xp.asarray(
                values, dtype=wide.dtype, device=array_api_compat.device(wide)
            )

        missing = xp.isnan(wide)
        standardized = xp.where(
            missing,
            constant(np.zeros(())),
            (wide - constant(self.mean_)) / constant(self.scale_),
        )
        components = constant(self.components_)
        reduced = standardized @ components.T - constant(self.offset_)
        return xp.where(
            xp.all(missing, axis=1)[:, None],
            constant(np.full((), np.nan)),
            reduced,
        )

    def _table(
        self, blocks: list[_typing.FloatArray], rest: pd.DataFrame
    ) -> pd.DataFrame:
        """The output table of some rows.

        Args:
            blocks: Each column's embeddings, shape (n_rows, width).
            rest: The other columns of the rows.

        Returns:
            The reduced embeddings, then the columns kept, with the index
            of ``rest``.
        """
        names = self.get_feature_names_out()
        reduced = self._reduced(blocks)
        table = pd.DataFrame(
            reduced, index=rest.index, columns=names[: reduced.shape[1]]
        )
        kept = self._kept(rest)
        if kept is None:
            return table
        kept = kept.set_axis(names[reduced.shape[1] :], axis=1)
        return pd.concat([table, kept], axis=1)

    def _kept(self, rest: pd.DataFrame) -> pd.DataFrame | None:
        """The other columns as ``remainder`` leaves them; ``None`` if none.

        Args:
            rest: The columns that are not in ``columns``.

        Returns:
            The frame of the columns kept, with the index of ``rest``.
        """
        if isinstance(self.remainder_, str):
            return rest if self.remainder_ == "passthrough" else None
        made = self.remainder_.transform(rest)
        if isinstance(made, pd.DataFrame):
            return made
        return pd.DataFrame(np.asarray(made), index=rest.index)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.object_]:
        """The names of the output columns.

        Args:
            input_features: Ignored; scikit-learn passes it.

        Returns:
            ``embedding__pc0`` onwards for the reduced columns, or
            ``<column>__<name>`` with each encoder's names
            (``notes__text_hashing_3``) for embeddings that pass unchanged;
            then ``remainder__<column>`` for each column kept.

        Raises:
            sklearn.exceptions.NotFittedError: If it is not fitted.
        """
        del input_features  # Unused: the fitted table names the columns.
        validation.check_is_fitted(self, "encoders_")
        if self.components_ is None:
            names: list[str] = []
            for column, encoder in self.encoders_.items():
                names.extend(
                    f"{column}__{name}"
                    for name in encoder.get_feature_names_out()
                )
        else:
            names = [f"embedding__pc{i}" for i in range(len(self.components_))]
        return np.array([*names, *self._remainder_names()], dtype=object)

    def _remainder_names(self) -> list[str]:
        """``remainder__<column>`` for each column kept, in order."""
        if isinstance(self.remainder_, str):
            if self.remainder_ == "drop":
                return []
            columns = getattr(
                self, "feature_names_in_", np.arange(self.n_features_in_)
            )
            names = list(np.delete(columns, self._positions))
        else:
            names = list(self.remainder_.get_feature_names_out())
        return [f"remainder__{name}" for name in names]


def _factors(weights: object, varying: list[int]) -> _typing.FloatArray:
    """The factor that multiplies each standardized embedding.

    A standardized embedding has a total variance of its number of columns
    that vary, so the factor that gives it the total variance ``w`` is
    ``sqrt(w / n_varying)``.

    Args:
        weights: ``weights`` as given.
        varying: The number of columns of each embedding that vary in the
            context.

    Returns:
        The factors, shape (n_item_columns,); 1 for an embedding with no
        column that varies.

    Raises:
        ValueError: If ``weights`` is not ``"equal"``, ``None`` or one
            positive finite number per column of items.
    """
    counts = np.maximum(np.array(varying, dtype=float), 1.0)
    if weights is None:
        return np.ones(len(varying))
    if isinstance(weights, str):
        if weights not in WEIGHTS:
            raise ValueError(
                "weights must be 'equal', None or a list of one number per"
                f" column of items: {weights=}"
            )
        return 1.0 / np.sqrt(counts)
    try:
        totals = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError):
        totals = np.empty(0)
    valid = totals.shape == (len(varying),) and bool(
        (np.isfinite(totals) & (totals > 0)).all()
    )
    if not valid:
        raise ValueError(
            "weights must be 'equal', None or a list of one positive number"
            f" per column of items, {len(varying)} here: {weights=}"
        )
    return np.asarray(np.sqrt(totals / counts))


def _frame(X: Any) -> pd.DataFrame:  # noqa: GS030 - scikit-learn's X, y.
    """A table as a DataFrame; an array's columns are named by position."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X)).infer_objects()


def _position(frame: pd.DataFrame, column: Hashable) -> int:
    """The position of a column of a table, by its label.

    Raises:
        ValueError: If the table has no column, or several, of that name.
    """
    try:
        position = frame.columns.get_loc(column)
    except (KeyError, TypeError):
        raise ValueError(
            f"columns names {column!r}, which is not a column of X"
        ) from None
    if not isinstance(position, int):
        raise ValueError(f"X has several columns named {column!r}")
    return position


def _encoder(column: Hashable, spec: object) -> _base.BaseEncoder:
    """A new encoder, unfitted, for one entry of ``columns``.

    Raises:
        TypeError: If ``spec`` is neither text nor an encoder.
        UnknownNameError: If ``spec`` names no modality and no encoder.
    """
    if isinstance(spec, _base.BaseEncoder):
        return sklearn_base.clone(spec)
    if not isinstance(spec, str):
        raise TypeError(
            f"columns[{column!r}] must be a modality, an encoder name or an"
            f" encoder; got {type(spec).__name__}"
        )
    return registry.get_encoder(spec)
