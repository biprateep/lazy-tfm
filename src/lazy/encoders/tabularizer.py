# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""From a table with columns of items to a table of numbers.

:class:`Tabularizer` encodes each column of a table that holds items
(images, texts, time series) with an encoder, reduces each embedding to a
few columns, and passes the other columns through. It is the step before
the model in a pipeline::

    tab = lazy.Tabularizer({"cutout": "image", "notes": "text"})
    model = lazy.make_pipeline(tab, lazy.LazyModel())
    model.fit(table, y)
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base as sklearn_base
from sklearn import compose
from sklearn import decomposition
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.utils import validation

from lazy import _typing
from lazy.encoders import _base
from lazy.encoders import registry

__all__ = ["Tabularizer"]

_FITTED = (
    "transformer_",
    "column_labels_",
    "n_features_in_",
    "feature_names_in_",
)


class Tabularizer(sklearn_base.TransformerMixin, sklearn_base.BaseEstimator):
    """Encodes the columns of items, reduces them and keeps the others.

    Each column named in ``columns`` is encoded by its encoder, which a
    modality (``"image"``, ``"text"`` or ``"time"``, through
    :data:`~lazy.encoders.DEFAULT_ENCODERS`), the name of a registered
    encoder, or an encoder names. An embedding wider than ``n_components`` is
    standardized and reduced to its first ``n_components`` principal
    components. A row of ``NaN``, from a missing item, stays a row of
    ``NaN``. The other columns pass through, or are dropped, as
    ``remainder`` says.

    The encoders are frozen. The standardization and the principal
    components are fitted on the rows given to ``fit``, the context, and
    a row's output depends on that row alone.

    Attributes:
        columns: Column name (or position, for an array) -> what encodes
            it: a modality, an encoder name or an encoder.
        n_components: The most columns each embedding keeps; ``None``
            keeps every column.
        remainder: What becomes of the columns not in ``columns``:
            ``"passthrough"`` keeps them, ``"drop"`` drops them, and a
            transformer transforms them, as in
            :class:`~sklearn.compose.ColumnTransformer`.
        transformer_: The fitted :class:`sklearn.compose.ColumnTransformer`
            that does the work.
    """

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        columns: Mapping[Hashable, str | _base.BaseEncoder],
        *,
        n_components: int | None = 8,
        remainder: Any = "passthrough",
    ) -> None:
        self.columns = columns
        self.n_components = n_components
        self.remainder = remainder

    def fit(self, X: Any, y: Any = None) -> Tabularizer:  # noqa: GS030 - scikit-learn's X, y.
        """Builds the encoders, and fits each reducer on the context.

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
                ``X`` does not have, ``n_components`` is not a positive
                integer or ``None``, ``remainder`` is not valid, or an
                embedding to reduce has no more rows without ``NaN`` (items
                that are not missing) than ``n_components``.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: Any, y: Any = None) -> npt.NDArray[Any]:  # noqa: GS030 - scikit-learn's X, y.
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
        if self.n_components is not None:
            _base.positive_integer("n_components", self.n_components)
        # Steps get names of their own: scikit-learn refuses a step name
        # with "__" or one that is an argument of ColumnTransformer.
        steps = [
            (
                f"column{i}",
                pipeline.make_pipeline(
                    _encoder(column, spec), _Reducer(self.n_components)
                ),
                [_position(X, column)],
            )
            for i, (column, spec) in enumerate(self.columns.items())
        ]
        transformer = compose.ColumnTransformer(
            steps, remainder=self.remainder, sparse_threshold=0
        )
        table = transformer.fit_transform(X)
        self.transformer_ = transformer
        self.column_labels_ = {
            f"column{i}": str(column) for i, column in enumerate(self.columns)
        }
        self.n_features_in_ = transformer.n_features_in_
        if hasattr(transformer, "feature_names_in_"):
            self.feature_names_in_ = transformer.feature_names_in_
        return table

    def transform(self, X: Any) -> npt.NDArray[Any]:  # noqa: GS030 - scikit-learn's X, y.
        """The encoded and reduced columns, then the columns kept.

        Args:
            X: A table with the columns of the one given to ``fit``.

        Returns:
            The table as an array, shape (n_rows, n_features_out), in the
            order of :meth:`get_feature_names_out`.

        Raises:
            sklearn.exceptions.NotFittedError: If it is not fitted.
        """
        validation.check_is_fitted(self, "transformer_")
        return self.transformer_.transform(X)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.object_]:
        """The names of the output columns.

        Args:
            input_features: Ignored; scikit-learn passes it.

        Returns:
            ``<column>__<name>`` for each column of an embedding, where
            the name is the encoder's (``notes__text_hashing_3``) or a
            principal component's (``notes__pc0``), then
            ``remainder__<column>`` for each column kept.

        Raises:
            sklearn.exceptions.NotFittedError: If it is not fitted.
        """
        del input_features  # Unused: the fitted table names the columns.
        validation.check_is_fitted(self, "transformer_")
        names = []
        for name in self.transformer_.get_feature_names_out():
            step, _, feature = name.partition("__")
            names.append(f"{self.column_labels_.get(step, step)}__{feature}")
        return np.array(names, dtype=object)


def _position(X: Any, column: Hashable) -> Hashable:  # noqa: GS030 - scikit-learn's X, y.
    """The position of a column of a DataFrame; ``column`` for an array.

    ColumnTransformer reads an integer as a position, even as the label of
    a column of a DataFrame, so a DataFrame's columns are given by position.

    Raises:
        ValueError: If a DataFrame has no column, or several, of that name.
    """
    if not isinstance(X, pd.DataFrame):
        return column
    try:
        position = X.columns.get_loc(column)
    except KeyError:
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


class _Reducer(sklearn_base.TransformerMixin, sklearn_base.BaseEstimator):
    """Standardizes an embedding and keeps its first principal components.

    An embedding no wider than ``n_components`` passes unchanged, with
    its names. Rows with a ``NaN`` are left out of the fit and stay
    ``NaN``.

    Attributes:
        n_components: The most columns to keep; ``None`` for all.
    """

    def __init__(self, n_components: int | None) -> None:
        self.n_components = n_components

    def fit(self, X: _typing.FloatArray, y: Any = None) -> _Reducer:  # noqa: GS030 - scikit-learn's X, y.
        """Fits the standardization and the components on complete rows.

        Raises:
            ValueError: If the embedding must be reduced and no row is
                complete.
        """
        del y  # Unused: the reduction is fitted without the target.
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        if self.n_components is None or X.shape[1] <= self.n_components:
            return self
        complete = X[~np.isnan(X).any(axis=1)]
        if len(complete) <= self.n_components:
            raise ValueError(
                f"an embedding to reduce to {self.n_components} components"
                f" has {len(complete)} rows without NaN (items that are not"
                " missing); it needs more rows than components"
            )
        self.pca_ = pipeline.make_pipeline(
            preprocessing.StandardScaler(),
            decomposition.PCA(self.n_components, svd_solver="full"),
        ).fit(complete)
        return self

    def transform(self, X: _typing.FloatArray) -> _typing.FloatArray:  # noqa: GS030 - scikit-learn's X, y.
        """The components of each complete row; ``NaN`` for the others."""
        validation.check_is_fitted(self, "n_features_in_")
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "pca_"):
            return X
        complete = ~np.isnan(X).any(axis=1)
        reduced = np.full((len(X), self._width()), np.nan)
        if complete.any():
            reduced[complete] = self.pca_.transform(X[complete])
        return reduced

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.object_]:
        """The encoder's names, which a pipeline passes, or ``pc0`` onwards."""
        validation.check_is_fitted(self, "n_features_in_")
        if not hasattr(self, "pca_"):
            return np.asarray(input_features, dtype=object)
        return np.array([f"pc{i}" for i in range(self._width())], dtype=object)

    def _width(self) -> int:
        """The number of principal components kept."""
        return int(self.pca_[-1].n_components_)
