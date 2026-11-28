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
import copy
import numbers
from typing import Any

import array_api_compat
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import linalg
from sklearn import base as sklearn_base
from sklearn import preprocessing
from sklearn.utils import extmath
from sklearn.utils import validation

from lazy import _steps
from lazy import _typing
from lazy import _warnings
from lazy.encoders import _adaptive
from lazy.encoders import _base
from lazy.encoders import registry
from lazy.models import _differentiable as differentiable
from lazy.models import _members

__all__ = ["Tabularizer"]

_FITTED = (
    "encoders_",
    "augmentations_",
    "remainder_",
    "mean_",
    "scale_",
    "components_",
    "offset_",
    "blocks_",
    "weights_",
    "network_",
    "history_",
    "best_step_",
    "_positions",
    "_rest_names",
    "_n_reduced",
    "_context",
    "n_features_in_",
    "feature_names_in_",
)


class Tabularizer(sklearn_base.TransformerMixin, sklearn_base.BaseEstimator):
    r"""Encodes the columns of items, reduces them, and keeps the others.

    Each column named in ``columns`` is encoded by its encoder, which a
    modality (``"image"``, ``"text"`` or ``"time"``, through
    :data:`~lazy.encoders.DEFAULT_ENCODERS`), the name of a registered
    encoder, or an encoder object names. A list gives the steps of
    :mod:`lazy.augment` that transform the column's items first, then its
    encoder, as ``[augment.Dihedral(), "image"]``.

    The embeddings share one budget of ``n_columns`` columns, whatever the
    number of columns of items. Each column of each embedding is
    standardized, and each embedding gets principal components of its
    own. All the components are then ranked together, by the variance
    each explains times its embedding's weight (``weights``), and the best
    ``n_columns`` are kept; every embedding keeps at least its first.
    :doc:`/guide/encoders` gives the equations. Embeddings that have no
    more than ``n_columns`` columns between them pass unchanged, with
    their encoders' names. The other columns of the table pass through
    unreduced, or are dropped, as ``remainder`` says.

    Each reduced column belongs to one column of items. A missing item
    is therefore ``NaN`` in its own columns alone, which the models read
    as missing features, and its row keeps what its other items say. It
    is left out of the fit of its embedding.

    The output is a DataFrame with the index of ``X`` and the names of
    :meth:`get_feature_names_out`. The encoded columns are float64, and a
    column that passes through keeps its type. ``set_output``, for example
    with ``transform="polars"``, converts it to another container.

    The encoders are frozen. The standardization and the principal
    components are fitted on the rows given to ``fit``, the context, and
    a row's output depends on that row alone.

    ``reducer="mlp"``, or a torch module, makes the reduction adaptive:
    a network that :meth:`lazy.pipeline.LazyPipeline.fit` trains through
    the frozen model that follows (:class:`~lazy.encoders.AdaptiveReducer`).
    It trains weights, so it is opt-in. Outside a ``LazyPipeline`` the
    network stays as it starts: PCA for ``"mlp"``, and the module as given.

    Attributes:
        columns: Column name (or position, for an array) -> what encodes
            it: a modality, an encoder name or an encoder, or a list of
            augmentations that ends with one of those.
        n_columns: The most columns that all the embeddings keep between
            them; ``None`` keeps every column of every embedding.
        weights: The share of each embedding in the budget: ``"equal"``
            ranks the components as if every embedding had the same total
            variance, 1, whatever its width; ``None`` leaves every
            standardized column at variance 1, so that a wider embedding
            takes more of the budget; a list gives each embedding its
            total variance, in the order of ``columns``.
        reducer: What reduces the weighted embeddings: ``"pca"``, the
            principal components; ``"mlp"``, the components plus a
            residual network, trained through the model; a
            ``torch.nn.Module`` from the embeddings to the columns,
            trained likewise; or an
            :class:`~lazy.encoders.AdaptiveReducer`, which also holds the
            settings of the training. An adaptive reducer always reduces,
            to at most ``n_columns`` columns.
        remainder: What becomes of the columns not in ``columns``:
            ``"passthrough"`` keeps them, ``"drop"`` drops them, and a
            transformer transforms them, as in
            :class:`~sklearn.compose.ColumnTransformer`.
        random_state: The seed of the columns' augmentations: augmentation
            ``j`` of column ``k`` whose own ``random_state`` is still its
            default is seeded from ``SeedSequence([random_state, k, j])``;
            one given another seed keeps it. ``None`` leaves each
            augmentation its own seed.
        encoders_: The fitted encoder of each column of items, by the
            column's name, in the order of ``columns``.
        augmentations_: The fitted augmentations of each column of items,
            a list in their order, by the column's name.
        mean_: The mean of each column of the embeddings, side by side,
            shape (n_embedding_columns,).
        scale_: The standard deviation of each of those columns, of that
            shape.
        weights_: The factor of each embedding, whose square multiplies
            the variance of its components when they are ranked, shape
            (n_item_columns,).
        components_: The components kept, each zero outside its own
            embedding, shape (n_columns_kept, n_embedding_columns), or
            ``None`` when the embeddings pass unchanged.
        offset_: What is subtracted from the projection, shape
            (n_columns_kept,), or ``None``.
        blocks_: The position in ``columns`` of the column of items that
            each kept component belongs to, shape (n_columns_kept,), or
            ``None``.
        network_: The adaptive reducer's network: for ``"mlp"``, float64
            arrays by name (``"components"``, ``"hidden_weight"``,
            ``"hidden_bias"``, ``"output_weight"`` and ``"output_bias"``);
            for a module, a trained copy of it; ``None`` for ``"pca"``.
        history_: Set when a pipeline trains the reducer: the mean NLL, in
            nats, of the training queries (``"train_nll"``) and of the
            validation queries (``"validation_nll"``) before each step's
            update, one row per step from 0.
        best_step_: Set with ``history_``: the kept step, the row of
            ``history_`` with the lowest ``"validation_nll"``; 0 if none
            is finite.
        remainder_: The fitted transformer of the other columns, or
            ``remainder`` itself for ``"passthrough"`` and ``"drop"``.
    """

    # Set by fit; declared for the type checker only.
    network_: Any

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        columns: Mapping[Hashable, Any],
        *,
        n_columns: int | None = 120,
        weights: str | Sequence[float] | None = "equal",
        reducer: Any = "pca",
        remainder: Any = "passthrough",
        random_state: int | None = None,
    ) -> None:
        self.columns = columns
        self.n_columns = n_columns
        self.weights = weights
        self.reducer = reducer
        self.remainder = remainder
        self.random_state = random_state

    # To a tensor of a table; a path of gradients starts inside, at the
    # leaves of :meth:`_leaves`.
    gradient_kind = "barrier"
    barrier_reason = "a Tabularizer takes a table of items, not their columns"

    @property
    def trains_through_model(self) -> bool:
        """Whether a pipeline trains this step through its model.

        :meth:`lazy.pipeline.LazyPipeline.fit` calls
        :meth:`_fit_through` on a step for which this is true.
        """
        return self._settings() is not None

    def _settings(self) -> _adaptive.AdaptiveReducer | None:
        """The settings of an adaptive ``reducer``; ``None`` for ``"pca"``.

        Raises:
            ValueError: If ``reducer`` is none of the values documented.
        """
        reducer = self.reducer
        if isinstance(reducer, _adaptive.AdaptiveReducer):
            return reducer
        if _adaptive.is_module(reducer):
            return _adaptive.AdaptiveReducer(network=reducer)
        if isinstance(reducer, str) and reducer in ("pca", "mlp"):
            return _adaptive.AdaptiveReducer() if reducer == "mlp" else None
        raise ValueError(
            "reducer must be 'pca', 'mlp', a torch.nn.Module or an"
            f" AdaptiveReducer: {reducer=}"
        )

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
                integer or ``None``, ``weights``, ``reducer`` or
                ``remainder`` is not valid, or the embeddings to reduce
                have fewer than two rows with an item that is not missing.
        """
        self.fit_transform(X, y)
        # Only a pipeline's training uses it.
        self.__dict__.pop("_context", None)
        return self

    def fit_transform(self, X: Any, y: Any = None) -> pd.DataFrame:  # noqa: GS030 - scikit-learn's X, y.
        """Fits as :meth:`fit` does, and returns the table it encoded.

        If the step trains through the model, it keeps the embeddings of
        ``X`` for :meth:`_fit_through`, which takes them; a fit, a clone
        or a pickle drops them.

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
        self._reset()
        if not isinstance(self.columns, Mapping):
            raise TypeError(
                "columns must map a column to a modality, an encoder name or"
                f" an encoder; got {type(self.columns).__name__}"
            )
        if not self.columns:
            raise ValueError("columns must name at least one column")
        if self.n_columns is not None:
            _base.positive_integer("n_columns", self.n_columns)
        if self.random_state is not None:
            validation.check_scalar(
                self.random_state, "random_state", numbers.Integral, min_val=0
            )
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
        encoders, augmentations = {}, {}
        blocks = []
        for index, (column, spec) in enumerate(self.columns.items()):
            steps, encoders[column] = _steps_of(
                column, spec, self.random_state, index
            )
            items = _read(
                encoders[column], steps, frame.iloc[:, positions[index]]
            )
            for step in steps:
                items = step.fit_transform(items)
            augmentations[column] = steps
            blocks.append(encoders[column].fit_transform(items))
        fitted = self._fit_reduction(blocks)
        rest = frame.drop(columns=frame.columns[positions])
        remainder = self.remainder
        if not plain:
            # A transformer has nothing to fit when no column is left.
            remainder = (
                sklearn_base.clone(self.remainder).fit(rest)
                if rest.shape[1]
                else "drop"
            )
        validation.validate_data(self, X, skip_check_array=True)
        self.encoders_ = encoders
        self.augmentations_ = augmentations
        self.remainder_ = remainder
        self._positions = positions
        self.__dict__.update(fitted)
        try:
            self._rest_names = self._names_of(rest)
            reduced = self._reduced(blocks)
            self._n_reduced = int(reduced.shape[1])
            if self.n_columns is not None and self._n_reduced > self.n_columns:
                raise ValueError(
                    f"the reducer's module gives {self._n_reduced} columns,"
                    f" more than n_columns={self.n_columns}; set n_columns="
                    f"{self._n_reduced} or None"
                )
            table = self._table(reduced, rest)
        except Exception:
            self._reset()
            raise
        if self.trains_through_model:
            # For _fit_through, which then encodes no item of X again.
            self._context = (X, blocks, rest)
        return table

    def _reset(self) -> None:
        """Removes what a fit set: a failed fit leaves it unfitted."""
        for name in _FITTED:
            self.__dict__.pop(name, None)

    def __getstate__(self) -> dict[str, Any]:
        """The state to pickle, without the embeddings kept for training.

        Returns:
            The instance's attributes, but the context that
            :meth:`fit_transform` keeps for :meth:`_fit_through`.
        """
        state = dict(super().__getstate__())
        state.pop("_context", None)
        return state

    def _fit_reduction(
        self, blocks: list[_typing.FloatArray]
    ) -> dict[str, Any]:
        """The fitted standardization, the components kept and the network.

        Each embedding is standardized and gets principal components of
        its own, fitted on its rows that have an item. All the components
        are then ranked together by the variance they explain, times
        their embedding's weight, and the best ``n_columns`` are kept; an
        embedding keeps at least its first, if there is room.

        Args:
            blocks: Each column's embeddings of the context, shape
                (n_rows, width), a missing item's row all ``NaN``.

        Returns:
            The fitted attributes ``mean_``, ``scale_``, ``weights_``,
            ``components_``, ``offset_``, ``blocks_`` and ``network_``, by
            name.

        Raises:
            ValueError: If ``weights`` or ``reducer`` is not valid, or an
                embedding to reduce has fewer than two rows with an item.
        """
        present = [~np.isnan(block).any(axis=1) for block in blocks]
        widths = [block.shape[1] for block in blocks]
        scalers = [
            preprocessing.StandardScaler().fit(block[rows])
            if rows.any()
            else None
            for block, rows in zip(blocks, present, strict=True)
        ]
        # A column that is constant has no variance to share.
        varying = [
            0 if scaler is None else int((scaler.var_ > 0).sum())
            for scaler in scalers
        ]
        factors = _factors(self.weights, varying)
        fitted: dict[str, Any] = {
            "mean_": np.concatenate(
                [
                    np.zeros(width) if scaler is None else scaler.mean_
                    for scaler, width in zip(scalers, widths, strict=True)
                ]
            ),
            "scale_": np.concatenate(
                [
                    np.ones(width) if scaler is None else scaler.scale_
                    for scaler, width in zip(scalers, widths, strict=True)
                ]
            ),
            "weights_": factors,
            "components_": None,
            "offset_": None,
            "blocks_": None,
            "network_": None,
        }
        adaptive = self._settings()
        if adaptive is not None:
            adaptive.check()
        if adaptive is not None and _adaptive.is_module(adaptive.network):
            fitted["network_"] = copy.deepcopy(adaptive.network).eval()
            return fitted
        target = sum(widths) if self.n_columns is None else self.n_columns
        if adaptive is None and sum(widths) <= target:
            return fitted
        starts = np.cumsum([0, *widths])
        fits: list[tuple[_typing.FloatArray, _typing.FloatArray]] = []
        scores: list[float] = []
        owners: list[int] = []
        numbers: list[int] = []
        for index, (block, rows) in enumerate(
            zip(blocks, present, strict=True)
        ):
            scaler = scalers[index]
            if scaler is None or rows.sum() < 2:
                raise ValueError(
                    f"an embedding to reduce has {rows.sum()} rows with an"
                    " item that is not missing; it needs at least 2"
                )
            mean, components, variances = _principal_components(
                scaler.transform(block[rows]),
                # The centred rows span one direction fewer than they are.
                # No embedding can keep more than all the columns asked.
                min(widths[index], int(rows.sum()) - 1, target),
            )
            # A copy, which frees the rest of the decomposition.
            fits.append((mean, components.copy()))
            for number in range(len(components)):
                # An embedding's first component is kept before any other.
                first = number == 0 and target >= len(blocks)
                value = variances[number] * factors[index] ** 2
                scores.append(np.inf if first else value)
                owners.append(index)
                numbers.append(number)
        # Stable: components that tie keep their order.
        kept = np.sort(np.argsort(-np.array(scores), kind="stable")[:target])
        if len(kept) < min(target, sum(widths)):
            _warnings.warn(
                f"the embeddings reduced to n_columns={target} have too few"
                f" rows with an item for that many components, so they keep"
                f" {len(kept)} columns",
                _warnings.LazyWarning,
            )
        # Only the kept components get a row over all the embeddings.
        rows_of = np.zeros((len(kept), starts[-1]))
        offsets = np.empty(len(kept))
        blocks_kept = np.array(owners)[kept]
        for row, (index, number) in enumerate(
            zip(blocks_kept, np.array(numbers)[kept], strict=True)
        ):
            mean, components = fits[index]
            rows_of[row, starts[index] : starts[index + 1]] = components[number]
            offsets[row] = mean @ components[number]
        fitted["components_"] = rows_of
        fitted["offset_"] = offsets
        fitted["blocks_"] = blocks_kept
        if adaptive is not None:
            fitted["network_"] = adaptive.initial(
                fitted["components_"],
                len(blocks),
                np.random.default_rng(adaptive.random_state),
            )
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
        return self._table(self._reduced(blocks), rest)

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
        blocks = [
            self.encoders_[column].transform(
                self._augmented(column, frame.iloc[:, position])
            )
            for column, position in zip(
                self.encoders_, self._positions, strict=True
            )
        ]
        return blocks, frame.drop(columns=frame.columns[self._positions])

    def _augmented(self, column: Hashable, items: Any) -> Any:
        """A column's items after its augmentations, in order."""
        steps = self.augmentations_[column]
        items = _read(self.encoders_[column], steps, items)
        for step in steps:
            items = step.transform(items)
        return items

    def _reduced(self, blocks: Sequence[Any]) -> Any:
        """The reduced columns of some rows' embeddings.

        One implementation for numpy arrays and for torch tensors, which
        keep their gradients. Each reduced column belongs to one
        embedding, and is ``NaN`` in a row whose item is missing there; an
        adaptive network reads, beside the components, a mask of which
        items a row has.

        Args:
            blocks: Each column's embeddings, in the order of ``columns``,
                arrays of one kind and of shape (n_rows, width).

        Returns:
            The embeddings side by side if they pass unchanged, else their
            components, or what the fitted network makes of them, of that
            kind and shape (n_rows, n_columns_kept).
        """
        xp = array_api_compat.array_namespace(*blocks)
        wide = xp.concat(list(blocks), axis=1)
        if self.components_ is None and self.network_ is None:
            return wide
        scaled, absent = self._scaled(blocks, wide)
        return self._mapped(scaled, absent, _owners(blocks), self.network_)

    def _scaled(self, blocks: Sequence[Any], wide: Any) -> tuple[Any, Any]:
        """Some rows' embeddings less ``mean_`` and over ``scale_``.

        Args:
            blocks: As :meth:`_reduced` takes them.
            wide: The same embeddings side by side, shape (n_rows,
                n_embedding_columns).

        Returns:
            A tuple (scaled, absent): the scaled embeddings, of the kind
            and shape of ``wide``; and which items a row lacks, bool of
            that kind and of shape (n_rows, n_item_columns).
        """
        xp = array_api_compat.array_namespace(wide)
        absent = xp.stack(
            [xp.any(xp.isnan(block), axis=1) for block in blocks], axis=1
        )
        scaled = (wide - _constant(self.mean_, wide)) / _constant(
            self.scale_, wide
        )
        return scaled, absent

    def _mapped(
        self,
        scaled: Any,
        absent: Any,
        owner: _typing.IntArray,
        network: Any,
        dropped: Any = None,
    ) -> Any:
        """The reduced columns of some rows' scaled embeddings.

        The part of :meth:`_reduced` after the scaling, which each step of
        training repeats on the context's embeddings, scaled once.

        Args:
            scaled: The embeddings side by side, less ``mean_`` and over
                ``scale_``, a numpy array or a torch tensor of shape
                (n_rows, n_embedding_columns).
            absent: Which items a row lacks, bool of that kind and of shape
                (n_rows, n_item_columns).
            owner: The column of items of each embedding column, shape
                (n_embedding_columns,).
            network: The adaptive network to map with, of that kind, or
                ``None`` for the components alone.
            dropped: For training, which items to treat as missing, bool
                of shape (n_rows, n_item_columns). A missing item's
                columns are then 0, their mean, since a differentiable
                pass takes no ``NaN``; ``None`` leaves them ``NaN``.

        Returns:
            The components, or what the network makes of them, of the kind
            of ``scaled`` and of shape (n_rows, n_columns_kept).
        """
        xp = array_api_compat.array_namespace(scaled)
        if dropped is not None:
            absent = absent | dropped
        zero, one = (
            _constant(np.zeros(()), scaled),
            _constant(np.ones(()), scaled),
        )
        standardized = xp.where(absent[:, owner], zero, scaled)
        mask = xp.where(absent, zero, one)
        if _adaptive.is_module(network):
            reduced = _through_module(
                network, xp.concat([standardized, mask], axis=1)
            )
            gone = xp.all(absent, axis=1)[:, None]
        else:
            components = _constant(self.components_, scaled)
            if network is not None:
                # The trained loadings, each within its own embedding.
                support = _constant(
                    (self.components_ != 0).astype(float), scaled
                )
                components = _constant(network["components"], scaled) * support
            gone = absent[:, self.blocks_]
            reduced = xp.where(
                gone,
                zero,
                standardized @ components.T - _constant(self.offset_, scaled),
            )
            if network is not None:
                weights = {
                    name: _constant(value, scaled)
                    for name, value in network.items()
                }
                reduced = _adaptive.residual(weights, reduced, mask, xp)
        fill = (
            _constant(np.full((), np.nan), scaled) if dropped is None else zero
        )
        return xp.where(gone, fill, reduced)

    def _fit_through(  # noqa: GS030 - scikit-learn's X, y.
        self, X: Any, y: Any, later: Sequence[Any], model: Any
    ) -> pd.DataFrame:
        """Trains the adaptive reducer through the model that follows.

        :meth:`lazy.pipeline.LazyPipeline.fit` calls it once this step and
        the later ones are fitted. If ``X`` is the table that
        :meth:`fit_transform` was given, its embeddings, which that kept,
        are taken and no item is encoded again; otherwise ``X`` is
        encoded. A clone of ``model`` is fitted on the support rows, and
        Adam lowers the NLL of the training queries through the forward of
        each later step and the clone's differentiable pass
        (:class:`~lazy.encoders.AdaptiveReducer`). It sets ``network_``,
        ``history_`` and ``best_step_``.

        Args:
            X: The context as this step takes it, with the columns of
                ``fit``.
            y: Its finite targets, shape (n_rows,).
            later: The fitted steps between this one and the model, in
                order; each needs a forward.
            model: The pipeline's model, which is cloned.

        Returns:
            The trained step's table of ``X``, on which the later steps
            and the model are fitted next.

        Raises:
            ValueError: If a column that passes through is not numeric, a
                feature is missing or infinite, a later step is a barrier,
                the model has no differentiable forward pass, or a part of
                the split has no row.
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        reducer = self._settings()
        if reducer is None:
            raise ValueError("reducer='pca' trains nothing")
        y = np.asarray(y, dtype=np.float64)
        context = self.__dict__.pop("_context", None)
        if context is not None and context[0] is X:
            blocks, rest = context[1:]
        else:
            blocks, rest = self._encoded(X)
        del context
        kept = self._kept(rest)
        tensors = [torch.as_tensor(block) for block in blocks]
        others = None
        if kept is not None:
            others = torch.tensor(_numeric_values(kept, _adaptive.LABEL))
        rng = np.random.default_rng(reducer.random_state)
        parts = reducer.split(len(y), rng)
        if _adaptive.is_module(self.network_):
            network = self.network_
            weights = list(network.parameters())

            def snapshot() -> Any:
                """A copy of the module's weights."""
                return copy.deepcopy(network.state_dict())

            def restore(kept_network: Any) -> None:
                """Sets the module's weights to a snapshot."""
                network.load_state_dict(kept_network)
        else:
            network = {
                name: torch.tensor(value, requires_grad=True)
                for name, value in self.network_.items()
            }
            weights = list(network.values())

            def snapshot() -> Any:
                """The network's weights, as float64 arrays."""
                return {
                    name: value.detach().numpy().copy()
                    for name, value in network.items()
                }

            def restore(kept_network: Any) -> None:
                """Keeps a snapshot as the fitted network."""
                self.network_ = kept_network

        generator = torch.Generator().manual_seed(int(rng.integers(2**31 - 1)))
        n_blocks = len(blocks)
        # Scaled once: a step changes only the dropped items and the network.
        scaled, native = self._scaled(tensors, torch.cat(tensors, dim=1))
        owner = _owners(blocks)
        del blocks, tensors
        drops = n_blocks > 1 and reducer.block_dropout > 0
        # Without dropout or a module, the plain features are the training
        # pass's, which train asks for next at the same weights.
        stochastic = drops or _adaptive.is_module(network)
        last: list[Any] = []

        def features(drop: bool) -> Any:
            """Every row's features as the model reads them, on the graph.

            With ``drop``, each item is treated as missing with the
            probability ``block_dropout``; a row that would lose every
            item it has loses none. A module is in training mode only
            then.
            """
            if last and not drop:
                return last.pop().detach()
            if _adaptive.is_module(network):
                network.train(drop)
            dropped = torch.zeros((len(y), n_blocks), dtype=torch.bool)
            if drop and drops:
                draws = torch.rand((len(y), n_blocks), generator=generator)
                dropped = draws < reducer.block_dropout
                dropped[(dropped | native).all(dim=1)] = False
            columns = self._mapped(scaled, native, owner, network, dropped)
            if others is not None:
                columns = torch.cat([columns, others], dim=1)
            for step in later:
                columns = _steps.forward(step, columns)
            if drop and not stochastic:
                last[:] = [columns]
            return columns

        label = _adaptive.LABEL
        with torch.no_grad():
            start = features(False).numpy()
        if not np.isfinite(start).all():
            raise ValueError(
                f"{label} needs the columns that pass through the"
                " Tabularizer without missing or infinite values"
            )
        support = parts[0]
        try:
            fitted = sklearn_base.clone(model).fit(start[support], y[support])
            kept_network, history, self.best_step_ = _adaptive.train(
                reducer,
                weights,
                features,
                y,
                parts,
                differentiable.backend(fitted, label),
                snapshot,
            )
        finally:
            if _adaptive.is_module(self.network_):
                self.network_.eval()
        restore(kept_network)
        self.history_ = pd.DataFrame(
            history, columns=["train_nll", "validation_nll"]
        ).rename_axis("step")
        if (
            reducer.n_steps
            and not np.isfinite(self.history_["validation_nll"]).any()
        ):
            _warnings.warn(
                "no step had a finite validation NLL, so the reducer stays"
                " as it started: a validation query's target is outside the"
                " model's buckets, which the support rows' targets set;"
                " another random_state splits the rows differently",
                _warnings.LazyWarning,
            )
        last.clear()
        with torch.no_grad():
            reduced = self._mapped(scaled, native, owner, self.network_)
        return self._table(reduced.numpy(), rest)

    def _leaves(  # noqa: GS030 - scikit-learn's X.
        self, X: Any, X_context: Any, items: bool, name: str
    ) -> tuple[list[_steps.Leaf], list[bool], list[_steps.Barrier]]:
        """What a gradient method differentiates of two tables.

        A column of items is a leaf of its own: the items themselves if
        ``items`` is set, its encoder has a forward and the items have one
        shape, and its embedding otherwise. The other columns, which must
        be numeric, are one last leaf, unless ``remainder`` drops them.

        Args:
            X: The rows to explain, with the columns of ``fit``.
            X_context: The context rows.
            items: Whether to go through the encoders to the items.
            name: The name of this step in its pipeline, for the barriers.

        Returns:
            A tuple (leaves, through, barriers): the leaves; whether each
            leaf goes through its encoder, in the order of ``columns``,
            and then whether the last leaf goes through the transformer
            of the other columns; and each encoder (if ``items`` is set)
            or transformer that the gradient cannot cross, with the
            reason.

        Raises:
            ValueError: If a column that passes through is not numeric.
        """
        frames = [_frame(X), _frame(X_context)]
        leaves, through, barriers = [], [], []
        for (column, encoder), position in zip(
            self.encoders_.items(), self._positions, strict=True
        ):
            given = [frame.iloc[:, position] for frame in frames]
            label = f"{name}[{column!r}]"
            found = _steps.barrier(label, encoder)
            for step in self.augmentations_[column]:
                found = found or _steps.barrier(label, step)
            stacked = None
            if items and found is None:
                stacked = [encoder._stacked(part) for part in given]  # noqa: SLF001 - the encoders' contract.
                if (
                    stacked[0] is None
                    or stacked[1] is None
                    or stacked[0].shape[1:] != stacked[1].shape[1:]
                ):
                    found = _steps.Barrier(
                        label, "its items do not all have one shape"
                    )
            through.append(found is None and items)
            if stacked is not None and found is None:
                leaves.append(
                    _steps.Leaf((str(column),), stacked[0], stacked[1], False)
                )
                continue
            if items and found is not None:
                barriers.append(found)
            leaves.append(
                _steps.Leaf(
                    tuple(
                        f"{column}__{feature}"
                        for feature in encoder.get_feature_names_out()
                    ),
                    encoder.transform(self._augmented(column, given[0])),
                    encoder.transform(self._augmented(column, given[1])),
                )
            )
        if not (isinstance(self.remainder_, str) and self.remainder_ == "drop"):
            rest = [
                frame.drop(columns=frame.columns[self._positions])
                for frame in frames
            ]
            closed = _steps.barrier(f"{name}[remainder]", self.remainder_)
            # Whether the leaf goes through the remainder's transformer.
            through.append(closed is None)
            if closed is not None:
                # Attribution ends at what the transformer gives.
                barriers.append(closed)
                names = self._remainder_names()
                rest = [
                    pd.DataFrame(self.remainder_.transform(part)).set_axis(
                        names, axis=1
                    )
                    for part in rest
                ]
            leaves.append(
                _steps.Leaf(
                    tuple(str(column) for column in rest[0].columns),
                    _numeric_values(rest[0], "a gradient"),
                    rest[1].to_numpy(dtype=np.float64),
                )
            )
        return leaves, through, barriers

    def _forward_leaves(
        self, tensors: Sequence[Any], through: Sequence[bool]
    ) -> Any:
        """The output columns of some rows, from their leaves, on the graph.

        Args:
            tensors: One torch tensor per leaf of :meth:`_leaves`.
            through: Whether each leaf of a column of items goes through
                its encoder, as :meth:`_leaves` returns it.

        Returns:
            What :meth:`transform` gives for the same rows, to float
            rounding, a torch tensor of shape (n_rows, n_features_out).
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        blocks = []
        for column, tensor, crossed in zip(
            self.encoders_, tensors, through, strict=False
        ):
            if crossed:
                for step in self.augmentations_[column]:
                    tensor = _steps.forward(step, tensor)
                tensor = self.encoders_[column].forward(tensor)
            blocks.append(tensor)
        parts = [self._reduced(blocks)]
        if len(tensors) > len(blocks):
            kept = tensors[-1]
            parts.append(
                _steps.forward(self.remainder_, kept) if through[-1] else kept
            )
        return torch.cat(parts, dim=1)

    def _table(
        self, reduced: _typing.FloatArray, rest: pd.DataFrame
    ) -> pd.DataFrame:
        """The output table of some rows.

        Args:
            reduced: Their reduced columns, as :meth:`_reduced` gives them.
            rest: The other columns of the rows.

        Returns:
            The reduced embeddings, then the columns kept, with the index
            of ``rest``.
        """
        names = self.get_feature_names_out()
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
            ``<column>__pc0`` onwards for each column's components
            (``<column>__z0`` onwards from the ``"mlp"`` reducer, and
            ``embedding__z0`` onwards from a module), or
            ``<column>__<name>`` with each encoder's names
            (``notes__text_hashing_3``) for embeddings that pass unchanged;
            then ``remainder__<column>`` for each column kept.

        Raises:
            sklearn.exceptions.NotFittedError: If it is not fitted.
        """
        del input_features  # Unused: the fitted table names the columns.
        validation.check_is_fitted(self, "encoders_")
        names: list[str] = []
        if self.components_ is None and self.network_ is None:
            for column, encoder in self.encoders_.items():
                names.extend(
                    f"{column}__{name}"
                    for name in encoder.get_feature_names_out()
                )
        elif self.components_ is None:
            names = [f"embedding__z{i}" for i in range(self._n_reduced)]
        else:
            kind = "pc" if self.network_ is None else "z"
            columns = list(self.encoders_)
            counts = dict.fromkeys(columns, 0)
            for block in self.blocks_:
                column = columns[block]
                names.append(f"{column}__{kind}{counts[column]}")
                counts[column] += 1
        return np.array([*names, *self._remainder_names()], dtype=object)

    def _remainder_names(self) -> list[str]:
        """``remainder__<column>`` for each column kept, in order."""
        return [f"remainder__{name}" for name in self._rest_names]

    def _names_of(self, rest: pd.DataFrame) -> list[Any]:
        """The names of the columns kept of the context's other columns.

        They are the columns' own labels if they pass through, the names
        the transformer gives if it has ``get_feature_names_out``, and
        else those of the frame it makes.
        """
        if hasattr(self.remainder_, "get_feature_names_out"):
            return list(self.remainder_.get_feature_names_out())
        kept = self._kept(rest)
        return [] if kept is None else list(kept.columns)


def _principal_components(
    standardized: _typing.FloatArray, n_components: int
) -> tuple[_typing.FloatArray, _typing.FloatArray, _typing.FloatArray]:
    """The leading principal components of standardized embedding rows.

    This is scikit-learn's ``PCA(svd_solver="full")``: the same SVD, sign
    convention and variances. LAPACK's default driver, ``gesdd``, fails to
    converge on some matrices and platforms; then the SVD runs again with the
    ``gesvd`` driver.

    Args:
        standardized: Standardized rows of one embedding, shape
            ``(n_rows, n_columns)``.
        n_components: Number of components to keep, at most
            ``min(n_rows - 1, n_columns)``.

    Returns:
        The column means, shape ``(n_columns,)``; the components, shape
        ``(n_components, n_columns)``; and the variance each explains, shape
        ``(n_components,)``.

    Raises:
        numpy.linalg.LinAlgError: If neither driver converges.
    """
    mean = standardized.mean(axis=0)
    centred = standardized - mean
    try:
        u, singular, vt = linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        u, singular, vt = linalg.svd(
            centred, full_matrices=False, lapack_driver="gesvd"
        )
    _, vt = extmath.svd_flip(u, vt, u_based_decision=False)
    variances = singular**2 / (len(standardized) - 1)
    return mean, vt[:n_components], variances[:n_components]


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
        if weights != "equal":
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


def _numeric_values(frame: pd.DataFrame, label: str) -> _typing.FloatArray:
    """The values of columns that pass through, which must be numeric.

    Args:
        frame: The columns that pass through the Tabularizer.
        label: What needs them numeric, to start the error.

    Returns:
        The values, float64, shape (n_rows, n_columns).

    Raises:
        ValueError: If a column is not numeric.
    """
    other = frame.select_dtypes(exclude="number").columns
    if len(other):
        raise ValueError(
            f"{label} needs the columns that pass through the Tabularizer"
            f" to be numeric; these are not: {list(other)}"
        )
    return frame.to_numpy(dtype=np.float64)


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


def _read(encoder: _base.BaseEncoder, steps: list[Any], items: Any) -> Any:
    """A column's items as its augmentations take them.

    The encoder reads each item first (``_read_item``), so that a file or
    a Pillow image reaches an augmentation as the array the encoder would
    read, in the encoder's own mode. A column without augmentations is
    left to the encoder.

    Args:
        encoder: The column's encoder.
        steps: The column's augmentations.
        items: The column, as given.

    Returns:
        ``items`` itself without augmentations, else a list of the items
        as read, a missing one as it is.
    """
    if not steps:
        return items
    return [
        item if _base._is_missing(item) else encoder._read_item(item)  # noqa: SLF001 - the encoders' contract.
        for item in _base._as_items(items)  # noqa: SLF001 - as above.
    ]


def _steps_of(
    column: Hashable, spec: object, random_state: int | None, index: int
) -> tuple[list[Any], _base.BaseEncoder]:
    """New augmentations and a new encoder for one entry of ``columns``.

    Args:
        column: The column's name, for the errors.
        spec: The entry: what names an encoder, or a list of augmentations
            that ends with it.
        random_state: The Tabularizer's seed, or ``None``.
        index: The position of the entry in ``columns``.

    Returns:
        A tuple (augmentations, encoder), all unfitted clones; augmentation
        ``j`` is seeded from ``member_seed(random_state, index, j)`` if
        ``random_state`` is given and its own seed is still its default.

    Raises:
        TypeError: If an entry of a list but the last is no transformer,
            or as :func:`_encoder` raises.
        UnknownNameError: As :func:`_encoder` raises.
        ValueError: If a list is empty.
    """
    if not isinstance(spec, (list, tuple)):
        return [], _encoder(column, spec)
    if not spec:
        raise ValueError(f"columns[{column!r}] must not be an empty list")
    steps = []
    for position, given in enumerate(spec[:-1]):
        if not hasattr(given, "fit_transform"):
            raise TypeError(
                f"columns[{column!r}] must list augmentations, then what"
                f" encodes the column; got {type(given).__name__} first"
            )
        step = sklearn_base.clone(given)
        if random_state is not None:
            _members.seed_if_default(
                step, _members.member_seed(random_state, index, position)
            )
        steps.append(step)
    return steps, _encoder(column, spec[-1])


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


def _constant(values: Any, like: Any) -> Any:
    """A fitted array, of the kind, type and device of ``like``.

    A torch tensor is returned as it is: a weight in training, which a copy
    would cut off the graph.
    """
    if array_api_compat.is_torch_array(values):
        return values
    xp = array_api_compat.array_namespace(like)
    return xp.asarray(
        values, dtype=like.dtype, device=array_api_compat.device(like)
    )


def _owners(blocks: Sequence[Any]) -> _typing.IntArray:
    """The position of the block of each column of blocks side by side."""
    return np.repeat(
        np.arange(len(blocks)), [block.shape[1] for block in blocks]
    )


def _through_module(module: Any, standardized: Any) -> Any:
    """A torch module's columns of standardized embeddings.

    Args:
        module: The module, which maps (n_rows, n_embedding_columns) to
            (n_rows, n_columns).
        standardized: The embeddings, a numpy array or a torch tensor,
            which keeps its gradient.

    Returns:
        The module's output, of the kind and type of ``standardized``.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    weight = next(iter(module.parameters()), None)
    dtype = standardized.dtype if weight is None else weight.dtype
    if array_api_compat.is_torch_array(standardized):
        return module(standardized.to(dtype)).to(standardized.dtype)
    with torch.no_grad():
        out = module(torch.as_tensor(standardized, dtype=dtype))
    return out.double().numpy()
