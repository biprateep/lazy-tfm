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
from typing import Any

import array_api_compat
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base as sklearn_base
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.utils import validation

from lazy import _steps
from lazy import _typing
from lazy import _warnings
from lazy.encoders import _adaptive
from lazy.encoders import _base
from lazy.encoders import registry
from lazy.models import _differentiable as differentiable

__all__ = ["REDUCERS", "WEIGHTS", "Tabularizer"]

WEIGHTS: tuple[str, ...] = ("equal",)
"""The named values of ``weights``."""

REDUCERS: tuple[str, ...] = ("pca", "mlp")
"""The named values of ``reducer``."""

_FITTED = (
    "encoders_",
    "augmentations_",
    "remainder_",
    "mean_",
    "scale_",
    "components_",
    "offset_",
    "weights_",
    "network_",
    "history_",
    "best_step_",
    "_positions",
    "_n_reduced",
    "n_features_in_",
    "feature_names_in_",
)


class Tabularizer(sklearn_base.TransformerMixin, sklearn_base.BaseEstimator):
    r"""Encodes the columns of items, reduces them together, keeps the others.

    Each column named in ``columns`` is encoded by its encoder, which a
    modality (``"image"``, ``"text"`` or ``"time"``, through
    :data:`~lazy.encoders.DEFAULT_ENCODERS`), the name of a registered
    encoder, or an encoder object names. A list gives the steps of
    :mod:`lazy.augment` that transform the column's items first, then its
    encoder, as ``[augment.Dihedral(), "image"]``. The embeddings of all
    the columns are then reduced together to ``n_columns`` columns,
    whatever the number of columns of items: each column of each embedding
    is standardized,
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
        weights: The share of each embedding in the reduction: ``"equal"``
            gives every embedding the same total variance, 1, whatever its
            width; ``None`` leaves every standardized column at variance
            1, so that a wider embedding weighs more; a list gives each
            embedding its total variance, in the order of ``columns``.
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
            ``j`` of column ``k`` is seeded from ``SeedSequence(
            [random_state, k, j])``, whatever its own ``random_state``.
            ``None`` leaves each augmentation its own seed.
        encoders_: The fitted encoder of each column of items, by the
            column's name, in the order of ``columns``.
        augmentations_: The fitted augmentations of each column of items,
            a list in their order, by the column's name.
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
        columns: Mapping[Hashable, str | _base.BaseEncoder],
        *,
        n_columns: int | None = 30,
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

    @property
    def trains_through_model(self) -> bool:
        """Whether a pipeline trains this step through its model.

        :meth:`lazy.pipeline.LazyPipeline.fit` calls
        :meth:`_fit_through` on a step for which this is true.
        """
        return self._adaptive() is not None

    def _adaptive(self) -> _adaptive.AdaptiveReducer | None:
        """The settings of an adaptive ``reducer``; ``None`` for ``"pca"``.

        Raises:
            ValueError: If ``reducer`` is none of the values documented.
        """
        reducer = self.reducer
        if isinstance(reducer, _adaptive.AdaptiveReducer):
            return reducer
        if _adaptive.is_module(reducer):
            return _adaptive.AdaptiveReducer(network=reducer)
        if isinstance(reducer, str) and reducer in REDUCERS:
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
        encoders, augmentations = {}, {}
        blocks = []
        for index, (column, spec) in enumerate(self.columns.items()):
            steps, encoders[column] = _steps_of(
                column, spec, self.random_state, index
            )
            items = frame.iloc[:, positions[index]]
            for step in steps:
                items = step.fit_transform(items)
            augmentations[column] = steps
            blocks.append(encoders[column].fit_transform(items))
        fitted = self._fit_reduction(blocks)
        rest = frame.drop(columns=frame.columns[positions])
        remainder = self.remainder
        if not plain:
            remainder = sklearn_base.clone(self.remainder).fit(rest)
        validation.validate_data(self, X, skip_check_array=True)
        self.encoders_ = encoders
        self.augmentations_ = augmentations
        self.remainder_ = remainder
        self._positions = positions
        self.__dict__.update(fitted)
        self._n_reduced = int(self._reduced(blocks).shape[1])
        if self.n_columns is not None and self._n_reduced > self.n_columns:
            width = self._n_reduced
            del self.encoders_
            raise ValueError(
                f"the reducer's module gives {width} columns, more than"
                f" n_columns={self.n_columns}; set n_columns={width} or None"
            )
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
        adaptive = self._adaptive()
        if adaptive is not None:
            adaptive.check()
        target = sum(widths) if self.n_columns is None else self.n_columns
        fitted["network_"] = None
        if adaptive is not None and _adaptive.is_module(adaptive.network):
            fitted["network_"] = copy.deepcopy(adaptive.network).eval()
            return fitted
        if adaptive is None and sum(widths) <= target:
            return fitted
        target = min(target, sum(widths))
        wide = np.hstack(blocks)
        present = ~np.isnan(wide).all(axis=1)
        if present.sum() < 2:
            raise ValueError(
                f"the embeddings to reduce have {present.sum()} rows with an"
                " item that is not missing; they need at least 2"
            )
        # The centred rows span at most one direction fewer than they are.
        n_kept = min(target, int(present.sum()) - 1)
        if n_kept < target:
            _warnings.warn(
                f"the embeddings reduced to n_columns={target} have"
                f" {present.sum()} rows with an item that is not missing,"
                f" so they keep {n_kept} columns",
                _warnings.LazyWarning,
            )
        standardized = np.nan_to_num((wide[present] - mean) / scale, nan=0.0)
        pca = decomposition.PCA(n_kept, svd_solver="full").fit(standardized)
        fitted["components_"] = pca.components_
        fitted["offset_"] = pca.mean_ @ pca.components_.T
        if adaptive is not None:
            fitted["network_"] = adaptive.initial(
                pca.components_, np.random.default_rng(adaptive.random_state)
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
            self.encoders_[column].transform(
                self._augmented(column, frame.iloc[:, position])
            )
            for column, position in zip(self.columns, positions, strict=True)
        ]
        return blocks, frame.drop(columns=frame.columns[positions])

    def _augmented(self, column: Hashable, items: Any) -> Any:
        """A column's items after its augmentations, in order."""
        for step in self.augmentations_[column]:
            items = step.transform(items)
        return items

    def _reduced(self, blocks: Sequence[Any], network: Any = None) -> Any:
        """The reduced columns of some rows' embeddings.

        One implementation for numpy arrays and for torch tensors, which
        keep their gradients.

        Args:
            blocks: Each column's embeddings, in the order of ``columns``,
                arrays of one kind and of shape (n_rows, width).
            network: The adaptive network to map with, of the kind of
                ``blocks``; ``None`` for the fitted ``network_``.

        Returns:
            The embeddings side by side if they pass unchanged, else their
            components, or what the network makes of them, of that kind
            and shape (n_rows, n_columns_kept).
        """
        xp = array_api_compat.array_namespace(*blocks)
        wide = xp.concat(list(blocks), axis=1)
        network = self.network_ if network is None else network
        if self.components_ is None and network is None:
            return wide

        def constant(values: Any) -> Any:
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
        if _adaptive.is_module(network):
            reduced = _through_module(network, standardized)
        elif network is None:
            reduced = standardized @ constant(self.components_).T - constant(
                self.offset_
            )
        else:
            weights = {name: constant(value) for name, value in network.items()}
            projected = standardized @ weights["components"].T - constant(
                self.offset_
            )
            reduced = _adaptive.residual(weights, projected, xp)
        return xp.where(
            xp.all(missing, axis=1)[:, None],
            constant(np.full((), np.nan)),
            reduced,
        )

    def _fit_through(  # noqa: GS030 - scikit-learn's X, y.
        self, X: Any, y: Any, later: Sequence[Any], model: Any
    ) -> None:
        """Trains the adaptive reducer through the model that follows.

        :meth:`lazy.pipeline.LazyPipeline.fit` calls it once every step is
        fitted. A clone of ``model`` is fitted on the support rows, and
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

        Raises:
            ValueError: If a column that passes through is not numeric, a
                feature is missing or infinite, a later step is a barrier,
                the model has no differentiable forward pass, or a part of
                the split has no row.
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        reducer = self._adaptive()
        if reducer is None:
            raise ValueError("reducer='pca' trains nothing")
        y = np.asarray(y, dtype=np.float64)
        blocks, rest = self._encoded(X)
        kept = self._kept(rest)
        tensors = [torch.tensor(block) for block in blocks]
        if kept is not None:
            other = kept.select_dtypes(exclude="number").columns
            if len(other):
                raise ValueError(
                    "an adaptive reducer needs the columns that pass through"
                    f" the Tabularizer to be numeric; these are not:"
                    f" {list(other)}"
                )
        others = (
            None
            if kept is None
            else torch.tensor(kept.to_numpy(dtype=np.float64))
        )
        rng = np.random.default_rng(reducer.random_state)
        parts = reducer.split(len(y), rng)
        if _adaptive.is_module(self.network_):
            network = self.network_
            weights = list(network.parameters())
            network.train()

            def snapshot() -> Any:
                """A copy of the module's weights."""
                return copy.deepcopy(network.state_dict())
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

        def features() -> Any:
            """Every row's features as the model reads them, on the graph."""
            columns = self._reduced(tensors, network)
            if others is not None:
                columns = torch.cat([columns, others], dim=1)
            for step in later:
                columns = _steps.forward(step, columns)
            return columns

        label = "an adaptive reducer"
        with torch.no_grad():
            start = features().numpy()
        if not np.isfinite(start).all():
            raise ValueError(
                f"{label} needs X without missing or infinite values"
            )
        support = parts[0]
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
        if _adaptive.is_module(self.network_):
            self.network_.load_state_dict(kept_network)
            self.network_.eval()
        else:
            self.network_ = kept_network
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
            leaf of a column of items goes through its encoder, in the
            order of ``columns``; and, if ``items`` is set, each encoder
            that the gradient cannot cross, with the reason.

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
                        f"{name}[{column!r}]",
                        "its items do not all have one shape",
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
            other = rest[0].select_dtypes(exclude="number").columns
            if len(other):
                raise ValueError(
                    "a gradient needs the columns that pass through the"
                    f" Tabularizer to be numeric; these are not: {list(other)}"
                )
            leaves.append(
                _steps.Leaf(
                    tuple(str(column) for column in rest[0].columns),
                    rest[0].to_numpy(dtype=np.float64),
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
            parts.append(_steps.forward(self.remainder_, tensors[-1]))
        return torch.cat(parts, dim=1)

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
            ``embedding__pc0`` onwards for the reduced columns
            (``embedding__z0`` onwards from an adaptive reducer), or
            ``<column>__<name>`` with each encoder's names
            (``notes__text_hashing_3``) for embeddings that pass unchanged;
            then ``remainder__<column>`` for each column kept.

        Raises:
            sklearn.exceptions.NotFittedError: If it is not fitted.
        """
        del input_features  # Unused: the fitted table names the columns.
        validation.check_is_fitted(self, "encoders_")
        if self.components_ is None and self.network_ is None:
            names: list[str] = []
            for column, encoder in self.encoders_.items():
                names.extend(
                    f"{column}__{name}"
                    for name in encoder.get_feature_names_out()
                )
        elif self.network_ is None:
            names = [f"embedding__pc{i}" for i in range(len(self.components_))]
        else:
            names = [f"embedding__z{i}" for i in range(self._n_reduced)]
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
        ``j`` is seeded from ``SeedSequence([random_state, index, j])`` if
        ``random_state`` is given and it takes a seed.

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
        if random_state is not None and "random_state" in step.get_params():
            seed = np.random.SeedSequence([random_state, index, position])
            step.set_params(random_state=int(seed.generate_state(1)[0]))
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
