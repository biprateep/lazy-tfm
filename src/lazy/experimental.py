# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Experimental estimators, which a release may change or remove.

Nothing in this module follows lazy's deprecation policy: a release may
change or remove any of it without a warning first.

:class:`ContextAdapter` learns a small network that maps some of a table's
columns to a few new ones, by the negative log-likelihood of a frozen lazy
model given the mapped context. Unlike the rest of lazy, it trains weights
(the network's, never the model's), so it is opt-in::

    adapter = experimental.ContextAdapter(
        lazy.LazyModel("tabpfn", transforms=("none",), outlier_threshold=None)
    )
    adapter.fit(X_train, y_train)
    pdfs = adapter.predict_proba(X_test)
    adapter.history_  # The training and validation NLL at each step.

The "Context adapter" page of the documentation
(``docs/guide/adapter.md``) defines the network and the loss.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import math
import numbers
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
from lazy import base
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _differentiable as differentiable
from lazy.models import _ensemble

__all__ = ["ContextAdapter"]

# The weights of the network that training updates; the standardization's
# "mean" and "scale" stay as fitted.
_TRAINED = (
    "components",
    "hidden_weight",
    "hidden_bias",
    "output_weight",
    "output_bias",
)

# Arrays of one kind, numpy or torch, by the names of ContextAdapter.network_.
_Network = Mapping[str, Any]


class ContextAdapter(base.BaseDensityRegressor):
    """A frozen model behind a network that it trains to map the columns.

    The network maps the ``columns`` of each row to ``n_components`` new
    columns: the row's first principal components, plus a residual
    multilayer perceptron (MLP) whose output layer starts at zero, so that
    the untrained network is principal component analysis (PCA). The
    model sees the new columns, then the other columns unchanged.

    ``fit`` splits the context once into support rows, training queries
    and validation queries. Each step of Adam lowers the mean negative
    log-likelihood (NLL) of the training queries under the model given
    the support rows, both mapped, through the model's differentiable
    forward pass; the model's own weights stay frozen. The step with the
    lowest validation NLL is kept, step 0 (PCA) included. A clone of
    ``model`` is then fitted on the whole context, mapped by the kept
    network, and answers every prediction.

    The network trains on the model's device, in float64, or in float32 on
    an Apple GPU, which has no float64.
    Each step runs the forward pass twice over the support rows, once with
    the training queries and their autograd graph and once with the
    validation queries, and ``fit`` fits the model twice besides: on the
    support rows, for the pass, and on the whole context.

    Args:
        model: A lazy model with a differentiable forward pass: a backend,
            or a :class:`~lazy.models.lazy_model.LazyModel`, set to
            ``transforms=("none",)`` and ``outlier_threshold=None``. It is
            cloned, never fitted itself.
        columns: The columns the network maps, as names (of ``X`` with
            column names) or positions; ``None`` for every column.
        n_components: The network's output columns; at most the number of
            ``columns`` and the rows of ``X``.
        hidden: The units of the MLP's one hidden layer, which applies
            ``tanh``.
        n_steps: The steps of Adam; 0 keeps PCA.
        learning_rate: Adam's learning rate.
        validation_fraction: The share of the context's rows held out as
            validation queries, in (0, 1).
        query_fraction: The share of them used as training queries, in
            (0, 1). The two fractions sum to less than 1; the other rows
            are the support rows.
        random_state: The seed of the split and of the MLP's initial
            weights, a non-negative integer, or ``None`` for fresh entropy.
        y_grid: Default output grid: a :class:`~lazy.grid.Grid`, bin
            centers, ``"native"``, or ``None`` for the fitted model's
            default grid.

    Attributes:
        fitted_model_: The clone of ``model`` fitted on the mapped context,
            which answers every prediction.
        network_: The kept network, as float64 arrays by name: ``"mean"``
            and ``"scale"``, the standardization, shape (n_mapped,);
            ``"components"``, shape (n_components, n_mapped);
            ``"hidden_weight"``, shape (hidden, n_components);
            ``"hidden_bias"``, shape (hidden,); ``"output_weight"``, shape
            (n_components, hidden); and ``"output_bias"``, shape
            (n_components,).
        is_mapped_: Which columns of ``X`` the network maps, shape
            (``n_features_in_``,).
        history_: The mean NLL, in nats, of the training queries
            (``"train_nll"``) and of the validation queries
            (``"validation_nll"``) before each step's update, one row per
            step from 0. Training stops early if the network's output
            stops being finite.
        best_step_: The kept step, the row of ``history_`` with the lowest
            ``"validation_nll"``; 0 if none is finite.
        native_grid_: The fitted model's native grid, or ``None``.

    Examples:
        >>> import lazy
        >>> model = lazy.LazyModel("tabpfn")
        >>> adapter = ContextAdapter(model, n_components=4)
        >>> adapter.get_params()["n_components"]
        4
        >>> adapter.name_
        'adapter(tabpfn:v3.5)'
    """

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        model: base.BaseDensityRegressor,
        *,
        columns: Sequence[Any] | None = None,
        n_components: int = 8,
        hidden: int = 64,
        n_steps: int = 200,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        query_fraction: float = 0.2,
        random_state: int | None = 0,
        y_grid: grid_lib.GridLike = None,
    ):
        self.model = model
        self.columns = columns
        self.n_components = n_components
        self.hidden = hidden
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.query_fraction = query_fraction
        self.random_state = random_state
        self.y_grid = y_grid

    def __sklearn_tags__(self) -> sklearn_base.Tags:
        """A regressor's tags, without ``allow_nan``: ``fit`` refuses it."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags

    @property
    def name_(self) -> str:
        """``"adapter(<model>)"``, the label that ``evaluate`` uses."""
        return f"adapter({self.model.name_})"

    def transform(self, X: Any) -> _typing.FloatArray:  # noqa: GS030 - scikit-learn's X.
        """The columns the fitted model sees: the new ones, then the others.

        Args:
            X: Features with the columns ``fit`` saw, shape
                (n_samples, n_features).

        Returns:
            The network's columns, then the columns it does not map, in
            their order, shape (n_samples, n_components + n_unmapped). A
            row with a missing value in a mapped column gets ``NaN`` in
            every new column.

        Raises:
            sklearn.exceptions.NotFittedError: If the adapter is not
                fitted.
        """
        validation.check_is_fitted(self)
        return self._adapted(self._check_features(X)).to_numpy()

    # -- fitting -----------------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: _typing.FloatArray) -> None:
        """Trains the network, then fits the model on the mapped context."""
        self._check_params()
        self.is_mapped_ = _mapped_columns(
            self.columns, X, "feature_names_in_" in vars(self)
        )
        values = self._check_context(X)
        mapped, unmapped = (
            values[:, self.is_mapped_],
            values[:, ~self.is_mapped_],
        )
        rng = np.random.default_rng(self.random_state)
        parts = self._split(len(X), rng)
        network = _initial_network(mapped, self.n_components, self.hidden, rng)
        self.network_, history, self.best_step_ = _train(
            self.model,
            network,
            mapped,
            unmapped,
            y,
            parts,
            self.n_steps,
            self.learning_rate,
        )
        self.history_ = pd.DataFrame(
            history, columns=["train_nll", "validation_nll"]
        ).rename_axis("step")
        if (
            self.n_steps
            and not np.isfinite(self.history_["validation_nll"]).any()
        ):
            _warnings.warn(
                "no step had a finite validation NLL, so the adapter keeps"
                " PCA: a validation query's target is outside the model's"
                " buckets, which the support rows' targets set; another"
                " random_state splits the rows differently",
                _warnings.LazyWarning,
            )
        self.fitted_model_ = sklearn_base.clone(self.model).fit(
            self._adapted(X).to_numpy(), y
        )
        self.native_grid_ = getattr(self.fitted_model_, "native_grid_", None)

    def _check_context(self, X: pd.DataFrame) -> _typing.FloatArray:  # noqa: GS030 - scikit-learn's X.
        """The context's values, once checked against the parameters.

        Args:
            X: Validated features, shape (n_rows, n_features).

        Returns:
            Their values, float64, shape (n_rows, n_features).

        Raises:
            ValueError: If a column is categorical, a value is missing or
                infinite, or ``n_components`` is above the mapped columns
                or the rows.
        """
        if self.is_categorical_.any():
            raise ValueError(
                "ContextAdapter needs numeric columns; X has categorical ones:"
                f" {list(X.columns[self.is_categorical_])}"
            )
        values = X.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                "ContextAdapter needs X without missing or infinite values"
            )
        limit = min(len(values), int(self.is_mapped_.sum()))
        if self.n_components > limit:
            raise ValueError(
                f"n_components must be at most {limit}, the number of mapped"
                " columns or of rows, whichever is fewer:"
                f" n_components={self.n_components}"
            )
        return values

    def _check_params(self) -> None:
        """Refuses a parameter that is not as the class documents it."""
        for name, low in (("n_components", 1), ("hidden", 1), ("n_steps", 0)):
            validation.check_scalar(
                getattr(self, name), name, numbers.Integral, min_val=low
            )
        validation.check_scalar(
            self.learning_rate,
            "learning_rate",
            numbers.Real,
            min_val=0.0,
            include_boundaries="neither",
        )
        for name in ("validation_fraction", "query_fraction"):
            validation.check_scalar(
                getattr(self, name),
                name,
                numbers.Real,
                min_val=0.0,
                max_val=1.0,
                include_boundaries="neither",
            )
        if self.random_state is not None:
            validation.check_scalar(
                self.random_state, "random_state", numbers.Integral, min_val=0
            )
        if self.validation_fraction + self.query_fraction >= 1:
            raise ValueError(
                "validation_fraction + query_fraction must be below 1, to"
                " leave support rows:"
                f" validation_fraction={self.validation_fraction},"
                f" query_fraction={self.query_fraction}"
            )

    def _split(
        self, n_rows: int, rng: np.random.Generator
    ) -> tuple[_typing.IntArray, _typing.IntArray, _typing.IntArray]:
        """The rows of the support, the training and the validation queries.

        Args:
            n_rows: The rows of the context.
            rng: The generator that draws the split.

        Returns:
            A tuple (support, queries, held_out) of sorted row positions,
            which partition ``range(n_rows)``.

        Raises:
            ValueError: If a part would have no row.
        """
        n_held = round(self.validation_fraction * n_rows)
        n_queries = round(self.query_fraction * n_rows)
        n_support = n_rows - n_held - n_queries
        if min(n_held, n_queries, n_support) < 1:
            raise ValueError(
                f"X has {n_rows} rows, which validation_fraction and"
                f" query_fraction split into {n_support} support rows,"
                f" {n_queries} training queries and {n_held} validation"
                " queries; each part needs a row"
            )
        order = rng.permutation(n_rows)
        held_out, queries, support = np.split(
            order, [n_held, n_held + n_queries]
        )
        return np.sort(support), np.sort(queries), np.sort(held_out)

    # -- predicting --------------------------------------------------------

    def _adapted(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validated features as the fitted model sees them; see transform."""
        values = X.to_numpy(dtype=np.float64)
        return pd.DataFrame(
            _features(
                self.network_,
                values[:, self.is_mapped_],
                values[:, ~self.is_mapped_],
            )
        )

    def _predict_pdf(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> _typing.FloatArray:
        """The fitted model's densities of the mapped rows."""
        return self.fitted_model_._predict_pdf(self._adapted(X), grid)  # noqa: SLF001 - delegate, as LazyModel does.

    def _density_blocks(
        self, X: pd.DataFrame, grid: grid_lib.Grid
    ) -> Iterator[tuple[slice, _typing.FloatArray]]:
        """The fitted model's densities a block at a time."""
        return self.fitted_model_._density_blocks(self._adapted(X), grid)  # noqa: SLF001 - delegate, as LazyModel does.

    def _predict_distribution(
        self, X: pd.DataFrame
    ) -> distributions.Distribution:
        """The fitted model's distributions of the mapped rows."""
        return self.fitted_model_._predict_distribution(self._adapted(X))  # noqa: SLF001 - delegate, as LazyModel does.

    def _distribution_chunks(
        self, X: pd.DataFrame
    ) -> Iterator[distributions.Distribution]:
        """The fitted model's distributions of the mapped rows, by chunk."""
        return self.fitted_model_._distribution_chunks(self._adapted(X))  # noqa: SLF001 - delegate, as LazyModel does.

    def _default_grid(self) -> grid_lib.Grid:
        """The constructor's grid, else the fitted model's default grid."""
        if self.y_grid is None:
            return self.fitted_model_.grid_
        return super()._default_grid()


def _mapped_columns(
    columns: Sequence[Any] | None,
    X: pd.DataFrame,
    named: bool,  # noqa: GS030 - scikit-learn's X.
) -> npt.NDArray[np.bool_]:
    """Which columns of ``X`` the network maps, as ``columns`` says.

    Args:
        columns: Column names (if ``named``) or positions, or ``None`` for
            every column.
        X: The features, shape (n_samples, n_features).
        named: Whether the caller named the columns.

    Returns:
        One boolean per column, shape (n_features,).

    Raises:
        ValueError: If an entry is neither a name nor a position of a
            column, ``columns`` is empty, or names are looked up among
            column names that repeat.
    """
    n_columns = X.shape[1]
    if columns is None:
        return np.ones(n_columns, dtype=bool)
    if isinstance(columns, str) or not isinstance(columns, Sequence):
        raise ValueError(
            "columns must be a list of column names or positions, or None;"
            f" got {columns!r}"
        )
    if named and not X.columns.is_unique:
        raise ValueError("columns needs X's column names to be unique")
    mask = np.zeros(n_columns, dtype=bool)
    for key in columns:
        if named and key in X.columns:
            mask[X.columns.get_loc(key)] = True
        elif _ensemble.is_integer(key) and -n_columns <= key < n_columns:
            mask[key] = True
        else:
            raise ValueError(
                "columns must be names (of X with column names) or positions"
                f" of X's {n_columns} columns; {key!r} is neither"
            )
    if not mask.any():
        raise ValueError("columns must name at least one column, or be None")
    return mask


def _initial_network(
    mapped: _typing.FloatArray,
    n_components: int,
    hidden: int,
    rng: np.random.Generator,
) -> dict[str, _typing.FloatArray]:
    """The network at step 0: PCA of the standardized columns, and an MLP.

    The hidden layer's weights and biases are drawn uniformly within
    ``1 / sqrt(n_components)`` of zero, as ``torch.nn.Linear`` draws them;
    the output layer is zero, so that the network is PCA.

    Args:
        mapped: The mapped columns of the context, shape (n_rows, n_mapped).
        n_components: The network's output columns.
        hidden: The units of the hidden layer.
        rng: The generator that draws the hidden layer.

    Returns:
        The network, float64, by the names of ``ContextAdapter.network_``.
    """
    scaler = preprocessing.StandardScaler().fit(mapped)
    pca = decomposition.PCA(n_components, svd_solver="full").fit(
        scaler.transform(mapped)
    )
    bound = 1.0 / math.sqrt(n_components)
    return {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "components": pca.components_,
        "hidden_weight": rng.uniform(-bound, bound, (hidden, n_components)),
        "hidden_bias": rng.uniform(-bound, bound, hidden),
        "output_weight": np.zeros((n_components, hidden)),
        "output_bias": np.zeros(n_components),
    }


def _map(network: _Network, mapped: Any) -> Any:
    """The network's columns, ``h + tanh(h A^T + a) B^T + b`` for PCA's ``h``.

    Args:
        network: The network's arrays, of the kind of ``mapped``.
        mapped: The mapped columns, a numpy array or a torch tensor of
            shape (n_rows, n_mapped).

    Returns:
        The new columns, of that kind, shape (n_rows, n_components).
    """
    xp = array_api_compat.array_namespace(mapped)
    standardized = (mapped - network["mean"]) / network["scale"]
    components = standardized @ network["components"].T
    hidden = xp.tanh(
        components @ network["hidden_weight"].T + network["hidden_bias"]
    )
    residual = hidden @ network["output_weight"].T + network["output_bias"]
    return components + residual


def _features(network: _Network, mapped: Any, unmapped: Any) -> Any:
    """The network's columns of some rows, then their unmapped columns.

    Args:
        network: The network's arrays, of the kind of ``mapped``.
        mapped: The mapped columns, shape (n_rows, n_mapped).
        unmapped: The other columns, of that kind, shape
            (n_rows, n_unmapped).

    Returns:
        The features, of that kind, shape (n_rows, n_components +
        n_unmapped).
    """
    xp = array_api_compat.array_namespace(mapped)
    return xp.concat([_map(network, mapped), unmapped], axis=1)


def _train(  # noqa: GS030 - scikit-learn's y.
    model: base.BaseDensityRegressor,
    network: dict[str, _typing.FloatArray],
    mapped: _typing.FloatArray,
    unmapped: _typing.FloatArray,
    y: _typing.FloatArray,
    parts: tuple[_typing.IntArray, _typing.IntArray, _typing.IntArray],
    n_steps: int,
    learning_rate: float,
) -> tuple[dict[str, _typing.FloatArray], list[tuple[float, float]], int]:
    """Trains the network by Adam, through a clone of the frozen model.

    Args:
        model: The unfitted model.
        network: The initial network, float64.
        mapped: The context's mapped columns, shape (n_rows, n_mapped).
        unmapped: Its other columns, shape (n_rows, n_unmapped).
        y: Its targets, shape (n_rows,).
        parts: The rows of the support, the training and the validation
            queries.
        n_steps: The steps of Adam.
        learning_rate: Adam's learning rate.

    Returns:
        A tuple (network, history, step): the network of the step with the
        lowest finite validation NLL, float64, or the initial one if none
        is finite; the (training, validation) NLL of each step; and the
        kept step.

    Raises:
        ValueError: If the model has no differentiable forward pass, or
            torch's inference mode is on.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    differentiable.refuse_inference_mode("ContextAdapter")
    support, queries, held_out = parts
    backend = _backend(
        model, _features(network, mapped, unmapped)[support], y[support]
    )
    device = torch.device(backend.device_)
    # An Apple GPU has no float64. The columns are standardized in float64
    # first, so that float32 keeps a column whose spread is small next to
    # its mean.
    dtype = torch.float32 if device.type == "mps" else torch.float64
    standardized = (mapped - network["mean"]) / network["scale"]
    unit = {
        "mean": np.zeros_like(network["mean"]),
        "scale": np.ones_like(network["scale"]),
    }
    # Copies: Adam updates them in place, and step 0 keeps ``network``.
    tensors = {
        name: torch.tensor(value, dtype=dtype, device=device)
        for name, value in {**network, **unit}.items()
    }
    weights = [tensors[name].requires_grad_() for name in _TRAINED]
    optimizer = torch.optim.Adam(weights, lr=learning_rate)
    inputs = [
        torch.as_tensor(array, dtype=dtype, device=device)
        for array in (standardized, unmapped)
    ]
    kept, best, best_step = network, math.inf, 0
    history = []
    # Inside torch.no_grad() too, which would build no graph.
    with torch.enable_grad():
        for step in range(n_steps + 1):
            features = _features(tensors, *inputs)
            if not bool(features.isfinite().all()):
                break
            loss = _nll(backend, features, y, support, queries)
            with torch.no_grad():
                held = float(_nll(backend, features, y, support, held_out))
            history.append((float(loss.detach()), held))
            if held < best:
                # Step 0 keeps the float64 network, exactly PCA.
                kept = network if step == 0 else _snapshot(network, tensors)
                best, best_step = held, step
            if step < n_steps:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return kept, history, best_step


def _backend(
    model: base.BaseDensityRegressor,
    support: _typing.FloatArray,
    y_support: _typing.FloatArray,
) -> Any:
    """A clone of the model fitted on the support rows, as its backend.

    Its differentiable pass then exists, and each step of training passes
    it the support rows mapped by the current network in place of these:
    the same rows, in the same order, with as many columns.

    Args:
        model: The unfitted model.
        support: The support rows mapped by the initial network, shape
            (n_support, n_features).
        y_support: Their targets, shape (n_support,).

    Returns:
        The fitted backend: the clone, or the backend of a ``LazyModel``.

    Raises:
        ValueError: If it has no differentiable forward pass.
    """
    fitted = sklearn_base.clone(model).fit(support, y_support)
    return differentiable.backend(fitted, "ContextAdapter")


def _nll(  # noqa: GS030 - scikit-learn's y.
    backend: Any,
    features: Any,
    y: _typing.FloatArray,
    support: _typing.IntArray,
    rows: _typing.IntArray,
) -> Any:
    """The mean NLL of some rows given the support rows, on the graph.

    Args:
        backend: The fitted backend, whose context is the support rows.
        features: Every row's features, a torch tensor of shape
            (n_rows, n_features).
        y: Every row's target, shape (n_rows,).
        support: The rows of the context.
        rows: The rows to score.

    Returns:
        The mean NLL in nats, a float64 torch scalar.

    Raises:
        ValueError: If the backend refuses the pass.
    """
    density = differentiable.density(
        backend, "ContextAdapter", features[rows], features[support], y[support]
    )
    return -density.log_prob(y[rows]).mean()


def _snapshot(
    network: dict[str, _typing.FloatArray], tensors: _Network
) -> dict[str, _typing.FloatArray]:
    """The network with the trained weights of ``tensors``, as float64.

    Args:
        network: The initial network, whose standardization stays.
        tensors: The current weights, torch tensors.

    Returns:
        A new network, float64 numpy arrays.
    """
    return {
        **network,
        **{
            name: tensors[name].detach().cpu().double().numpy()
            for name in _TRAINED
        },
    }
