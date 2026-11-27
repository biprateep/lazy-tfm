# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""A reducer that is trained through the frozen model that follows it.

:class:`AdaptiveReducer` configures the reducer of a
:class:`~lazy.encoders.Tabularizer` that trains weights: a small network
that maps the standardized embeddings to the columns the model reads, by
the negative log-likelihood of the frozen model given the mapped context.
Unlike the rest of lazy, it trains weights (the network's, never the
model's), so it is opt-in::

    tab = lazy.Tabularizer({"notes": "text"}, n_columns=8, reducer="mlp")
    model = lazy.make_pipeline(
        tab, lazy.LazyModel("tabpfn", transforms=("none",),
                            outlier_threshold=None)
    )
    model.fit(table, y)
    model[0].history_  # The training and validation NLL at each step.

A scikit-learn step never sees the model after it, so
:meth:`lazy.pipeline.LazyPipeline.fit` trains the reducer, through the
forward of each step between the two (:mod:`lazy._steps`) and the model's
differentiable forward pass. The "Adaptive reducer" page of the
documentation (``docs/guide/adapter.md``) defines the network and the
loss.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import numbers
from typing import Any

import numpy as np
from sklearn import base as sklearn_base
from sklearn.utils import validation

from lazy import _typing
from lazy.models import _differentiable as differentiable

__all__ = ["AdaptiveReducer"]

LABEL = "an adaptive reducer"
"""What the errors of training call the reducer."""


class AdaptiveReducer(sklearn_base.BaseEstimator):
    """The settings of a reducer that trains through the model.

    ``Tabularizer(reducer="mlp")`` is ``Tabularizer(reducer=
    AdaptiveReducer())``, and a torch module given as ``reducer`` is
    ``AdaptiveReducer(network=module)``.

    The context is split once into support rows, training queries and
    validation queries. Each step of Adam lowers the mean negative
    log-likelihood (NLL) of the training queries under the model given the
    support rows, both mapped by the network. The step with the lowest
    validation NLL is kept, step 0 included.

    Attributes:
        network: ``"mlp"``, the principal components plus a residual
            multilayer perceptron (MLP) whose output layer starts at zero,
            so that the untrained network is principal component analysis
            (PCA); or a ``torch.nn.Module`` that maps the standardized
            embeddings, side by side, then the mask of the items a row
            has, shape (n_rows, n_embedding_columns + n_item_columns), to
            the columns the model reads.
        hidden: For ``"mlp"``, the units of the one hidden layer, which
            applies ``tanh``.
        n_steps: The steps of Adam; 0 keeps the network as it starts.
        learning_rate: Adam's learning rate.
        validation_fraction: The share of the context's rows held out as
            validation queries, in (0, 1).
        query_fraction: The share of them used as training queries, in
            (0, 1). The two fractions sum to less than 1; the other rows
            are the support rows.
        block_dropout: The probability that a training step treats an
            item of a row as missing, in [0, 1), so that the network
            learns to map a row that lacks an item; a row that would lose
            every item it has loses none, so a table with one column of
            items drops none. The
            network reads a mask of the items a row has.
        random_state: The seed of the split, of the dropout and of the
            MLP's initial weights, a non-negative integer, or ``None`` for
            fresh entropy.
    """

    def __init__(
        self,
        network: Any = "mlp",
        *,
        hidden: int = 64,
        n_steps: int = 200,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        query_fraction: float = 0.2,
        block_dropout: float = 0.1,
        random_state: int | None = 0,
    ) -> None:
        self.network = network
        self.hidden = hidden
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.query_fraction = query_fraction
        self.block_dropout = block_dropout
        self.random_state = random_state

    def check(self) -> None:
        """Refuses a setting that is not as the class documents it.

        Raises:
            ValueError: If a setting is out of its range, or ``network`` is
                neither ``"mlp"`` nor a module.
            TypeError: If a count is not an integer or a rate not a number.
        """
        if not (is_module(self.network) or self.network == "mlp"):
            raise ValueError(
                "network must be 'mlp' or a torch.nn.Module:"
                f" network={self.network!r}"
            )
        for name, low in (("hidden", 1), ("n_steps", 0)):
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
        validation.check_scalar(
            self.block_dropout,
            "block_dropout",
            numbers.Real,
            min_val=0.0,
            max_val=1.0,
            include_boundaries="left",
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

    def split(
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

    def initial(
        self,
        components: _typing.FloatArray,
        n_blocks: int,
        rng: np.random.Generator,
    ) -> dict[str, _typing.FloatArray]:
        """The ``"mlp"`` network at step 0: the components, and an MLP.

        The hidden layer reads the components and the mask of the items a
        row has. Its weights and biases are drawn uniformly within
        ``1 / sqrt(n_inputs)`` of zero, as ``torch.nn.Linear`` draws them;
        the output layer is zero, so that the network is PCA.

        Args:
            components: The principal components, shape (n_columns,
                n_embedding_columns).
            n_blocks: The number of columns of items.
            rng: The generator that draws the hidden layer.

        Returns:
            The network as float64 arrays by name, which are
            ``components``, the hidden layer and the output layer.
        """
        n_columns = len(components)
        n_inputs = n_columns + n_blocks
        bound = 1.0 / math.sqrt(n_inputs)
        return {
            "components": components.copy(),
            "hidden_weight": rng.uniform(
                -bound, bound, (self.hidden, n_inputs)
            ),
            "hidden_bias": rng.uniform(-bound, bound, self.hidden),
            "output_weight": np.zeros((n_columns, self.hidden)),
            "output_bias": np.zeros(n_columns),
        }


def is_module(network: object) -> bool:
    """Whether ``network`` is a ``torch.nn.Module``, without importing torch."""
    return any(
        cls.__module__.startswith("torch.") and cls.__name__ == "Module"
        for cls in type(network).__mro__
    )


def residual(
    network: Mapping[str, Any], components: Any, mask: Any, xp: Any
) -> Any:
    """The ``"mlp"`` network's columns, ``h + tanh([h, m] A^T + a) B^T + b``.

    Args:
        network: The network's arrays, of the kind of ``components``.
        components: The projection ``h`` of some rows, 0 where an item is
            missing, a numpy array or a torch tensor of shape (n_rows,
            n_columns).
        mask: ``m``, 1 for each item a row has and 0 for one it lacks, of
            that kind, shape (n_rows, n_item_columns).
        xp: The array namespace of ``components``.

    Returns:
        The new columns, of that kind and shape.
    """
    inputs = xp.concat([components, mask], axis=1)
    hidden = xp.tanh(
        inputs @ network["hidden_weight"].T + network["hidden_bias"]
    )
    return (
        components
        + hidden @ network["output_weight"].T
        + network["output_bias"]
    )


def train(  # noqa: GS030 - scikit-learn's y.
    reducer: AdaptiveReducer,
    weights: list[Any],
    features: Callable[[bool], Any],
    y: _typing.FloatArray,
    parts: tuple[_typing.IntArray, _typing.IntArray, _typing.IntArray],
    backend: Any,
    snapshot: Callable[[], Any],
) -> tuple[Any, list[tuple[float, float]], int]:
    """Trains a network's weights by Adam, through a frozen backend.

    Args:
        reducer: The settings.
        weights: The torch tensors to train, which require gradients.
        features: Gives every row's features as the model reads them, a
            torch tensor of shape (n_rows, n_features) on the graph of
            ``weights`` at their current values; its argument says
            whether to drop items at random, as a training step does.
        y: Every row's target, shape (n_rows,).
        parts: The rows of the support, the training and the validation
            queries.
        backend: A backend fitted on the support rows, with a
            differentiable forward pass.
        snapshot: Gives a copy of the network at its current weights.

    Returns:
        A tuple (kept, history, step): the snapshot of the step with the
        lowest finite validation NLL, or of step 0 if none is finite; the
        (training, validation) NLL of each step; and the kept step.

    Raises:
        ValueError: If the backend refuses the pass, or torch's inference
            mode is on.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    label = LABEL
    differentiable.refuse_inference_mode(label)
    support, queries, held_out = parts

    def nll(columns: Any, rows: _typing.IntArray) -> Any:
        """The mean NLL of some rows given the support rows, on the graph."""
        # The network moves the context's columns at every step.
        density = differentiable.density(
            backend,
            label,
            columns[rows],
            columns[support],
            y[support],
            context_moves=True,
        )
        return -density.log_prob(y[rows]).mean()

    optimizer = torch.optim.Adam(weights, lr=reducer.learning_rate)
    kept, best, best_step = snapshot(), math.inf, 0
    history = []
    # Inside torch.no_grad() too, which would build no graph.
    with torch.enable_grad():
        for step in range(reducer.n_steps + 1):
            columns = features(True)
            if not bool(columns.isfinite().all()):
                break
            loss = nll(columns, queries)
            with torch.no_grad():
                held = float(nll(features(False), held_out))
            history.append((float(loss.detach()), held))
            if held < best:
                kept, best, best_step = snapshot(), held, step
            if step < reducer.n_steps:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return kept, history, best_step
