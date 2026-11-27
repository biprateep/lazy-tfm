# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The differentiable path from a model's inputs to its densities.

A gradient method of :mod:`lazy.interp` differentiates a statistic of each
predicted density in some inputs. :func:`build` finds those inputs, the
*leaves* (:class:`lazy._steps.Leaf`), and the function from them to the
densities, for each kind of model:

* a backend, or a ``LazyModel``: the columns of the table, through the
  backend's differentiable forward pass;
* a :class:`~lazy.pipeline.LazyPipeline`: the output of one of its steps,
  the *stage*, through the forward of each later step (:mod:`lazy._steps`)
  and then the model's pass. With no stage named, the path starts as far
  back as a gradient reaches: after the last step that is a barrier, or at
  the raw input. A ``Tabularizer`` at the start is entered column by
  column, down to the items of each encoder that has a forward;
* a :class:`~lazy.models.ensemble.LazyEnsembleModel`: every member's
  leaves, through each member's path to the pooled density, so that one
  backward pass gives every member its share.

Typical usage example:

  path = _paths.build(model, X, X_context, y_context, label="gradient")
  density = path.density(path.table(queries), path.table(contexts))
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.utils import validation

from lazy import _arrays
from lazy import _inputs
from lazy import _steps
from lazy import _typing
from lazy import grid as grid_lib
from lazy import pipeline
from lazy.encoders import tabularizer
from lazy.models import _differentiable as differentiable
from lazy.models import ensemble

__all__ = ["Path", "build"]

# From one tensor per leaf to the table a model reads, on the graph.
_Table = Callable[[Sequence[Any]], Any]


@dataclasses.dataclass(frozen=True)
class Path:
    """How a gradient reaches a fitted model's densities.

    Attributes:
        leaves: The inputs to differentiate, in order.
        stage: The step of a pipeline whose output the leaves are, or
            ``None`` for the model's raw input, or for an ensemble.
        barriers: The steps on the way from the raw input that a gradient
            cannot cross, with the reason.
    """

    leaves: tuple[_steps.Leaf, ...]
    stage: str | None
    barriers: tuple[_steps.Barrier, ...]
    _table: _Table
    _density: Callable[[Any, Any], differentiable.TorchDensity]

    def table(self, tensors: Sequence[Any]) -> Any:
        """What the model reads of some rows, on the autograd graph.

        Args:
            tensors: One float torch tensor per leaf, of the leaf's
                trailing shape, with one row per row.

        Returns:
            The rows as :meth:`density` takes them.
        """
        return self._table(tensors)

    def density(self, query: Any, context: Any) -> differentiable.TorchDensity:
        """The densities of query rows given context rows, on the graph.

        Args:
            query: The query rows, as :meth:`table` returns them.
            context: The rows of the whole context, in the order of the
                fit, as :meth:`table` returns them.

        Returns:
            One density per query row.

        Raises:
            ValueError: If the model refuses the pass.
        """
        return self._density(query, context)


def build(  # noqa: GS030 - scikit-learn's X, y.
    model: Any,
    X: Any,
    X_context: Any,
    y_context: _typing.FloatArray,
    *,
    label: str,
    at: str | None = None,
    items: bool = True,
) -> Path:
    """The path of a gradient through a fitted model.

    Args:
        model: A fitted backend, ``LazyModel``, ``LazyPipeline`` or
            ``LazyEnsembleModel``.
        X: The rows to explain, as the model's ``predict`` takes them.
        X_context: The context the model was fitted on, in the same form.
        y_context: Its targets, shape (n_context,).
        label: What takes the gradient, for the errors, such as
            ``"method='gradient'"``.
        at: The step of a pipeline whose output to differentiate in;
            ``None`` for as far back as a gradient reaches.
        items: Whether a path that reaches a ``Tabularizer`` goes through
            its encoders to the items; if not, it starts at the embeddings.

    Returns:
        The path.

    Raises:
        sklearn.exceptions.NotFittedError: If the model is not fitted.
        ValueError: If the model has no differentiable forward pass, ``at``
            names no step before the model of a pipeline, a barrier stands
            between the stage and the model, or a leaf is not numeric and
            finite.
    """
    validation.check_is_fitted(model)
    if isinstance(model, ensemble.LazyEnsembleModel):
        return _pooled(model, X, X_context, y_context, label, at, items)
    if isinstance(model, pipeline.LazyPipeline):
        return _piped(model, X, X_context, y_context, label, at, items)
    if at is not None:
        raise ValueError(
            f"at names a step of a LazyPipeline; {type(model).__name__} has"
            f" none: {at=}"
        )
    backend = differentiable.backend(model, label)
    frames = [_inputs.to_frame(part) for part in (X, X_context)]
    # Where each of the fit's columns is in X, by predict's rule for names.
    positions = pd.DataFrame(
        [np.arange(frames[0][0].shape[1])], columns=frames[0][0].columns
    )
    aligned = backend._align_to_fit(positions, frames[0][1])  # noqa: SLF001 - predict's rule.
    order = aligned.to_numpy()[0].copy()
    leaf = _leaf(frames[0][0], frames[1][0], frames[0][1], label)
    return Path(
        (leaf,),
        None,
        (),
        lambda tensors: tensors[0][:, order],
        _pass(backend, label, y_context),
    )


def _pass(
    backend: Any, label: str, y_context: _typing.FloatArray
) -> Callable[[Any, Any], differentiable.TorchDensity]:
    """A backend's differentiable pass, as a function of two tables."""

    def density(query: Any, context: Any) -> differentiable.TorchDensity:
        """The densities of ``query`` given ``context``."""
        return differentiable.density(backend, label, query, context, y_context)

    return density


def _leaf(
    queries: pd.DataFrame, context: pd.DataFrame, named: bool, label: str
) -> _steps.Leaf:
    """A table of numbers as one leaf, each column a feature.

    Args:
        queries: The rows to explain, shape (n_rows, n_columns).
        context: The context rows, shape (n_context, n_columns).
        named: Whether the table names its columns; if not, they are named
            ``x0`` onwards.
        label: What takes the gradient, for the errors.

    Returns:
        The leaf.

    Raises:
        ValueError: If a column is not numeric, or a value is missing or
            infinite.
    """
    names = (
        tuple(str(column) for column in queries.columns)
        if named
        else tuple(f"x{i}" for i in range(queries.shape[1]))
    )
    return _steps.Leaf(
        names,
        _floats(queries, "X", label),
        _floats(context, "X_context", label),
    )


def _floats(frame: pd.DataFrame, name: str, label: str) -> _typing.FloatArray:
    """A table's values, which gradients need numeric and present.

    Args:
        frame: Features, shape (n, p).
        name: The argument's name, for the errors.
        label: What takes the gradient, for the errors.

    Returns:
        The values, float64, shape (n, p).

    Raises:
        ValueError: If a column is not numeric, or a value is missing or
            infinite.
    """
    other = frame.select_dtypes(exclude="number").columns
    if len(other):
        raise ValueError(
            f"{label} needs numeric columns; {name} has others: {list(other)}"
        )
    return _finite(frame.to_numpy(dtype=np.float64), name, label)


def _finite(
    values: _typing.FloatArray, name: str, label: str
) -> _typing.FloatArray:
    """``values``, checked to hold no missing or infinite value."""
    if not np.isfinite(values).all():
        raise ValueError(
            f"{label} needs {name} without missing or infinite values"
        )
    return values


def _flattened(
    pipe: pipeline.LazyPipeline,
) -> tuple[list[tuple[str, Any]], Any]:
    """A pipeline's steps before its model, and the model.

    A pipeline at the end of a pipeline is opened, its steps named
    ``<pipeline>__<step>``, and a step that a pipeline skips is left out.

    Args:
        pipe: The fitted pipeline.

    Returns:
        A tuple (steps, model): the named steps in order, and the last
        estimator.
    """
    steps = list(pipe.steps[:-1])
    name, model = pipe.steps[-1]
    while isinstance(model, pipeline.LazyPipeline):
        steps.extend(
            (f"{name}__{inner}", step) for inner, step in model.steps[:-1]
        )
        name, model = f"{name}__{model.steps[-1][0]}", model.steps[-1][1]
    kept = [
        (name, step)
        for name, step in steps
        if not (step is None or isinstance(step, str))
    ]
    return kept, model


def _start(steps: list[tuple[str, Any]], at: str | None, label: str) -> int:
    """The number of steps that run before the leaves.

    Args:
        steps: A pipeline's named steps before its model.
        at: The step whose output is the stage, or ``None``.
        label: What takes the gradient, for the errors.

    Returns:
        The position of the first step that the gradient passes.

    Raises:
        ValueError: If ``at`` names no step, or a later step is a barrier.
    """
    names = [name for name, _ in steps]
    if at is None:
        for index in reversed(range(len(steps))):
            step = steps[index][1]
            # A Tabularizer is entered column by column, from the table.
            if isinstance(step, tabularizer.Tabularizer):
                return index
            if _steps.gradient_kind(step) == "barrier":
                return index + 1
        return 0
    if at not in names:
        raise ValueError(
            f"at must name a step before the model, one of {names}: {at=}"
        )
    start = names.index(at) + 1
    for index, (name, step) in enumerate(steps[start:]):
        entered = index == 0 and isinstance(step, tabularizer.Tabularizer)
        found = None if entered else _closed(name, step)
        if found is not None:
            raise ValueError(
                f"{label} cannot start at {at=}: the gradient does not cross"
                f" the later step {found.step!r} ({found.reason})"
            )
    return start


def _closed(name: str, step: Any) -> _steps.Barrier | None:
    """The barrier that a step is to a tensor of a table, if it is one."""
    if isinstance(step, tabularizer.Tabularizer):
        return _steps.Barrier(
            name, "a Tabularizer takes a table of items, not their columns"
        )
    return _steps.barrier(name, step)


def _piped(  # noqa: GS030 - scikit-learn's X, y.
    pipe: pipeline.LazyPipeline,
    X: Any,
    X_context: Any,
    y_context: _typing.FloatArray,
    label: str,
    at: str | None,
    items: bool,
) -> Path:
    """The path through a fitted pipeline; see :func:`build`."""
    steps, model = _flattened(pipe)
    start = _start(steps, at, label)
    barriers = []
    if at is None:
        barriers = [
            found
            for name, step in steps[:start]
            if (found := _closed(name, step)) is not None
        ]
    queries, context = X, X_context
    for _, step in steps[:start]:
        queries, context = step.transform(queries), step.transform(context)
    tail = [step for _, step in steps[start:]]
    if tail and isinstance(tail[0], tabularizer.Tabularizer):
        first = tail.pop(0)
        leaves, through, closed = first._leaves(  # noqa: SLF001 - lazy's own step.
            queries, context, items, steps[start][0]
        )
        barriers.extend(closed)
        for leaf in leaves:
            _finite(leaf.queries, "X", label)
            _finite(leaf.context, "X_context", label)

        def enter(tensors: Sequence[Any]) -> Any:
            """The Tabularizer's columns, from its leaves."""
            return first._forward_leaves(tensors, through)  # noqa: SLF001 - lazy's own step.
    else:
        frames = [_inputs.to_frame(part) for part in (queries, context)]
        leaves = [_leaf(frames[0][0], frames[1][0], frames[0][1], label)]

        def enter(tensors: Sequence[Any]) -> Any:
            """The one leaf, a table."""
            return tensors[0]

    def table(tensors: Sequence[Any]) -> Any:
        """The model's columns, through every step after the leaves."""
        columns = enter(tensors)
        for step in tail:
            columns = _steps.forward(step, columns)
        return columns

    return Path(
        tuple(leaves),
        steps[start - 1][0] if start else None,
        tuple(barriers),
        table,
        _pass(differentiable.backend(model, label), label, y_context),
    )


def _pooled(  # noqa: GS030 - scikit-learn's X, y.
    model: ensemble.LazyEnsembleModel,
    X: Any,
    X_context: Any,
    y_context: _typing.FloatArray,
    label: str,
    at: str | None,
    items: bool,
) -> Path:
    """The path through a fitted ensemble; see :func:`build`.

    Every member gets leaves of its own, even of the same columns, so a
    feature's attribution is the sum over the leaves that carry its name.

    Raises:
        ValueError: If the ensemble pools quantile functions, or as
            :func:`build` raises for a member.
    """
    if model.pooling_ == "quantile":
        raise ValueError(
            f"{label} needs an ensemble that pools densities, with"
            " pooling='geometric' or 'linear'; pooling='quantile' has no"
            " differentiable pass"
        )
    members = model.estimators_
    tables = [
        model._member_features(model._check_features(part), members)  # noqa: SLF001 - predict's own reading.
        for part in (X, X_context)
    ]
    paths = [
        build(
            member,
            queries,
            context,
            y_context,
            label=label,
            at=at,
            items=items,
        )
        for member, queries, context in zip(members, *tables, strict=True)
    ]
    counts = np.cumsum([0, *(len(path.leaves) for path in paths)])

    def table(tensors: Sequence[Any]) -> list[Any]:
        """Each member's table, from its own leaves."""
        return [
            path.table(tensors[counts[i] : counts[i + 1]])
            for i, path in enumerate(paths)
        ]

    def density(query: list[Any], context: list[Any]) -> Any:
        """The members' densities, pooled as the ensemble pools them."""
        parts = [
            path.density(q, c)
            for path, q, c in zip(paths, query, context, strict=True)
        ]
        return _pool(parts, model.weights_, model.pooling_, model.native_grid_)

    stages = {path.stage for path in paths}
    leaves: list[_steps.Leaf] = []
    for path in paths:
        leaves.extend(path.leaves)
    return Path(
        tuple(leaves),
        stages.pop() if len(stages) == 1 else None,
        tuple(_named_barriers(model, paths)),
        table,
        density,
    )


def _named_barriers(
    model: ensemble.LazyEnsembleModel, paths: list[Path]
) -> list[_steps.Barrier]:
    """Each member's barriers, their steps named ``<member>__<step>``."""
    found: list[_steps.Barrier] = []
    for name, path in zip(model.named_estimators_, paths, strict=True):
        found.extend(
            dataclasses.replace(barrier, step=f"{name}__{barrier.step}")
            for barrier in path.barriers
        )
    return found


def _pool(
    parts: list[differentiable.TorchDensity],
    weights: npt.NDArray[np.float64],
    pooling: str,
    grid: grid_lib.Grid,
) -> differentiable.TorchHistogram:
    """The members' densities pooled on the pooling grid, on the graph.

    As :func:`lazy.models.ensemble._pool` pools their numpy twins: each
    member's masses in the grid's bins, renormalized, then their weighted
    mean (``"linear"``) or the weighted mean of the logarithms of their
    floored densities (``"geometric"``).

    Args:
        parts: One density per member, all with the same rows.
        weights: The members' weights, shape (n_members,), summing to one.
        pooling: ``"geometric"`` or ``"linear"``.
        grid: The pooling grid.

    Returns:
        The pooled histogram on the grid's bins.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    widths = torch.as_tensor(grid.widths)
    floor = ensemble.LOG_FLOOR / (grid.y_max - grid.y_min)
    terms = []
    for part in parts:
        masses = part._bin_densities(grid) * widths  # noqa: SLF001 - one implementation.
        masses = masses / masses.sum(dim=1, keepdim=True)
        terms.append(
            masses
            if pooling == "linear"
            else torch.log(masses / widths + floor)
        )
    total = _arrays.weighted_total(weights, terms)
    if pooling == "linear":
        return differentiable.TorchHistogram(grid.edges, torch.log(total))
    total = total - total.max(dim=1, keepdim=True).values
    return differentiable.TorchHistogram(grid.edges, total + torch.log(widths))
