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
import functools
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
        table: What the model reads of some rows, on the autograd graph:
            a function of one float torch tensor per leaf, of the leaf's
            trailing shape, with one row per row.
        density: The densities of query rows given the rows of the whole
            context, in the order of the fit, on the graph: a function of
            the two as ``table`` returns them. It raises ``ValueError`` if
            the model refuses the pass.
        ranked: Of some rows' tables, as ``table`` returns them, those
            that a backend reads whose network ranks each query cell among
            the context's values (its ``ranking_versions``), a step
            function of the query that it smooths under
            :func:`lazy.models._differentiable.smoothed_ranks`: a list of
            tensors, empty if no backend on the path ranks.
    """

    leaves: tuple[_steps.Leaf, ...]
    stage: str | None
    barriers: tuple[_steps.Barrier, ...]
    table: _Table
    density: Callable[[Any, Any], differentiable.TorchDensity]
    ranked: Callable[[Any], list[Any]]


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
        raise no_steps(model, at)
    backend = differentiable.backend(model, label)
    (queries, named), (context, _) = (
        _inputs.to_frame(part) for part in (X, X_context)
    )
    # Where each of the fit's columns is in X, by predict's rule for names.
    positions = pd.DataFrame(
        [np.arange(queries.shape[1])], columns=queries.columns
    )
    aligned = backend._align_to_fit(positions, named)  # noqa: SLF001 - predict's rule.
    order = aligned.to_numpy()[0].copy()
    leaf = _leaf(queries, context, named, label)
    return Path(
        (leaf,),
        None,
        (),
        lambda tensors: tensors[0][:, order],
        _pass(backend, label, y_context),
        _ranked(backend),
    )


def no_steps(model: Any, at: str) -> ValueError:
    """The error for ``at`` given with a model that is no pipeline."""
    return ValueError(
        f"at names a step of a LazyPipeline; {type(model).__name__} has"
        f" none: {at=}"
    )


def position(at: str, names: Sequence[str]) -> int:
    """The position of the step ``at`` among a pipeline's steps.

    Args:
        at: The name given.
        names: The names of the steps before the model, in order.

    Returns:
        The position of ``at`` in ``names``.

    Raises:
        ValueError: If ``at`` is none of ``names``.
    """
    if at not in names:
        raise ValueError(
            f"at must name a step before the model, one of {list(names)}: {at=}"
        )
    return list(names).index(at)


def _pass(
    backend: Any, label: str, y_context: _typing.FloatArray
) -> Callable[[Any, Any], differentiable.TorchDensity]:
    """A backend's differentiable pass, as a function of two tables."""
    return functools.partial(
        differentiable.density, backend, label, y_context=y_context
    )


def _ranked(backend: Any) -> Callable[[Any], list[Any]]:
    """A path's ``ranked`` for a table that ``backend`` reads."""
    ranks = backend._ranks_queries()  # noqa: SLF001 - lazy's own backends.
    return lambda table: [table] if ranks else []


def _leaf(
    queries: pd.DataFrame,
    context: pd.DataFrame,
    named: bool,
    label: str,
    hint: str = "",
) -> _steps.Leaf:
    """A table of numbers as one leaf, each column a feature.

    Args:
        queries: The rows to explain, shape (n_rows, n_columns).
        context: The context rows, shape (n_context, n_columns).
        named: Whether the table names its columns; if not, they are named
            ``x0`` onwards.
        label: What takes the gradient, for the errors.
        hint: What to do about a missing value, to end the error.

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
        _floats(queries, "X", label, hint),
        _floats(context, "X_context", label, hint),
    )


def _floats(
    frame: pd.DataFrame, name: str, label: str, hint: str = ""
) -> _typing.FloatArray:
    """A table's values, which gradients need numeric and present.

    Args:
        frame: Features, shape (n, p).
        name: The argument's name, for the errors.
        label: What takes the gradient, for the errors.
        hint: What to do about a missing value, to end the error.

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
    return _finite(frame.to_numpy(dtype=np.float64), name, label, hint)


def _finite(
    values: _typing.FloatArray, name: str, label: str, hint: str = ""
) -> _typing.FloatArray:
    """``values``, checked to hold no missing or infinite value."""
    if not np.isfinite(values).all():
        raise ValueError(
            f"{label} needs {name} without missing or infinite values{hint}"
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
            # Such a step is entered leaf by leaf, from its table.
            if _opens(step):
                return index
            if _steps.gradient_kind(step) == "barrier":
                return index + 1
        return 0
    start = position(at, names) + 1
    for index, (name, step) in enumerate(steps[start:]):
        entered = index == 0 and _opens(step)
        found = None if entered else _steps.barrier(name, step)
        if found is not None:
            raise ValueError(
                f"{label} cannot start at {at=}: the gradient does not cross"
                f" the later step {found.step!r} ({found.reason})"
            )
    return start


def _fills(step: Any) -> bool:
    """Whether a step fills missing values, so that a path may start after."""
    return _steps.gradient_kind(step) == "mask"


def _order(
    columns: pd.Index, named: bool, reader: Any
) -> npt.NDArray[np.intp] | None:
    """Where each column that a fitted step reads is in a table.

    ``predict`` matches a named table's columns to the fit's by name, so
    the forward of the first step must read them in the fit's order.

    Args:
        columns: The table's column names.
        named: Whether the caller named them.
        reader: The fitted step, or model, that reads the table.

    Returns:
        The positions in the table of the reader's columns, in its order,
        or ``None`` if the table is unnamed, the reader recorded no names,
        or the names differ, which ``predict`` refuses.
    """
    fitted = getattr(reader, "feature_names_in_", None)
    if not named or fitted is None:
        return None
    if sorted(map(str, fitted)) != sorted(map(str, columns)):
        return None
    return columns.get_indexer(fitted)


def _opens(step: Any) -> bool:
    """Whether a path can start inside a step, at leaves of its own.

    Such a step, a barrier to a tensor of a table, gives its leaves with
    ``_leaves`` and its output from them with ``_forward_leaves``.
    """
    return hasattr(step, "_leaves") and hasattr(step, "_forward_leaves")


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
    if at is None and start:
        # Where attribution ends; an earlier barrier is never reached.
        found = _steps.barrier(*steps[start - 1])
        if found is not None:
            barriers.append(found)
    queries, context = X, X_context
    for _, step in steps[:start]:
        queries, context = step.transform(queries), step.transform(context)
    tail = [step for _, step in steps[start:]]
    if tail and _opens(tail[0]):
        first = tail.pop(0)
        leaves, through, closed = first._leaves(  # noqa: SLF001 - lazy's own step.
            queries, context, items, steps[start][0]
        )
        barriers.extend(closed)
        for leaf in leaves:
            if not leaf.padded:
                _finite(leaf.queries, "X", label)
                _finite(leaf.context, "X_context", label)

        def enter(tensors: Sequence[Any]) -> Any:
            """The Tabularizer's columns, from its leaves."""
            return first._forward_leaves(tensors, through)  # noqa: SLF001 - lazy's own step.
    else:
        (queries, named), (context, _) = (
            _inputs.to_frame(part) for part in (queries, context)
        )
        fills = [name for name, step in steps[start:] if _fills(step)]
        hint = (
            f"; at={fills[0]!r} starts after the step that fills them"
            if fills
            else ""
        )
        leaves = [_leaf(queries, context, named, label, hint)]
        order = _order(queries.columns, named, tail[0] if tail else model)

        def enter(tensors: Sequence[Any]) -> Any:
            """The one leaf, a table, its columns in the order of the fit."""
            return tensors[0] if order is None else tensors[0][:, order]

    def table(tensors: Sequence[Any]) -> Any:
        """The model's columns, through every step after the leaves."""
        columns = enter(tensors)
        for step in tail:
            columns = _steps.forward(step, columns)
        return columns

    backend = differentiable.backend(model, label)
    return Path(
        tuple(leaves),
        steps[start - 1][0] if start else None,
        tuple(barriers),
        table,
        _pass(backend, label, y_context),
        _ranked(backend),
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
    # A member that starts at a stage has features of its own, which
    # share only their placeholder names with another member's.
    paths = [
        path
        if path.stage is None
        else dataclasses.replace(
            path,
            leaves=tuple(
                dataclasses.replace(
                    leaf,
                    names=tuple(f"{name}__{feature}" for feature in leaf.names),
                )
                for leaf in path.leaves
            ),
        )
        for name, path in zip(model.named_estimators_, paths, strict=True)
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

    def ranked(tables: list[Any]) -> list[Any]:
        """The members' tables that a ranking backend reads."""
        found = []
        for path, member_table in zip(paths, tables, strict=True):
            found.extend(path.ranked(member_table))
        return found

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
        ranked,
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
    masses = []
    for part in parts:
        inside = part._bin_densities(grid) * widths  # noqa: SLF001 - one implementation.
        masses.append(inside / inside.sum(dim=1, keepdim=True))
    if pooling == "linear":
        total = _arrays.weighted_total(weights, masses)
        # A bin that no member fills: log(0) there would make the
        # gradient of the whole row NaN.
        filled = total > 0
        log_total = torch.where(
            filled,
            torch.log(torch.where(filled, total, torch.ones_like(total))),
            torch.full_like(total, -torch.inf),
        )
        return differentiable.TorchHistogram(grid.edges, log_total)
    total = _arrays.weighted_total(
        weights, [torch.log(mass / widths + floor) for mass in masses]
    )
    total = total - total.max(dim=1, keepdim=True).values
    return differentiable.TorchHistogram(grid.edges, total + torch.log(widths))
