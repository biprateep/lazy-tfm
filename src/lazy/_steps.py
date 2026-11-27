# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""How a gradient passes through one step of a pipeline.

A *step* is one stage of a pipeline: a transform, an encoder, the
``Tabularizer`` or an augmentation. Beside its scikit-learn ``transform``,
on numpy arrays, a step may have a *forward*, the same computation on torch
tensors, which keeps the autograd graph. Each step has one gradient kind,
one of :data:`GRADIENT_KINDS`:

``"differentiable"``
    The gradient passes: a scaling, a logarithm, a projection.
``"index_map"``
    The gradient passes after an index mapping: each output is one input,
    as after a flip, a crop or a column subset.
``"mask"``
    The gradient passes, and a masked input gets none.
``"barrier"``
    The gradient stops at the step's output, where attribution ends.

A lazy step declares its kind as ``gradient_kind`` and implements
``forward``; a barrier may say why in ``barrier_reason``. A fitted
scikit-learn transformer gets its kind and its forward from the registry
that :func:`register_forward` fills (:mod:`lazy._sklearn_forwards` holds
lazy's entries). Any other step is a barrier.

Typical usage example:

  if _steps.gradient_kind(step) != "barrier":
      columns = _steps.forward(step, columns)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import dataclasses
import importlib
from typing import Any

__all__ = [
    "GRADIENT_KINDS",
    "Barrier",
    "BarrierError",
    "Leaf",
    "barrier",
    "chained",
    "forward",
    "gradient_kind",
    "register_forward",
]

GRADIENT_KINDS: tuple[str, ...] = (
    "differentiable",
    "index_map",
    "mask",
    "barrier",
)
"""The gradient kinds a step can declare, from the most to the least open."""


@dataclasses.dataclass(frozen=True)
class Barrier:
    """A step that a gradient cannot cross.

    Attributes:
        step: The name of the step, as its pipeline names it.
        reason: Why the gradient stops there.
    """

    step: str
    reason: str


@dataclasses.dataclass(frozen=True)
class Leaf:
    """Inputs at which a gradient is taken, and the features they are.

    A leaf is a block of what a path differentiates: the columns of a
    table, or one column of items of one shape, such as images.

    Attributes:
        names: The features of the block: one per column for a table, and
            one for a column of items.
        queries: The rows to explain, float64, shape (n_rows, n_columns)
            for a table and (n_rows, ...) for items.
        context: The context rows, of the same trailing shape, shape
            (n_context, ...).
        per_column: Whether each column is a feature, as in a table; if
            not, the whole of each item is one feature.
    """

    names: tuple[str, ...]
    queries: Any
    context: Any
    per_column: bool = True


class BarrierError(ValueError):
    """A forward was asked of a step that has none."""


# A fitted transformer's kind, and its forward on a torch tensor.
_Kind = Callable[[Any], str]
_Forward = Callable[[Any, Any], Any]

# By class: the kind (or a function of an instance that gives it), the
# forward, and what to use in place of an instance that is a barrier.
_REGISTRY: dict[
    type, tuple[str | _Kind, _Forward, str | Callable[[Any], str]]
] = {}


def register_forward(
    cls: type,
    kind: str | _Kind = "differentiable",
    hint: str | Callable[[Any], str] = "",
) -> Callable[[_Forward], _Forward]:
    """Registers the forward of a class of fitted transformers.

    The forward serves instances of exactly ``cls``: a subclass may
    transform differently, so it is a barrier until it is registered.

    Args:
        cls: The class of transformer.
        kind: Its gradient kind, or a function from a fitted instance to
            its kind, for a class whose kind depends on its parameters.
        hint: What to use in place of an instance that is a barrier, a
            sentence that ends the barrier's reason; empty for none. A
            function from the instance gives the whole reason instead, as
            a container does for the step inside it that is the barrier.

    Returns:
        A decorator of the forward, a function from a fitted instance and a
        float torch tensor of shape (n_rows, n_features_in) to the tensor
        that ``transform`` would give, shape (n_rows, n_features_out).

    Raises:
        ValueError: If ``kind`` is text that is not a gradient kind.
    """
    if isinstance(kind, str) and kind not in GRADIENT_KINDS:
        raise ValueError(f"kind must be one of {GRADIENT_KINDS}: {kind=}")

    def decorator(function: _Forward) -> _Forward:
        """Files the forward under ``cls``."""
        _REGISTRY[cls] = (kind, function, hint)
        return function

    return decorator


# From the kind a chain keeps longest to the one any step imposes on it.
_CHAIN_ORDER = ("index_map", "differentiable", "mask", "barrier")


def chained(kinds: Iterable[str]) -> str:
    """The gradient kind of steps in a chain, or side by side.

    A barrier anywhere makes a barrier; else a mask makes a mask; else a
    step that mixes its inputs makes the chain differentiable; and only
    index maps make an index map.

    Args:
        kinds: Gradient kinds, each one of :data:`GRADIENT_KINDS`.

    Returns:
        The kind of the whole; ``"index_map"`` for no step.
    """
    return max(kinds, key=_CHAIN_ORDER.index, default="index_map")


def _resolve(step: Any) -> tuple[str, _Forward | None, str]:
    """A fitted step's gradient kind, its forward and its barrier's reason.

    The one place that reads a step's declaration, or its class's entry in
    the registry.

    Args:
        step: A fitted transformer, or what a pipeline skips (``None`` or
            ``"passthrough"``).

    Returns:
        A tuple (kind, forward, reason): the kind; a function from the
        step and a tensor to the transformed tensor, or ``None`` for a
        barrier; and why the step is a barrier, empty if it is not one.
    """
    if step is None or (isinstance(step, str) and step == "passthrough"):
        return "index_map", lambda _, X: X, ""
    reason = (
        f"{type(step).__name__} has no differentiable forward that lazy knows"
    )
    declared = getattr(step, "gradient_kind", None)
    if declared in GRADIENT_KINDS:
        function = getattr(type(step), "forward", None)
        if declared != "barrier" and function is not None:
            return declared, function, ""
        return "barrier", None, getattr(step, "barrier_reason", "") or reason
    # By name: that module imports this one, and registers on import.
    importlib.import_module("lazy._sklearn_forwards")
    entry = _REGISTRY.get(type(step))
    if entry is None:
        return "barrier", None, reason
    kind = entry[0] if isinstance(entry[0], str) else entry[0](step)
    if kind != "barrier":
        return kind, entry[1], ""
    hint = entry[2]
    if callable(hint):
        return kind, None, hint(step)
    return kind, None, reason + (f"; {hint}" if hint else "")


def gradient_kind(step: Any) -> str:
    """The gradient kind of a fitted step.

    Args:
        step: A fitted transformer, or what a pipeline skips (``None`` or
            ``"passthrough"``).

    Returns:
        One of :data:`GRADIENT_KINDS`; ``"barrier"`` for a step that
        declares no kind and whose class has no registered forward, and
        for one that declares a kind but has no ``forward``.
    """
    return _resolve(step)[0]


def barrier(name: str, step: Any) -> Barrier | None:
    """The barrier that a fitted step is, if it is one.

    Args:
        name: The name of the step in its pipeline.
        step: The fitted step.

    Returns:
        The barrier with its reason, or ``None`` if a gradient passes.
    """
    kind, _, reason = _resolve(step)
    return Barrier(name, reason) if kind == "barrier" else None


def forward(step: Any, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """A fitted step's ``transform`` on a torch tensor, on the graph.

    Args:
        step: A fitted transformer, or what a pipeline skips.
        X: What the step takes, as float torch tensors: for a step of a
            table, its columns, shape (n_rows, n_features_in).

    Returns:
        What ``step.transform`` gives for the same values, to float
        rounding, as a torch tensor that keeps the gradient of ``X``.

    Raises:
        BarrierError: If the step is a barrier.
    """
    _, function, reason = _resolve(step)
    if function is None:
        raise BarrierError(reason)
    return function(step, X)
