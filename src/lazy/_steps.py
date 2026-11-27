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
``forward``. A fitted scikit-learn transformer gets both from the registry
that :func:`register_forward` fills (:mod:`lazy._sklearn_forwards` holds
lazy's entries). Any other step is a barrier.

Typical usage example:

  if _steps.gradient_kind(step) != "barrier":
      columns = _steps.forward(step, columns)
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import importlib
from typing import Any

__all__ = [
    "GRADIENT_KINDS",
    "Barrier",
    "BarrierError",
    "barrier",
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


class BarrierError(ValueError):
    """A forward was asked of a step that has none."""


# A fitted transformer's kind, and its forward on a torch tensor.
_Kind = Callable[[Any], str]
_Forward = Callable[[Any, Any], Any]

_REGISTRY: dict[type, tuple[_Kind, _Forward, str]] = {}


def register_forward(
    cls: type, kind: str | _Kind = "differentiable", hint: str = ""
) -> Callable[[_Forward], _Forward]:
    """Registers the forward of a class of fitted transformers.

    The forward serves instances of exactly ``cls``: a subclass may
    transform differently, so it is a barrier until it is registered.

    Args:
        cls: The class of transformer.
        kind: Its gradient kind, or a function from a fitted instance to
            its kind, for a class whose kind depends on its parameters.
        hint: What to use in place of an instance that is a barrier, a
            sentence that ends the barrier's reason; empty for none.

    Returns:
        A decorator of the forward, a function from a fitted instance and a
        float torch tensor of shape (n_rows, n_features_in) to the tensor
        that ``transform`` would give, shape (n_rows, n_features_out).

    Raises:
        ValueError: If ``kind`` is text that is not a gradient kind.
    """
    if isinstance(kind, str):
        if kind not in GRADIENT_KINDS:
            raise ValueError(f"kind must be one of {GRADIENT_KINDS}: {kind=}")
        fixed = kind

        def kind(step: Any) -> str:
            """The kind of every instance."""
            del step  # Unused: the kind is the class's.
            return fixed

    def decorator(function: _Forward) -> _Forward:
        """Files the forward under ``cls``."""
        _REGISTRY[cls] = (kind, function, hint)
        return function

    return decorator


def _is_skipped(step: Any) -> bool:
    """Whether a pipeline skips the step: ``None`` or ``"passthrough"``."""
    return step is None or (isinstance(step, str) and step == "passthrough")


def gradient_kind(step: Any) -> str:
    """The gradient kind of a fitted step.

    Args:
        step: A fitted transformer, or what a pipeline skips (``None`` or
            ``"passthrough"``).

    Returns:
        One of :data:`GRADIENT_KINDS`; ``"barrier"`` for a step that
        declares no kind and whose class has no registered forward.
    """
    if _is_skipped(step):
        return "differentiable"
    declared = getattr(step, "gradient_kind", None)
    if declared in GRADIENT_KINDS:
        return str(declared)
    _load_registry()
    entry = _REGISTRY.get(type(step))
    return "barrier" if entry is None else entry[0](step)


def barrier(name: str, step: Any) -> Barrier | None:
    """The barrier that a fitted step is, if it is one.

    Args:
        name: The name of the step in its pipeline.
        step: The fitted step.

    Returns:
        The barrier with its reason, or ``None`` if a gradient passes.
    """
    if gradient_kind(step) != "barrier":
        return None
    reason = getattr(step, "barrier_reason", None)
    if not isinstance(reason, str):
        reason = (
            f"{type(step).__name__} has no differentiable forward that lazy"
            " knows"
        )
        hint = _REGISTRY.get(type(step), (None, None, ""))[2]
        reason += f"; {hint}" if hint else ""
    return Barrier(name, reason)


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
    if _is_skipped(step):
        return X
    if gradient_kind(step) == "barrier":
        found = barrier("step", step)
        raise BarrierError(found.reason if found else "the step is a barrier")
    if hasattr(step, "forward"):
        return step.forward(X)
    return _REGISTRY[type(step)][1](step, X)


def _load_registry() -> None:
    """Imports lazy's own entries, which register themselves on import."""
    # By name: that module imports this one.
    importlib.import_module("lazy._sklearn_forwards")
