# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Kumo Tabular's random recipe steps, drawn the same on every device.

Upstream's ``FlipSign`` and ``ShuffleColumns`` draw each member's sign flips
and column permutations on the data's device, and a CUDA generator gives
another stream than a CPU one for the same seed
(NVIDIA/structured-data-models#1339). The subclasses here, ``FlipSign`` and
``ShuffleColumns``, run upstream's own draw on a table with no rows on the
generator's device, a CPU one in :mod:`lazy.models.kumo`, and move the drawn
buffers to the data's device. Upstream's draws read only the table's shape,
dtype and member layout, so on a CPU these are upstream's draws. They wrap
upstream's private ``_fit_ensemble`` and ``EnsembleTable._apply_tensor`` as
of :data:`lazy.models.kumo.REPOSITORY_COMMIT`.

``sdm`` is an optional extra, so the classes are built on first access, as
attributes of this module; pickled recipes find them the same way.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any

_STEPS = ("FlipSign", "ShuffleColumns")


def __getattr__(name: str) -> type:
    """``FlipSign`` and ``ShuffleColumns``, built on first access."""
    if name in _STEPS:
        return _steps()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@functools.cache
def _steps() -> dict[str, type]:
    """Upstream's two random steps, subclassed to draw on the generator."""
    from sdm import processing  # noqa: PLC0415 - sdm is an optional extra.

    class FlipSign(processing.FlipSign):
        """Upstream's ``FlipSign``, drawn on the generator's device."""

        def _fit_ensemble(
            self, ensemble_table: Any, *, generator: Any = None
        ) -> None:
            _fit_on_generator(
                self, super()._fit_ensemble, ensemble_table, generator
            )

    class ShuffleColumns(processing.ShuffleColumns):
        """Upstream's ``ShuffleColumns``, drawn on the generator's device."""

        def _fit_ensemble(
            self, ensemble_table: Any, *, generator: Any = None
        ) -> None:
            _fit_on_generator(
                self, super()._fit_ensemble, ensemble_table, generator
            )

    steps: dict[str, type] = {
        "FlipSign": FlipSign,
        "ShuffleColumns": ShuffleColumns,
    }
    for name, step in steps.items():
        # Pickle finds a class by its module and qualified name.
        step.__module__ = __name__
        step.__qualname__ = name
    return steps


def _fit_on_generator(
    step: Any,
    fit: Callable[..., None],
    ensemble_table: Any,
    generator: Any,
) -> None:
    """Runs upstream's draw on the generator's device, then moves the step.

    Args:
        step: The step being fitted; its buffers move to the data's device.
        fit: Upstream's ``_fit_ensemble``, bound to ``step``.
        ensemble_table: The members' tables, on the data's device.
        generator: The ``torch.Generator`` to draw from, or None for torch's
            global one on the data's device, as upstream draws.
    """
    if generator is None:
        fit(ensemble_table, generator=None)
        return
    empty = ensemble_table._apply_tensor(  # noqa: SLF001 - upstream's map.
        lambda table: table.narrow(-2, 0, 0).to(generator.device)
    )
    fit(empty, generator=generator)
    step.to(ensemble_table.device)
