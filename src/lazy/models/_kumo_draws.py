# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Kumo Tabular's random recipe steps, drawn the same on every device.

Upstream's ``FlipSign`` and ``ShuffleColumns`` draw each member's sign flips
and column permutations on the data's device, and a CUDA generator gives
another stream than a CPU one for the same seed
(NVIDIA/structured-data-models#1339). These subclasses run upstream's own
draw on a table with no rows on the generator's device, a CPU one in
:mod:`lazy.models.kumo`, and move the drawn buffers to the data's device.
Upstream's draws read only the table's shape, dtype and member layout, so on
a CPU these are upstream's draws. They wrap upstream's private
``_fit_ensemble`` and ``EnsembleTable._apply_tensor`` as of
:data:`lazy.models.kumo.REPOSITORY_COMMIT`.

This module imports ``sdm``, so :mod:`lazy.models.kumo` imports it only when
it builds a recipe. Pickled recipes refer to its classes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sdm import processing


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
