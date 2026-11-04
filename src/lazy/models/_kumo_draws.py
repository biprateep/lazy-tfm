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

from typing import Any

from sdm import processing


class FlipSign(processing.FlipSign):
    """Upstream's ``FlipSign``, drawn on the generator's device."""

    def _fit_ensemble(self, ensemble_table: Any, *, generator: Any) -> None:
        super()._fit_ensemble(
            _empty_on(ensemble_table, generator.device), generator=generator
        )
        self.to(ensemble_table.device)


class ShuffleColumns(processing.ShuffleColumns):
    """Upstream's ``ShuffleColumns``, drawn on the generator's device."""

    def _fit_ensemble(self, ensemble_table: Any, *, generator: Any) -> None:
        super()._fit_ensemble(
            _empty_on(ensemble_table, generator.device), generator=generator
        )
        self.to(ensemble_table.device)


def _empty_on(ensemble_table: Any, device: Any) -> Any:
    """The ensemble table with no rows, on ``device``: its shape alone."""
    return ensemble_table._apply_tensor(  # noqa: SLF001 - upstream's map.
        lambda table: table.narrow(-2, 0, 0).to(device)
    )
