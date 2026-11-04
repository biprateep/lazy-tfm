# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Kumo Tabular's random recipe steps, drawn the same on every device.

Upstream's ``FlipSign`` and ``ShuffleColumns`` draw each member's sign flips
and column permutations with the generator on the data's device, and a CUDA
generator gives another stream than a CPU one for the same seed
(NVIDIA/structured-data-models#1339). These subclasses draw on the
generator's own device, a CPU one in :mod:`lazy.models.kumo`, and move the
draws to the data's device. On a CPU they are upstream's own draws. They
override upstream's private ``_fit_ensemble`` as of
:data:`lazy.models.kumo.REPOSITORY_COMMIT`.

This module imports ``sdm``, so :mod:`lazy.models.kumo` imports it only when
it builds a recipe. Pickled recipes refer to its classes.
"""

from __future__ import annotations

from typing import Any

from sdm import processing
import torch


class FlipSign(processing.FlipSign):
    """Upstream's ``FlipSign``, drawn on the generator's device."""

    _signs: Any  # Upstream's list of buffers, one per member.

    def _fit_ensemble(
        self, ensemble_table: Any, *, generator: Any = None
    ) -> None:
        signs = []
        for member_id in range(len(ensemble_table)):
            numerical = ensemble_table[member_id].numerical
            sign = torch.empty(
                (*numerical.shape[:-2], 1, numerical.shape[-1]),
                dtype=numerical.dtype,
                device=_draw_device(generator, numerical),
            )
            sign.bernoulli_(self.probability, generator=generator)
            signs.append(sign.mul_(-2).add_(1).to(numerical.device))
        self._signs = type(self._signs)(signs)


class ShuffleColumns(processing.ShuffleColumns):
    """Upstream's Latin ``ShuffleColumns``, drawn on the generator's device.

    Raises:
        ValueError: At fit, for any method but ``"latin"``.
    """

    _permutations: Any  # Upstream's list of buffers, one per member.

    def _fit_ensemble(
        self, ensemble_table: Any, *, generator: Any = None
    ) -> None:
        if self.method != "latin":
            raise ValueError(
                f"only the latin method is drawn, not {self.method!r}"
            )
        numerical = ensemble_table[0].numerical
        widths = [
            ensemble_table[member_id].numerical.shape[-1]
            for member_id in range(len(ensemble_table))
        ]
        width = max(widths)
        device = _draw_device(generator, numerical)
        base_rank = torch.randperm(width, generator=generator, device=device)
        row_rank = torch.randperm(width, generator=generator, device=device)
        permutations = []
        for member_id, n_features in enumerate(widths):
            if n_features <= 1:
                permutations.append(torch.arange(n_features, device=device))
                continue
            base = base_rank[base_rank < n_features]
            rows = row_rank[row_rank < n_features]
            pattern = member_id % n_features
            permutations.append(base[(pattern - rows) % n_features])
        self._permutations = type(self._permutations)(
            [permutation.to(numerical.device) for permutation in permutations]
        )
        self._host_permutations = tuple(
            tuple(permutation.tolist()) for permutation in permutations
        )


def _draw_device(generator: Any, data: Any) -> Any:
    """The generator's device, or the data's for torch's global generator."""
    return data.device if generator is None else generator.device
