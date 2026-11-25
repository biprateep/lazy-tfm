# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Predicted densities that keep torch's autograd graph.

Their :meth:`log_prob` takes gradients, for a loss or an attribution, with
respect to the features the densities came from.

Typical usage example:

  density = model._differentiable_density(query, context, y_context)
  (-density.log_prob(y_query).mean()).backward()
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from lazy import _typing
from lazy import distributions


@dataclasses.dataclass(frozen=True)
class TorchHistogram:
    """Probability masses over fixed buckets, as a torch tensor.

    Attributes:
        edges: The bucket edges in the target's units, non-decreasing, shape
            (n_buckets + 1,), float64.
        log_masses: The log-probability of each bucket for each row, a
            torch tensor of shape (n_rows, n_buckets) that may carry
            gradients.
    """

    edges: _typing.FloatArray
    log_masses: Any

    def log_prob(self, y: Any) -> Any:
        """The log-density of each row at its own value.

        The density inside a bucket is its mass over its width, as in
        :meth:`lazy.distributions.HistogramDistribution.pdf`; the last edge
        belongs to the last bucket, and the log-density is ``-inf`` outside
        the buckets and in a zero-width bucket.

        Args:
            y: One value per row, shape (n_rows,), a torch tensor or array.

        Returns:
            The log-densities, a float64 torch tensor on the CPU, of shape
            (n_rows,).

        Raises:
            ValueError: If ``y`` does not have one value per row.
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        values = torch.as_tensor(y).detach().cpu().double().numpy()
        if values.shape != (self.log_masses.shape[0],):
            raise ValueError(
                f"y must have one value per row, shape "
                f"({self.log_masses.shape[0]},); got {values.shape}"
            )
        # The bucket carries no gradient, so numpy finds it.
        index = distributions._closed_bucket(self.edges, values)  # noqa: SLF001 - HistogramDistribution.pdf's rule.
        widths = np.diff(self.edges)
        inside = (index >= 0) & (index < widths.size)
        index = np.clip(index, 0, widths.size - 1)
        index_t = torch.as_tensor(index, device=self.log_masses.device)
        # On the CPU from here, in float64, which an Apple GPU lacks.
        rows = self.log_masses.gather(1, index_t[:, None])[:, 0].cpu().double()
        log_widths = np.log(np.where(widths[index] > 0, widths[index], 1.0))
        valid = torch.as_tensor(inside & (widths[index] > 0))
        return torch.where(valid, rows - torch.as_tensor(log_widths), -math.inf)


@dataclasses.dataclass(frozen=True)
class TorchMixture:
    """A weighted mixture of densities, as a model pools its member groups.

    Attributes:
        parts: The components, each with :meth:`log_prob`.
        weights: Their weights, summing to 1, shape (n_parts,).
    """

    parts: tuple[TorchHistogram, ...]
    weights: _typing.FloatArray

    def log_prob(self, y: Any) -> Any:
        """The log-density of each row at its own value.

        Args:
            y: One value per row, shape (n_rows,), a torch tensor or array.

        Returns:
            The log-densities, a float64 torch tensor of shape (n_rows,).
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        terms = torch.stack([part.log_prob(y) for part in self.parts])
        log_weights = torch.log(torch.as_tensor(self.weights))
        return torch.logsumexp(terms + log_weights[:, None], dim=0)
