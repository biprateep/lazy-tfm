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


def frozen(*networks: Any) -> None:
    """Stops the networks' weights from collecting gradients, for good.

    lazy trains no weight, so a backward pass through a shared network
    computes no gradient for its weights, and leaves none on them.

    Args:
        *networks: ``torch.nn.Module`` instances.
    """
    for network in networks:
        network.requires_grad_(False)


def standardized(context: Any, query: Any, eps: float) -> tuple[Any, Any]:
    """Both tables z-scored by the context's columns, on the graph.

    The standard deviation is the population one, plus ``eps``, as the
    models' own scalers compute it; a column constant in the context is
    scaled by ``eps`` alone, with a finite gradient, since the square root
    of a zero variance has none.

    Args:
        context: Context features, a torch tensor of shape (n_rows,
            n_features).
        query: Query features, shape (n_queries, n_features).
        eps: Added to each standard deviation.

    Returns:
        A tuple (context, query), standardized.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    mean = context.mean(dim=0)
    var = context.var(dim=0, correction=0)
    std = torch.where(var > 0, torch.where(var > 0, var, 1.0).sqrt(), 0.0)
    std = std + eps
    return (context - mean) / std, (query - mean) / std


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


@dataclasses.dataclass(frozen=True)
class TorchQuantiles:
    """Values at fixed quantile levels, as a torch tensor.

    Its density is :meth:`lazy.distributions.QuantileDistribution.pdf`'s,
    on the graph.

    Attributes:
        levels: The cumulative probabilities, increasing inside [0, 1],
            shape (n_levels,).
        locs: The value at each level for each row, a float64 torch tensor
            of shape (n_rows, n_levels) that may carry gradients; made
            non-decreasing on construction, as the numpy class makes them.
    """

    levels: _typing.FloatArray
    locs: Any

    def __post_init__(self) -> None:
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        object.__setattr__(
            self, "locs", torch.cummax(self.locs.double(), dim=1).values
        )

    @classmethod
    def average(
        cls, parts: list[TorchQuantiles], weights: _typing.FloatArray
    ) -> TorchQuantiles:
        """The weighted average of quantile functions.

        Args:
            parts: Densities at the same levels, with the same rows.
            weights: Their weights, summing to 1, shape (n_parts,).

        Returns:
            The average, as
            :meth:`lazy.distributions.QuantileDistribution.average` gives.

        Raises:
            ValueError: If the parts' levels differ.
        """
        if any(not np.array_equal(p.levels, parts[0].levels) for p in parts):
            raise ValueError("every part must share the same levels")
        locs = sum(
            float(w) * part.locs for part, w in zip(parts, weights, strict=True)
        )
        return cls(parts[0].levels, locs)

    @classmethod
    def trimmed_average(
        cls, parts: list[TorchQuantiles], trim_fraction: float
    ) -> TorchQuantiles:
        """The trimmed mean of quantile functions.

        Args:
            parts: Densities at the same levels, with the same rows.
            trim_fraction: The fraction of parts dropped from each tail, in
                [0, 0.5).

        Returns:
            The trimmed mean, as
            :meth:`lazy.distributions.QuantileDistribution.trimmed_average`
            gives.

        Raises:
            ValueError: If the parts' levels differ.
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        if any(not np.array_equal(p.levels, parts[0].levels) for p in parts):
            raise ValueError("every part must share the same levels")
        locs = torch.stack([part.locs for part in parts])
        cut = int(trim_fraction * len(parts))
        if cut > 0:
            locs = torch.sort(locs, dim=0).values[cut : len(parts) - cut]
        return cls(parts[0].levels, locs.mean(dim=0))

    def log_prob(self, y: Any) -> Any:
        """The log-density of each row at its own value.

        Args:
            y: One value per row, shape (n_rows,), a torch tensor or array.

        Returns:
            The log-densities, a float64 torch tensor on the CPU, of shape
            (n_rows,).

        Raises:
            ValueError: If ``y`` does not have one value per row.
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        locs = self.locs.cpu()
        values = torch.as_tensor(y).detach().cpu().double().numpy()
        if values.shape != (locs.shape[0],):
            raise ValueError(
                f"y must have one value per row, shape ({locs.shape[0]},); "
                f"got {values.shape}"
            )
        # The piece holding each value carries no gradient; its width does.
        mass, gap = distributions._quantile_pieces(  # noqa: SLF001 - QuantileDistribution.pdf's rule.
            np.asarray(self.levels, dtype=np.float64),
            locs.detach().numpy(),
            values[:, None],
        )
        mass, gap = mass[:, 0], torch.as_tensor(gap[:, 0])
        rows = torch.arange(len(values))
        width = locs[rows, gap + 1] - locs[rows, gap]
        valid = torch.as_tensor(mass > 0) & (width > 0)
        log_mass = torch.log(torch.as_tensor(mass))
        return torch.where(
            valid,
            log_mass - torch.log(torch.where(valid, width, 1.0)),
            -math.inf,
        )
