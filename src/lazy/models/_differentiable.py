# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Predicted densities that keep torch's autograd graph.

Their :meth:`log_prob`, :meth:`on_grid` and :meth:`mean` take gradients, for
a loss or an attribution, with respect to the features the densities came
from. ``on_grid`` and ``mean`` run the code of their numpy twins in
:mod:`lazy.distributions`, which is written for any array, so they give what
the twins give, to float rounding. As there, every search for an index (a
bucket, a piece of a quantile function) runs in numpy on detached values and
carries no gradient; the gradients pass through the masses and the values.

Typical usage example:

  density = model._differentiable_density(query, context, y_context)
  (-density.log_prob(y_query).mean()).backward()
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from lazy import _arrays
from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy import metrics


def frozen(*networks: Any) -> None:
    """Stops the networks' weights from collecting gradients, for good.

    lazy trains no weight, so a backward pass through a shared network
    computes no gradient for its weights, and leaves none on them.

    Args:
        *networks: ``torch.nn.Module`` instances.
    """
    for network in networks:
        network.requires_grad_(False)


def mean_of_softmaxes(log_probabilities: list[Any]) -> Any:
    """The log of the members' average probabilities, on the graph.

    Args:
        log_probabilities: Each member's log-probabilities, torch tensors of
            the same shape, the buckets on the last axis.

    Returns:
        ``log(mean(exp(...)))`` over the members, of that shape.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    stacked = torch.stack(log_probabilities)
    return torch.logsumexp(stacked, dim=0) - math.log(len(log_probabilities))


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


def standardized_members(
    context: Any, query: Any, keep: Any, shuffles: list[Any]
) -> Any:
    """Each member's view of the features, standardized, on the graph.

    The ``"none"`` view of in-context models whose members standardize
    finite numeric features and shuffle their columns: the columns kept at
    fit, standardized on the context with ``eps`` 1e-6 and clipped to
    [-100, 100], then each member's column order, the context rows before
    the queries.

    Args:
        context: Context features, a torch tensor of shape (n_rows,
            n_features).
        query: Query features, a torch tensor of shape (n_queries,
            n_features).
        keep: Which columns the model kept, bool, shape
            (n_features,).
        shuffles: Each member's column order over the kept columns.

    Returns:
        The members' features, float32 on the CPU, shape (n_members, n_rows
        + n_queries, n_kept).
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    keep = torch.as_tensor(keep)
    # On the CPU for float64, which an Apple GPU lacks.
    context, query = standardized(
        context.cpu()[:, keep].double(), query.cpu()[:, keep].double(), 1e-6
    )
    both = torch.cat([context, query]).clamp(-100, 100)
    return torch.stack([both[:, shuffle] for shuffle in shuffles]).float()


class _OnGrid:
    """Gives a torch density :meth:`on_grid` from its ``_bin_densities``."""

    def _bin_densities(self, grid: grid_lib.Grid) -> Any:
        """The mass in each bin of ``grid`` over its width, unnormalized."""
        raise NotImplementedError

    def on_grid(self, grid: grid_lib.Grid) -> Any:
        """Densities on ``grid``, as the numpy twin's ``on_grid`` gives them.

        Args:
            grid: The grid to put the densities on.

        Returns:
            Normalized densities, a float64 torch tensor on the CPU of
            shape (n_rows, grid.n_bins).
        """
        return metrics._normalized(  # noqa: SLF001 - Grid.normalize's code.
            grid.centers, grid.histogram_edges, self._bin_densities(grid)
        )


@dataclasses.dataclass(frozen=True)
class TorchHistogram(_OnGrid):
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
        index = distributions._closed_bucket(self.edges, values)  # noqa: SLF001 - HistogramDistribution.pdf's rule.
        widths = np.diff(self.edges)
        inside = (index >= 0) & (index < widths.size)
        index = np.clip(index, 0, widths.size - 1)
        index_t = torch.as_tensor(index, device=self.log_masses.device)
        # On the CPU from here, in float64, which an Apple GPU lacks.
        rows = self.log_masses.gather(1, index_t[:, None])[:, 0].cpu().double()
        log_widths = np.log(np.where(widths[index] > 0, widths[index], 1.0))
        # An empty bucket's log-mass is -inf, whose gradient would be NaN.
        valid = torch.as_tensor(inside & (widths[index] > 0)) & rows.isfinite()
        return torch.where(valid, rows - torch.as_tensor(log_widths), -math.inf)

    def _probabilities(self) -> Any:
        """The masses, each row renormalized, a float64 tensor on the CPU.

        As :attr:`lazy.distributions.HistogramDistribution.probabilities`
        gives them for ``exp(log_masses)``.
        """
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        masses = torch.exp(self.log_masses.cpu().double())
        return distributions._probabilities(masses, np.diff(self.edges))  # noqa: SLF001 - HistogramDistribution's code.

    def _bin_densities(self, grid: grid_lib.Grid) -> Any:
        """HistogramDistribution's unnormalized densities on ``grid``."""
        return distributions._histogram_bin_densities(  # noqa: SLF001 - HistogramDistribution's code.
            self.edges, self._probabilities(), grid.edges
        )

    def mean(self) -> Any:
        """The mean of each row, a float64 tensor on the CPU, (n_rows,)."""
        return distributions._histogram_mean(self.edges, self._probabilities())  # noqa: SLF001 - HistogramDistribution's code.


@dataclasses.dataclass(frozen=True)
class TorchMixture(_OnGrid):
    """A weighted mixture of densities, as a model pools its parts.

    Attributes:
        parts: The components, each with :meth:`log_prob`.
        weights: Their weights, summing to 1, shape (n_parts,).
    """

    parts: tuple[TorchDensity, ...]
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

    def _bin_densities(self, grid: grid_lib.Grid) -> Any:
        """The parts' unnormalized densities on ``grid``, weighted.

        :meth:`on_grid` normalizes their sum, as
        :meth:`lazy.distributions.MixtureDistribution.on_grid` does.
        """
        return _arrays.weighted_total(
            self.weights,
            (part._bin_densities(grid) for part in self.parts),  # noqa: SLF001 - one implementation.
        )

    def mean(self) -> Any:
        """The mean of each row, a float64 tensor on the CPU, (n_rows,)."""
        return _arrays.weighted_total(
            self.weights, (part.mean() for part in self.parts)
        )


@dataclasses.dataclass(frozen=True)
class TorchQuantiles(_OnGrid):
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

    def _bin_densities(self, grid: grid_lib.Grid) -> Any:
        """QuantileDistribution's unnormalized densities on ``grid``."""
        return distributions._quantile_bin_densities(  # noqa: SLF001 - QuantileDistribution's code.
            np.asarray(self.levels, dtype=np.float64),
            self.locs.cpu(),
            grid.edges,
        )

    def mean(self) -> Any:
        """The mean of each row, a float64 tensor on the CPU, (n_rows,)."""
        return distributions._quantile_mean(  # noqa: SLF001 - QuantileDistribution's code.
            np.asarray(self.levels, dtype=np.float64), self.locs.cpu()
        )


TorchDensity = TorchHistogram | TorchMixture | TorchQuantiles
"""A density on torch's autograd graph, with ``log_prob``, ``on_grid`` and
``mean``."""
