# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Scoring on the common redshift grid, and the DC1 metric for comparison.

Every method hands over its distributions as the probability in each of the
4,000 bins of :data:`FINE_GRID` (width 0.0005 on [0, 2]), computed exactly from
its own representation, with the probability outside [0, 2] dropped and each
row renormalised. The CDE loss on that grid treats the density as constant in
each bin, so it is exact for the masses given. For comparison with Schmidt et
al. (2020), :func:`score` also rebins the masses to the 200 bins of the Data
Challenge and scores them as the challenge did, by the trapezoid rule over the
bin centres with the truth at the nearest centre. Both go through
``lazy.metrics``, which the foundation models are scored with as well.

Typical usage example:

  masses = scoring.gaussian_masses(loc, scale)
  per_object = scoring.score(z_true, masses)
  scoring.summary(z_true, per_object)
"""

from __future__ import annotations

import dataclasses
from typing import Any

from lazy import datasets
from lazy import grid as grid_lib
from lazy import metrics
import numpy as np
import numpy.typing as npt
from scipy import special

from fm4pz import config

__all__ = [
    "BIN_WIDTH",
    "EDGES",
    "FINE_GRID",
    "gaussian_masses",
    "normalize_masses",
    "score",
    "summary",
]

FloatArray = npt.NDArray[np.float64]

#: The common grid: 4,000 bins of width 0.0005 on [0, 2], histogram convention.
FINE_GRID = grid_lib.Grid.linear(
    config.Z_MIN, config.Z_MAX, config.FINE_BINS, normalization="histogram"
)
EDGES: FloatArray = np.asarray(FINE_GRID.edges, dtype=np.float64)
BIN_WIDTH = float(EDGES[1] - EDGES[0])

# Fine bins per bin of the Data Challenge's 200-bin grid.
_PER_DC1_BIN = config.FINE_BINS // datasets.DC1_GRID.n_bins


def normalize_masses(masses: npt.ArrayLike) -> FloatArray:
    """Clips negative masses to zero and makes every row sum to one.

    A row with no probability on the grid at all becomes uniform, which is
    the most honest answer to "nothing landed in [0, 2]".

    Args:
        masses: Probability per bin, shape (n_rows, 4000).

    Returns:
        The normalised masses, float64, same shape.
    """
    out = np.clip(np.asarray(masses, dtype=np.float64), 0.0, None)
    total = out.sum(axis=1, keepdims=True)
    empty = total[:, 0] <= 0.0
    out[empty] = 1.0
    total[empty] = out.shape[1]
    return out / total


def gaussian_masses(loc: npt.ArrayLike, scale: npt.ArrayLike) -> FloatArray:
    """Exact bin masses of Gaussians truncated to the grid.

    Args:
        loc: Means, shape (n_rows,), in redshift.
        scale: Standard deviations, shape (n_rows,), positive.

    Returns:
        Masses on :data:`FINE_GRID`, shape (n_rows, 4000), rows summing to 1.
    """
    loc = np.asarray(loc, dtype=np.float64)[:, None]
    scale = np.asarray(scale, dtype=np.float64)[:, None]
    cdf = special.ndtr((EDGES[None, :] - loc) / scale)
    return normalize_masses(np.diff(cdf, axis=1))


def score(
    z_true: npt.ArrayLike, masses: npt.ArrayLike
) -> dict[str, FloatArray]:
    """Per-object scores of distributions on the common grid.

    Args:
        z_true: True redshifts, shape (n_rows,).
        masses: Probability per bin of :data:`FINE_GRID`, shape
            (n_rows, 4000); renormalised here.

    Returns:
        Arrays of shape (n_rows,): ``cde`` and ``pit`` on the common grid,
        ``cde_dc1`` and ``pit_dc1`` on the Data Challenge's 200-bin grid,
        and the point estimates ``z_peak`` and ``z_mean`` from the common
        grid.
    """
    z_true = np.asarray(z_true, dtype=np.float64)
    masses = normalize_masses(masses)
    density = masses / BIN_WIDTH
    cde, pit = metrics.per_object_scores(z_true, FINE_GRID, density)
    coarse = masses.reshape(len(masses), -1, _PER_DC1_BIN).sum(axis=2)
    dc1_width = _PER_DC1_BIN * BIN_WIDTH
    cde_dc1, pit_dc1 = metrics.per_object_scores(
        z_true, datasets.DC1_GRID, coarse / dc1_width
    )
    points = metrics.grid_point_estimates(FINE_GRID, density)
    return {
        "cde": np.asarray(cde),
        "pit": np.asarray(pit),
        "cde_dc1": np.asarray(cde_dc1),
        "pit_dc1": np.asarray(pit_dc1),
        "z_peak": np.asarray(points["mode"]),
        "z_mean": np.asarray(points["mean"]),
    }


def summary(
    z_true: npt.ArrayLike, per_object: dict[str, FloatArray]
) -> dict[str, Any]:
    """The scalar metrics of a run, from its per-object scores.

    Args:
        z_true: True redshifts, shape (n_rows,).
        per_object: What :func:`score` returned.

    Returns:
        ``cde_loss`` and ``cde_loss_dc1`` (means), the point metrics of the
        mode on the (1 + z) scale (``sigma_iqr``, ``bias``, ``outlier_rate``
        and the rest of ``lazy.metrics.PointMetrics``), and the PIT
        statistics on the common grid.
    """
    z_true = np.asarray(z_true, dtype=np.float64)
    point = metrics.point_metrics(z_true, per_object["z_peak"], scale="1+y")
    out: dict[str, Any] = {
        "cde_loss": float(np.mean(per_object["cde"])),
        "cde_loss_dc1": float(np.mean(per_object["cde_dc1"])),
        "n_test": int(z_true.size),
    }
    out.update(
        {
            key: float(value)
            for key, value in dataclasses.asdict(point).items()
            if key != "scale"
        }
    )
    out.update(
        {
            key: float(value)
            for key, value in metrics.pit_statistics(per_object["pit"]).items()
        }
    )
    return out
