# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Backend-free stand-ins that implement the per-backend interface.

Each member is a k-nearest-neighbour model on its own rows, columns and
transform, so every piece of the shared layer -- planning, scaffolded
transforms, bagging, combination, chunking, native grids -- runs on CPU in
milliseconds with nothing downloaded. Not photo-z methods.
"""

import types

import numpy as np

from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _ensemble


def _power(features):
    """A cheap stand-in for a model's own power transform."""
    return np.sign(features) * np.log1p(np.abs(features))


class _KNNStandIn(_ensemble.ContextEnsembleEstimator):
    """The machinery both stand-ins share."""

    display_name = "StandIn"
    extra = "none"
    cpu_friendly = True  # Milliseconds anywhere; never the CPU warning.

    def __init__(
        self,
        *,
        version="v0",
        n_estimators=8,
        transforms="auto",
        feature_shuffle=True,
        bag_size=None,
        kv_cache=True,
        z_grid=None,
        device="auto",
        random_state=0,
        chunk_size=8_192,
        softmax_temperature="auto",
        mixed_precision=True,
        outlier_threshold="auto",
        progress="auto",
        verbose=False,
        n_neighbors=12,
    ):
        self.version = version
        self.n_estimators = n_estimators
        self.transforms = transforms
        self.feature_shuffle = feature_shuffle
        self.bag_size = bag_size
        self.kv_cache = kv_cache
        self.z_grid = z_grid
        self.device = device
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.softmax_temperature = softmax_temperature
        self.mixed_precision = mixed_precision
        self.outlier_threshold = outlier_threshold
        self.progress = progress
        self.verbose = verbose
        self.n_neighbors = n_neighbors

    def _import_backend(self):
        return types.ModuleType("standin")

    def _load_checkpoint(self):
        self.checkpoint_ = None
        self.provenance_ = {"backend": "standin", "version": self.version}

    def _members(self, X, y, group):
        """Each member's rows, columns and transform, as the model would."""
        rng = np.random.default_rng(group.seed)
        members = []
        for j in range(group.n_members):
            rows = (
                np.arange(len(X))
                if group.member_rows is None
                else group.member_rows[j]
            )
            columns = (
                rng.permutation(X.shape[1])
                if group.feature_shuffle
                else np.arange(X.shape[1])
            )
            tokens = group.native_transforms or self.auto_tokens
            token = tokens[j % len(tokens)]
            members.append((X[rows][:, columns], y[rows], columns, token))
        return members

    def _neighbours(self, member, X):
        """The redshifts of each query's nearest context rows."""
        context, redshifts, columns, token = member
        query = X[:, columns]
        if token == "power":
            context, query = _power(context), _power(query)
        context = np.nan_to_num(context)
        query = np.nan_to_num(query)
        # Columns summed one after another, as the members are below: a
        # reduction over an axis may pair its sums by the array's shape.
        distance = sum(
            (query[:, None, j] - context[None, :, j]) ** 2
            for j in range(query.shape[1])
        )
        k = min(self.n_neighbors, len(redshifts))
        return redshifts[np.argsort(distance, axis=1)[:, :k]]


class HistogramStandIn(_KNNStandIn):
    """A bar-distribution stand-in with native transforms and bagging."""

    native_output = "histogram"
    native_transforms = {"none": "none", "power": "power"}
    auto_tokens = ("none", "power")
    supports_native_bagging = True

    def _fit_group(self, X, y, group):
        # Buckets placed by the group's own targets, as bar models do.
        borders = y.mean() + y.std() * np.linspace(-4.0, 4.0, 65)
        self.borders_ = borders
        return {"members": self._members(X, y, group), "borders": borders}

    def _predict_group(self, handle, X):
        borders = handle["borders"]
        masses = np.zeros((len(X), borders.size - 1))
        for member in handle["members"]:
            neighbours = self._neighbours(member, X)
            for row, values in enumerate(neighbours):
                counts, _ = np.histogram(values, bins=borders)
                masses[row] += counts + 1e-3
        return distributions.HistogramDistribution(borders, masses)

    def _native_grid(self):
        return grid_lib.Grid.from_edges(
            self.handles_[0]["borders"], normalization="histogram"
        )

    def _progress_postfix(self, dist):
        return {"buckets": len(dist.bins) - 1}


class ScaffoldedHistogramStandIn(HistogramStandIn):
    """The same, without native bagging: bags become one group per member."""

    supports_native_bagging = False


class QuantileStandIn(_KNNStandIn):
    """A quantile stand-in with one native transform and no bagging."""

    native_output = "quantiles"
    native_transforms = {"none": "none"}
    auto_tokens = ("none",)
    member_combination = "quantile_average"
    levels = np.linspace(0.05, 0.95, 19)

    def _fit_group(self, X, y, group):
        self.support_ = (float(y.min()), float(y.max()))
        return {"members": self._members(X, y, group)}

    def _predict_group(self, handle, X):
        # Members summed one after another, so every row's average is the
        # same arithmetic whatever the chunk: np.mean over an axis may pair
        # its sums differently for arrays of different shapes.
        members = [
            np.quantile(self._neighbours(m, X), self.levels, axis=1).T
            for m in handle["members"]
        ]
        locs = sum(members[1:], members[0]) / len(members)
        return distributions.QuantileDistribution(self.levels, locs)

    def _native_grid(self):
        low, high = self.support_
        pad = 0.02 * (high - low)
        return grid_lib.Grid.linear(
            low - pad, high + pad, 100, normalization="histogram"
        )

    def _progress_postfix(self, dist):
        return {"quantiles": dist.locs.shape[1]}


STANDINS = (HistogramStandIn, ScaffoldedHistogramStandIn, QuantileStandIn)
