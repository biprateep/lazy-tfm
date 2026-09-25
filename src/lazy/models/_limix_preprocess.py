# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LimiX's per-member feature preprocessing, fitted on the context alone.

Upstream's ``LimiXPredictor`` builds each ensemble member a pipeline --
drop constant columns, rebalance the feature distributions (a power or
quantile transform, optionally with the original columns and SVD components
appended), encode categoricals, shuffle the columns -- and runs it on the
context and query rows stacked together. Only the column filter actually
looks at the query rows, so here every step is fitted on the context and then
applied to any rows: a query's features never depend on which other queries
share its chunk. The distribution and shuffle steps are upstream's own
classes, seeded exactly as upstream seeds them; the column filter is
upstream's rule restricted to the context; categorical encoding is skipped,
as every feature is treated as numeric.

Each member is described by a *token* naming its rebalancing step (see
:data:`PIPELINES`); :data:`AUTO_TOKENS` is upstream's recommended regression
recipe, ``config/reg_default_noretrieval_v2.json``.

Typical usage example:

  seeds = member_seeds(seed=0, n_members=8)
  pipeline = MemberPipeline("power", seeds[0], shuffle=True).fit(context)
  prepared = pipeline.transform(queries)
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from lazy import _typing
from lazy.models import _limix_source

__all__ = [
    "AUTO_TOKENS",
    "NATIVE_TRANSFORMS",
    "PIPELINES",
    "MemberPipeline",
    "member_seeds",
]

# Upstream's name for each transform in the shared vocabulary.
_WORKER_TAGS: dict[str, str | None] = {
    "none": None,
    "power": "power",
    "quantile": "quantile_norm_all_data",
    "quantile_uniform": "quantile_uniform_all_data",
    "robust": "robust",
}

#: Upstream ``RebalanceFeatureDistribution`` settings per token: every
#: vocabulary transform LimiX has, alone and ``+original``, and the quantile
#: member of upstream's recipe (which also appends SVD components).
PIPELINES: dict[str, dict[str, Any]] = {
    **{
        name: {"worker_tags": [tag], "original_flag": False, "svd_tag": None}
        for name, tag in _WORKER_TAGS.items()
    },
    **{
        f"{name}+original": {
            "worker_tags": [tag],
            "original_flag": True,
            "svd_tag": None,
        }
        for name, tag in _WORKER_TAGS.items()
    },
    "quantile_uniform+original+svd": {
        "worker_tags": ["quantile_uniform_all_data"],
        "original_flag": True,
        "svd_tag": "svd",
    },
}

#: Vocabulary transform name to token, for the transforms LimiX has.
NATIVE_TRANSFORMS: dict[str, str] = {
    name: name for name in PIPELINES if not name.endswith("+svd")
}

#: Upstream's recommended recipe, member by member: four quantile members
#: (with the original columns and SVD components), then four power members.
AUTO_TOKENS: tuple[str, ...] = ("quantile_uniform+original+svd",) * 4 + (
    "power",
) * 4

# Upstream reserves this many seeds per member, one per possible step.
_SEEDS_PER_MEMBER = 10
# The steps' positions in upstream's pipeline: the column filter, the
# rebalancing, the categorical encoder, the shuffler.
_REBALANCE_STEP, _SHUFFLE_STEP = 1, 3


def member_seeds(seed: int, n_members: int) -> list[tuple[int, int]]:
    """Each member's (rebalance, shuffle) seeds, as upstream derives them.

    Upstream seeds Python's ``random`` with the predictor's seed and draws
    ten integers per member; step ``j`` of member ``i`` takes draw
    ``10 * i + j``. A private ``random.Random`` gives the same draws without
    touching the global generator.

    Args:
        seed: The group's seed.
        n_members: How many members it serves.

    Returns:
        One ``(rebalance_seed, shuffle_seed)`` pair per member.

    Examples:
        >>> member_seeds(0, 2)
        [(6890, 4242), (3578, 2281)]
    """
    rng = random.Random(seed)
    draws = [
        rng.randint(0, 10_000) for _ in range(n_members * _SEEDS_PER_MEMBER)
    ]
    return [
        (
            draws[_SEEDS_PER_MEMBER * i + _REBALANCE_STEP],
            draws[_SEEDS_PER_MEMBER * i + _SHUFFLE_STEP],
        )
        for i in range(n_members)
    ]


class MemberPipeline:
    """One member's preprocessing, fitted on the context and then reused.

    Args:
        token: The rebalancing step, a key of :data:`PIPELINES`.
        seeds: The member's ``(rebalance, shuffle)`` seeds.
        shuffle: Whether the member permutes its columns (upstream's
            ``FeatureShuffler`` in ``"shuffle"`` mode).

    Attributes:
        keep_: Which input columns carry information on the context.
        n_features_out_: How many columns the network sees.
    """

    def __init__(
        self, token: str, seeds: tuple[int, int], *, shuffle: bool
    ) -> None:
        if token not in PIPELINES:
            raise ValueError(
                f"unknown LimiX pipeline {token!r}; known: {sorted(PIPELINES)}"
            )
        self.token = token
        self.seeds = seeds
        self.shuffle = shuffle

    def fit(self, context: _typing.FloatArray) -> MemberPipeline:
        """Fits every step on the context rows.

        Args:
            context: Context features, shape (n_rows, n_features); NaN marks
                a missing value.

        Returns:
            This pipeline, fitted.

        Raises:
            ValueError: If every column is constant or missing on the
                context.
        """
        preprocess = _limix_source.load().preprocess
        # Upstream's rule -- drop a column equal to its first row everywhere
        # -- plus its drop of all-missing columns, on the context alone.
        constant = (context[:1] == context).mean(axis=0) == 1.0
        self.keep_ = ~constant & ~np.isnan(context).all(axis=0)
        if not self.keep_.any():
            raise ValueError(
                "every feature is constant or missing on the context"
            )
        kept = context[:, self.keep_]
        rebalance_seed, shuffle_seed = self.seeds
        self._rebalance = preprocess.RebalanceFeatureDistribution(
            **PIPELINES[self.token], enable_parallel=False
        )
        self._rebalance.fit(kept, [], rebalance_seed)
        rebalanced, _ = self._rebalance.transform(kept)
        self._shuffler = preprocess.FeatureShuffler(
            mode="shuffle" if self.shuffle else None
        )
        self._shuffler.fit(rebalanced, [], shuffle_seed)
        self.n_features_out_ = int(rebalanced.shape[1])
        return self

    def transform(self, features: _typing.FloatArray) -> _typing.FloatArray:
        """Applies the fitted steps to any rows.

        Args:
            features: Features in the context's columns, shape
                (n_rows, n_features).

        Returns:
            The member's view, shape (n_rows, ``n_features_out_``).
        """
        rebalanced, _ = self._rebalance.transform(features[:, self.keep_])
        shuffled, _ = self._shuffler.transform(rebalanced)
        return np.asarray(shuffled, dtype=np.float64)
