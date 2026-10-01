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
:data:`PIPELINES`). :data:`RECIPE` pins upstream's recommended regression
recipe, ``config/reg_default_noretrieval_v2.json``, and the predictor
settings around it, written out so that the installed LimiX cannot change
them.

Typical usage example:

  seeds = member_seeds(seed=0, n_members=8)
  pipeline = MemberPipeline("power", seeds[0], shuffle=True).fit(context)
  prepared = pipeline.transform(queries)
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import random
import types
from typing import Any

import numpy as np

from lazy import _typing
from lazy.models import _limix_source

__all__ = [
    "AUTO_TOKENS",
    "NATIVE_TRANSFORMS",
    "PIPELINES",
    "RECIPE",
    "MemberPipeline",
    "Recipe",
    "member_seeds",
]


@dataclasses.dataclass(frozen=True)
class Recipe:
    """LimiX-2's own regression recipe, written out here rather than read.

    Every setting of upstream's ``LimiXPredictor`` that shapes a regression
    prediction under ``transforms="auto"``, copied from LimiX at
    :data:`lazy.models._limix_source.LIMIX_COMMIT`: the ensemble from
    ``config/reg_default_noretrieval_v2.json`` and the rest from
    ``inference/v2_0/predictor.py``. A different upstream release changes
    nothing here.

    Attributes:
        pipelines: Each member pipeline's ``RebalanceFeatureDistribution``
            settings, by token. Every pipeline then shuffles its columns
            (``FeatureShuffler(mode="shuffle")``); its categorical encoder
            has nothing to encode, as every feature is numeric.
        block: The config's members in order, as tokens.
        seeds_per_member: Seeds upstream draws per member
            (``preprocess_num``), one per possible step.
        rebalance_step: The rebalancing step's position in a member's
            pipeline (after the column filter), which picks its seed.
        shuffle_step: The shuffler's position (after the categorical
            encoder), which picks its seed.
        softmax_temperature: The temperature on the bucket logits
            (``LimiXPredictor``'s default).
        target_ddof: The degrees of freedom of the target's standard
            deviation, by which the context targets are standardised (a zero
            one is replaced by 1).
    """

    pipelines: Mapping[str, Mapping[str, Any]]
    block: tuple[str, ...]
    seeds_per_member: int
    rebalance_step: int
    shuffle_step: int
    softmax_temperature: float
    target_ddof: int


#: The pinned recipe; see :class:`Recipe`. Upstream at commit 516bf39.
RECIPE = Recipe(
    pipelines=types.MappingProxyType(
        {
            # Uniform quantiles (n // 5 of them), after the original columns,
            # then TruncatedSVD components of the two.
            "auto_quantile": {
                "worker_tags": ["quantile_uniform_all_data"],
                "discrete_flag": False,
                "original_flag": True,
                "svd_tag": "svd",
            },
            # Yeo-Johnson (upstream's RobustPowerTransformer), missing values
            # imputed to the mean, standardised again.
            "auto_power": {
                "worker_tags": ["power"],
                "discrete_flag": False,
                "original_flag": False,
                "svd_tag": None,
            },
        }
    ),
    block=("auto_quantile",) * 4 + ("auto_power",) * 4,
    seeds_per_member=10,
    rebalance_step=1,
    shuffle_step=3,
    softmax_temperature=0.9,
    target_ddof=1,
)

#: Upstream's recommended recipe, member by member: four quantile members
#: (with the original columns and SVD components), then four power members.
AUTO_TOKENS: tuple[str, ...] = RECIPE.block

# Upstream's name for each transform in the shared vocabulary.
_WORKER_TAGS: dict[str, str | None] = {
    "none": None,
    "power": "power",
    "quantile": "quantile_norm_all_data",
    "quantile_uniform": "quantile_uniform_all_data",
    "robust": "robust",
}

#: Upstream ``RebalanceFeatureDistribution`` settings per token: every
#: vocabulary transform LimiX has, alone and ``+original``, and the members
#: of upstream's recipe.
PIPELINES: dict[str, Mapping[str, Any]] = {
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
    **RECIPE.pipelines,
}

#: Vocabulary transform name to token, for the transforms LimiX has.
NATIVE_TRANSFORMS: dict[str, str] = {
    name: name for name in PIPELINES if name not in RECIPE.pipelines
}


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
    per_member = RECIPE.seeds_per_member
    draws = [rng.randint(0, 10_000) for _ in range(n_members * per_member)]
    return [
        (
            draws[per_member * i + RECIPE.rebalance_step],
            draws[per_member * i + RECIPE.shuffle_step],
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
