# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""LimiX-2, through lazy-tfm, with the bucket masses of its bar distribution.

Beyond the settings every foundation model has (:mod:`fm4pz.tfms`), the
search covers the softmax temperature (0.9 is LimiX-2's released value) and,
where the context exceeds 20,000 rows, bagging. LimiX-2 was pretrained on
tables of up to 20,000 rows and its loss on DC1 grows beyond them; in the
submitted paper's study, members that each read a bag of 20,000 rows, or of
half the training set, recovered it. A bagged ensemble then has 16 or 32
members, so that the bags cover the context.

Typical usage example:

  model = METHOD.fit(params, train, z_train, seed=0)
  masses = METHOD.masses(METHOD.predict(model, test), post)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna

from fm4pz import config
from fm4pz import tfms

__all__ = ["METHOD", "LimiX"]

#: The ``transforms`` values searched: LimiX-2's own recipe, none and two of
#: lazy's uniform transforms.
TRANSFORMS: tuple[str, ...] = ("auto", "none", "power", "quantile")

#: The largest context LimiX-2 was pretrained on, in rows.
MAX_CONTEXT = 20_000

#: Query rows LimiX-2 predicts at a time: a production run's chunk of the
#: test set (:mod:`fm4pz.produce`) in one pass. A bagged member whose
#: context cache does not fit on the GPU runs its context again for every
#: chunk, and a query's answer does not depend on the others in its chunk,
#: up to float rounding.
QUERY_CHUNK = 10_000

#: Each bagging choice's ``bag_size``.
BAGS: dict[str, int | float | None] = {
    "none": None,
    "20k": MAX_CONTEXT,
    "half": 0.5,
}


class LimiX(tfms.FoundationModel):
    """LimiX-2 as a :class:`fm4pz.cv.Method`."""

    name = "limix"
    backend = "limix"
    package = "torch"

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Proposes the inputs, ensemble, preprocessing, temperature, bags.

        Bagging is proposed only when the folds' context, four fifths of
        the training set, exceeds :data:`MAX_CONTEXT`.

        Args:
            trial: The Optuna trial.

        Returns:
            The fit parameters.
        """
        params = self.suggest_common(trial, TRANSFORMS)
        params["softmax_temperature"] = trial.suggest_categorical(
            "softmax_temperature", [0.8, 0.9, 1.0, 1.1]
        )
        n_context = tfms.study_size(trial) * (config.CV_FOLDS - 1)
        if n_context // config.CV_FOLDS > MAX_CONTEXT:
            params["bag"] = trial.suggest_categorical("bag", list(BAGS))
            if params["bag"] != "none":
                params["n_estimators"] = trial.suggest_categorical(
                    "n_bags", [16, 32]
                )
        else:
            params["bag"] = "none"
        return params

    def model_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The ``LazyModel`` arguments of a setting.

        Args:
            params: The fit parameters.

        Returns:
            Keyword arguments of ``lazy.LazyModel("limix", ...)``.
        """
        return {
            "version": "v2",
            "softmax_temperature": float(params["softmax_temperature"]),
            "bag_size": BAGS[params.get("bag", "none")],
            "chunk_size": QUERY_CHUNK,
            **tfms.common_model_params(params),
        }

    def warm_starts(self) -> list[dict[str, Any]]:
        """LimiX-2's defaults, and with the bags of the submitted paper.

        Returns:
            Optuna parameters; the bagged ones apply only where bagging is
            searched.
        """
        default = {
            "feature_set": "common12",
            "n_estimators": 8,
            "transforms": "auto",
            "outlier": "auto",
            "softmax_temperature": 0.9,
            "bag": "none",
        }
        return [
            default,
            {**default, "bag": "half", "n_bags": 32},
            {**default, "bag": "20k", "n_bags": 16},
            {**default, "feature_set": "common12_logerrs"},
        ]


METHOD = LimiX()
