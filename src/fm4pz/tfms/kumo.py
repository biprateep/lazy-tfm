# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Kumo Tabular, through lazy-tfm, with densities from its predicted quantiles.

Kumo's head predicts quantiles, which lazy-tfm turns into a distribution
with spread tails and puts on the common grid exactly. The search covers
the model's size (large, medium or small) beside the settings every
foundation model has (:mod:`fm4pz.tfms`); there is no softmax to set a
temperature on, so the sharpening power alone adjusts the widths.

Typical usage example:

  model = METHOD.fit(params, train, z_train, seed=0)
  masses = METHOD.masses(METHOD.predict(model, test), post)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna

from fm4pz import tfms

__all__ = ["METHOD", "Kumo"]

#: The ``transforms`` values searched: Kumo's own recipe, none and three of
#: lazy's uniform transforms.
TRANSFORMS: tuple[str, ...] = ("auto", "none", "power", "quantile", "robust")

#: The sizes of Kumo Tabular, in lazy-tfm's vocabulary.
VERSIONS: tuple[str, ...] = ("v1-large", "v1-medium", "v1-small")


class Kumo(tfms.FoundationModel):
    """Kumo Tabular as a :class:`fm4pz.cv.Method`."""

    name = "kumo"
    backend = "kumo"
    package = "structured-data-models"

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Proposes the model size, inputs, ensemble and preprocessing.

        Args:
            trial: The Optuna trial.

        Returns:
            The fit parameters.
        """
        params = self.suggest_common(trial, TRANSFORMS)
        params["version"] = trial.suggest_categorical("version", list(VERSIONS))
        return params

    def model_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The ``LazyModel`` arguments of a setting.

        Args:
            params: The fit parameters.

        Returns:
            Keyword arguments of ``lazy.LazyModel("kumo", ...)``.
        """
        return {
            "version": params["version"],
            **tfms.common_model_params(params),
        }

    def warm_starts(self) -> list[dict[str, Any]]:
        """lazy-tfm's defaults, on the large and medium sizes.

        Returns:
            Optuna parameters.
        """
        default = {
            "feature_set": "common12",
            "n_estimators": 8,
            "transforms": "auto",
            "outlier": "auto",
            "version": "v1-large",
        }
        return [
            default,
            {**default, "version": "v1-medium"},
            {**default, "feature_set": "common12_logerrs"},
        ]


METHOD = Kumo()
