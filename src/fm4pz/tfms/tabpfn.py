# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""TabPFN-3.5, through lazy-tfm, with the bucket masses of its bar distribution.

Beyond the settings every foundation model has (:mod:`fm4pz.tfms`), the
search covers the softmax temperature on the bucket logits, which sets how
sharp the densities are before the ensemble is averaged; 1.0 is the value
TabPFN-3.5 was calibrated with. Bagging is not searched: it hurt TabPFN in
the submitted paper's study, whose context limit DC1 never reaches.

Typical usage example:

  model = METHOD.fit(params, train, z_train, seed=0)
  masses = METHOD.masses(METHOD.predict(model, test), post)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna

from fm4pz import tfms

__all__ = ["METHOD", "TabPFN"]

#: The ``transforms`` values searched: TabPFN-3.5's own recipe, none, two
#: of lazy's uniform transforms, and LimiX-2's recipe.
TRANSFORMS: tuple[str, ...] = ("auto", "none", "power", "quantile", "limix")


class TabPFN(tfms.FoundationModel):
    """TabPFN-3.5 as a :class:`fm4pz.cv.Method`."""

    name = "tabpfn"
    backend = "tabpfn"
    package = "tabpfn"

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Proposes the inputs, ensemble, preprocessing and temperature.

        Args:
            trial: The Optuna trial.

        Returns:
            The fit parameters.
        """
        params = self.suggest_common(trial, TRANSFORMS)
        params["softmax_temperature"] = trial.suggest_categorical(
            "softmax_temperature", [0.8, 0.9, 1.0, 1.1]
        )
        return params

    def model_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The ``LazyModel`` arguments of a setting.

        Args:
            params: The fit parameters.

        Returns:
            Keyword arguments of ``lazy.LazyModel("tabpfn", ...)``.
        """
        return {
            "version": "v3.5",
            "softmax_temperature": float(params["softmax_temperature"]),
            **tfms.common_model_params(params),
        }

    def warm_starts(self) -> list[dict[str, Any]]:
        """The submitted paper's setting, and its two other input sets.

        Returns:
            Optuna parameters.
        """
        default = {
            "feature_set": "common12",
            "n_estimators": 8,
            "transforms": "auto",
            "outlier": "auto",
            "softmax_temperature": 1.0,
        }
        return [
            default,
            {**default, "feature_set": "common12_logerrs"},
            {**default, "feature_set": "mags_errs"},
        ]


METHOD = TabPFN()
