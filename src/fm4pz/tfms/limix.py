# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""LimiX-2, through lazy-tfm, with the bucket masses of its bar distribution.

Beyond the settings every foundation model has (:mod:`fm4pz.tfms`), the
search covers the softmax temperature (0.9 is LimiX-2's released value) and
bagging. LimiX-2 was pretrained on tables of up to 20,000 rows and its loss
on DC1 grows beyond them; in the submitted paper's study, members that each
read a bag of 20,000 rows recovered it. A bag is a number of rows (10,000,
15,000 or 20,000), so that a production fit reads bags of the size its
search validated, and a search offers only the bags smaller than its folds'
context. A bagged ensemble has 16 or 32 members, so that the bags cover the
context. (Round 1 also offered bags of half the training set, which made a
production fit's bags a quarter larger than its search's.)

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

#: Each bagging choice's ``bag_size``. ``"half"`` was searched in round 1
#: only, and is kept so that its settings can still be fitted.
BAGS: dict[str, int | float | None] = {
    "none": None,
    "10k": 10_000,
    "15k": 15_000,
    "20k": MAX_CONTEXT,
    "half": 0.5,
}

#: The bags a search chooses between, as rows per member.
SEARCHED_BAGS: dict[str, int] = {
    "10k": 10_000,
    "15k": 15_000,
    "20k": MAX_CONTEXT,
}


def offered_bags(n_train: int) -> list[str]:
    """The bagging choices of a search, smaller bags than its folds' context.

    Args:
        n_train: The training size of the search.

    Returns:
        ``"none"`` and every name in :data:`SEARCHED_BAGS` whose rows are
        fewer than the context of a fold, four fifths of ``n_train``.
    """
    fit_rows = n_train * (config.CV_FOLDS - 1) // config.CV_FOLDS
    return ["none"] + [
        name for name, rows in SEARCHED_BAGS.items() if rows < fit_rows
    ]


class LimiX(tfms.FoundationModel):
    """LimiX-2 as a :class:`fm4pz.cv.Method`."""

    name = "limix"
    backend = "limix"
    package = "torch"

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Proposes the inputs, ensemble, preprocessing, temperature, bags.

        Bagging is proposed only where :func:`offered_bags` offers a bag.

        Args:
            trial: The Optuna trial.

        Returns:
            The fit parameters.
        """
        params = self.suggest_common(trial, TRANSFORMS)
        params["softmax_temperature"] = trial.suggest_categorical(
            "softmax_temperature", [0.8, 0.9, 1.0, 1.1]
        )
        bags = offered_bags(tfms.study_size(trial))
        params["bag"] = (
            trial.suggest_categorical("bag", bags) if len(bags) > 1 else "none"
        )
        if params["bag"] != "none":
            params["n_estimators"] = trial.suggest_categorical(
                "n_bags", [16, 32]
            )
        return params

    def adapt_warm_start(
        self, params: Mapping[str, Any], n_train: int
    ) -> dict[str, Any] | None:
        """A setting with its bag translated to the bags this size offers.

        A round-1 bag of half the training set becomes the offered bag
        nearest half of ``n_train``; a bag this size does not offer is left
        out, with its member count, for the sampler to draw.

        Args:
            params: Optuna parameters of the setting.
            n_train: The training size of the search.

        Returns:
            The parameters to queue.
        """
        out = dict(params)
        bags = offered_bags(n_train)
        if out.get("bag") == "half":
            rows = [name for name in bags if name != "none"]
            out["bag"] = min(
                rows,
                key=lambda name: abs(SEARCHED_BAGS[name] - n_train / 2),
                default="none",
            )
        if "bag" in out and out["bag"] not in bags:
            del out["bag"]
            out.pop("n_bags", None)
        if out.get("bag") == "none":
            out.pop("n_bags", None)
        return out

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
            {**default, "bag": "20k", "n_bags": 16},
            {**default, "bag": "15k", "n_bags": 32},
            {**default, "feature_set": "common12_logerrs"},
        ]


METHOD = LimiX()
