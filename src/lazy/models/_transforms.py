# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The uniform vocabulary of per-member feature transforms.

Every backend takes ``transforms``: ``"auto"`` (the model's own tuned
recipe, untouched), a recipe name, one transform name, or a sequence of them
that ensemble members cycle through. The names are shared by all models:

========================  ==================================================
``none``                  the features as given
``power``                 Yeo-Johnson power transform, standardised
``quantile``              quantile transform to a normal distribution
``quantile_uniform``      quantile transform to a uniform distribution
``quantile_rtdl``         quantile transform to normal after a little noise
``robust``                centring on the median, scaling by the IQR
========================  ==================================================

Any name takes a ``+original`` suffix, which appends the untransformed
features to the transformed ones. Recipes name a sequence: ``"limix"`` is
LimiX-2's own, ``("quantile_uniform+original", "power")``.

A backend translates a name to its model's own implementation when it has
one; otherwise the transform is applied here, by :class:`ScaffoldTransform`,
before the model sees the features (the model's internal preprocessing then
still runs on top). It is fitted on the context rows of the member group it
serves, never on query rows: under scaffolded bagging each member is a group
of its own, so that is the member's own bag; for a model that subsamples
rows natively, the group's members share the whole context, so the
transform is fitted on all of it and the model then draws each member's
rows from the transformed features.

Typical usage example:

  specs = _transforms.parse(("quantile_uniform+original", "power"))
  transform = _transforms.ScaffoldTransform(specs[0], seed=0).fit(X_context)
"""

from collections.abc import Sequence
import dataclasses

import numpy as np
from sklearn import preprocessing

from lazy import _typing

__all__ = [
    "AUTO",
    "BASE_TRANSFORMS",
    "RECIPES",
    "ScaffoldTransform",
    "TransformSpec",
    "parse",
]

#: The ``transforms`` value meaning each model's own tuned recipe.
AUTO = "auto"

#: The transform names every backend accepts.
BASE_TRANSFORMS: tuple[str, ...] = (
    "none",
    "power",
    "quantile",
    "quantile_uniform",
    "quantile_rtdl",
    "robust",
)

#: Named recipes: sequences of transforms that members cycle through.
RECIPES: dict[str, tuple[str, ...]] = {
    "limix": ("quantile_uniform+original", "power"),
}

_ORIGINAL = "+original"


@dataclasses.dataclass(frozen=True)
class TransformSpec:
    """One member's feature transform.

    Attributes:
        base: One of :data:`BASE_TRANSFORMS`.
        original: Whether the untransformed features are appended.
    """

    base: str
    original: bool = False

    @property
    def name(self) -> str:
        """The spec in the vocabulary, e.g. ``"quantile_uniform+original"``."""
        return self.base + (_ORIGINAL if self.original else "")


def parse(transforms: str | Sequence[str]) -> tuple[TransformSpec, ...] | None:
    """Parses a ``transforms`` value into member specs, or None for auto.

    Args:
        transforms: ``"auto"``, a recipe name, a transform name, or a
            non-empty sequence of transform names.

    Returns:
        The specs members cycle through, or None for the model's own recipe.

    Raises:
        ValueError: If a name is not in the vocabulary.
    """
    if isinstance(transforms, str):
        if transforms == AUTO:
            return None
        names: Sequence[str] = RECIPES.get(transforms, (transforms,))
    else:
        names = tuple(transforms)
        if not names:
            raise ValueError("transforms must not be an empty sequence")
    return tuple(_parse_one(name) for name in names)


def _parse_one(name: str) -> TransformSpec:
    """One transform name as a spec."""
    if not isinstance(name, str):
        raise ValueError(f"transform names must be strings: {name=}")
    original = name.endswith(_ORIGINAL)
    base = name.removesuffix(_ORIGINAL) if original else name
    if base not in BASE_TRANSFORMS:
        raise ValueError(
            f"unknown transform {name!r}; use 'auto', a recipe "
            f"{sorted(RECIPES)}, or one of {list(BASE_TRANSFORMS)} "
            f"(optionally with '{_ORIGINAL}')"
        )
    return TransformSpec(base, original)


class ScaffoldTransform:
    """A transform applied by the package for a model that lacks it.

    Fitted on its member group's context rows only (see the module
    docstring), so a query row's features never depend on the other query
    rows. Missing values (NaN) are ignored
    when fitting and stay NaN, for the model to handle.

    Attributes:
        spec: The transform.
        seed: The seed for its randomness (quantile subsampling, noise).
    """

    def __init__(self, spec: TransformSpec, seed: int):
        """Initialises an unfitted transform."""
        self.spec = spec
        self.seed = seed
        self._transformer: object | None = None

    def fit(self, features: _typing.FloatArray) -> "ScaffoldTransform":
        """Fits on context features, shape (n_rows, n_features).

        Returns:
            The fitted transform itself.
        """
        n_rows = len(features)
        base = self.spec.base
        rng = np.random.default_rng(self.seed)
        if base == "none":
            self._transformer = None
        elif base == "power":
            self._transformer = preprocessing.PowerTransformer(
                method="yeo-johnson", standardize=True
            ).fit(features)
        elif base == "robust":
            self._transformer = preprocessing.RobustScaler().fit(features)
        elif base == "quantile_rtdl":
            # RTDL's recipe: a little noise so ties do not collapse, and a
            # quantile count that scales with the context.
            noisy = features + rng.normal(
                0.0, 1e-3, features.shape
            ) * np.nanstd(features, axis=0)
            self._transformer = _quantile(
                "normal", max(min(n_rows // 30, 1000), 10), n_rows, self.seed
            ).fit(noisy)
        else:
            # The uniform settings are LimiX-2's (quantile_uniform_all_data).
            distribution = "uniform" if base == "quantile_uniform" else "normal"
            n_quantiles = (
                max(n_rows // 5, 2)
                if base == "quantile_uniform"
                else min(1000, n_rows)
            )
            self._transformer = _quantile(
                distribution, n_quantiles, n_rows, self.seed
            ).fit(features)
        return self

    def transform(self, features: _typing.FloatArray) -> _typing.FloatArray:
        """Applies the transform, shape (n_rows, n_features[* 2]).

        With ``+original`` the transformed columns come first, then the
        untransformed ones.
        """
        if self._transformer is None:
            out = np.asarray(features, dtype=np.float64)
        else:
            out = np.asarray(
                self._transformer.transform(features),  # type: ignore[attr-defined]  # an sklearn transformer
                dtype=np.float64,
            )
        if self.spec.original:
            out = np.hstack([out, np.asarray(features, dtype=np.float64)])
        return out


def _quantile(
    distribution: str, n_quantiles: int, n_rows: int, seed: int
) -> preprocessing.QuantileTransformer:
    """A quantile transformer fitted on all context rows (no subsampling)."""
    return preprocessing.QuantileTransformer(
        n_quantiles=max(min(n_quantiles, n_rows), 1),
        output_distribution=distribution,
        subsample=max(n_rows, 1),
        random_state=seed,
    )
