# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The uniform vocabulary of per-member feature transforms.

Every backend takes ``transforms``: ``"auto"`` (the model's own tuned
recipe, written out by its backend so that the installed upstream version
cannot change it), a recipe name, one transform name, or a sequence of them
that ensemble members cycle through. The names are shared by all models and
mean the same transform on each:

========================  ==================================================
``none``                  no transform
``power``                 Yeo-Johnson power transform, standardised; a
                          column with at most one distinct value in the
                          context (constant, or all missing) is passed
                          through unchanged, as it has nothing to fit
``quantile``              quantile transform to a normal distribution
``quantile_uniform``      quantile transform to a uniform distribution
``quantile_rtdl``         quantile transform to normal after a little noise
``robust``                centring on the median, scaling by the IQR
========================  ==================================================

Any name takes a ``+original`` suffix, which appends the untransformed
features to the transformed ones. Recipes name a sequence: ``"limix"`` is
LimiX-2's own, ``("quantile_uniform+original", "power")``.

Under an explicit ``transforms`` value the model sees that transform and
nothing else optional: each backend turns off its model's extra steps (extra
feature columns, target transforms, outlier clipping unless
``outlier_threshold`` asks for it). What every model still does is its
unavoidable input handling -- standardising the columns, dropping constant
ones, filling missing values where it cannot take them -- which each
backend's documentation lists.

A backend translates a name to its model's own implementation only when that
implementation is the one defined here; otherwise the transform is applied
here, by :class:`ScaffoldTransform`, before the model sees the features. It is
fitted on the context rows of the member group it serves, never on query
rows. Under bagging a member with a scaffolded transform is a group of its
own on every model, so the transform, and the :class:`SoftClip` after it,
are fitted on that member's own bag.

Typical usage example:

  specs = _transforms.parse(("quantile_uniform+original", "power"))
  transform = _transforms.ScaffoldTransform(specs[0], seed=0).fit(X_context)
"""

from collections.abc import Sequence
import dataclasses
import warnings

import numpy as np
from sklearn import preprocessing

from lazy import _typing

__all__ = [
    "AUTO",
    "BASE_TRANSFORMS",
    "RECIPES",
    "ScaffoldTransform",
    "SoftClip",
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
        # Columns the transformer is fitted on and applied to; None for all.
        self._columns: _typing.IntArray | None = None

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
            # Yeo-Johnson cannot fit a column with no spread (sklearn fails
            # cryptically on an all-NaN one), so such columns pass through.
            self._columns = np.flatnonzero(
                [np.unique(c[~np.isnan(c)]).size > 1 for c in features.T]
            )
            self._transformer = (
                preprocessing.PowerTransformer(
                    method="yeo-johnson", standardize=True
                ).fit(features[:, self._columns])
                if self._columns.size
                else None
            )
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
        out = np.array(features, dtype=np.float64)
        if self._transformer is not None:
            columns = slice(None) if self._columns is None else self._columns
            out[:, columns] = self._transformer.transform(  # type: ignore[attr-defined]  # an sklearn transformer
                out[:, columns]
            )
        if self.spec.original:
            out = np.hstack([out, np.asarray(features, dtype=np.float64)])
        return out


class SoftClip:
    """The soft outlier clip TabPFN, TabICL and TabFM share, for any model.

    Fitted on context features in two passes: the bounds
    ``mean +- threshold * std`` of each column, then the same bounds
    recomputed without the values that fell outside the first ones. A value
    beyond a bound is pulled back to it plus the logarithm of its magnitude,
    ``min(x, upper + log1p(|x|))`` (and its mirror below), so extreme values
    keep their order but lose their leverage. Missing values are ignored when
    fitting and stay missing.

    This is TabICL's and TabFM's ``OutlierRemover`` and TabPFN's
    ``TorchSoftClipOutliers`` (the standard deviation with ``ddof=1``, floored
    at 1e-6); it is what ``outlier_threshold`` means on every backend.

    Attributes:
        threshold: The bound, in standard deviations.
    """

    def __init__(self, threshold: float):
        """Initialises an unfitted clip at ``threshold`` standard deviations."""
        self.threshold = float(threshold)
        self._lower: _typing.FloatArray | None = None
        self._upper: _typing.FloatArray | None = None

    def fit(self, features: _typing.FloatArray) -> "SoftClip":
        """Fits the bounds on context features, shape (n_rows, n_features).

        Returns:
            The fitted clip itself.
        """
        features = np.asarray(features, dtype=np.float64)
        ddof = 1 if len(features) > 1 else 0
        lower, upper = self._bounds(features, ddof)
        clean = np.where(
            (features < lower) | (features > upper), np.nan, features
        )
        self._lower, self._upper = self._bounds(clean, ddof)
        return self

    def _bounds(
        self, features: _typing.FloatArray, ddof: int
    ) -> tuple[_typing.FloatArray, _typing.FloatArray]:
        """``mean -+ threshold * std`` of each column, ignoring NaN."""
        # An all-missing column has no bounds; numpy warns, and the NaN
        # bounds leave its (missing) values alone.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(features, axis=0)
            std = np.maximum(np.nanstd(features, axis=0, ddof=ddof), 1e-6)
        return mean - self.threshold * std, mean + self.threshold * std

    def transform(self, features: _typing.FloatArray) -> _typing.FloatArray:
        """Applies the clip, shape (n_rows, n_features)."""
        if self._lower is None or self._upper is None:
            raise RuntimeError("SoftClip.transform before fit")
        out = np.asarray(features, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            out = np.maximum(-np.log1p(np.abs(out)) + self._lower, out)
            out = np.minimum(np.log1p(np.abs(out)) + self._upper, out)
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
