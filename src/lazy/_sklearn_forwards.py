# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Differentiable forwards of fitted scikit-learn transformers.

Each forward reads a fitted transformer's parameters, such as ``mean_`` and
``scale_``, and computes its ``transform`` on a torch tensor, so that a
pipeline of scikit-learn's own classes passes gradients
(:mod:`lazy._steps`). The fitted parameters are constants: a gradient does
not see how they depend on the rows they were fitted on. As in
:mod:`lazy.models._differentiable`, every search for an index (a piece of
a quantile function) runs on detached values and carries no gradient.

The classes covered are the scalers (standard, min-max, max-abs, robust),
``PowerTransformer``, ``QuantileTransformer``, ``PCA``, ``TruncatedSVD``,
``SimpleImputer`` (a mask), ``FunctionTransformer`` with one of
:data:`UFUNCS`, and ``Pipeline`` and ``ColumnTransformer``, by recursion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn import compose
from sklearn import decomposition
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing

from lazy import _steps

__all__ = ["UFUNCS"]

UFUNCS: dict[Any, str] = {
    np.log: "log",
    np.log1p: "log1p",
    np.log2: "log2",
    np.log10: "log10",
    np.exp: "exp",
    np.expm1: "expm1",
    np.sqrt: "sqrt",
    np.square: "square",
    np.arcsinh: "arcsinh",
    np.sinh: "sinh",
    np.tanh: "tanh",
    np.arctan: "arctan",
    np.abs: "abs",
    np.negative: "negative",
    np.reciprocal: "reciprocal",
}
"""The numpy functions of a ``FunctionTransformer`` that have a forward, by
the name of torch's function of the same values."""

# QuantileTransformer's own constant: how near a bound counts as on it.
_BOUNDS_THRESHOLD = 1e-7


def _constant(values: Any, like: Any) -> Any:
    """A fitted parameter as a tensor of the kind of ``like``."""
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    return torch.as_tensor(
        np.asarray(values), dtype=like.dtype, device=like.device
    )


@_steps.register_forward(preprocessing.StandardScaler)
def _standard(step: preprocessing.StandardScaler, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """``(X - mean_) / scale_``, each where the scaler has it."""
    if step.with_mean:
        X = X - _constant(step.mean_, X)
    if step.with_std:
        X = X / _constant(step.scale_, X)
    return X


@_steps.register_forward(preprocessing.MinMaxScaler)
def _min_max(step: preprocessing.MinMaxScaler, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """``X * scale_ + min_``, clipped to the range if the scaler clips."""
    X = X * _constant(step.scale_, X) + _constant(step.min_, X)
    return X.clamp(*step.feature_range) if step.clip else X


@_steps.register_forward(preprocessing.MaxAbsScaler)
def _max_abs(step: preprocessing.MaxAbsScaler, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """``X / scale_``."""
    return X / _constant(step.scale_, X)


@_steps.register_forward(preprocessing.RobustScaler)
def _robust(step: preprocessing.RobustScaler, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """``(X - center_) / scale_``, each where the scaler has it."""
    if step.with_centering:
        X = X - _constant(step.center_, X)
    if step.with_scaling:
        X = X / _constant(step.scale_, X)
    return X


@_steps.register_forward(preprocessing.PowerTransformer)
def _power(step: preprocessing.PowerTransformer, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """Yeo-Johnson or Box-Cox at the fitted ``lambdas_``, then the scaler.

    Each branch is computed where it is defined, so that the branch a
    value does not take leaves no ``NaN`` in its gradient.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    lambdas = _constant(step.lambdas_, X)
    tiny = float(np.spacing(1.0))
    if step.method == "box-cox":
        positive = torch.where(X > 0, X, 1.0)
        zero = lambdas.abs() < tiny
        safe = torch.where(zero, 1.0, lambdas)
        out = torch.where(
            zero,
            positive.log(),
            torch.expm1(safe * positive.log()) / safe,
        )
        out = torch.where(X > 0, out, torch.nan)
    else:
        above = torch.log1p(X.clamp(min=0))
        below = torch.log1p((-X).clamp(min=0))
        zero, two = lambdas.abs() < tiny, (lambdas - 2).abs() < tiny
        up = torch.where(zero, 1.0, lambdas)
        down = torch.where(two, 1.0, 2 - lambdas)
        out = torch.where(
            X >= 0,
            torch.where(zero, above, torch.expm1(up * above) / up),
            -torch.where(two, below, torch.expm1(down * below) / down),
        )
        out = torch.where(X.isnan(), X, out)
    return _standard(step._scaler, out) if step.standardize else out  # noqa: SLF001 - the fitted scaler.


def _interpolated(x: Any, knots: Any, values: Any) -> Any:
    """``np.interp(x, knots, values)`` on the graph of ``x``.

    Args:
        x: Points, a torch tensor of shape (n,), without ``NaN``.
        knots: Non-decreasing knots, a float64 array of shape (k,), k >= 2.
        values: The value at each knot, of that shape.

    Returns:
        The linear interpolant at ``x``, constant outside the knots, of
        the shape and kind of ``x``.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    points = x.detach().cpu().double().numpy()
    piece = np.clip(
        np.searchsorted(knots, points, side="right") - 1, 0, knots.size - 2
    )
    width = knots[piece + 1] - knots[piece]
    slope = np.where(
        width > 0, np.diff(values)[piece] / np.where(width > 0, width, 1.0), 0.0
    )
    inside = (points >= knots[0]) & (points < knots[-1])
    line = _constant(values[piece], x) + _constant(slope, x) * (
        x - _constant(knots[piece], x)
    )
    edge = _constant(np.where(points < knots[0], values[0], values[-1]), x)
    return torch.where(torch.as_tensor(inside, device=x.device), line, edge)


@_steps.register_forward(preprocessing.QuantileTransformer)
def _quantile(step: preprocessing.QuantileTransformer, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """The piecewise-linear map to the references, then the normal's ppf.

    As ``QuantileTransformer._transform_col``: the mean of the
    interpolation from below and from above, the bounds set to 0 and 1,
    and for a normal output the ppf, clipped where it would be infinite.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    normal = step.output_distribution == "normal"
    references = np.asarray(step.references_, dtype=np.float64)
    columns = []
    for index, quantiles in enumerate(step.quantiles_.T):
        x = X[:, index]
        missing = x.isnan()
        filled = torch.where(missing, float(quantiles[0]), x)
        out = 0.5 * (
            _interpolated(filled, quantiles, references)
            - _interpolated(-filled, -quantiles[::-1], -references[::-1])
        )
        margin = _BOUNDS_THRESHOLD if normal else 0.0
        low = (
            filled - margin < quantiles[0] if normal else filled == quantiles[0]
        )
        high = (
            filled + margin > quantiles[-1]
            if normal
            else filled == quantiles[-1]
        )
        out = torch.where(high, 1.0, out)
        out = torch.where(low, 0.0, out)
        if normal:
            bound = _BOUNDS_THRESHOLD - float(np.spacing(1.0))
            out = torch.special.ndtri(out).clamp(
                float(stats.norm.ppf(bound)), float(stats.norm.ppf(1 - bound))
            )
        columns.append(torch.where(missing, x, out))
    return torch.stack(columns, dim=1)


@_steps.register_forward(decomposition.PCA)
def _pca(step: decomposition.PCA, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """``(X - mean_) @ components_.T``, whitened if the PCA whitens."""
    out = (X - _constant(step.mean_, X)) @ _constant(step.components_, X).T
    if step.whiten:
        scale = np.sqrt(step.explained_variance_)
        out = out / _constant(np.maximum(scale, np.finfo(scale.dtype).eps), X)
    return out


@_steps.register_forward(decomposition.TruncatedSVD)
def _svd(step: decomposition.TruncatedSVD, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """``X @ components_.T``."""
    return X @ _constant(step.components_, X).T


def _imputer_kind(step: impute.SimpleImputer) -> str:
    """A mask, unless the imputer adds columns or reads another marker."""
    plain = not step.add_indicator and (
        isinstance(step.missing_values, float) and np.isnan(step.missing_values)
    )
    return "mask" if plain else "barrier"


@_steps.register_forward(
    impute.SimpleImputer,
    _imputer_kind,
    hint="use missing_values=nan and add_indicator=False",
)
def _imputed(step: impute.SimpleImputer, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """Each ``NaN`` as its column's fitted statistic, a constant.

    A column that the imputer dropped at ``fit``, for want of any value,
    is dropped here too.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    statistics = np.asarray(step.statistics_, dtype=np.float64)
    kept = np.ones(statistics.size, dtype=bool)
    if not step.keep_empty_features:
        kept = ~np.isnan(statistics)
    X = X[:, torch.as_tensor(kept, device=X.device)]
    return torch.where(X.isnan(), _constant(statistics[kept], X), X)


def _function_kind(step: preprocessing.FunctionTransformer) -> str:
    """Differentiable for no function, or one of ``UFUNCS`` alone."""
    plain = step.func is None or (
        isinstance(step.func, np.ufunc)
        and step.func in UFUNCS
        and not step.kw_args
    )
    return "differentiable" if plain else "barrier"


@_steps.register_forward(
    preprocessing.FunctionTransformer,
    _function_kind,
    hint=(
        "give it a numpy function such as np.log1p or np.arcsinh with no"
        " kw_args, or use a step of lazy.augment"
    ),
)
def _function(step: preprocessing.FunctionTransformer, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """Torch's function of the name that ``UFUNCS`` gives ``func``."""
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    return X if step.func is None else getattr(torch, UFUNCS[step.func])(X)


def _weakest(kinds: list[str]) -> str:
    """The least open of some gradient kinds, in ``GRADIENT_KINDS``."""
    return max(kinds, key=_steps.GRADIENT_KINDS.index, default="differentiable")


def _pipeline_kind(step: pipeline.Pipeline) -> str:
    """The least open kind of the pipeline's steps."""
    return _weakest([_steps.gradient_kind(inner) for _, inner in step.steps])


@_steps.register_forward(pipeline.Pipeline, _pipeline_kind)
def _chained(step: pipeline.Pipeline, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """Each step's forward, in order."""
    for _, inner in step.steps:
        X = _steps.forward(inner, X)
    return X


def _parts(step: compose.ColumnTransformer) -> list[tuple[Any, list[int]]]:
    """The fitted transformers that give columns, with their input columns."""
    return [
        (inner, list(step._transformer_to_input_indices[name]))  # noqa: SLF001 - the fitted columns, by position.
        for name, inner, _ in step.transformers_
        if not (isinstance(inner, str) and inner == "drop")
        and len(step._transformer_to_input_indices[name])  # noqa: SLF001 - as above.
    ]


def _columns_kind(step: compose.ColumnTransformer) -> str:
    """The least open kind of the transformers; an index map at best."""
    kinds = [_steps.gradient_kind(inner) for inner, _ in _parts(step)]
    return _weakest(["index_map", *kinds])


@_steps.register_forward(compose.ColumnTransformer, _columns_kind)
def _by_columns(step: compose.ColumnTransformer, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """Each transformer's forward of its columns, side by side."""
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    return torch.cat(
        [
            _steps.forward(inner, X[:, columns])
            for inner, columns in _parts(step)
        ],
        dim=1,
    )
