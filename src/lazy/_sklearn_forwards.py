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
from scipy import special
from sklearn import compose
from sklearn import decomposition
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.preprocessing import _data as sklearn_data

from lazy import _arrays
from lazy import _steps

__all__ = ["UFUNCS"]

UFUNCS: frozenset[np.ufunc] = frozenset(
    {
        np.log,
        np.log1p,
        np.log2,
        np.log10,
        np.exp,
        np.expm1,
        np.sqrt,
        np.square,
        np.arcsinh,
        np.sinh,
        np.tanh,
        np.arctan,
        np.abs,
        np.negative,
        np.reciprocal,
    }
)
"""The numpy functions of a ``FunctionTransformer`` that have a forward:
torch's function of the same name gives the same values."""

# How near a bound counts as on it, as QuantileTransformer reads it, and the
# normal quantiles it clips to, where the ppf would be infinite.
_BOUNDS_THRESHOLD = sklearn_data.BOUNDS_THRESHOLD
_NORMAL_CLIP = (
    float(special.ndtri(_BOUNDS_THRESHOLD - np.spacing(1.0))),
    float(special.ndtri(1 - (_BOUNDS_THRESHOLD - np.spacing(1.0)))),
)

_constant = _arrays.constant


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


def _interpolated(x: Any, points: Any, knots: Any, values: Any) -> Any:
    """``np.interp`` of each column of points, on the graph of the points.

    The search for each point's piece carries no gradient. It runs on the
    CPU in float64, which an Apple GPU lacks, so that a float32 tensor
    takes the pieces that float64 takes.

    Args:
        x: Points, a torch tensor of shape (n_columns, n), without ``NaN``.
        points: Their values off the graph, float64 on the CPU, C-ordered.
        knots: Each column's non-decreasing knots, a float64 tensor on the
            CPU of shape (n_columns, k).
        values: The value at each knot, a float64 tensor on the CPU of
            shape (k,).

    Returns:
        ``np.interp(x[j], knots[j], values)`` for each column ``j``, the
        linear interpolant, constant outside the knots, of the shape and
        kind of ``x``.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    n_knots = knots.shape[1]
    if n_knots == 1:
        # One knot: the interpolant is constant, with no slope.
        return x * 0 + float(values[0])
    piece = (torch.searchsorted(knots, points, right=True) - 1).clamp(
        0, n_knots - 2
    )
    start = knots.gather(1, piece)
    width = knots.gather(1, piece + 1) - start
    rise = (values[1:] - values[:-1])[piece]
    slope = torch.where(
        width > 0, rise / torch.where(width > 0, width, 1.0), 0.0
    )
    inside = (points >= knots[:, :1]) & (points < knots[:, -1:])
    edge = torch.where(points < knots[:, :1], values[0], values[-1])

    def like(constant: Any) -> Any:
        """A constant of the search, of the kind of ``x``."""
        return constant.to(dtype=x.dtype, device=x.device)

    line = like(values[piece]) + like(slope) * (x - like(start))
    return torch.where(inside.to(x.device), line, like(edge))


@_steps.register_forward(preprocessing.QuantileTransformer)
def _quantile(step: preprocessing.QuantileTransformer, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """The piecewise-linear map to the references, then the normal's ppf.

    As ``QuantileTransformer._transform_col``, for every column at once:
    the mean of the interpolation from below and from above, the bounds
    set to 0 and 1, and for a normal output the ppf, clipped where it
    would be infinite. A value on a bound, or beyond it, has no gradient.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    normal = step.output_distribution == "normal"
    knots = torch.as_tensor(np.ascontiguousarray(step.quantiles_.T))
    references = torch.as_tensor(np.ascontiguousarray(step.references_))
    low, high = knots[:, :1], knots[:, -1:]
    missing = X.T.isnan()
    x = torch.where(missing, low.to(dtype=X.dtype, device=X.device), X.T)
    points = x.detach().cpu().double().contiguous()
    out = 0.5 * (
        _interpolated(x, points, knots, references)
        - _interpolated(-x, -points, -knots.flip(1), -references.flip(0))
    )
    # The bounds are read in float64, as the search is.
    margin = _BOUNDS_THRESHOLD if normal else 0.0
    at_low = (points - margin < low) if normal else (points == low)
    at_high = (points + margin > high) if normal else (points == high)
    out = torch.where(at_high.to(X.device), 1.0, out)
    out = torch.where(at_low.to(X.device), 0.0, out)
    if normal:
        # The ppf is infinite at 0 and 1, where its gradient would be NaN:
        # it is taken strictly inside, and the ends get their clip bounds.
        inside = (out > 0) & (out < 1)
        ppf = torch.special.ndtri(torch.where(inside, out, 0.5))
        lowest, highest = (out.new_tensor(bound) for bound in _NORMAL_CLIP)
        ends = torch.where(out <= 0, lowest, highest)
        out = torch.where(inside, ppf, ends).clamp(*_NORMAL_CLIP)
    return torch.where(missing, X.T, out).T


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
    """An index map for no function; differentiable for one of ``UFUNCS``."""
    plain = step.func is None or (
        isinstance(step.func, np.ufunc)
        and step.func in UFUNCS
        and not step.kw_args
    )
    if step.func is None:
        return "index_map"
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

    return X if step.func is None else getattr(torch, step.func.__name__)(X)


def _inner_reason(parts: list[tuple[str, Any]]) -> str:
    """Why a container is a barrier: its first step that is one."""
    for name, inner in parts:
        found = _steps.barrier(name, inner)
        if found is not None:
            return f"its step {name!r} is a barrier: {found.reason}"
    return "a step of it is a barrier"


def _pipeline_kind(step: pipeline.Pipeline) -> str:
    """The kind of the pipeline's steps in a chain."""
    return _steps.chained(
        _steps.gradient_kind(inner) for _, inner in step.steps
    )


@_steps.register_forward(
    pipeline.Pipeline,
    _pipeline_kind,
    hint=lambda step: _inner_reason(step.steps),
)
def _chained(step: pipeline.Pipeline, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """Each step's forward, in order."""
    for _, inner in step.steps:
        X = _steps.forward(inner, X)
    return X


def _parts(
    step: compose.ColumnTransformer,
) -> list[tuple[str, Any, list[int]]]:
    """The fitted transformers that give columns: name, step, input columns."""
    return [
        (name, inner, list(step._transformer_to_input_indices[name]))  # noqa: SLF001 - the fitted columns, by position.
        for name, inner, _ in step.transformers_
        if not (isinstance(inner, str) and inner == "drop")
        and len(step._transformer_to_input_indices[name])  # noqa: SLF001 - as above.
    ]


def _columns_kind(step: compose.ColumnTransformer) -> str:
    """The kind of the transformers side by side; an index map at best."""
    return _steps.chained(
        _steps.gradient_kind(inner) for _, inner, _ in _parts(step)
    )


@_steps.register_forward(
    compose.ColumnTransformer,
    _columns_kind,
    hint=lambda step: _inner_reason([part[:2] for part in _parts(step)]),
)
def _by_columns(step: compose.ColumnTransformer, X: Any) -> Any:  # noqa: GS030 - scikit-learn's X.
    """Each transformer's forward of its columns, weighted, side by side."""
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    weights = step.transformer_weights or {}
    return torch.cat(
        [
            _steps.forward(inner, X[:, columns]) * weights.get(name, 1.0)
            for name, inner, columns in _parts(step)
        ],
        dim=1,
    )
