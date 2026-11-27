# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The step contract, and the forwards of scikit-learn's transformers."""

import numpy as np
import pytest
from sklearn import compose
from sklearn import decomposition
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing

from lazy import _steps

torch = pytest.importorskip("torch")


def _table(n_rows=80, seed=0):
    """Skewed columns, one with ties and one with negative values."""
    rng = np.random.default_rng(seed)
    X = rng.lognormal(size=(n_rows, 4))
    X[:, 1] = np.round(X[:, 1])
    X[:, 2] -= 2.0
    return X


_TRANSFORMERS = [
    preprocessing.StandardScaler(),
    preprocessing.StandardScaler(with_mean=False),
    preprocessing.MinMaxScaler(feature_range=(-1, 2), clip=True),
    preprocessing.MaxAbsScaler(),
    preprocessing.RobustScaler(),
    preprocessing.PowerTransformer(),
    preprocessing.PowerTransformer(standardize=False),
    preprocessing.QuantileTransformer(n_quantiles=20),
    preprocessing.QuantileTransformer(
        n_quantiles=20, output_distribution="normal"
    ),
    decomposition.PCA(2),
    decomposition.PCA(3, whiten=True),
    decomposition.TruncatedSVD(2, random_state=0),
    # arpack keeps its components as a view with a negative stride.
    decomposition.TruncatedSVD(2, algorithm="arpack"),
    impute.SimpleImputer(strategy="median"),
    preprocessing.FunctionTransformer(np.arcsinh),
    preprocessing.FunctionTransformer(),
    pipeline.make_pipeline(preprocessing.RobustScaler(), decomposition.PCA(2)),
    compose.ColumnTransformer(
        [
            ("scaled", preprocessing.StandardScaler(), [3, 0]),
            ("dropped", "drop", [1]),
        ],
        remainder="passthrough",
    ),
    compose.ColumnTransformer(
        [("scaled", preprocessing.StandardScaler(), [0, 1])],
        remainder="passthrough",
        transformer_weights={"scaled": 3.0},
    ),
]


@pytest.mark.parametrize(
    "transformer", _TRANSFORMERS, ids=lambda t: repr(t)[:60]
)
def test_a_forward_gives_what_transform_gives(transformer):
    # Queries beyond the fitted range on both sides, and on its bounds.
    X = _table()
    transformer.fit(X)
    queries = np.vstack([_table(30, seed=1) * 3 - 1, X[:5], X.min(0), X.max(0)])
    got = _steps.forward(transformer, torch.tensor(queries))
    np.testing.assert_allclose(
        got.numpy(), transformer.transform(queries), rtol=1e-9, atol=1e-9
    )


def test_a_quantile_transformer_of_one_quantile_is_constant():
    X = _table()
    transformer = preprocessing.QuantileTransformer(n_quantiles=1).fit(X)
    got = _steps.forward(transformer, torch.tensor(X[:5]))
    np.testing.assert_allclose(got.numpy(), transformer.transform(X[:5]))


def test_box_cox_gives_what_transform_gives():
    X = np.abs(_table()) + 0.1
    transformer = preprocessing.PowerTransformer(method="box-cox").fit(X)
    got = _steps.forward(transformer, torch.tensor(X))
    np.testing.assert_allclose(got.numpy(), transformer.transform(X))


@pytest.mark.parametrize(
    "transformer",
    [
        preprocessing.PowerTransformer(),
        preprocessing.QuantileTransformer(
            n_quantiles=20, output_distribution="normal"
        ),
        decomposition.PCA(2),
    ],
    ids=["power", "quantile", "pca"],
)
def test_a_forwards_gradient_is_the_slope_of_transform(transformer):
    X = _table()
    transformer.fit(X)
    # Inside the fitted range and away from the knots' kinks by chance.
    point = torch.tensor(np.median(X, axis=0)[None] + 0.013, requires_grad=True)
    _steps.forward(transformer, point).sum().backward()
    step = 1e-6
    for column in range(X.shape[1]):
        shift = np.zeros((1, X.shape[1]))
        shift[0, column] = step
        slope = (
            transformer.transform(point.detach().numpy() + shift).sum()
            - transformer.transform(point.detach().numpy() - shift).sum()
        ) / (2 * step)
        assert point.grad[0, column].item() == pytest.approx(
            slope, rel=1e-4, abs=1e-6
        )


def test_missing_values_stay_missing_and_an_imputed_one_gets_no_gradient():
    X = _table()
    queries = X[:6].copy()
    queries[0, 0] = queries[3, 2] = np.nan
    for transformer in (
        preprocessing.PowerTransformer(),
        preprocessing.QuantileTransformer(n_quantiles=20),
    ):
        got = _steps.forward(transformer.fit(X), torch.tensor(queries))
        np.testing.assert_allclose(
            got.numpy(), transformer.transform(queries), equal_nan=True
        )
    imputer = impute.SimpleImputer().fit(X)
    point = torch.tensor(queries, requires_grad=True)
    _steps.forward(imputer, point).sum().backward()
    assert point.grad[0, 0] == 0 and point.grad[1, 0] == 1


def test_each_kind_is_reported():
    X = _table()
    kinds = {
        "differentiable": preprocessing.StandardScaler().fit(X),
        "mask": impute.SimpleImputer().fit(X),
        "index_map": compose.ColumnTransformer(
            [("kept", "passthrough", [2, 0])]
        ).fit(X),
        "barrier": preprocessing.KBinsDiscretizer(3).fit(X),
    }
    for kind, step in kinds.items():
        assert _steps.gradient_kind(step) == kind
    assert _steps.gradient_kind("passthrough") == "index_map"
    # A pipeline is as closed as its most closed step.
    chained = pipeline.make_pipeline(kinds["differentiable"], kinds["mask"])
    assert _steps.gradient_kind(chained) == "mask"


def test_a_barrier_names_its_reason_and_refuses_a_forward():
    X = _table()
    function = preprocessing.FunctionTransformer(lambda x: x).fit(X)
    found = _steps.barrier("identity", function)
    assert found.step == "identity"
    assert "FunctionTransformer" in found.reason and "np.log1p" in found.reason
    with pytest.raises(_steps.BarrierError, match="FunctionTransformer"):
        _steps.forward(function, torch.tensor(X))
    assert (
        _steps.barrier("scaler", preprocessing.StandardScaler().fit(X)) is None
    )


def test_a_subclass_of_a_registered_class_is_a_barrier():
    class Scaler(preprocessing.StandardScaler):
        def transform(self, X, copy=None):
            return super().transform(X, copy) + 1

    assert _steps.gradient_kind(Scaler().fit(_table())) == "barrier"


def test_a_lazy_step_declares_its_kind_and_its_forward():
    class Doubled:
        gradient_kind = "differentiable"

        def forward(self, X):
            return 2 * X

    assert _steps.forward(Doubled(), torch.ones(2)).tolist() == [2.0, 2.0]
    with pytest.raises(ValueError, match="kind must be one of"):
        _steps.register_forward(Doubled, "open")


def test_a_step_that_declares_a_kind_without_a_forward_is_a_barrier():
    class Declared:
        gradient_kind = "differentiable"

    assert _steps.gradient_kind(Declared()) == "barrier"
    with pytest.raises(_steps.BarrierError, match="Declared has no differ"):
        _steps.forward(Declared(), torch.ones(2))


def test_a_barrier_is_found_before_any_kind_was_asked():
    # The registry loads on the first question of any kind.
    X = _table()
    function = preprocessing.FunctionTransformer(np.sign).fit(X)
    assert "np.log1p" in _steps.barrier("sign", function).reason


def test_a_float32_tensor_takes_the_pieces_that_float64_takes():
    X = _table()
    transformer = preprocessing.QuantileTransformer(n_quantiles=20).fit(X)
    queries = X[:30].astype(np.float32)
    got = _steps.forward(transformer, torch.tensor(queries))
    assert got.dtype == torch.float32
    np.testing.assert_allclose(
        got.numpy(), transformer.transform(queries.astype(float)), atol=1e-6
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_a_value_on_a_normal_quantiles_bound_has_a_finite_gradient(dtype):
    # Large values, where the bound's margin of 1e-7 is below the spacing
    # of floats, and a column of one value, whose bounds coincide.
    X = _table()
    X[:, 0] = 1.7e9 + np.arange(len(X)) * 3600.0
    X[:, 3] = 2.5
    transformer = preprocessing.QuantileTransformer(
        n_quantiles=20, output_distribution="normal"
    ).fit(X)
    rows = X[[X[:, 0].argmin(), X[:, 0].argmax(), 5]].astype(
        np.float32 if dtype is torch.float32 else np.float64
    )
    # float32 holds a value near 1.7e9 to about 128, so an inner value
    # moves along its piece by that much.
    tolerance = 1e-3 if dtype is torch.float32 else 1e-9
    point = torch.tensor(rows, requires_grad=True)
    out = _steps.forward(transformer, point)
    np.testing.assert_allclose(
        out.detach().numpy(),
        transformer.transform(rows.astype(float)),
        atol=tolerance,
    )
    out.sum().backward()
    assert point.grad.isfinite().all()


def test_a_chain_is_an_index_map_only_if_every_step_is_one():
    X = _table()
    subset = compose.ColumnTransformer([("kept", "passthrough", [0, 1, 2])])
    mixed = pipeline.make_pipeline(subset, decomposition.PCA(2)).fit(X)
    assert _steps.gradient_kind(mixed) == "differentiable"
    assert _steps.gradient_kind(subset.fit(X)) == "index_map"
    assert _steps.chained(["index_map", "mask", "differentiable"]) == "mask"


def test_a_container_names_the_step_inside_it_that_is_the_barrier():
    X = _table()
    blocked = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        preprocessing.FunctionTransformer(np.sign),
    ).fit(X)
    reason = _steps.barrier("p", blocked).reason
    assert "'functiontransformer'" in reason and "np.log1p" in reason
    columns = compose.ColumnTransformer(
        [("bins", preprocessing.KBinsDiscretizer(3, encode="ordinal"), [0])]
    ).fit(X)
    assert "'bins' is a barrier: KBinsDiscretizer" in (
        _steps.barrier("c", columns).reason
    )
