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
    assert _steps.gradient_kind("passthrough") == "differentiable"
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
