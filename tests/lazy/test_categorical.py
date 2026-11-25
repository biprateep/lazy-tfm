# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Categorical columns through the shared layer, on the stand-ins.

Each backend's own handling of categories is tested beside the backend;
these tests cover what every backend shares: the codes reach each
member's view intact, the wrappers pass them on, and lazy.interp keeps
them.
"""

import numpy as np
import pandas as pd
import pytest
import standins

import lazy
from lazy import interp
from lazy.models import _transforms
from lazy.models import registry

N_ROWS = 60


@pytest.fixture(name="table")
def _table():
    """A numeric column, a categorical one, and a target that uses both."""
    rng = np.random.default_rng(4)
    kind = rng.choice(["red", "green", "blue"], N_ROWS)
    frame = pd.DataFrame(
        {"x": rng.normal(size=N_ROWS), "kind": pd.Categorical(kind)}
    )
    y = frame["x"] + (kind == "red") + rng.normal(0, 0.1, N_ROWS)
    return frame, y.to_numpy()


class _Recording(standins.ScaffoldedHistogramStandIn):
    """Records each group's view and the mask it is told of."""

    def _fit_group(self, X, y, group):
        views = getattr(self, "views_", [])
        self.views_ = [*views, (X.copy(), self._categorical_columns(group))]
        return super()._fit_group(X, y, group)


def test_each_view_holds_the_codes_where_its_mask_says(table):
    frame, y = table
    codes = frame["kind"].cat.codes.to_numpy(dtype=float)
    model = _Recording(
        n_estimators=3,
        transforms=("quantile+original",),
        outlier_threshold=0.5,
        progress=False,
    ).fit(frame, y)
    for view, mask in model.views_:
        assert mask.sum() == 2  # "+original" doubles the columns.
        for column in view[:, mask].T:
            np.testing.assert_array_equal(column, codes)
        assert not np.isin(view[:, ~mask], [0.0, 1.0, 2.0]).all()


def test_a_scaffolded_transform_leaves_the_codes_and_fits_the_rest():
    rng = np.random.default_rng(5)
    features = np.column_stack(
        [rng.integers(0, 4, 50), rng.normal(size=50)]
    ).astype(float)
    spec = _transforms.parse(("power",))[0]
    categorical = np.array([True, False])
    out = _transforms.ScaffoldTransform(spec, 0, categorical)
    out = out.fit(features).transform(features)
    alone = _transforms.ScaffoldTransform(spec, 0).fit(features[:, 1:])
    np.testing.assert_array_equal(out[:, 0], features[:, 0])
    np.testing.assert_allclose(out[:, 1:], alone.transform(features[:, 1:]))


def test_the_wrapper_answers_as_its_backend_does(table, monkeypatch):
    frame, y = table
    monkeypatch.setitem(
        registry.ESTIMATORS, "histogram", standins.HistogramStandIn
    )
    options = {"n_estimators": 2, "progress": False}
    wrapped = lazy.LazyModel("histogram", **options).fit(frame, y)
    alone = standins.HistogramStandIn(**options).fit(frame, y)
    np.testing.assert_array_equal(
        wrapped.estimator_.is_categorical_, alone.is_categorical_
    )
    queries = frame.iloc[:10]
    np.testing.assert_array_equal(
        wrapped.predict_quantiles(queries, [0.2, 0.5]),
        alone.predict_quantiles(queries, [0.2, 0.5]),
    )


@pytest.mark.filterwarnings("ignore::lazy.models.ensemble.EnsembleSizeWarning")
def test_an_ensemble_tells_each_member_which_columns_hold_codes(table):
    frame, y = table
    members = [
        standins.HistogramStandIn(n_estimators=2, progress=False),
        standins.QuantileStandIn(n_estimators=2, progress=False),
    ]
    model = lazy.LazyEnsembleModel(members).fit(frame, y)
    for member in model.estimators_:
        assert member.is_categorical_.tolist() == [False, True]


def test_interp_refits_on_columns_marked_by_position(table):
    frame, y = table
    values = frame.astype(object).to_numpy()
    model = standins.HistogramStandIn(
        n_estimators=2, progress=False, categorical_features=[1]
    )
    explanation = interp.explain(
        model, values[:20], y[:20], values[20:], y[20:], method="loco"
    )
    assert explanation.feature_names == ("x0", "x1")
    assert np.isfinite(explanation.values).all()
