# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Every supported feature table is read the same way, following sklearn."""

import numpy as np
import pandas as pd
import pytest

import lazy
from lazy import _inputs
from lazy import base


class _Echo(base.BaseDensityRegressor):
    """Densities that depend on the features, so column mix-ups show."""

    def __init__(self, *, z_grid=None):
        self.z_grid = z_grid

    def _fit(self, X, y):
        self.columns_seen_ = list(X.columns)

    def _predict_pdf(self, X, grid):
        centre = np.nan_to_num(X.to_numpy()) @ np.arange(1.0, X.shape[1] + 1)
        centre = 0.5 + 0.1 * np.tanh(centre)
        return np.exp(-0.5 * ((grid.centers - centre[:, None]) / 0.1) ** 2)


@pytest.fixture
def table():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {"g_r": rng.normal(size=40), "r_i": rng.normal(size=40)}
    )
    return frame, rng.uniform(0.1, 1.5, 40)


def _forms(frame):
    """The same table as every supported form, named ones first."""
    records = frame.to_records(index=False)
    return {
        "DataFrame": frame,
        "recarray": records,
        "structured": records.view(np.ndarray),
        "ndarray": frame.to_numpy(),
        "list": frame.to_numpy().tolist(),
    }


def test_every_form_gives_the_same_predictions(table):
    frame, z = table
    reference = _Echo().fit(frame, z).predict_proba(frame)
    for form, data in _forms(frame).items():
        model = _Echo().fit(data, z)
        np.testing.assert_array_equal(
            model.predict_proba(data), reference, err_msg=form
        )


def test_named_forms_keep_their_names(table):
    frame, z = table
    for form in ("DataFrame", "recarray", "structured"):
        model = _Echo().fit(_forms(frame)[form], z)
        assert list(model.feature_names_in_) == ["g_r", "r_i"], form


def test_unnamed_forms_have_no_feature_names(table):
    frame, z = table
    model = _Echo().fit(frame.to_numpy(), z)
    assert not hasattr(model, "feature_names_in_")
    assert model.columns_seen_ == ["x0", "x1"]


def test_an_astropy_table_is_read_by_column_name(table):
    astropy_table = pytest.importorskip("astropy.table")
    units = pytest.importorskip("astropy.units")
    frame, z = table
    qtable = astropy_table.QTable(
        {"g_r": frame["g_r"].to_numpy() * units.mag, "r_i": frame["r_i"]}
    )
    reference = _Echo().fit(frame, z).predict_proba(frame)
    model = _Echo().fit(qtable, z)
    assert list(model.feature_names_in_) == ["g_r", "r_i"]
    np.testing.assert_array_equal(model.predict_proba(qtable), reference)


def test_masked_astropy_entries_become_nan():
    astropy_table = pytest.importorskip("astropy.table")
    masked = astropy_table.Table(
        {"a": np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])}
    )
    frame, named = _inputs.as_feature_frame(masked)
    assert named
    assert np.isnan(frame["a"].iloc[1]) and frame["a"].iloc[2] == 3.0


def test_missing_values_pass_through_as_nan(table):
    frame, z = table
    frame = frame.copy()
    frame.iloc[::5, 0] = np.nan
    frame["empty"] = np.nan
    model = _Echo().fit(frame, z)
    assert model.predict_proba(frame).shape == (40, lazy.DC1_GRID.n_bins)
    converted, _ = _inputs.as_feature_frame(frame)
    assert converted["empty"].isna().all()


def test_infinities_are_rejected_by_column(table):
    frame, z = table
    frame = frame.copy()
    frame.iloc[3, 1] = np.inf
    with pytest.raises(ValueError, match=r"infinite.*'r_i'"):
        _Echo().fit(frame, z)


def test_text_columns_are_rejected(table):
    frame, z = table
    frame = frame.assign(name="galaxy")
    with pytest.raises(ValueError, match=r"not numeric: \['name'\]"):
        _Echo().fit(frame, z)


def test_multidimensional_fields_are_rejected():
    array = np.zeros(4, dtype=[("flux", np.float64, (3,))])
    with pytest.raises(ValueError, match="multidimensional"):
        _inputs.as_feature_frame(array)


def test_an_unnamed_table_meets_a_named_fit_with_a_warning(table):
    frame, z = table
    model = _Echo().fit(frame, z)
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        pdfs = model.predict_proba(frame.to_numpy())
    np.testing.assert_array_equal(pdfs, model.predict_proba(frame))


def test_a_named_table_meets_an_unnamed_fit_with_a_warning(table):
    frame, z = table
    model = _Echo().fit(frame.to_numpy(), z)
    with pytest.warns(UserWarning, match="fitted without feature names"):
        model.predict_proba(frame)


def test_masked_targets_are_rejected(table):
    frame, z = table
    masked = np.ma.array(z, mask=np.zeros(z.size, bool))
    masked.mask[0] = True
    with pytest.raises(ValueError, match="non-finite"):
        _Echo().fit(frame, masked)


def test_quantity_targets_give_their_values(table):
    units = pytest.importorskip("astropy.units")
    frame, z = table
    np.testing.assert_array_equal(
        _inputs.as_target(z * units.dimensionless_unscaled), z
    )


def test_a_masked_column_vector_target_is_flattened():
    target = np.ma.masked_array(
        [[0.1], [0.2], [0.3]], mask=[[False], [True], [False]]
    )
    values = _inputs.as_target(target)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [0.1, np.nan, 0.3])


def test_a_one_dimensional_x_gets_scikit_learns_reshape_hint():
    with pytest.raises(ValueError, match=r"array\.reshape\(-1, 1\)"):
        _inputs.as_feature_frame(np.arange(5.0))
