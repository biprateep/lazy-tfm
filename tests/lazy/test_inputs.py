# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Every supported feature table is read the same way, following sklearn."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn import exceptions

import lazy
from lazy import _inputs
from lazy import base
from lazy.models import ensemble


class _Echo(base.BaseDensityRegressor):
    """Densities that depend on the features, so column mix-ups show."""

    def __init__(self, *, y_grid=None):
        self.y_grid = y_grid

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


def _astropy_masked(values, mask, unit=None):
    """An astropy Masked array, or quantity with a unit; the -99 is masked."""
    masked = pytest.importorskip("astropy.utils.masked")
    if unit is not None:
        units = pytest.importorskip("astropy.units")
        values = values * getattr(units, unit)
    return masked.Masked(values, mask=mask)


@pytest.mark.parametrize("unit", [None, "mag"], ids=["array", "quantity"])
def test_astropy_masked_features_become_nan(unit):
    values = np.array([[1.0, -99.0], [3.0, 4.0]])
    features = _astropy_masked(values, values == -99.0, unit)
    frame, named = _inputs.as_feature_frame(features)
    assert not named
    np.testing.assert_array_equal(frame.to_numpy(), [[1.0, np.nan], [3.0, 4.0]])


@pytest.mark.parametrize("unit", [None, "mag"], ids=["array", "quantity"])
def test_a_masked_qtable_column_becomes_nan(unit):
    astropy_table = pytest.importorskip("astropy.table")
    values = np.array([1.0, -99.0, 3.0])
    column = _astropy_masked(values, values == -99.0, unit)
    frame, _ = _inputs.as_feature_frame(astropy_table.QTable({"m": column}))
    np.testing.assert_array_equal(frame["m"], [1.0, np.nan, 3.0])
    np.testing.assert_array_equal(
        _inputs.as_target(astropy_table.QTable({"z": column})["z"]),
        [1.0, np.nan, 3.0],
    )


@pytest.mark.parametrize("unit", [None, "dimensionless_unscaled"])
def test_astropy_masked_targets_are_rejected(table, unit):
    frame, z = table
    z = z.copy()
    z[0] = -99.0  # a sentinel, masked: never a training target
    target = _astropy_masked(z, z == -99.0, unit)
    np.testing.assert_array_equal(_inputs.as_target(target)[1:], z[1:])
    assert np.isnan(_inputs.as_target(target)[0])
    with pytest.raises(ValueError, match="non-finite"):
        _Echo().fit(frame, target)


def test_a_series_is_not_taken_for_a_masked_array(table):
    _, z = table
    np.testing.assert_array_equal(_inputs.as_target(pd.Series(z)), z)


def test_missing_values_pass_through_as_nan(table):
    frame, z = table
    frame = frame.copy()
    frame.iloc[::5, 0] = np.nan
    frame["empty"] = np.nan
    model = _Echo().fit(frame, z)
    assert model.predict_proba(frame).shape == (
        40,
        lazy.datasets.DC1_GRID.n_bins,
    )
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
    with pytest.warns(exceptions.DataConversionWarning, match="column-vector"):
        values = _inputs.as_target(target)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [0.1, np.nan, 0.3])


def test_a_target_with_several_columns_is_refused():
    # Half as many rows of two columns have as many values as X has rows.
    with pytest.raises(ValueError, match="1d array"):
        _inputs.as_target(np.ones((3, 2)))


def test_a_complex_target_is_refused_not_truncated():
    with pytest.raises(ValueError, match="Complex data not supported"):
        _inputs.as_target(np.array([1 + 2j, 3 + 0j]))


def test_sparse_features_are_refused_with_a_dense_alternative():
    with pytest.raises(TypeError, match="sparse X is not supported"):
        _inputs.as_feature_frame(sparse.csr_matrix(np.eye(3)))


@pytest.mark.parametrize("masked", [False, True])
def test_complex_features_are_refused_not_truncated(masked):
    features = np.ones((3, 2)) * (1 + 1j)
    if masked:
        features = np.ma.masked_array(features, mask=np.eye(3, 2, dtype=bool))
    with pytest.raises(ValueError, match="Complex data not supported"):
        _inputs.as_feature_frame(features)


def test_a_one_dimensional_x_gets_scikit_learns_reshape_hint():
    with pytest.raises(ValueError, match=r"array\.reshape\(-1, 1\)"):
        _inputs.as_feature_frame(np.arange(5.0))


# -- Deprecated lenient coercions, through an ensemble of two members --------


@pytest.fixture
def ensemble_data(standin_backends):
    rng = np.random.default_rng(5)
    n = ensemble.SMALL_CONTEXT_ROWS  # No small-context warning.
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + rng.normal(0.0, 0.2, n)
    model = lazy.LazyEnsembleModel(["tabpfn", "tabicl"], chunk_size=20)
    return model, X, y


def _deprecations(record):
    return [w for w in record if w.category is FutureWarning]


def _text(X):
    return X.astype(str)


def _bytes(X):
    return X.astype("S32")


def _list_of_text(X):
    return X.astype(str).tolist()


@pytest.mark.parametrize("convert", [_text, _bytes, _list_of_text])
def test_numeric_strings_still_work_and_warn_once_in_every_entry_point(
    ensemble_data, convert
):
    model, X, y = ensemble_data
    expected = model.fit(X, y).predict_proba(X[:30])
    text = convert(X)
    with pytest.warns(FutureWarning, match=r"0\.7\.0.*ValueError") as fit:
        model.fit(text, y)
    assert len(_deprecations(fit)) == 1
    assert fit[0].filename == __file__
    for call in (
        model.predict,
        model.predict_proba,
        model.predict_quantiles,
        lambda X: model.score(X, y[:30]),
        lambda X: model.evaluate(X, y[:30]),
    ):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            call(convert(X[:30]))
        (warning,) = _deprecations(record)
        assert warning.filename == __file__
    with pytest.warns(FutureWarning):
        got = model.predict_proba(convert(X[:30]))
    np.testing.assert_array_equal(got, expected)


def test_mixed_column_names_still_work_and_warn_once(ensemble_data):
    model, X, y = ensemble_data
    plain = model.fit(X, y).predict_proba(X[:30])
    mixed = pd.DataFrame(X, columns=["a", 1, "c"])
    with pytest.warns(FutureWarning, match=r"0\.7\.0.*TypeError") as fit:
        model.fit(mixed, y)
    assert len(_deprecations(fit)) == 1
    assert fit[0].filename == __file__
    assert not hasattr(model, "feature_names_in_")
    with pytest.warns(FutureWarning) as record:
        got = model.predict_proba(mixed[:30])
    assert len(_deprecations(record)) == 1
    np.testing.assert_array_equal(got, plain)


def test_a_dataframe_of_strings_is_accepted_silently_as_in_scikit_learn(
    ensemble_data,
):
    model, X, y = ensemble_data
    expected = model.fit(X, y).predict_proba(X[:30])
    for dtype in (object, "string"):
        frame = pd.DataFrame(X.astype(str), columns=["a", "b", "c"])
        frame = frame.astype(dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            model.fit(frame, y)
            got = model.predict_proba(frame[:30])
        assert got.shape == expected.shape


def test_integer_column_names_are_unnamed_without_a_warning(ensemble_data):
    model, X, y = ensemble_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.fit(pd.DataFrame(X), y)


def test_a_column_vector_target_warns_once_and_fits(ensemble_data):
    model, X, y = ensemble_data
    expected = model.fit(X, y).predict_proba(X[:30])
    with pytest.warns(exceptions.DataConversionWarning) as record:
        model.fit(X, y[:, None])
    assert len(record) == 1
    assert record[0].filename == __file__
    np.testing.assert_array_equal(model.predict_proba(X[:30]), expected)


def test_the_non_numeric_error_names_only_the_failing_columns():
    features = np.array([["1", "a"], ["3", "b"]])
    with pytest.raises(ValueError, match=r"not numeric: \['1'\]$"):
        _inputs.as_feature_frame(features)
