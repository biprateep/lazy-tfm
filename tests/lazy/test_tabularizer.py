# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The Tabularizer: columns of items to encoded, reduced columns."""

import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn import decomposition
from sklearn import exceptions
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
import standins

import lazy
from lazy import encoders

N_ROWS = 40


@pytest.fixture(name="table")
def _table():
    """A table with a number, time series, texts and images per row."""
    rng = np.random.default_rng(1)
    words = ["lazy", "model", "frozen", "context", "grid"]
    return pd.DataFrame(
        {
            "number": rng.normal(size=N_ROWS),
            "curve": [
                rng.normal(size=rng.integers(5, 60)).cumsum()
                for _ in range(N_ROWS)
            ],
            "notes": [
                " ".join(rng.choice(words, size=rng.integers(1, 6)))
                for _ in range(N_ROWS)
            ],
            "cutout": [rng.uniform(size=(6, 6, 3)) for _ in range(N_ROWS)],
        }
    )


def test_each_column_is_encoded_reduced_and_named(table):
    tab = lazy.Tabularizer(
        {"curve": "time", "notes": "text", "cutout": "image_statistics"},
        n_components=3,
    ).fit(table)
    pairs = itertools.product(("curve", "notes", "cutout"), range(3))
    expected = [f"{column}__pc{i}" for column, i in pairs]
    assert list(tab.get_feature_names_out()) == [*expected, "remainder__number"]
    out = tab.transform(table)
    assert out.shape == (N_ROWS, 10)
    np.testing.assert_array_equal(out[:, -1], table["number"])


def test_a_reduced_embedding_is_the_principal_components_of_its_scaled_rows(
    table,
):
    tab = lazy.Tabularizer(
        {"curve": "time"}, n_components=4, remainder="drop"
    ).fit(table)
    embedding = lazy.time.Statistics().fit_transform(table["curve"])
    expected = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        decomposition.PCA(4, svd_solver="full"),
    ).fit_transform(embedding)
    np.testing.assert_allclose(tab.transform(table)[:, :4], expected)


def test_a_narrow_embedding_keeps_its_columns_and_names(table):
    tab = lazy.Tabularizer(
        {"curve": "time"}, n_components=None, remainder="drop"
    ).fit(table)
    np.testing.assert_array_equal(
        tab.transform(table),
        lazy.time.Statistics().fit_transform(table["curve"]),
    )
    assert tab.get_feature_names_out()[0] == "curve__time_statistics_mean"
    wide = lazy.Tabularizer({"curve": "time"}, n_components=13)
    assert wide.fit_transform(table[["number", "curve"]]).shape == (N_ROWS, 14)


def test_a_missing_item_stays_missing_and_is_left_out_of_the_fit(table):
    gappy = table.copy()
    gappy.loc[[2, 7], "notes"] = None
    tab = lazy.Tabularizer(
        {"notes": "text"}, n_components=3, remainder="drop"
    ).fit(gappy)
    out = tab.transform(gappy)
    assert np.isnan(out[[2, 7], :3]).all()
    assert not np.isnan(np.delete(out, [2, 7], axis=0)).any()
    complete = gappy.drop(index=[2, 7])
    alone = lazy.Tabularizer(
        {"notes": "text"}, n_components=3, remainder="drop"
    ).fit(complete)
    np.testing.assert_allclose(
        out[~gappy["notes"].isna()], alone.transform(complete)
    )


def test_the_reduction_is_fitted_on_the_context_only(table):
    context, queries = table.iloc[:30], table.iloc[30:]
    tab = lazy.Tabularizer(
        {"curve": "time"}, n_components=3, remainder="drop"
    ).fit(context)
    one_at_a_time = [tab.transform(queries.iloc[[i]]) for i in range(10)]
    np.testing.assert_allclose(np.vstack(one_at_a_time), tab.transform(queries))


def test_an_encoder_given_is_cloned_and_a_name_builds_one(table):
    given = lazy.time.Spectrum(n_octaves=3)
    tab = lazy.Tabularizer(
        {"curve": given, "notes": "text_statistics"}, n_components=None
    ).fit(table)
    encoders_used = [
        step[0] for _, step, _ in tab.transformer_.transformers_[:2]
    ]
    assert encoders_used[0].get_params() == given.get_params()
    assert isinstance(encoders_used[1], lazy.text.Statistics)
    assert not hasattr(given, "_n_features_out")


def test_a_model_behind_it_predicts_and_a_search_tunes_it(table):
    table = table.drop(columns="cutout")
    y = 3 * table["number"] + np.random.default_rng(2).normal(0, 0.1, N_ROWS)
    model = lazy.make_pipeline(
        lazy.Tabularizer({"curve": "time", "notes": "text"}, n_components=2),
        standins.HistogramStandIn(n_estimators=2, progress=False),
    )
    model.fit(table, y)
    assert model.predict_quantiles(table, [0.5]).shape == (N_ROWS, 1)
    search = model_selection.GridSearchCV(
        model, {"tabularizer__n_components": [1, 2]}, cv=2
    ).fit(table, y)
    assert search.best_params_["tabularizer__n_components"] in (1, 2)


class _Recording(lazy.text.Statistics):
    """Text statistics that count the items they encode."""

    def _encode(self, items):
        self.n_encoded_ = getattr(self, "n_encoded_", 0) + len(items)
        return super()._encode(items)


def test_fitting_in_a_pipeline_encodes_each_item_once(table):
    tab = lazy.Tabularizer({"notes": _Recording()}, remainder="drop")
    tab.fit_transform(table)
    assert (
        tab.transformer_.named_transformers_["column0"][0].n_encoded_ == N_ROWS
    )


@pytest.mark.parametrize(
    ("params", "error", "message"),
    [
        ({"columns": ["curve"]}, TypeError, "columns must map a column"),
        ({"columns": {}}, ValueError, "at least one column"),
        ({"columns": {"curve": 3}}, TypeError, r"columns\['curve'\] must be"),
        (
            {"columns": {"curve": "times"}},
            lazy.UnknownNameError,
            "unknown encoder 'times'; known: .'image', 'text', 'time'",
        ),
        (
            {"columns": {"curve": "time"}, "n_components": 0},
            ValueError,
            "n_components must be a positive integer",
        ),
        (
            {"columns": {"curve": "time"}, "remainder": "keep"},
            ValueError,
            "remainder",
        ),
    ],
)
def test_bad_arguments_are_refused_at_fit_by_name(
    table, params, error, message
):
    with pytest.raises(error, match=message):
        lazy.Tabularizer(**params).fit(table)


def test_a_column_of_only_missing_items_cannot_be_reduced(table):
    empty = table.assign(notes=None)
    with pytest.raises(ValueError, match="has 0 rows without NaN"):
        lazy.Tabularizer({"notes": "text"}).fit(empty)
    few = table.copy()
    few.loc[3:, "notes"] = None
    with pytest.raises(ValueError, match="has 3 rows without NaN"):
        lazy.Tabularizer({"notes": "text"}, n_components=3).fit(few)


@pytest.mark.parametrize("name", ["flux__g", "remainder", "verbose"])
def test_any_column_name_works_and_names_its_output(table, name):
    renamed = table.rename(columns={"curve": name})
    tab = lazy.Tabularizer({name: "time"}, n_components=2).fit(renamed)
    assert list(tab.get_feature_names_out()[:2]) == [
        f"{name}__pc0",
        f"{name}__pc1",
    ]


def test_an_integer_label_of_a_dataframe_is_a_label_not_a_position(table):
    labelled = table[["notes", "curve"]].set_axis([1, 0], axis=1)
    tab = lazy.Tabularizer({0: "time"}, n_components=None, remainder="drop")
    np.testing.assert_array_equal(
        tab.fit_transform(labelled),
        lazy.time.Statistics().fit_transform(table["curve"]),
    )
    with pytest.raises(ValueError, match="names 2, which is not a column"):
        lazy.Tabularizer({2: "time"}).fit(labelled)


def test_a_failed_fit_leaves_the_tabularizer_unfitted(table):
    tab = lazy.Tabularizer({"curve": "time"}).fit(table)
    with pytest.raises(ValueError, match="remainder"):
        tab.set_params(remainder="keep").fit(table)
    with pytest.raises(exceptions.NotFittedError):
        tab.transform(table)
    refit = tab.set_params(columns={1: "time"}, remainder="drop")
    refit.fit(table.to_numpy())
    assert not hasattr(refit, "feature_names_in_")


def test_each_modality_names_a_registered_encoder_of_its_own():
    assert set(encoders.DEFAULT_ENCODERS) == set(encoders.MODALITIES)
    for modality in encoders.MODALITIES:
        assert encoders.get_encoder(modality).modality == modality
