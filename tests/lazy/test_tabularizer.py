# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The Tabularizer: columns of items to encoded, reduced columns."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn import decomposition
from sklearn import exceptions
from sklearn import model_selection
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


def test_the_columns_are_encoded_reduced_together_and_named(table):
    tab = lazy.Tabularizer(
        {
            "curve": "time_statistics",
            "notes": "text_hashing",
            "cutout": "image_statistics",
        },
        n_columns=5,
    ).fit(table)
    expected = [f"embedding__pc{i}" for i in range(5)]
    assert list(tab.get_feature_names_out()) == [*expected, "remainder__number"]
    assert list(tab.encoders_) == ["curve", "notes", "cutout"]
    out = tab.transform(table.set_axis(range(100, 100 + N_ROWS)))
    assert list(out.columns) == list(tab.get_feature_names_out())
    assert out.index.tolist() == list(range(100, 100 + N_ROWS))
    np.testing.assert_array_equal(out["remainder__number"], table["number"])


def test_a_column_passed_through_keeps_its_type(table):
    kinds = pd.Categorical(["a", "b"] * (N_ROWS // 2))
    typed = table.assign(kind=kinds, count=np.arange(N_ROWS))
    out = lazy.Tabularizer({"notes": "text_hashing"}, n_columns=2).fit(typed)
    out = out.transform(typed)
    assert out["embedding__pc0"].dtype == np.float64
    assert out["remainder__kind"].dtype == kinds.dtype
    assert out["remainder__count"].dtype == np.int64


def _standardized(table, column, encoder):
    """A column's embedding, standardized, and its columns that vary."""
    embedding = encoder.fit_transform(table[column])
    scaler = preprocessing.StandardScaler().fit(embedding)
    return scaler.transform(embedding), int((scaler.var_ > 0).sum())


@pytest.mark.parametrize(
    ("weights", "totals"),
    [("equal", (1.0, 1.0)), (None, None), ([4.0, 0.25], (4.0, 0.25))],
)
def test_the_reduction_is_the_pca_of_the_weighted_standardized_embeddings(
    table, weights, totals
):
    tab = lazy.Tabularizer(
        {"curve": "time_statistics", "cutout": "image_pixels"},
        n_columns=4,
        weights=weights,
        remainder="drop",
    ).fit(table)
    blocks = []
    for index, (column, encoder) in enumerate(
        (("curve", lazy.time.Statistics()), ("cutout", lazy.image.Pixels()))
    ):
        block, n_varying = _standardized(table, column, encoder)
        if totals is not None:
            block = block * np.sqrt(totals[index] / n_varying)
            # The embedding's total variance is its weight.
            assert block.var(axis=0).sum() == pytest.approx(totals[index])
        blocks.append(block)
    expected = decomposition.PCA(4, svd_solver="full").fit_transform(
        np.hstack(blocks)
    )
    np.testing.assert_allclose(tab.transform(table), expected, atol=1e-10)


def test_narrow_embeddings_keep_their_columns_and_names(table):
    tab = lazy.Tabularizer(
        {"curve": "time_statistics"}, n_columns=None, remainder="drop"
    ).fit(table)
    np.testing.assert_array_equal(
        tab.transform(table),
        lazy.time.Statistics().fit_transform(table["curve"]),
    )
    assert tab.get_feature_names_out()[0] == "curve__time_statistics_mean"
    assert tab.components_ is None
    wide = lazy.Tabularizer({"curve": "time_statistics"}, n_columns=13)
    assert wide.fit_transform(table[["number", "curve"]]).shape == (N_ROWS, 14)


def test_a_missing_item_takes_the_contexts_mean_embedding(table):
    gappy = table.copy()
    gappy.loc[[2, 7], "notes"] = None
    gappy.loc[[7, 9], "curve"] = None
    columns = {"notes": "text_hashing", "curve": "time_statistics"}
    tab = lazy.Tabularizer(columns, n_columns=3, remainder="drop").fit(gappy)
    out = tab.transform(gappy)
    # Row 7 has no item at all; rows 2 and 9 keep what their other item says.
    assert out.loc[7].isna().all()
    assert not out.drop(index=7).isna().any(axis=None)
    blocks, _ = tab._encoded(gappy.loc[[2]])
    blocks[0] = tab.mean_[None, : blocks[0].shape[1]]
    np.testing.assert_allclose(out.loc[[2]], tab._reduced(blocks))
    # Without a reduction, a missing item stays a row of NaN in its columns.
    plain = lazy.Tabularizer(columns, n_columns=None, remainder="drop")
    out = plain.fit_transform(gappy)
    assert out.loc[2].isna().sum() == blocks[0].shape[1]


def test_the_reduction_is_fitted_on_the_context_only(table):
    context, queries = table.iloc[:30], table.iloc[30:]
    tab = lazy.Tabularizer(
        {"curve": "time_statistics"}, n_columns=3, remainder="drop"
    ).fit(context)
    one_at_a_time = [tab.transform(queries.iloc[[i]]) for i in range(10)]
    np.testing.assert_allclose(np.vstack(one_at_a_time), tab.transform(queries))


def test_an_encoder_given_is_cloned_and_a_name_builds_one(table):
    given = lazy.time.Spectrum(n_octaves=3)
    tab = lazy.Tabularizer(
        {"curve": given, "notes": "text_statistics"}, n_columns=None
    ).fit(table)
    encoders_used = list(tab.encoders_.values())
    assert encoders_used[0].get_params() == given.get_params()
    assert isinstance(encoders_used[1], lazy.text.Statistics)
    assert not hasattr(given, "_n_features_out")


def test_a_model_behind_it_predicts_and_a_search_tunes_it(table):
    table = table.drop(columns="cutout")
    y = 3 * table["number"] + np.random.default_rng(2).normal(0, 0.1, N_ROWS)
    model = lazy.make_pipeline(
        lazy.Tabularizer(
            {"curve": "time_statistics", "notes": "text_hashing"},
            n_columns=2,
        ),
        standins.HistogramStandIn(n_estimators=2, progress=False),
    )
    model.fit(table, y)
    assert model[-1].feature_names_in_[0] == "embedding__pc0"
    assert model.predict_quantiles(table, [0.5]).shape == (N_ROWS, 1)
    search = model_selection.GridSearchCV(
        model, {"tabularizer__n_columns": [1, 2]}, cv=2
    ).fit(table, y)
    assert search.best_params_["tabularizer__n_columns"] in (1, 2)


class _Recording(lazy.text.Statistics):
    """Text statistics that count the items they encode."""

    def _encode(self, items):
        self.n_encoded_ = getattr(self, "n_encoded_", 0) + len(items)
        return super()._encode(items)


def test_fitting_in_a_pipeline_encodes_each_item_once(table):
    tab = lazy.Tabularizer({"notes": _Recording()}, remainder="drop")
    tab.fit_transform(table)
    assert tab.encoders_["notes"].n_encoded_ == N_ROWS


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
            {"columns": {"curve": "time_statistics"}, "n_columns": 0},
            ValueError,
            "n_columns must be a positive integer",
        ),
        (
            {"columns": {"curve": "time_statistics"}, "weights": "even"},
            ValueError,
            "weights must be 'equal', None or a list",
        ),
        (
            {"columns": {"curve": "time_statistics"}, "weights": [1.0, 2.0]},
            ValueError,
            "one positive number per column of items, 1 here",
        ),
        (
            {"columns": {"curve": "time_statistics"}, "weights": [0.0]},
            ValueError,
            "one positive number per column of items",
        ),
        (
            {"columns": {"curve": "time_statistics"}, "remainder": "keep"},
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


def test_a_short_context_keeps_fewer_components_and_says_so(table):
    few = table.copy()
    few.loc[5:, "notes"] = None
    tab = lazy.Tabularizer({"notes": "text_hashing"}, remainder="drop")
    with pytest.warns(lazy.LazyWarning, match="so they keep 4 columns"):
        out = tab.fit_transform(few)
    assert out.shape == (N_ROWS, 4)
    assert not out[:5].isna().any(axis=None)
    with pytest.raises(ValueError, match="have 1 rows with an item"):
        tab.fit(table.assign(notes=[*table["notes"][:1], *[None] * 39]))


@pytest.mark.parametrize("name", ["flux__g", "remainder", "verbose"])
def test_any_column_name_works_and_names_its_output(table, name):
    renamed = table.rename(columns={"curve": name})
    tab = lazy.Tabularizer({name: "time_statistics"}, n_columns=None)
    names = tab.fit(renamed).get_feature_names_out()
    assert names[0] == f"{name}__time_statistics_mean"
    assert list(names[-3:]) == [
        "remainder__number",
        "remainder__notes",
        "remainder__cutout",
    ]


def test_an_integer_label_of_a_dataframe_is_a_label_not_a_position(table):
    labelled = table[["notes", "curve"]].set_axis([1, 0], axis=1)
    tab = lazy.Tabularizer(
        {0: "time_statistics"}, n_columns=None, remainder="drop"
    )
    np.testing.assert_array_equal(
        tab.fit_transform(labelled),
        lazy.time.Statistics().fit_transform(table["curve"]),
    )
    with pytest.raises(ValueError, match="names 2, which is not a column"):
        lazy.Tabularizer({2: "time_statistics"}).fit(labelled)


def test_a_failed_fit_leaves_the_tabularizer_unfitted(table):
    tab = lazy.Tabularizer({"curve": "time_statistics"}).fit(table)
    with pytest.raises(ValueError, match="remainder"):
        tab.set_params(remainder="keep").fit(table)
    with pytest.raises(exceptions.NotFittedError):
        tab.transform(table)
    refit = tab.set_params(columns={1: "time_statistics"}, remainder="drop")
    refit.fit(table.to_numpy())
    assert not hasattr(refit, "feature_names_in_")


def test_each_modality_names_a_registered_encoder_of_its_own():
    assert set(encoders.DEFAULT_ENCODERS) == set(encoders.MODALITIES)
    for modality in encoders.MODALITIES:
        assert encoders.get_encoder(modality).modality == modality


def test_a_fitted_tabularizer_pickles(table):
    tab = lazy.Tabularizer({"curve": "time_statistics"}, n_columns=2)
    tab.fit(table)
    again = pickle.loads(pickle.dumps(tab))
    pd.testing.assert_frame_equal(again.transform(table), tab.transform(table))


def test_a_transformer_of_the_remainder_is_cloned_fitted_and_named(table):
    scaler = preprocessing.StandardScaler()
    tab = lazy.Tabularizer(
        {"curve": "time_statistics", "notes": "text", "cutout": "image_pixels"},
        n_columns=2,
        remainder=scaler,
    )
    tab.set_params(columns={"curve": "time_statistics"})
    out = tab.fit_transform(table[["number", "curve"]])
    assert list(out.columns)[-1] == "remainder__number"
    assert out["remainder__number"].std(ddof=0) == pytest.approx(1.0)
    assert not hasattr(scaler, "mean_")


def test_a_table_with_other_columns_is_refused_at_transform(table):
    tab = lazy.Tabularizer({"curve": "time_statistics"}).fit(table)
    with pytest.raises(ValueError, match="feature names"):
        tab.transform(table.rename(columns={"number": "count"}))
