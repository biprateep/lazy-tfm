# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The Tabularizer: columns of items to encoded, reduced columns."""

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import linalg
from sklearn import decomposition
from sklearn import exceptions
from sklearn import model_selection
from sklearn import preprocessing
import standins

import lazy
from lazy import encoders
from lazy.encoders import tabularizer

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


def test_the_columns_are_encoded_reduced_and_named_by_their_column(table):
    tab = lazy.Tabularizer(
        {
            "curve": "time_statistics",
            "notes": "text_hashing",
            "cutout": "image_statistics",
        },
        n_columns=5,
    ).fit(table)
    names = list(tab.get_feature_names_out())
    assert len(names) == 6 and names[-1] == "remainder__number"
    # Every embedding keeps its first component, and all share the rest.
    assert {name.split("__")[0] for name in names[:5]} == {
        "curve",
        "notes",
        "cutout",
    }
    assert names[0] == "curve__pc0"
    assert list(tab.encoders_) == ["curve", "notes", "cutout"]
    out = tab.transform(table.set_axis(range(100, 100 + N_ROWS)))
    assert list(out.columns) == names
    assert out.index.tolist() == list(range(100, 100 + N_ROWS))
    np.testing.assert_array_equal(out["remainder__number"], table["number"])


def test_a_column_passed_through_keeps_its_type(table):
    kinds = pd.Categorical(["a", "b"] * (N_ROWS // 2))
    typed = table.assign(kind=kinds, count=np.arange(N_ROWS))
    out = lazy.Tabularizer({"notes": "text_hashing"}, n_columns=2).fit(typed)
    out = out.transform(typed)
    assert out["notes__pc0"].dtype == np.float64
    assert out["remainder__kind"].dtype == kinds.dtype
    assert out["remainder__count"].dtype == np.int64


def _components(table, column, encoder):
    """A column's principal components, and the variance of each."""
    embedding = encoder.fit_transform(table[column])
    scaler = preprocessing.StandardScaler().fit(embedding)
    pca = decomposition.PCA(svd_solver="full").fit(scaler.transform(embedding))
    scores = pca.transform(scaler.transform(embedding))
    n_varying = int((scaler.var_ > 0).sum())
    return scores, pca.explained_variance_, n_varying


@pytest.mark.parametrize(
    ("weights", "totals"),
    [("equal", (1.0, 1.0)), (None, None), ([4.0, 0.25], (4.0, 0.25))],
)
def test_the_best_weighted_components_of_each_embedding_are_kept(
    table, weights, totals
):
    tab = lazy.Tabularizer(
        {"curve": "time_statistics", "cutout": "image_pixels"},
        n_columns=6,
        weights=weights,
        remainder="drop",
    ).fit(table)
    out = tab.transform(table)
    ranked, scores = [], {}
    for index, (column, encoder) in enumerate(
        (("curve", lazy.time.Statistics()), ("cutout", lazy.image.Pixels()))
    ):
        scores[column], variance, n_varying = _components(
            table, column, encoder
        )
        weight = 1.0 if totals is None else totals[index] / n_varying
        ranked += [
            # An embedding's first component is kept before any other.
            (np.inf if j == 0 else weight * value, column, j)
            for j, value in enumerate(variance)
        ]
    kept = sorted(ranked, key=lambda entry: -entry[0])[:6]
    expected = sorted((column, j) for _, column, j in kept)
    expected = [entry for entry in expected if entry[0] == "curve"] + [
        entry for entry in expected if entry[0] == "cutout"
    ]
    assert list(out.columns) == [
        f"{column}__pc{i}"
        for i, (column, _) in zip(_running(expected), expected, strict=True)
    ]
    for name, (column, j) in zip(out.columns, expected, strict=True):
        np.testing.assert_allclose(out[name], scores[column][:, j], atol=1e-10)


def _running(entries):
    """The position of each entry among those of its own column."""
    seen = {}
    for column, _ in entries:
        seen[column] = seen.get(column, -1) + 1
        yield seen[column]


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


def test_a_missing_item_is_nan_in_its_own_columns_alone(table):
    gappy = table.copy()
    gappy.loc[[2, 7], "notes"] = None
    gappy.loc[[7, 9], "curve"] = None
    columns = {"notes": "text_hashing", "curve": "time_statistics"}
    tab = lazy.Tabularizer(columns, n_columns=4, remainder="drop").fit(gappy)
    out = tab.transform(gappy)
    notes = [name for name in out.columns if name.startswith("notes__")]
    curve = [name for name in out.columns if name.startswith("curve__")]
    assert notes and curve
    assert out.loc[[2, 7], notes].isna().all(axis=None)
    assert out.loc[[7, 9], curve].isna().all(axis=None)
    assert not out.loc[2, curve].isna().any()
    assert not out.loc[9, notes].isna().any()
    assert not out.drop(index=[2, 7, 9]).isna().any(axis=None)
    # A row's columns of one item do not depend on its other items, and
    # the rows without an item are left out of that item's fit.
    alone = lazy.Tabularizer(
        {"curve": "time_statistics"}, n_columns=len(curve), remainder="drop"
    ).fit(gappy.drop(index=[7, 9]))
    np.testing.assert_allclose(
        out.loc[[2, 5], curve], alone.transform(gappy.loc[[2, 5]])
    )


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
    assert model[-1].feature_names_in_[0] == "curve__pc0"
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
    with pytest.raises(ValueError, match="has 1 rows with an item"):
        tab.fit(table.assign(notes=[*table["notes"][:1], *[None] * 39]))


def test_the_principal_components_are_scikit_learns_full_pca():
    rows = np.random.default_rng(0).normal(size=(60, 8)) * np.arange(1, 9)
    pca = decomposition.PCA(5, svd_solver="full").fit(rows)
    got = tabularizer._principal_components(rows, 5)
    want = (pca.mean_, pca.components_, pca.explained_variance_)
    for got_array, want_array in zip(got, want, strict=True):
        np.testing.assert_array_equal(got_array, want_array)


def test_a_failed_svd_driver_falls_back_to_gesvd_and_reduces_the_same(
    monkeypatch,
):
    rows = np.random.default_rng(0).normal(size=(60, 8)) * np.arange(1, 9)
    expected = tabularizer._principal_components(rows, 5)
    default_svd = linalg.svd
    drivers = []

    def gesdd_fails(*args, lapack_driver="gesdd", **kwargs):
        drivers.append(lapack_driver)
        if lapack_driver == "gesdd":
            raise np.linalg.LinAlgError("SVD did not converge")
        return default_svd(*args, lapack_driver=lapack_driver, **kwargs)

    monkeypatch.setattr(linalg, "svd", gesdd_fails)
    fallback = tabularizer._principal_components(rows, 5)
    assert drivers == ["gesdd", "gesvd"]
    for got, want in zip(fallback, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


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


def test_a_remainder_transformer_without_names_keeps_the_columns_names():
    frame = pd.DataFrame(
        {"notes": ["a b", "c d", "e f", "g h"], "num": [1.0, 2.0, 3.0, 4.0]}
    )
    out = lazy.Tabularizer(
        {"notes": "text_hashing"},
        n_columns=2,
        remainder=preprocessing.FunctionTransformer(np.log1p),
    ).fit_transform(frame)
    assert list(out.columns)[-1] == "remainder__num"
    np.testing.assert_allclose(out["remainder__num"], np.log1p(frame["num"]))


def test_a_remainder_transformer_is_skipped_when_no_column_is_left():
    frame = pd.DataFrame({"notes": ["a b", "c d", "e f", "g h"]})
    tab = lazy.Tabularizer(
        {"notes": "text_hashing"},
        n_columns=2,
        remainder=preprocessing.StandardScaler(),
    )
    out = tab.fit_transform(frame)
    assert list(out.columns) == ["notes__pc0", "notes__pc1"]
    assert tab.transform(frame).shape == (4, 2)


def test_columns_that_pass_through_are_named_by_their_integer_labels():
    frame = pd.DataFrame({10: ["a b", "c d", "e f", "g h"], 20: [1.0, 2, 3, 4]})
    out = lazy.Tabularizer({10: "text_hashing"}, n_columns=2).fit_transform(
        frame
    )
    assert list(out.columns) == ["10__pc0", "10__pc1", "remainder__20"]


@pytest.mark.parametrize("seed", ["a", 1.5, -1])
def test_a_random_state_that_is_no_seed_is_refused(seed):
    frame = pd.DataFrame({"notes": ["a b", "c d", "e f"]})
    tab = lazy.Tabularizer({"notes": "text_hashing"}, random_state=seed)
    with pytest.raises((TypeError, ValueError), match="random_state"):
        tab.fit(frame)


def _curves():
    """Each of 120 rows a series of six values, as an image of one pixel."""
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(120, 6))
    return pd.DataFrame(
        {
            "curve": [curve.reshape(1, 1, 6) for curve in curves],
            "number": rng.normal(size=120),
        }
    )


def _reducing(**reducer):
    """A Tabularizer of the curves, reduced to 3 columns."""
    return lazy.Tabularizer(
        {"curve": lazy.image.Pixels(size=1, n_channels=6)},
        n_columns=3,
        reducer=encoders.AdaptiveReducer(**reducer) if reducer else "pca",
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"hidden": 0}, "hidden == 0, must be >= 1"),
        ({"n_steps": -1}, "n_steps == -1, must be >= 0"),
        ({"learning_rate": 0.0}, "learning_rate == 0.0, must be > 0"),
        ({"validation_fraction": 1.0}, "validation_fraction == 1.0, must"),
        (
            {"validation_fraction": 0.6, "query_fraction": 0.5},
            "must be below 1, to leave support rows",
        ),
        ({"random_state": -1}, "random_state == -1, must be >= 0"),
        ({"block_dropout": 1.0}, "block_dropout == 1.0, must be < 1"),
        ({"network": "cnn"}, "network must be 'mlp' or a torch.nn.Module"),
    ],
)
def test_a_bad_setting_of_the_adaptive_reducer_is_refused_by_name(
    params, message
):
    with pytest.raises(ValueError, match=message):
        _reducing(**params).fit(_curves())


def test_outside_a_pipeline_the_mlp_stays_pca_and_pickles():
    X = _curves()
    tab = _reducing(n_steps=3).fit(X)
    assert tab.trains_through_model and not hasattr(tab, "history_")
    np.testing.assert_allclose(
        tab.transform(X), _reducing().fit(X).transform(X)
    )
    again = pickle.loads(pickle.dumps(tab))
    pd.testing.assert_frame_equal(again.transform(X), tab.transform(X))
