# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The encoder contract, its registry and the embedding cache, on stand-ins."""

import numpy as np
import pandas as pd
import pytest
from sklearn import base as sklearn_base
from sklearn import compose

import lazy
from lazy import encoders
from lazy.encoders import _conformance

TEXTS = ["a tabular model", "", "lazy", "frozen weights", "context"]


class _Counts(encoders.BaseEncoder):
    """Each text's length and its count of one letter; counts its calls."""

    modality = "text"

    def __init__(self, *, batch_size=64, cache=False, letter="a"):
        self.batch_size = batch_size
        self.cache = cache
        self.letter = letter

    def _width(self):
        return 2

    def _record(self, items):
        """Keeps each batch, for the tests to count the calls."""
        self.batches_ = [*getattr(self, "batches_", []), list(items)]

    def _encode(self, items):
        self._record(items)
        return np.array(
            [[len(item), item.count(self.letter)] for item in items],
            dtype=float,
        )


class _Series(_Counts):
    """The mean and the spread of each time series."""

    modality = "time"

    def _encode(self, items):
        self._record(items)
        return np.array([[np.mean(item), np.std(item)] for item in items])


@pytest.fixture(name="cache")
def _cache_directory(tmp_path):
    """A cache directory of this test's own."""
    return tmp_path / "embeddings"


def _encoded(items, **params):
    """The stand-in's embeddings of ``items``, and the fitted stand-in."""
    encoder = _Counts(**params).fit(items)
    return encoder.transform(items), encoder


@pytest.mark.parametrize(
    "column",
    [
        TEXTS,
        tuple(TEXTS),
        pd.Series(TEXTS, index=range(5, 10)),
        pd.DataFrame({"text": TEXTS}),
        np.array(TEXTS, dtype=object),
        np.array(TEXTS, dtype=object).reshape(-1, 1),
        np.array(TEXTS).reshape(-1, 1),
    ],
    ids=["list", "tuple", "series", "frame", "array", "column", "text-column"],
)
def test_a_column_in_any_form_gives_one_row_per_item(column):
    embeddings, _ = _encoded(column)
    assert embeddings.dtype == np.float64
    np.testing.assert_array_equal(
        embeddings, [[15, 3], [0, 0], [4, 1], [14, 0], [7, 0]]
    )


def test_numeric_rows_are_one_item_each_as_an_array_or_a_table():
    series = np.arange(12.0).reshape(3, 4)
    for column in (series, pd.DataFrame(series)):
        np.testing.assert_allclose(
            _Series().fit(column).transform(column)[:, 0], [1.5, 5.5, 9.5]
        )


def test_a_missing_item_gets_nan_and_never_reaches_the_encoder():
    items = ["lazy", None, float("nan"), pd.NA, np.float32("nan"), "context"]
    embeddings, encoder = _encoded(items)
    np.testing.assert_array_equal(
        np.isnan(embeddings[:, 0]), [0, 1, 1, 1, 1, 0]
    )
    assert encoder.batches_ == [["lazy", "context"]]


def test_a_row_does_not_depend_on_its_batch():
    whole, encoder = _encoded(TEXTS)
    by_two, batched = _encoded(TEXTS, batch_size=2)
    np.testing.assert_array_equal(by_two, whole)
    assert [len(batch) for batch in batched.batches_] == [2, 2, 1]
    assert [len(batch) for batch in encoder.batches_] == [5]


def test_the_input_is_not_modified():
    frame = pd.DataFrame({"text": [*TEXTS, None]})
    before = frame.copy()
    _encoded(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_an_encoder_is_a_step_of_a_column_transformer():
    frame = pd.DataFrame({"text": TEXTS, "number": np.arange(5.0)})
    table = compose.ColumnTransformer(
        [("text", _Counts(), ["text"])], remainder="passthrough"
    ).fit_transform(frame)
    np.testing.assert_array_equal(table[:, 0], [15, 0, 4, 14, 7])
    np.testing.assert_array_equal(table[:, 2], frame["number"])


@pytest.mark.parametrize(
    ("column", "error", "message"),
    [
        (pd.DataFrame({"a": TEXTS, "b": TEXTS}), ValueError, "one column"),
        ("one text", TypeError, "must be a column of items"),
        (3.0, TypeError, "must be a column of items"),
        (np.array(3.0), TypeError, "not a scalar"),
        (np.array([TEXTS, TEXTS]), ValueError, r"shape \(2, 5\)"),
        (pd.DataFrame(index=range(3)), ValueError, "one column"),
    ],
)
def test_what_is_not_one_column_of_items_is_refused(column, error, message):
    with pytest.raises(error, match=message):
        _Counts().fit(column)


class _Broken(_Counts):
    """An encoder whose answers break the contract."""

    def __init__(self, *, batch_size=64, cache=False, fault="narrow"):
        super().__init__(batch_size=batch_size, cache=cache)
        self.fault = fault

    def _encode(self, items):
        rows = super()._encode(items)
        return rows[:, :1] if self.fault == "narrow" else rows + np.inf


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("narrow", r"shape \(5, 1\).*expected \(5, 2\)"),
        ("infinite", "not finite"),
    ],
)
def test_an_answer_that_breaks_the_contract_is_refused(fault, message):
    encoder = _Broken(fault=fault).fit(TEXTS)
    with pytest.raises(ValueError, match=message):
        encoder.transform(TEXTS)


def test_a_cached_item_is_not_encoded_again(cache):
    first, encoder = _encoded(TEXTS, cache=cache)
    again = sklearn_base.clone(encoder).fit(TEXTS)
    np.testing.assert_array_equal(again.transform(TEXTS[::-1]), first[::-1])
    assert not hasattr(again, "batches_")
    more = [*TEXTS[:2], "new", None]
    np.testing.assert_array_equal(again.transform(more)[:2], first[:2])
    assert again.batches_ == [["new"]]


def test_the_cache_keeps_encoders_with_other_outputs_apart(cache):
    first, _ = _encoded(TEXTS, cache=cache)
    other, encoder = _encoded(TEXTS, cache=cache, letter="e")
    assert encoder.batches_ == [TEXTS]
    assert not np.array_equal(other, first)
    np.testing.assert_array_equal(
        _encoded(TEXTS, cache=cache, batch_size=1)[0], first
    )


def test_a_damaged_cache_file_counts_as_missing(cache):
    first, encoder = _encoded(TEXTS[:1], cache=cache)
    (file,) = cache.rglob("*.npy")
    assert file.parent.parent.parent.name == "v1"
    assert "_Counts" in (file.parent.parent / "identity.json").read_text()
    file.write_bytes(b"PK\x03\x04 not an array")
    again = sklearn_base.clone(encoder).fit(TEXTS[:1])
    np.testing.assert_array_equal(again.transform(TEXTS[:1]), first)
    assert again.batches_ == [TEXTS[:1]]
    assert not list(cache.rglob("*.partial"))


def test_arrays_are_cached_by_content_shape_and_type(cache):
    series = np.arange(12.0).reshape(3, 4)
    encoder = _Series(cache=cache).fit(series)
    encoder.transform(series)
    encoder.transform(series.copy())
    assert len(encoder.batches_) == 1
    encoder.transform(series.reshape(6, 2))
    encoder.transform(series.astype(np.float32))
    assert len(encoder.batches_) == 3


def test_an_item_without_content_bytes_cannot_be_cached(cache):
    encoder = _Counts(cache=cache).fit(TEXTS)
    for item in (object(), np.ma.array([1.0, 2.0], mask=[0, 1])):
        with pytest.raises(TypeError, match="implement _item_bytes"):
            encoder.transform([item])


def test_the_default_cache_is_under_data_home_and_can_be_cleared(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LAZY_DATA_HOME", str(tmp_path))
    _encoded(TEXTS, cache=True)
    assert len(list((tmp_path / "embeddings" / "v1").rglob("*.npy"))) == 5
    encoders.clear_embedding_cache()
    assert not (tmp_path / "embeddings").exists()


def test_a_parameter_the_identity_cannot_hold_refuses_the_cache(cache):
    encoder = _Counts(cache=cache, letter=np.zeros(2000)).fit(TEXTS)
    with pytest.raises(TypeError, match="override _cache_identity"):
        encoder.transform(TEXTS)


@pytest.mark.parametrize(
    "params",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": 2.0},
        {"cache": None},
        {"cache": ""},
    ],
)
def test_a_bad_parameter_is_refused_at_fit_by_name(params):
    with pytest.raises(ValueError, match=next(iter(params))):
        _Counts(**params).fit(TEXTS)
    _Counts(batch_size=np.int64(2)).fit(TEXTS)


def test_a_class_of_a_script_or_a_function_must_name_itself(cache, monkeypatch):
    monkeypatch.setattr(_Counts, "__module__", "__main__")
    encoder = _Counts(cache=cache).fit(TEXTS)
    with pytest.raises(TypeError, match="override _cache_identity"):
        encoder.transform(TEXTS)


def test_a_cache_that_cannot_be_written_warns_and_still_encodes(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    encoder = _Counts(cache=blocked, batch_size=2).fit(TEXTS)
    with pytest.warns(RuntimeWarning, match="cannot be written") as caught:
        embeddings = encoder.transform(TEXTS)
    assert len(caught) == 1
    np.testing.assert_array_equal(embeddings, _encoded(TEXTS)[0])


@pytest.fixture(name="registry")
def _registry(monkeypatch):
    """The registry with two stand-ins, restored after the test."""
    monkeypatch.setattr(encoders.registry, "ENCODERS", {})
    encoders.register("counts", _Counts)
    encoders.register("series", _Series)
    return encoders.registry.ENCODERS


def test_a_registered_encoder_is_built_by_name(registry):
    assert encoders.list_encoders() == ["counts", "series"]
    assert encoders.list_encoders("time") == ["series"]
    encoder = encoders.get_encoder("counts", letter="e")
    assert isinstance(encoder, _Counts)
    assert encoder.letter == "e"


def test_an_unknown_name_and_what_is_no_class_are_refused(registry):
    with pytest.raises(lazy.UnknownNameError, match="known: .*'counts'"):
        encoders.get_encoder("count")
    with pytest.raises(lazy.UnknownNameError, match="unknown modality"):
        encoders.list_encoders("texts")
    with pytest.raises(TypeError, match="is not a BaseEncoder subclass"):
        encoders.register("counts", _Counts())
    with pytest.raises(ValueError, match="'text' is a modality"):
        encoders.register("text", _Counts)


class _Positional(_Counts):
    """Breaks the contract: its parameters, and a modality of its own."""

    modality = "Text"

    def __init__(self, letter, version, batch_size=32, **kwargs):
        super().__init__(letter=letter, batch_size=batch_size)


def test_a_class_that_breaks_the_contract_is_not_registered(registry):
    with pytest.raises(TypeError, match="cannot be registered") as caught:
        encoders.register("positional", _Positional)
    for problem in (
        "modality = 'Text', not one of",
        "takes **kwargs",
        "lacks the uniform parameter 'cache'",
        "'batch_size' must be keyword-only",
        "'batch_size' defaults to 32, not 64",
        "'version' has no default",
    ):
        assert problem in str(caught.value)
    assert "positional" not in registry
    assert "abstract methods are not implemented: ['_encode', '_width']" in (
        _conformance.problems(encoders.BaseEncoder)
    )


class _Misnamed(_Counts):
    """Names one column of its two."""

    def _names(self):
        return ["length"]


def test_columns_are_named_by_modality_class_and_name_or_number():
    assert list(_Counts().fit(TEXTS).get_feature_names_out()) == [
        "text__counts_0",
        "text__counts_1",
    ]
    with pytest.raises(ValueError, match="gave 1 names for 2 columns"):
        _Misnamed().fit(TEXTS).get_feature_names_out()
