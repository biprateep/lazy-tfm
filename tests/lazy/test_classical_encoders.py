# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The classical encoders of time series, images and text."""

import numpy as np
from PIL import Image
import pytest
from scipy import stats
from sklearn import base as sklearn_base

import lazy
from lazy import encoders
from lazy.encoders import _pretrained

RNG = np.random.default_rng(0)
ITEMS = {
    "time": [RNG.normal(size=n).cumsum() for n in (40, 7, 128, 1, 33)],
    "image": [
        RNG.uniform(size=(h, w, 3)) for h, w in ((9, 12), (1, 1), (20, 5))
    ],
    "text": ["A lazy model, frozen.", "", "42 rows of context", "CONTEXT"],
}


CLASSICAL = [
    name
    for name, cls in encoders.ENCODERS.items()
    if not issubclass(cls, _pretrained.PretrainedEncoder)
]


@pytest.mark.parametrize("name", CLASSICAL)
def test_every_registered_encoder_meets_the_contract(name, tmp_path):
    encoder = encoders.get_encoder(name)
    items = ITEMS[encoder.modality]
    embeddings = encoder.fit_transform(items)
    names = encoder.get_feature_names_out()
    assert embeddings.shape == (len(items), len(names))
    assert len(set(names)) == len(names)
    # A row depends on its own item alone, and the cache gives it back.
    alone = sklearn_base.clone(encoder).set_params(batch_size=1, cache=tmp_path)
    for _ in range(2):
        np.testing.assert_array_equal(
            alone.fit(items[:1]).transform(items), embeddings
        )


def test_the_time_statistics_agree_with_direct_computations():
    series = ITEMS["time"][0]
    row = dict(
        zip(
            lazy.time.STATISTICS,
            lazy.time.Statistics().fit_transform([series])[0],
            strict=True,
        )
    )
    time = np.arange(series.size)
    centred = series - series.mean()
    expected = {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "q25": np.quantile(series, 0.25),
        "median": np.median(series),
        "q75": np.quantile(series, 0.75),
        "max": series.max(),
        "skewness": stats.skew(series),
        "kurtosis": stats.kurtosis(series),
        "slope": np.polyfit(time, series, 1)[0],
        "autocorrelation": centred[:-1] @ centred[1:] / (centred @ centred),
        "roughness": np.abs(series[1:] - series[:-1]).mean(),
        "length": series.size,
    }
    assert row == pytest.approx(expected, rel=1e-10)


def test_a_series_too_short_or_constant_for_a_statistic_gives_zero():
    series = [[3.0], [2.0, 2.0, 2.0]]
    encoder = lazy.time.Statistics().fit(series)
    rows = dict(
        zip(encoder.get_feature_names_out(), encoder.transform(series).T)
    )
    np.testing.assert_array_equal(rows["time_statistics_mean"], [3, 2])
    for name in ("std", "skewness", "kurtosis", "slope", "autocorrelation"):
        np.testing.assert_array_equal(rows[f"time_statistics_{name}"], 0)
    np.testing.assert_array_equal(rows["time_statistics_roughness"], 0)
    np.testing.assert_array_equal(lazy.time.Spectrum().fit_transform(series), 0)


def test_a_constant_series_with_rounding_residue_is_still_constant():
    series = [[0.1] * 7]  # 0.1 - mean(0.1, ...) is not exactly 0.
    rows = lazy.time.Statistics().fit_transform(series)
    np.testing.assert_array_equal(rows[0, 7:12], 0)
    np.testing.assert_array_equal(lazy.time.Spectrum().fit_transform(series), 0)


def test_values_that_are_not_finite_are_dropped_from_a_series():
    gappy = [1.0, np.nan, 4.0, np.inf, 2.0, 8.0]
    for encoder in (lazy.time.Statistics(), lazy.time.Spectrum()):
        np.testing.assert_array_equal(
            encoder.fit_transform([gappy]),
            encoder.transform([[1.0, 4.0, 2.0, 8.0]]),
        )


@pytest.mark.parametrize(
    ("item", "message"),
    [([[1.0, 2.0]], "one-dimensional"), ([np.nan, np.inf], "no finite value")],
)
def test_what_is_no_time_series_is_refused(item, message):
    for encoder in (lazy.time.Statistics(), lazy.time.Spectrum()):
        with pytest.raises(ValueError, match=message):
            encoder.fit_transform([item, [1.0]])


@pytest.mark.parametrize("length", [64, 100, 1000])
def test_a_sine_has_its_power_in_its_octave_whatever_the_length(length):
    # 0.1 cycles per sample is in octave 2: (1/16, 1/8].
    cycles = round(0.1 * length)
    series = np.sin(2 * np.pi * cycles * np.arange(length) / length)
    row = lazy.time.Spectrum(n_octaves=4).fit_transform([series])[0]
    np.testing.assert_allclose(row, [0, 0, 1, 0, cycles / length], atol=1e-12)


def test_the_shares_are_shares_of_the_variance_up_to_the_nyquist_frequency():
    # Two parts of variance 1 each: at 1/2 and at 1/8 cycles per sample.
    t = np.arange(64)
    series = (-1.0) ** t + np.sqrt(2) * np.sin(2 * np.pi * t / 8)
    row = lazy.time.Spectrum(n_octaves=3).fit_transform([series])[0]
    np.testing.assert_allclose(row[:3], [0.5, 0, 0.5], atol=1e-12)


def test_the_shares_of_the_octaves_sum_to_one_and_the_last_takes_the_rest():
    series = ITEMS["time"][2]
    fine = lazy.time.Spectrum(n_octaves=6).fit_transform([series])[0]
    coarse = lazy.time.Spectrum(n_octaves=2).fit_transform([series])[0]
    assert fine[:-1].sum() == pytest.approx(1)
    np.testing.assert_allclose(coarse, [fine[0], fine[1:-1].sum(), fine[-1]])


def test_the_image_statistics_agree_with_direct_computations():
    image = ITEMS["image"][0]
    encoder = lazy.image.Statistics().fit([image])
    row = dict(
        zip(encoder.get_feature_names_out(), encoder.transform([image])[0])
    )
    green = image[:, :, 1]
    steps = np.concatenate(
        [np.diff(green, axis=0).ravel(), np.diff(green, axis=1).ravel()]
    )
    assert row["image_statistics_mean1"] == pytest.approx(green.mean())
    assert row["image_statistics_std1"] == pytest.approx(green.std())
    assert row["image_statistics_q751"] == pytest.approx(
        np.quantile(green, 0.75)
    )
    assert row["image_statistics_max2"] == image[:, :, 2].max()
    assert row["image_statistics_gradient1"] == pytest.approx(
        np.abs(steps).mean()
    )


def test_pixels_average_whole_blocks_and_keep_the_mean_of_any_image():
    image = RNG.uniform(size=(6, 8, 2))
    small = lazy.image.Pixels(size=2, n_channels=2).fit_transform([image])
    blocks = image.reshape(2, 3, 2, 4, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(small[0], blocks.ravel())
    for odd in (*ITEMS["image"], np.ones((1, 1, 3))):
        thumbnail = lazy.image.Pixels(size=4).fit_transform([odd])[0]
        np.testing.assert_allclose(
            thumbnail.reshape(16, 3).mean(axis=0), odd.mean(axis=(0, 1))
        )


@pytest.mark.parametrize("name", ["image_statistics", "image_pixels"])
def test_a_file_a_pillow_image_and_their_array_give_the_same_row(
    name, tmp_path
):
    array = RNG.integers(0, 256, size=(6, 5, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(array).save(path)
    encoder = encoders.get_encoder(name).fit([array])
    grey = encoders.get_encoder(name, n_channels=1)
    with Image.open(path) as image:
        rows = encoder.transform([array, str(path), path, image])
        # A grey image is the same file read with one channel.
        np.testing.assert_array_equal(
            grey.fit_transform([path]),
            grey.transform([np.asarray(image.convert("L"))]),
        )
    np.testing.assert_array_equal(rows[1:], rows[:1].repeat(3, axis=0))


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32])
def test_an_image_of_more_than_8_bits_keeps_its_values(dtype, tmp_path):
    array = (np.arange(48).reshape(6, 8) * 1000.5).astype(dtype)
    path = tmp_path / "image.tif"
    Image.fromarray(array).save(path)
    encoder = lazy.image.Pixels(size=2, n_channels=1)
    rows = encoder.fit_transform([path, array])
    np.testing.assert_array_equal(rows[0], rows[1])
    with pytest.raises(ValueError, match="set n_channels=1"):
        lazy.image.Pixels().fit_transform([path])


def test_palette_images_are_cached_by_their_colours(tmp_path):
    images = []
    for colour in ([255, 0, 0], [0, 0, 255]):
        image = Image.new("P", (4, 4))
        image.putpalette(colour * 256)
        images.append(image)
    encoder = lazy.image.Statistics(cache=tmp_path)
    np.testing.assert_array_equal(
        encoder.fit_transform(images)[:, :3], [[255, 0, 0], [0, 0, 255]]
    )
    np.testing.assert_array_equal(
        encoder.transform(images[::-1])[:, :3], [[0, 0, 255], [255, 0, 0]]
    )


def test_a_numpy_integer_parameter_is_cached_as_its_python_value(tmp_path):
    numpy = lazy.image.Pixels(size=np.int64(2), cache=tmp_path)
    python = lazy.image.Pixels(size=2, cache=tmp_path)
    assert numpy._cache_identity() == python._cache_identity()
    np.testing.assert_array_equal(
        numpy.fit_transform(ITEMS["image"]),
        python.fit_transform(ITEMS["image"]),
    )


def test_a_file_is_cached_by_its_content_not_by_its_path(tmp_path):
    path = tmp_path / "image.png"
    encoder = lazy.image.Statistics(cache=tmp_path / "cache")
    rows = []
    for value in (10, 200):
        Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)
        rows.append(encoder.fit_transform([path])[0, 0])
    assert rows == [10, 200]


@pytest.mark.parametrize(
    ("image", "params", "message"),
    [
        (np.ones((4, 4)), {}, "as n_channels says"),
        (np.ones((4, 4, 3, 1)), {}, "as n_channels says"),
        (np.ones((0, 4, 3)), {}, "must have pixels"),
        (np.full((4, 4, 3), np.nan), {}, "all of them finite"),
        (Image.new("RGB", (4, 4)), {"n_channels": 2}, "1, 3 or 4 channels"),
    ],
)
def test_what_is_no_image_of_n_channels_is_refused(image, params, message):
    with pytest.raises(ValueError, match=message):
        lazy.image.Pixels(**params).fit_transform([image])


def test_the_text_statistics_count_characters_and_words():
    rows = lazy.text.Statistics().fit_transform(["Ab ab 1,2 X", "", " \n"])
    expected = [11, 4, 2, 3 / 4, 2 / 11, 2 / 11, 1 / 11, 3 / 11]
    np.testing.assert_allclose(rows[0], expected)
    np.testing.assert_array_equal(rows[1], 0)
    np.testing.assert_array_equal(rows[2], [2, 0, 0, 0, 0, 0, 0, 1])


def test_hashing_keeps_no_vocabulary_and_gives_rows_of_unit_length():
    texts = ITEMS["text"]
    rows = (
        lazy.text.Hashing(n_features=32).fit(["other words"]).transform(texts)
    )
    np.testing.assert_array_equal(
        rows, lazy.text.Hashing(n_features=32).fit_transform(texts)
    )
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), [1, 0, 1, 1])
    assert rows[3] @ rows[2] > rows[3] @ rows[0]  # "CONTEXT" is in "context".


def test_an_item_that_is_no_text_is_refused():
    for encoder in (lazy.text.Statistics(), lazy.text.Hashing()):
        with pytest.raises(TypeError, match="a text must be a str.*bytes"):
            encoder.fit_transform(["text", b"bytes"])


@pytest.mark.parametrize(
    ("encoder", "message"),
    [
        (lazy.time.Spectrum(n_octaves=0), "n_octaves must be a positive"),
        (lazy.image.Pixels(size=2.5), "size must be a positive"),
        (lazy.image.Statistics(n_channels=True), "n_channels must be a"),
        (lazy.text.Hashing(n_features=0), "n_features"),
        (lazy.text.Hashing(n_features=True), "n_features must be a"),
        (lazy.text.Hashing(analyzer="words"), "analyzer"),
        (lazy.text.Hashing(ngram_range=(3, 2)), "ngram_range"),
    ],
)
def test_a_bad_parameter_of_an_encoder_is_refused_at_fit(encoder, message):
    with pytest.raises(ValueError, match=message):
        encoder.fit(ITEMS[encoder.modality])
