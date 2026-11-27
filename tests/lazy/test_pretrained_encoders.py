# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The pretrained encoders: their own code on fakes, and golden embeddings.

The fakes (``fakes/encoders.py``) stand in for what each encoder loads
from ``transformers`` and ``chronos``, so these tests need those packages
and run in the backends job. The golden test runs the real weights, on
request.
"""

from importlib import metadata
import json
import sys

from fakes import encoders as fake_encoders
import golden_data
import numpy as np
from PIL import Image
import pytest

import lazy
from lazy import encoders
from lazy.encoders import _pretrained

MODELS = sorted(encoders.CHECKPOINTS)
"""Every pinned model, as ``"<modality>:<model>"``."""

PRETRAINED = {
    name: cls
    for name, cls in encoders.ENCODERS.items()
    if issubclass(cls, _pretrained.PretrainedEncoder)
}


def _encoder(key, **params):
    """The encoder of one pinned model, on the CPU, without the cache."""
    modality, _, model = key.partition(":")
    params = {"device": "cpu", "cache": False, **params}
    return encoders.get_encoder(modality, model=model, **params)


@pytest.fixture(name="fakes")
def _install_fakes(monkeypatch, tmp_path):
    """The fakes in place of the downloads and the upstream loaders.

    Returns:
        How many times each model's network loaded, by its key.
    """
    loads = fake_encoders.install(monkeypatch, tmp_path / "weights")
    lazy.clear_model_cache()
    yield loads
    lazy.clear_model_cache()


@pytest.mark.parametrize("key", MODELS)
def test_a_row_depends_on_its_item_alone_whatever_its_batch(fakes, key):
    items = golden_data.encoder_items()[key.partition(":")[0]]
    encoder = _encoder(key, batch_size=2)
    together = encoder.fit_transform(items)
    alone = [encoder.transform(items[i : i + 1]) for i in range(len(items))]
    reversed_ = encoder.transform(items[::-1])[::-1]
    np.testing.assert_allclose(np.vstack(alone), together, rtol=1e-6)
    np.testing.assert_allclose(reversed_, together, rtol=1e-6)
    assert np.abs(together).max() < fake_encoders.PAD / 10
    assert together.shape == (len(items), len(encoder.get_feature_names_out()))


@pytest.mark.parametrize("key", MODELS)
def test_the_network_loads_once_per_process_and_device(fakes, key):
    items = golden_data.encoder_items()[key.partition(":")[0]]
    for _ in range(3):
        _encoder(key).fit_transform(items)
    assert fakes[key] == 1


def test_the_cache_serves_an_embedding_without_the_network(fakes, tmp_path):
    items = golden_data.encoder_items()["text"]
    cache = tmp_path / "embeddings"
    rows = lazy.text.Pretrained(device="cpu", cache=cache).fit_transform(items)
    lazy.clear_model_cache()
    again = lazy.text.Pretrained(device="cpu", cache=cache).fit_transform(items)
    np.testing.assert_array_equal(again, rows)
    assert fakes["text:e5-small-v2"] == 1


def test_the_cache_is_keyed_by_version_and_weights_not_device():
    pytest.importorskip("transformers")
    identity = json.loads(lazy.text.Pretrained()._cache_identity())
    checkpoint = lazy.get_checkpoint("text")
    assert identity[:5] == [
        lazy.__version__,
        metadata.version("transformers"),
        metadata.version("torch"),
        checkpoint.repo_id,
        checkpoint.revision,
    ]
    cpu = lazy.text.Pretrained(device="cpu")
    assert cpu._cache_identity() == lazy.text.Pretrained()._cache_identity()
    other = lazy.text.Pretrained(model="bge-small-en-v1.5")
    assert other._cache_identity() != cpu._cache_identity()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("e5-small-v2", "mean"),
        ("bge-small-en-v1.5", "first"),
        ("all-MiniLM-L6-v2", "mean"),
        ("Qwen3-Embedding-0.6B", "last"),
    ],
)
def test_text_is_pooled_as_its_model_says_and_scaled_to_unit_length(
    fakes, model, expected
):
    texts = ["a lazy model", "x " * 300]
    encoder = _encoder(f"text:{model}")
    rows = encoder.fit_transform(texts)
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1, rtol=1e-6)
    prefix = [6] if model == "e5-small-v2" else []  # "query:" is a word.
    codes = np.array([101, *prefix, 1, 4, 5, 102])
    states = np.cos(codes[:, None] * np.arange(1, encoder._width() + 1))
    # The first text is padded: its last token is not the batch's last.
    pooled = {
        "mean": states.mean(axis=0),
        "first": states[0],
        "last": states[-1],
    }[expected]
    np.testing.assert_allclose(
        rows[0], pooled / np.linalg.norm(pooled), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    ("model", "patch"),
    [("chronos-t5-tiny", 1), ("chronos-bolt-small", 16), ("chronos-2", 16)],
)
def test_chronos_pools_the_observed_tokens_of_the_last_values(
    fakes, model, patch
):
    encoder = _encoder(f"time:{model}")
    context = encoder._recipe().context
    long = np.arange(10_000.0)
    gappy = [1.0, np.nan, 3.0, np.inf, 5.0]
    rows = encoder.fit_transform([long, long[-context:], gappy])
    np.testing.assert_allclose(rows[0], rows[1], rtol=1e-6)
    np.testing.assert_allclose(rows[2, -2:], [3, np.sqrt(8 / 3)])
    # The fake's token state is its mean value times a ramp to 1.
    tokens = [1.0, 3.0, 5.0] if patch == 1 else [3.0]
    expected = np.mean([*tokens, 3.0])
    assert rows[2, -3] == pytest.approx(expected)
    names = encoder.get_feature_names_out()
    assert list(names[-2:]) == ["time_pretrained_mean", "time_pretrained_scale"]
    with pytest.raises(ValueError, match="no finite value"):
        encoder.transform([[np.nan, np.inf]])


@pytest.mark.parametrize("model", ["timesfm-2.0-500m", "timesfm-2.5-200m"])
def test_timesfm_pools_the_observed_patches_of_the_filled_series(fakes, model):
    encoder = _encoder(f"time:{model}")
    context = encoder._recipe().context
    long = np.arange(20_000.0)
    # Three patches of 32 values, the first padded on the left. The second
    # is all missing: filled in, read, and left out of the pooling.
    gappy = np.arange(70.0) ** 2
    gappy[6:38] = np.nan
    gappy[50] = np.inf
    rows = encoder.fit_transform([long, long[-context:], gappy])
    np.testing.assert_allclose(rows[0], rows[1], rtol=1e-6)
    known = np.isfinite(gappy)
    np.testing.assert_allclose(
        rows[2, -2:], [gappy[known].mean(), gappy[known].std()]
    )
    filled = np.interp(np.arange(70), np.flatnonzero(known), gappy[known])
    read = (filled - filled.mean()) / filled.std()
    # The fake's token state is the mean value it read times a ramp to 1,
    # plus the frequency indicator: 0, high frequency, for TimesFM 2.0.
    expected = np.mean([read[:6].mean(), read[38:].mean()])
    assert rows[2, -3] == pytest.approx(expected, rel=1e-5)


def test_timesfm_runs_on_transformers_without_chronos(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setitem(sys.modules, "chronos", None)
    timesfm = lazy.time.Pretrained(model="timesfm-2.5-200m", device="cpu")
    timesfm.fit(golden_data.encoder_items()["time"])
    identity = json.loads(timesfm._cache_identity())
    assert identity[1:3] == [
        metadata.version("transformers"),
        metadata.version("torch"),
    ]


def test_trailing_padding_of_a_series_is_dropped_before_its_window(fakes):
    encoder = _encoder("time:chronos-t5-tiny")
    series = np.arange(300.0)
    padded = np.concatenate([[np.nan] * 5, series, [np.nan] * 2000])
    rows = encoder.fit_transform([series, padded])
    np.testing.assert_allclose(rows[1], rows[0])


def test_a_subclass_loads_its_own_network(fakes):
    class _Again(lazy.text.Pretrained):
        """Loads the same weights, which must not come from the base's."""

    items = golden_data.encoder_items()["text"]
    lazy.text.Pretrained(device="cpu", cache=False).fit_transform(items)
    _Again(device="cpu", cache=False).fit_transform(items)
    assert fakes["text:e5-small-v2"] == 2


def test_an_rgba_array_is_read_as_its_rgb(fakes):
    rgba = np.random.default_rng(1).integers(0, 256, (6, 5, 4), dtype=np.uint8)
    encoder = lazy.image.Pretrained(device="cpu", cache=False)
    rows = encoder.fit_transform([rgba, rgba[:, :, :3]])
    np.testing.assert_array_equal(rows[0], rows[1])


def test_an_image_file_pillow_image_and_array_give_the_same_row(
    fakes, tmp_path
):
    rgb = np.random.default_rng(0).integers(0, 256, (6, 5, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(rgb).save(path)
    encoder = lazy.image.Pretrained(device="cpu", cache=False)
    with Image.open(path) as image:
        rows = encoder.fit_transform([rgb, path, str(path), image])
    np.testing.assert_allclose(rows, rows[:1].repeat(4, axis=0))


@pytest.fixture(name="corner")
def _corner_processor(fakes, monkeypatch):
    """The fake image processor, reading each image's top-left pixel only."""
    import torch  # noqa: PLC0415 - the backends job's extra.

    def corner(self, images, return_tensors):
        pixels = [np.asarray(image, float)[0, 0] for image in images]
        return {"pixel_values": torch.tensor(np.array(pixels))}

    monkeypatch.setattr(fake_encoders.ImageProcessor, "__call__", corner)


def test_each_view_is_embedded_and_the_views_averaged(corner):
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (5, 4, 3), dtype=np.uint8)
    plain = lazy.image.Pretrained(device="cpu", cache=False).fit([image])
    turned = lazy.image.Pretrained(
        views=["rotate_90"], device="cpu", cache=False
    )
    np.testing.assert_array_equal(
        turned.fit_transform([image]), plain.transform([np.rot90(image)])
    )
    d4 = lazy.image.Pretrained(views="d4", device="cpu", cache=False)
    views = [
        image
        if view is None
        else np.asarray(Image.fromarray(image).transpose(view))
        for view in lazy.image.VIEWS.values()
    ]
    expected = plain.transform(views).mean(axis=0)
    # The network gives float32, of values up to 1 (spacing 6e-8), and the
    # views are summed in another order.
    np.testing.assert_allclose(
        d4.fit_transform([image])[0], expected, rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize(
    ("views", "message"),
    [
        ("rotate_90", "views must be"),
        ([], "views must be"),
        (["identity", "identity"], "views must be"),
        (90, "views must be"),
        ([["rotate_90"]], "views must be"),
        (["spin"], "unknown view 'spin'"),
    ],
)
def test_views_that_name_no_distinct_views_are_refused(views, message):
    encoder = lazy.image.Pretrained(views=views, device="cpu", cache=False)
    with pytest.raises(ValueError, match=message):
        encoder.fit([np.zeros((2, 2), dtype=np.uint8)])


def test_zoobot_reads_the_magnified_centre_in_the_unit_interval():
    pytest.importorskip("torch")
    # A grey square of 1/1.2 of the side, inside a red frame.
    image = np.zeros((96, 120, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    image[8:88, 20:100] = 51
    pixels = lazy.image._centred([Image.fromarray(image)], 224, 1.2).numpy()
    assert pixels.shape == (1, 3, 224, 224)
    np.testing.assert_allclose(pixels[..., 2:-2, 2:-2], 0.2, rtol=1e-6)
    # The square ends at the frame: only its outermost pixels, which the
    # bilinear filter reads across the edge, show the red.
    red = pixels[0, 0]
    for edge in (red[0], red[-1], red[:, 0], red[:, -1]):
        assert (edge > 0.25).all()


def test_an_image_that_is_not_8_bit_is_stretched_band_by_band():
    ramp = np.linspace(-1.0, 3.0, 101)
    image = np.stack([ramp, 10 * ramp, np.full(101, 5.0)], axis=-1)[None]
    stretched = np.asarray(lazy.image._rgb(image))[0]
    low, high = np.percentile(ramp, [1, 99])
    x = (ramp - low) / (high - low)
    expected = np.rint(np.clip(np.arcsinh(20 * x) / np.arcsinh(20), 0, 1) * 255)
    np.testing.assert_array_equal(stretched[:, 0], expected)
    np.testing.assert_array_equal(stretched[:, 1], expected)  # scale-free
    np.testing.assert_array_equal(stretched[:, 2], 0)  # no contrast
    grey = np.full((2, 2), 7, dtype=np.uint8)
    np.testing.assert_array_equal(np.asarray(lazy.image._rgb(grey)), 7)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((4, 4, 5)), r"or \(height, width, 3\)"),
        (np.full((4, 4), np.nan), "all of them finite"),
    ],
)
def test_what_a_network_cannot_read_as_rgb_is_refused(array, message):
    with pytest.raises(ValueError, match=message):
        lazy.image._rgb(array)


@pytest.mark.parametrize("name", sorted(PRETRAINED))
def test_each_modality_defaults_to_its_pretrained_encoder(name):
    cls = PRETRAINED[name]
    default = encoders.get_encoder(cls.modality)
    assert type(default) is cls
    assert default.cache is True


@pytest.mark.parametrize("name", sorted(PRETRAINED))
def test_an_unknown_model_is_refused_by_name(name):
    cls = PRETRAINED[name]
    with pytest.raises(lazy.UnknownNameError, match="unknown model 'large'"):
        cls(model="large").fit(golden_data.encoder_items()[cls.modality])


@pytest.mark.parametrize("name", sorted(PRETRAINED))
def test_a_missing_extra_is_named(monkeypatch, name):
    cls = PRETRAINED[name]
    monkeypatch.setitem(sys.modules, cls.package, None)
    with pytest.raises(ImportError, match=r"lazy-tfm\[encoders\]"):
        cls().fit(golden_data.encoder_items()[cls.modality])
    with pytest.raises(ImportError, match=rf"lazy\.{cls.modality}\.Pretrained"):
        cls().fit(golden_data.encoder_items()[cls.modality])


def test_a_missing_package_of_one_network_is_named_at_fit(monkeypatch):
    """Zoobot runs on timm as well as transformers."""
    monkeypatch.setitem(sys.modules, "timm", None)
    zoobot = lazy.image.Pretrained(model="zoobot-encoder-convnext_nano")
    with pytest.raises(ImportError, match=r"lazy-tfm\[encoders\]"):
        zoobot.fit(golden_data.encoder_items()["image"])


def test_zoobot_is_cached_by_the_versions_of_transformers_and_timm():
    pytest.importorskip("timm")
    zoobot = lazy.image.Pretrained(model="zoobot-encoder-convnext_nano")
    identity = json.loads(zoobot._cache_identity())
    assert identity[1:4] == [
        metadata.version("transformers"),
        metadata.version("timm"),
        metadata.version("torch"),
    ]


class _Unpinned(lazy.text.Pretrained):
    """Breaks the contract: a model with no pinned weights, and no device."""

    package = ""
    recipes = {**lazy.text.Pretrained.recipes, "tiny": None}

    def __init__(self, *, model="tiny", batch_size=64, cache=True):
        super().__init__(model=model, batch_size=batch_size, cache=cache)


def test_a_pretrained_encoder_needs_a_package_device_and_pinned_models(
    monkeypatch,
):
    monkeypatch.setattr(encoders.registry, "ENCODERS", {})
    with pytest.raises(TypeError, match="cannot be registered") as caught:
        encoders.register("unpinned", _Unpinned)
    for problem in (
        "package, which loads the weights, is not declared",
        "are not the models pinned for 'text'",
        "lacks the uniform parameter 'device'",
        "'model' defaults to 'tiny', not 'e5-small-v2'",
    ):
        assert problem in str(caught.value)


@pytest.mark.checkpoint
@pytest.mark.parametrize("key", MODELS)
def test_the_real_weights_give_the_golden_embeddings(key):
    with (
        np.load(golden_data.ENCODERS_PATH) as golden,
        golden_data.pinned_threads(),
    ):
        np.testing.assert_allclose(
            golden_data.embed(key),
            golden[key.replace(":", "__")],
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.checkpoint
@pytest.mark.parametrize("model", ["timesfm-2.0-500m", "timesfm-2.5-200m"])
def test_a_timesfm_row_does_not_depend_on_a_longer_series_in_its_batch(model):
    """TimesFM pads a batch to its longest series, which it masks."""
    generator = np.random.default_rng(0)
    short = generator.normal(size=70).cumsum()
    long = generator.normal(size=1500).cumsum()
    encoder = lazy.time.Pretrained(model=model, device="cpu", cache=False)
    with golden_data.pinned_threads():
        alone = encoder.fit_transform([short])
        together = encoder.transform([short, long])[:1]
    # Measured: the rows differ by float32 rounding, at most 4.1e-5.
    np.testing.assert_allclose(together, alone, rtol=1e-4, atol=1e-4)
