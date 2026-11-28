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
import types

import fakes as backend_fakes
from fakes import encoders as fake_encoders
import golden_data
import numpy as np
import pandas as pd
from PIL import Image
import pytest

import lazy
from lazy import _steps as steps
from lazy import encoders
from lazy.encoders import _pretrained


def _needing_its_package(key):
    """``key`` as a test parameter, skipped without the package it needs.

    TiRex's package comes only with its own extra, which ``all`` leaves
    out.
    """
    modality, _, model = key.partition(":")
    encoder = encoders.get_encoder(modality).set_params(model=model)
    module = encoder._package().partition(".")[0]
    reason = f"{module}, which loads {key}, is not installed"
    skip = pytest.mark.skipif(
        not fake_encoders.installed(module), reason=reason
    )
    return pytest.param(key, marks=skip)


MODELS = [_needing_its_package(key) for key in sorted(encoders.CHECKPOINTS)]
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


@pytest.mark.parametrize(
    "key",
    [
        _needing_its_package(f"time:{model}")
        for model in ("timesfm-2.0-500m", "timesfm-2.5-200m", "timesfm-3.0")
    ],
)
def test_timesfm_pools_the_observed_patches_of_the_filled_series(fakes, key):
    encoder = _encoder(key)
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


def test_tirex_normalizes_each_layers_mean_over_the_observed_patches(fakes):
    pytest.importorskip("tirex")
    encoder = _encoder("time:tirex")
    context = encoder._recipe().context
    long = np.arange(5_000.0)
    # Three patches of 32 values, the first padded on the left. The second
    # is all missing, and left out of the pooling.
    gappy = np.arange(70.0) ** 2
    gappy[6:38] = np.nan
    gappy[50] = np.inf
    rows = encoder.fit_transform([long, long[-context:], gappy])
    np.testing.assert_allclose(rows[0], rows[1], rtol=1e-6)
    known = np.isfinite(gappy)
    np.testing.assert_allclose(
        rows[2, -2:], [gappy[known].mean(), gappy[known].std()]
    )
    # The tokenizer scales the window by its own mean and deviation.
    observed = np.where(known, gappy, np.nan)
    read = (observed - gappy[known].mean()) / gappy[known].std()
    means = np.mean([np.nanmean(read[:6]), np.nanmean(read[38:])])
    # The fake's state of a patch in layer l is a ramp plus (l + 1) times
    # the mean value it read.
    layers = np.linspace(0, 1, 512) + means * np.arange(1, 13)[:, None]
    layers /= np.linalg.norm(layers, axis=1, keepdims=True)
    # A layer norm, whose epsilon is 1e-5.
    expected = (layers - layers.mean()) / np.sqrt(layers.var() + 1e-5)
    np.testing.assert_allclose(rows[2, :-2], expected.ravel(), atol=1e-6)


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


def test_an_augmentation_shows_the_network_another_view(fakes):
    rng = np.random.default_rng(2)
    images = [rng.integers(0, 256, (5, 4, 3), dtype=np.uint8) for _ in range(3)]
    table = pd.DataFrame({"cutout": images})
    encoder = lazy.image.Pretrained(device="cpu", cache=False)
    turned = lazy.Tabularizer(
        {"cutout": [lazy.augment.Dihedral("rotate_90"), encoder]},
        n_columns=None,
    ).fit_transform(table)
    plain = encoder.fit_transform([np.rot90(image) for image in images])
    np.testing.assert_allclose(turned.to_numpy(), plain)
    assert not np.allclose(plain, encoder.transform(images))


def test_zoobot_reads_the_magnified_centre_in_the_unit_interval():
    torch = pytest.importorskip("torch")
    # A grey square of 1/1.2 of the side, inside a red frame.
    image = np.zeros((96, 120, 3))
    image[:, :, 0] = 255
    image[8:88, 20:100] = 51
    recipe = lazy.image.Pretrained.recipes["zoobot-encoder-convnext_nano"]
    pixels = recipe.pixels(torch.from_numpy(image)[None])[0].numpy()
    assert pixels.shape == (3, 224, 224)
    np.testing.assert_allclose(pixels[..., 2:-2, 2:-2], 0.2, rtol=1e-12)
    # The square ends at the frame: only its outermost pixels, which the
    # bilinear filter reads across the edge, show the red.
    red = pixels[0]
    for edge in (red[0], red[-1], red[:, 0], red[:, -1]):
        assert (edge > 0.25).all()


def _dense_resampling(n_in, span, n_out, resample):
    """The weights of :func:`lazy.image._resampling`, shape (n_out, n_in)."""
    weights = np.zeros((n_out, n_in))
    indices, values = lazy.image._resampling(n_in, *span, n_out, resample)
    weights[tuple(indices)] = values
    return weights


@pytest.mark.parametrize("model", sorted(lazy.image.Pretrained.recipes))
@pytest.mark.parametrize("shape", [(37, 90, 3), (300, 41)])
def test_the_pixels_are_the_resampled_image_clipped_and_normalized(
    model, shape
):
    torch = pytest.importorskip("torch")
    recipe = lazy.image.Pretrained.recipes[model]
    images = np.random.default_rng(8).uniform(0, 255, (2, *shape))
    rgb = images if images.ndim == 4 else np.stack([images] * 3, axis=-1)
    height, width = shape[:2]
    rows, columns = (
        _dense_resampling(
            n, recipe.span(n, min(height, width)), 224, recipe.resample
        )
        for n in (height, width)
    )
    across = np.clip(rgb.transpose(0, 3, 1, 2) @ columns.T, 0, 255)
    square = np.clip(rows @ across, 0, 255) / 255
    mean, std = (np.array(v)[:, None, None] for v in (recipe.mean, recipe.std))
    np.testing.assert_allclose(
        recipe.pixels(torch.from_numpy(images)).numpy(),
        (square - mean) / std,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("resample", ["bilinear", "bicubic"])
@pytest.mark.parametrize(
    "box", [(0, 0, 40, 48), (6.5, 9.625, 33.25, 41.5), (0, 12.5, 40, 30)]
)
def test_the_resampling_weights_are_pillows(resample, box):
    """Pillow's float images keep its weights unrounded.

    The spans are exact in float32, in which Pillow takes them; the rest
    differs by Pillow's float32 storage between its two passes.
    """
    filters = {"bilinear": Image.Resampling.BILINEAR}
    filters["bicubic"] = Image.Resampling.BICUBIC
    image = np.random.default_rng(5).uniform(0, 255, (48, 40))
    expected = Image.fromarray(image.astype(np.float32), "F").resize(
        (224, 200), filters[resample], box=box
    )
    rows = _dense_resampling(48, box[1::2], 200, resample)
    columns = _dense_resampling(40, box[::2], 224, resample)
    np.testing.assert_allclose(
        rows @ image @ columns.T, np.asarray(expected), rtol=0, atol=1e-4
    )


def test_an_image_that_is_not_8_bit_is_stretched_band_by_band():
    ramp = np.linspace(-1.0, 3.0, 101)
    image = np.stack([ramp, 10 * ramp, np.full(101, 5.0)], axis=-1)[None]
    stretched = lazy.image._image(image)[0]
    low, high = np.percentile(ramp, [1, 99])
    x = (ramp - low) / (high - low)
    expected = np.clip(np.arcsinh(20 * x) / np.arcsinh(20), 0, 1) * 255
    np.testing.assert_allclose(stretched[:, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(stretched[:, 1], expected, rtol=1e-12)
    np.testing.assert_array_equal(stretched[:, 2], 0)  # no contrast
    grey = np.full((2, 2), 7, dtype=np.uint8)
    np.testing.assert_array_equal(lazy.image._image(grey), 7)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((4, 4, 5)), r"or \(height, width, 3\)"),
        (np.full((4, 4), np.nan), "all of them finite"),
    ],
)
def test_what_a_network_cannot_read_as_rgb_is_refused(array, message):
    with pytest.raises(ValueError, match=message):
        lazy.image._image(array)


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


def test_an_image_is_cached_by_the_version_of_its_preprocessing():
    pytest.importorskip("transformers")
    identity = json.loads(lazy.image.Pretrained()._cache_identity())
    assert f"preprocessing {lazy.image._PREPROCESSING}" in identity


def test_tirex_without_its_package_names_its_own_extra(monkeypatch):
    """tirex-ts is not open source, so the encoders extra leaves it out."""
    monkeypatch.setitem(sys.modules, "tirex", None)
    tirex = lazy.time.Pretrained(model="tirex")
    with pytest.raises(ImportError, match=r"lazy-tfm\[tirex\]"):
        tirex.fit(golden_data.encoder_items()["time"])


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
@pytest.mark.parametrize(
    "key",
    [
        _needing_its_package(f"time:{model}")
        for model in (
            "timesfm-2.0-500m",
            "timesfm-2.5-200m",
            "timesfm-3.0",
            "tirex",
        )
    ],
)
def test_a_row_does_not_depend_on_a_longer_series_in_its_batch(key):
    """A batch pads each series to its longest, or to the context."""
    model = key.partition(":")[2]
    generator = np.random.default_rng(0)
    short = generator.normal(size=70).cumsum()
    long = generator.normal(size=1500).cumsum()
    encoder = lazy.time.Pretrained(model=model, device="cpu", cache=False)
    with golden_data.pinned_threads():
        alone = encoder.fit_transform([short])
        together = encoder.transform([short, long])[:1]
    # Measured: the rows differ by float32 rounding, at most 4.1e-5.
    np.testing.assert_allclose(together, alone, rtol=1e-4, atol=1e-4)


@pytest.mark.checkpoint
def test_tirex_pools_as_its_authors_embed_a_series():
    """TiRexEmbedding, in float32, is tirex-ts's own embedding."""
    torch = pytest.importorskip("torch")
    embedding = pytest.importorskip("tirex.models.embedding")
    encoder = lazy.time.Pretrained(model="tirex", device="cpu", cache=False)
    path = encoder._checkpoint().download()
    network = encoder._recipe().load(path, "cpu").float()
    reference = embedding.TiRexEmbedding.__new__(embedding.TiRexEmbedding)
    torch.nn.Module.__init__(reference)
    reference.model = network
    reference.batch_size = 512
    reference.data_augmentation = False
    series = golden_data.encoder_items()["time"][0]
    with torch.inference_mode():
        ours = encoder._recipe().pooled(network, [torch.tensor(series)])
        theirs = reference(
            torch.tensor(series, dtype=torch.float32)[None, None]
        )
    # Measured: identical in float32.
    np.testing.assert_allclose(
        ours.numpy(), theirs.numpy(), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    ("model", "side"),
    [("e5-small-v2", "right"), ("Qwen3-Embedding-0.6B", "left")],
)
def test_long_texts_are_split_into_passes_under_the_token_budget(
    fakes, monkeypatch, model, side
):
    pytest.importorskip("torch")
    monkeypatch.setattr(fake_encoders.Tokenizer, "padding_side", side)
    rng = np.random.default_rng(0)
    sizes = [1, 40, 3, 2, 38, 5, 4, 1]
    texts = [" ".join(rng.choice(["ab", "abc", "a"], n)) for n in sizes]
    shapes = []
    call = fake_encoders.TextModel.__call__

    def recording(self, input_ids, attention_mask):
        shapes.append(tuple(input_ids.shape))
        return call(self, input_ids, attention_mask)

    monkeypatch.setattr(fake_encoders.TextModel, "__call__", recording)
    settings = {"model": model, "device": "cpu", "cache": False}
    whole = lazy.text.Pretrained(**settings).fit_transform(texts)
    # A first and a last token beside 40 words, and one for e5's prefix.
    assert shapes == [(8, 43 if model == "e5-small-v2" else 42)]
    shapes.clear()
    split = lazy.text.Pretrained(max_batch_tokens=50, **settings).fit_transform(
        texts
    )
    # Each pass fits the budget, padded; the six short texts share one.
    assert all(n * length <= 50 for n, length in shapes) and len(shapes) == 3
    assert sorted(n for n, _ in shapes) == [1, 1, 6]
    # A padded state is 1e6, so pooling that read other padding would show.
    np.testing.assert_allclose(split, whole, rtol=1e-6)
    # A text longer than the budget is a pass of its own.
    shapes.clear()
    alone = lazy.text.Pretrained(max_batch_tokens=10, **settings).fit_transform(
        texts
    )
    assert all(n == 1 or n * length <= 10 for n, length in shapes)
    assert whole.dtype == alone.dtype == np.float64
    np.testing.assert_allclose(alone, whole, rtol=1e-6)
    with pytest.raises(ValueError, match="max_batch_tokens must be a posit"):
        lazy.text.Pretrained(max_batch_tokens=0, device="cpu").fit(texts)


# -- the forwards, on torch's autograd graph ----------------------------------

DIFFERENTIABLE = [
    _needing_its_package(key)
    for key in sorted(encoders.CHECKPOINTS)
    if key.startswith("image:")
    or (
        key.startswith("time:")
        and not lazy.time.Pretrained.recipes[key.partition(":")[2]].quantized
    )
]
"""Each network whose embedding has a forward, as ``"<modality>:<model>"``."""

DIFFERENTIABLE_SERIES = [
    key for key in DIFFERENTIABLE if key.values[0].startswith("time:")
]
"""The networks of :data:`DIFFERENTIABLE` that read series."""


def _forward_and_transform(encoder, items):
    """An encoder's forward and transform of the same items.

    Returns:
        A tuple (forward, transform, gradient): the forward, a numpy array;
        the transform; and the gradient of a weighted sum of the forward
        in the items, shape (n_items, ...).
    """
    torch = pytest.importorskip("torch")
    stacked = encoder.fit(items)._stacked(items)
    tensor = torch.tensor(stacked, requires_grad=True)
    embedding = encoder.forward(tensor)
    weights = torch.tensor(
        np.random.default_rng(1).normal(size=embedding.shape[1])
    )
    (embedding * weights).sum().backward()
    return embedding.detach().numpy(), encoder.transform(items), tensor.grad


def _items(key):
    """Items of one shape for the modality of ``key``.

    Series: four of one length, one constant and one constant at first; the
    fourth opens with 40 equal values, more than a patch of TimesFM or
    TiRex, which normalize by the values up to each patch. Images: three
    8-bit RGB, the last of them with each channel constant.
    """
    generator = np.random.default_rng(2)
    if key.startswith("image:"):
        images = generator.integers(0, 256, (3, 40, 52, 3), dtype=np.uint8)
        images[2] = [[[10, 200, 90]]]
        return list(images)
    t = np.arange(70)
    return [
        np.sin(2 * np.pi * t / 25) + 0.1 * generator.normal(size=70),
        generator.normal(size=70).cumsum(),
        np.full(70, 2.5),
        np.append(np.full(40, -1.0), generator.normal(size=30).cumsum()),
    ]


def _check_gradient(gradient):
    """Asserts that each item's gradient is finite and not all zero."""
    assert gradient.isfinite().all()
    assert (gradient.reshape(len(gradient), -1).abs().sum(dim=1) > 0).all()


@pytest.mark.parametrize("key", DIFFERENTIABLE)
def test_a_forward_gives_what_transform_gives_and_has_gradients(fakes, key):
    forward, transform, gradient = _forward_and_transform(
        _encoder(key), _items(key)
    )
    np.testing.assert_allclose(forward, transform, rtol=1e-6, atol=1e-6)
    _check_gradient(gradient)


def test_an_image_forward_refuses_a_tensor_that_is_not_grey_or_rgb(fakes):
    torch = pytest.importorskip("torch")
    encoder = _encoder("image:dinov2-with-registers-small")
    with pytest.raises(steps.BarrierError, match="grey or RGB"):
        encoder.forward(torch.zeros((2, 8, 8, 4), dtype=torch.float64))


def _ragged():
    """Four series of other lengths, with gaps and trailing padding.

    The last is constant but for a gap.
    """
    generator = np.random.default_rng(6)
    gappy = generator.normal(size=70).cumsum()
    gappy[[10, 11, 40]] = np.nan
    trailing = np.append(generator.normal(size=50).cumsum(), [np.nan] * 2)
    steady = np.full(30, 1.0)
    steady[5] = np.inf
    return [gappy, generator.normal(size=45).cumsum(), trailing, steady]


@pytest.mark.parametrize("key", DIFFERENTIABLE_SERIES)
def test_a_forward_reads_padding_and_gaps_as_missing_values(fakes, key):
    encoder = _encoder(key)
    stacked = encoder.fit(_ragged())._stacked(_ragged())
    # Padded on the left to the longest, NaN where a value is missing.
    assert stacked.shape == (4, 70)
    assert np.isnan(stacked[1, :25]).all() and np.isnan(stacked[0, 10])
    forward, transform, gradient = _forward_and_transform(encoder, _ragged())
    np.testing.assert_allclose(forward, transform, rtol=1e-6, atol=1e-6)
    assert gradient.isfinite().all()
    missing = _tensor(np.isnan(stacked))
    assert (gradient[missing] == 0).all()
    assert (gradient.abs().sum(dim=1) > 0).all()


def _tensor(array):
    """``array`` as a torch tensor."""
    return pytest.importorskip("torch").from_numpy(array)


@pytest.mark.parametrize(
    "key", ["time:chronos-bolt-small", "image:dinov2-with-registers-small"]
)
def test_a_forward_runs_in_passes_of_at_most_batch_size_items(
    fakes, monkeypatch, key
):
    torch = pytest.importorskip("torch")
    encoder = _encoder(key, batch_size=3)
    passes = []
    embedded = type(encoder)._embedded

    def recorded(self, network, batch):
        passes.append(len(batch))
        return embedded(self, network, batch)

    monkeypatch.setattr(type(encoder), "_embedded", recorded)
    some = _items(key)
    items = [*some, *some, *some][:7]
    forward = encoder.fit(items).forward(torch.tensor(encoder._stacked(items)))
    assert passes == [3, 3, 1]
    np.testing.assert_allclose(forward.numpy(), encoder.transform(items))


def test_a_loaded_network_keeps_no_gradient_on_its_weights():
    torch = pytest.importorskip("torch")
    network = torch.nn.Linear(2, 1)
    pipeline = types.SimpleNamespace(model=torch.nn.Linear(2, 1))
    loaded = ("a preprocessor", network)
    assert _pretrained._frozen(loaded) is loaded
    assert _pretrained._frozen(pipeline) is pipeline
    for module in (network, pipeline.model):
        assert not any(p.requires_grad for p in module.parameters())


def test_chronos_t5_is_a_barrier_to_gradients(fakes):
    torch = pytest.importorskip("torch")
    series = _items("time:chronos-t5-tiny")
    encoder = _encoder("time:chronos-t5-tiny").fit(series)
    barrier = steps.barrier("chronos", encoder)
    assert "Chronos-T5 reads each value as one of a set of tokens" in (
        barrier.reason
    )
    with pytest.raises(steps.BarrierError, match="Chronos-T5"):
        encoder.forward(torch.tensor(encoder._stacked(series)))


def test_tirex_scales_its_windows_as_its_tokenizer_does():
    torch = pytest.importorskip("torch")
    tirex = pytest.importorskip("tirex")
    context = torch.tensor(np.random.default_rng(3).normal(size=(3, 64)))
    context[:, :20] = np.nan
    context[1, 40] = np.nan
    context[2, 20:] = 2.5
    recipe = lazy.time.Pretrained.recipes["tirex"]
    tokenizer = tirex.models.patcher.PatchedTokenizer(recipe.patch)
    expected, _ = tokenizer.input_transform(context)
    torch.testing.assert_close(
        recipe._scaled(context), expected, equal_nan=True, rtol=0, atol=0
    )


def test_a_gradient_reaches_each_value_of_a_series_through_its_network(
    monkeypatch, tmp_path
):
    generator = np.random.default_rng(4)
    curves = [generator.normal(size=40).cumsum() for _ in range(30)]
    X = pd.DataFrame({"curve": curves})
    y = np.array([curve[-1] for curve in curves])
    model = _series_pipeline(monkeypatch, tmp_path).fit(X[:24], y[:24])
    explanation = lazy.interp.explain(
        model, X[24:], y[24:], X[:24], y[:24], method="gradient"
    )
    assert explanation.feature_names == ("curve",)
    assert explanation.barriers == ()
    values = explanation.item_values["curve"]
    assert values.shape == (3, 6, 40)
    np.testing.assert_allclose(values.sum(axis=-1), explanation.values[..., 0])


def test_integrated_gradients_reach_series_of_other_lengths_with_gaps(
    monkeypatch, tmp_path
):
    generator = np.random.default_rng(7)
    curves = [
        generator.normal(size=generator.integers(20, 40)).cumsum()
        for _ in range(30)
    ]
    for curve in curves[::4]:
        curve[3] = np.nan
    # An explained series longer than every series of the context.
    curves[-1] = generator.normal(size=50).cumsum()
    X = pd.DataFrame({"curve": curves})
    y = np.array([np.nanmean(curve) for curve in curves])
    model = _series_pipeline(monkeypatch, tmp_path).fit(X[:24], y[:24])
    explanation = lazy.interp.explain(
        model,
        X[24:],
        y[24:],
        X[:24],
        y[:24],
        method="integrated_gradients",
        n_steps=128,
    )
    assert explanation.barriers == ()
    values = explanation.item_values["curve"]
    # Padded on the left to the longest series; padding and gaps get 0.
    assert values.shape == (3, 6, 50)
    for row, curve in enumerate(curves[24:]):
        own = values[:, row, 50 - len(curve) :]
        assert (values[:, row, : 50 - len(curve)] == 0).all()
        assert (own[:, np.isnan(curve)] == 0).all()
    np.testing.assert_allclose(values.sum(axis=-1), explanation.values[..., 0])
    # Completeness: a right Riemann sum of 128 steps is within 5% of the
    # largest gain.
    gained = explanation.full_values - explanation.base_values
    np.testing.assert_allclose(
        explanation.values.sum(axis=-1),
        gained,
        atol=0.05 * np.abs(gained).max(),
    )
    # The context's series too, with their padding and gaps.
    context = lazy.interp.explain_context(
        model, X[24:], y[24:], X[:24], y[:24], through_encoders=True
    )
    assert np.isfinite(context.values).all()


def test_padded_series_behind_an_augmentation_are_a_barrier(fakes):
    tab = lazy.Tabularizer(
        {
            "curve": [
                lazy.augment.Rescale(),
                _encoder("time:chronos-bolt-small"),
            ]
        },
        remainder="drop",
    )
    X = pd.DataFrame({"curve": _ragged()})
    tab.fit(X)
    _, through, barriers = tab._leaves(X, X, True, "tabularizer")
    assert through == [False]
    assert "its items are padded, or lack some elements" in barriers[0].reason


@pytest.mark.parametrize(
    ("augmentation", "flux"),
    [("Crop", True), ("Mask", True), ("Rescale", False)],
)
def test_images_stretched_behind_an_augmentation_are_a_barrier(
    fakes, augmentation, flux
):
    """The stretch, by percentiles, does not commute with the augmentation.

    A flux image is stretched; an 8-bit image is too once ``Rescale``
    makes it float.
    """
    generator = np.random.default_rng(9)
    images = generator.integers(0, 256, (4, 12, 10, 3), dtype=np.uint8)
    X = pd.DataFrame({"cutout": list(images / 7.0 if flux else images)})
    tab = lazy.Tabularizer(
        {
            "cutout": [
                getattr(lazy.augment, augmentation)(),
                _encoder("image:dinov2-with-registers-small"),
            ]
        },
        n_columns=None,
        remainder="drop",
    ).fit(X)
    _, through, barriers = tab._leaves(X, X, True, "tabularizer")
    assert through == [False]
    assert "such as a stretch" in barriers[0].reason


def test_8_bit_images_behind_a_crop_pass_the_gradient(fakes):
    images = np.random.default_rng(9).integers(0, 256, (4, 12, 10, 3))
    X = pd.DataFrame({"cutout": list(images.astype(np.uint8))})
    tab = lazy.Tabularizer(
        {
            "cutout": [
                lazy.augment.Crop(),
                _encoder("image:dinov2-with-registers-small"),
            ]
        },
        n_columns=None,
        remainder="drop",
    ).fit(X)
    _, through, barriers = tab._leaves(X, X, True, "tabularizer")
    assert (through, barriers) == ([True], [])


def test_chronos_normalizes_as_its_instance_norm_without_nan_gradients():
    torch = pytest.importorskip("torch")
    chronos_bolt = pytest.importorskip("chronos.chronos_bolt")
    context = torch.tensor(
        np.random.default_rng(8).normal(size=(4, 48)), dtype=torch.float32
    )
    context[:, :10] = np.nan
    context[1, 30] = np.nan
    context[2, 10:] = 2.5
    context[3] = np.nan
    for use_arcsinh in (False, True):
        upstream = chronos_bolt.InstanceNorm(use_arcsinh=use_arcsinh)
        masked = lazy.time._masked_norm(upstream)
        expected, (loc, scale) = upstream(context)
        leaf = context.clone().requires_grad_()
        found, (found_loc, found_scale) = masked(leaf)
        torch.testing.assert_close(
            found, expected, equal_nan=True, rtol=0, atol=0
        )
        torch.testing.assert_close(found_loc, loc, rtol=0, atol=0)
        torch.testing.assert_close(found_scale, scale, rtol=0, atol=0)
        torch.nan_to_num(found).sum().backward()
        assert leaf.grad.isfinite().all()
        assert (leaf.grad[context.isnan()] == 0).all()


def _series_pipeline(monkeypatch, tmp_path):
    """A Tabularizer of Chronos-Bolt's fake before TabPFN's fake, unfitted."""
    pytest.importorskip("torch")
    backend = backend_fakes.install("tabpfn", monkeypatch)(
        n_estimators=2,
        device="cpu",
        progress=False,
        transforms=("none",),
        outlier_threshold=None,
    )
    # After the backend's, whose download the encoders' replaces.
    fake_encoders.install(monkeypatch, tmp_path)
    tab = lazy.Tabularizer(
        {"curve": _encoder("time:chronos-bolt-small")},
        n_columns=8,
        remainder="drop",
    )
    return lazy.make_pipeline(tab, backend)


def test_a_gradient_reaches_each_pixel_of_an_image_through_its_network(
    monkeypatch, tmp_path
):
    pytest.importorskip("torch")
    generator = np.random.default_rng(6)
    images = generator.integers(0, 256, (30, 12, 10, 3), dtype=np.uint8)
    X = pd.DataFrame({"cutout": list(images)})
    y = images.mean(axis=(1, 2, 3)) / 255
    backend = backend_fakes.install("tabpfn", monkeypatch)(
        n_estimators=2,
        device="cpu",
        progress=False,
        transforms=("none",),
        outlier_threshold=None,
    )
    # After the backend's, whose download the encoders' replaces.
    fake_encoders.install(monkeypatch, tmp_path)
    tab = lazy.Tabularizer(
        {"cutout": _encoder("image:dinov2-with-registers-small")},
        n_columns=8,
        remainder="drop",
    )
    model = lazy.make_pipeline(tab, backend).fit(X[:24], y[:24])
    explanation = lazy.interp.explain(
        model, X[24:], y[24:], X[:24], y[:24], method="gradient"
    )
    assert explanation.feature_names == ("cutout",)
    assert explanation.barriers == ()
    values = explanation.item_values["cutout"]
    assert values.shape == (3, 6, 12, 10, 3)
    np.testing.assert_allclose(
        values.sum(axis=(-3, -2, -1)), explanation.values[..., 0]
    )


def _forward_cases():
    """The networks with a forward, each with the items to check it on."""
    cases = []
    for keys, items in (
        (DIFFERENTIABLE, "one shape"),
        (DIFFERENTIABLE_SERIES, "ragged"),
    ):
        for key in keys:
            name = key.values[0]
            cases.append(
                pytest.param(name, items, marks=key.marks, id=f"{name}-{items}")
            )
    return cases


@pytest.mark.checkpoint
@pytest.mark.parametrize(("key", "items"), _forward_cases())
def test_the_real_weights_give_a_forward_equal_to_transform(key, items):
    encoder = _encoder(key)
    items = _items(key) if items == "one shape" else _ragged()
    stacked = encoder.fit(items)._stacked(items)
    with golden_data.pinned_threads():
        forward, transform, gradient = _forward_and_transform(encoder, items)
    np.testing.assert_allclose(forward, transform, rtol=1e-5, atol=1e-5)
    assert (gradient[_tensor(np.isnan(stacked))] == 0).all()
    _check_gradient(gradient)


@pytest.mark.checkpoint
@pytest.mark.parametrize(
    "key",
    [
        key
        for key in DIFFERENTIABLE_SERIES
        if lazy.time.Pretrained.recipes[key.values[0][5:]].context <= 8192
    ],
)
def test_a_constant_window_of_a_varying_series_has_a_finite_gradient(key):
    """The network normalizes the window it reads, not the whole series."""
    context = lazy.time.Pretrained.recipes[key.partition(":")[2]].context
    walk = np.random.default_rng(5).normal(size=context + 30).cumsum()
    steady = np.append(walk[:30], np.full(context, 2.0))
    with golden_data.pinned_threads():
        forward, transform, gradient = _forward_and_transform(
            _encoder(key), [steady, walk]
        )
    np.testing.assert_allclose(forward, transform, rtol=1e-5, atol=1e-5)
    assert gradient.isfinite().all()


@pytest.mark.checkpoint
@pytest.mark.parametrize(
    "model",
    [
        model
        for model, recipe in lazy.image.Pretrained.recipes.items()
        if recipe.zoom is None
    ],
)
def test_images_are_preprocessed_as_the_pinned_processor_does(model):
    """It differs only by the processor's rounding to 8 bits.

    Pillow rounds each of its two passes to 8 bits, by at most half a
    level, and the bicubic kernel's weights of the second pass add up to
    at most 1.25 in absolute value: at most 1.125 levels in all, and
    float32 rounding.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path = encoders.CHECKPOINTS[f"image:{model}"].download()
    processor = transformers.AutoImageProcessor.from_pretrained(
        path, backend="pil"
    )
    recipe = lazy.image.Pretrained.recipes[model]
    generator = np.random.default_rng(7)
    for shape in [(48, 40, 3), (31, 97, 3), (300, 457, 3), (224, 224, 3)]:
        image = generator.integers(0, 256, shape, dtype=np.uint8)
        expected = processor(images=[image], return_tensors="pt")
        pixels = recipe.pixels(torch.from_numpy(image[None] / 1.0))[0]
        std = torch.tensor(recipe.std)[:, None, None]
        levels = (pixels - expected["pixel_values"][0]) * std * 255
        assert levels.abs().max() <= 1.13
