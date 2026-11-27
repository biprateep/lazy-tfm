# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The augmentations: one transform of every row or item, seeded."""

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from sklearn import base as sklearn_base
from sklearn import exceptions

import lazy
from lazy import _steps
from lazy import augment
from lazy import image as image_lib


def _images(n=4, shape=(8, 6, 3), seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(size=shape) for _ in range(n)]


NUMERIC = [
    augment.Rescale("asinh", scale=0.5),
    augment.Rescale("identity", scale=4.0),
    augment.Dihedral("rotate_90"),
    augment.Dihedral("transverse"),
    augment.Dihedral(random_state=5),
    augment.Crop(0.5, random_state=2),
    augment.Mask(0.4, patch=2, random_state=1),
]


@pytest.mark.parametrize("step", NUMERIC, ids=repr)
def test_a_forward_gives_what_transform_gives_and_declares_its_kind(step):
    torch = pytest.importorskip("torch")
    images = _images()
    transformed = step.fit(images).transform(images)
    tensor = torch.tensor(np.stack(images), requires_grad=True)
    forward = _steps.forward(step, tensor)
    np.testing.assert_allclose(forward.detach().numpy(), np.stack(transformed))
    forward.sum().backward()
    if _steps.gradient_kind(step) == "mask":
        # A masked pixel gets no gradient, and the others get all of it.
        assert set(tensor.grad.unique().tolist()) == {0.0, 1.0}
    assert _steps.gradient_kind(step) in ("differentiable", "index_map", "mask")


@pytest.mark.parametrize(
    "step",
    [*NUMERIC, augment.Noise(0.2, relative=True, random_state=3)],
    ids=repr,
)
def test_a_rows_augmentation_does_not_depend_on_the_rows_with_it(step):
    images = _images()
    together = step.fit(images).transform(images)
    alone = [step.transform([image])[0] for image in images]
    np.testing.assert_array_equal(np.stack(together), np.stack(alone))


@pytest.mark.parametrize("view", augment.VIEWS)
def test_each_view_is_pillows_transpose_of_that_name(view):
    image = (np.arange(24).reshape(4, 6) % 251).astype(np.uint8)
    (turned,) = augment.Dihedral(view).fit([image]).transform([image])
    if view == "identity":
        expected = image
    else:
        method = getattr(Image.Transpose, view.upper())
        expected = np.asarray(Image.fromarray(image).transpose(method))
    np.testing.assert_array_equal(turned, expected)


def test_the_seed_sets_a_random_augmentation():
    images = _images()
    views = {augment.Dihedral(random_state=seed)._view() for seed in range(40)}
    assert views == set(augment.VIEWS)
    for make in (
        augment.Noise,
        lambda random_state: augment.Crop(0.5, random_state=random_state),
        lambda random_state: augment.Mask(
            0.5, patch=1, random_state=random_state
        ),
    ):
        first, again, other = (
            np.stack(make(random_state=seed).fit(images).transform(images))
            for seed in (0, 0, 1)
        )
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)


def test_a_crop_keeps_a_window_and_a_mask_zeroes_whole_patches():
    images = _images(shape=(8, 8, 1))
    (cropped,) = augment.Crop(0.5).fit(images).transform(images[:1])
    assert cropped.shape == (4, 4, 1)
    series = [np.arange(10.0)]
    (window,) = augment.Crop(0.3, random_state=1).fit(series).transform(series)
    assert window.size == 3 and np.all(np.diff(window) == 1)
    (masked,) = augment.Mask(0.5, patch=4).fit(images).transform(images[:1])
    blocks = masked.reshape(2, 4, 2, 4).transpose(0, 2, 1, 3).reshape(4, -1)
    # Each of the four patches is all zero or untouched.
    assert all((block == 0).all() or (block != 0).all() for block in blocks)


def test_moving_pixels_keeps_an_images_type_and_reads_a_file(tmp_path):
    image = (np.arange(48).reshape(4, 4, 3) * 5).astype(np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(image).save(path)
    for item in (image, Image.fromarray(image), path, str(path)):
        (turned,) = augment.Dihedral("rotate_90").fit([item]).transform([item])
        np.testing.assert_array_equal(turned, np.rot90(image))
        assert turned.dtype == np.uint8
    (masked,) = augment.Mask(0.5, patch=1).fit([image]).transform([image])
    assert masked.dtype == np.uint8
    (scaled,) = augment.Rescale().fit([image]).transform([image])
    assert scaled.dtype == np.float64


def test_a_table_keeps_its_frame_and_a_missing_item_stays_missing():
    table = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=[7, 9])
    out = augment.Rescale("log1p").fit(table).transform(table)
    pd.testing.assert_frame_equal(out, np.log1p(table))
    items = [np.ones((2, 2)), None]
    assert augment.Dihedral("rotate_180").fit(items).transform(items)[1] is None


def test_column_dropout_drops_columns_or_whole_prefixes():
    torch = pytest.importorskip("torch")
    names = ["cutout__0", "cutout__1", "notes__0", "notes__1", "price"]
    table = pd.DataFrame(np.arange(10.0).reshape(2, 5), columns=names)
    by_column = augment.ColumnDropout(0.4, random_state=1).fit(table)
    assert by_column.transform(table).shape == (2, 3)
    by_prefix = augment.ColumnDropout(0.34, by="prefix", random_state=0)
    kept = by_prefix.fit(table).get_feature_names_out()
    # One of the three units went, whole.
    assert {name.split("__")[0] for name in kept} in (
        {"cutout", "notes"},
        {"cutout", "price"},
        {"notes", "price"},
    )
    assert list(by_prefix.transform(table).columns) == list(kept)
    assert _steps.gradient_kind(by_prefix) == "index_map"
    forward = _steps.forward(by_prefix, torch.tensor(table.to_numpy()))
    np.testing.assert_array_equal(forward.numpy(), by_prefix.transform(table))
    # A fraction that would drop everything keeps one unit.
    assert augment.ColumnDropout(0.99).fit(table).kept_.size == 1
    with pytest.raises(ValueError, match="needs a table with column names"):
        augment.ColumnDropout(by="prefix").fit(table.to_numpy())


def test_text_augmentations_keep_words_in_order_and_are_barriers():
    texts = ["the quick brown fox jumps over the lazy dog", "one", None]
    cut = augment.Truncate(3).fit(texts).transform(texts)
    assert cut == ["the quick brown", "one", None]
    dropout = augment.TokenDropout(0.5, random_state=4).fit(texts)
    dropped = dropout.transform(texts)
    words = iter(texts[0].split())
    assert all(word in words for word in dropped[0].split())
    assert 0 < len(dropped[0].split()) < 9 and dropped[1] == "one"
    assert dropout.transform(texts[:1]) == dropped[:1]
    assert _steps.barrier("notes", dropout).reason.endswith("no gradient")


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (augment.Rescale("sqrt"), "function must be one of"),
        (augment.Rescale(scale=0.0), "scale == 0.0, must be > 0"),
        (augment.Noise(-1.0), "scale == -1.0, must be >= 0"),
        (augment.Noise(random_state=None), "random_state must be an instance"),
        (augment.Dihedral("spin"), "view must be None or one of"),
        (augment.Crop(0.0), "fraction == 0.0, must be > 0"),
        (augment.Mask(1.0), "fraction == 1.0, must be < 1"),
        (augment.Mask(patch=0), "patch must be a positive integer"),
        (augment.ColumnDropout(by="row"), "by must be 'column' or 'prefix'"),
        (augment.TokenDropout(1.5), "fraction == 1.5, must be < 1"),
        (augment.Truncate(0), "max_words must be a positive integer"),
    ],
)
def test_a_bad_parameter_is_refused_at_fit_by_name(step, message):
    with pytest.raises((ValueError, TypeError), match=message):
        step.fit(pd.DataFrame({"a": [1.0]}))
    with pytest.raises(exceptions.NotFittedError):
        sklearn_base.clone(step).transform([np.ones((2, 2))])


def test_a_tabularizer_augments_a_column_before_its_encoder():
    images = _images(n=12, shape=(4, 4, 1))
    table = pd.DataFrame({"cutout": images, "number": np.arange(12.0)})
    pixels = image_lib.Pixels(size=4, n_channels=1)
    step = augment.Dihedral("flip_left_right")
    tab = lazy.Tabularizer({"cutout": [step, pixels]}, n_columns=None)
    out = tab.fit_transform(table)
    flipped = pixels.fit_transform([image[:, ::-1] for image in images])
    np.testing.assert_array_equal(out.iloc[:, :-1], flipped)
    assert not hasattr(step, "fitted_")
    for spec, error in (
        ([], ValueError),
        (["flip", "image_pixels"], TypeError),
    ):
        with pytest.raises(error, match=r"columns\['cutout'\] must"):
            lazy.Tabularizer({"cutout": spec}).fit(table)


def test_a_tabularizers_seed_seeds_each_columns_augmentations():
    images = _images(n=6, shape=(4, 4, 1))
    table = pd.DataFrame({"a": images, "b": images})
    spec = [augment.Dihedral(), image_lib.Pixels(size=4, n_channels=1)]

    def views(random_state):
        tab = lazy.Tabularizer(
            {"a": spec, "b": spec}, n_columns=None, random_state=random_state
        ).fit(table)
        return [steps[0]._view() for steps in tab.augmentations_.values()]

    # Without a seed, each augmentation keeps its own; with one, each
    # column draws for itself.
    assert len(set(views(None))) == 1
    assert len({tuple(views(seed)) for seed in range(8)}) > 1
    assert views(3) == views(3)
    assert any(len(set(views(seed))) == 2 for seed in range(8))


@pytest.mark.parametrize("relative", [False, True])
def test_noise_keeps_its_draw_when_a_value_moves_in_its_last_bit(relative):
    torch = pytest.importorskip("torch")
    row = np.array([[0.1, 0.2, 0.3, -1.5]])
    noise = augment.Noise(0.5, relative=relative).fit(row)
    moved = np.nextafter(row, 1.0)
    np.testing.assert_allclose(
        noise.transform(moved) - moved, noise.transform(row) - row, rtol=1e-12
    )
    # The draw is held fixed, so the gradient is that of x + s * eps.
    point = torch.tensor(row, requires_grad=True)
    out = _steps.forward(noise, point)
    np.testing.assert_allclose(out.detach().numpy(), noise.transform(row))
    out.sum().backward()
    if relative:
        draws = (noise.transform(row) - row) / row.std()
        expected = 1 + draws.sum() * (row - row.mean()) / (row.size * row.std())
    else:
        expected = np.ones_like(row)
    np.testing.assert_allclose(point.grad.numpy(), expected)


def test_a_grey_image_is_masked_alike_with_and_without_its_channel_axis():
    mask = augment.Mask(0.5, patch=2, random_state=3)
    np.testing.assert_array_equal(
        mask._kept((8, 8)), mask._kept((8, 8, 1))[:, :, 0]
    )


def test_a_file_is_read_in_the_encoders_mode_before_it_is_augmented(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for index in range(4):
        pixels = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
        path = tmp_path / f"{index}.png"
        # A palette image: its values index colours, they are not levels.
        Image.fromarray(pixels).convert("P").save(path)
        paths.append(str(path))
    table = pd.DataFrame({"cutout": paths})
    encoder = image_lib.Statistics(n_channels=3)
    augmented = lazy.Tabularizer(
        {"cutout": [augment.Dihedral("identity"), encoder]}, n_columns=None
    ).fit_transform(table)
    np.testing.assert_allclose(augmented, encoder.fit_transform(paths))


def test_a_tables_bool_column_and_its_names_pass_an_augmentation():
    table = pd.DataFrame({"a": [1.0, 2.0], "flag": [True, False]})
    noise = augment.Noise(0.1).fit(table)
    out = noise.transform(table)
    assert out["flag"].tolist() == [True, False] and out["flag"].dtype == bool
    assert list(noise.get_feature_names_out()) == ["a", "flag"]
    with pytest.raises(ValueError, match="fitted on no table"):
        augment.Noise().fit([np.ones(3)]).get_feature_names_out()


def test_a_failed_refit_leaves_an_augmentation_unfitted():
    table = pd.DataFrame({"a__0": [1.0], "b__0": [2.0]})
    step = augment.ColumnDropout(0.5, by="prefix").fit(table)
    with pytest.raises(ValueError, match="needs a table with column names"):
        step.fit(table.to_numpy())
    with pytest.raises(exceptions.NotFittedError):
        step.transform(table)


def test_a_tabularizers_seed_leaves_a_seed_that_was_given():
    images = _images(n=4, shape=(4, 4, 1))
    table = pd.DataFrame({"a": images})
    pixels = image_lib.Pixels(size=4, n_channels=1)
    spec = [augment.Crop(0.5, random_state=5), augment.Dihedral(), pixels]
    tab = lazy.Tabularizer({"a": spec}, n_columns=None, random_state=9)
    crop, turn = tab.fit(table).augmentations_["a"]
    assert crop.random_state == 5 and turn.random_state != 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_column_dropout_drops_a_unit_of_two(seed):
    table = np.arange(8.0).reshape(2, 4)
    kept = augment.ColumnDropout(0.2, random_state=seed).fit(table[:, :2]).kept_
    assert len(kept) == 1
