# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Attributions through pipelines and ensembles, and at any stage."""

import fakes
import numpy as np
import pandas as pd
import pytest
from sklearn import preprocessing
import standins

import lazy
from lazy import interp

torch = pytest.importorskip("torch")

N_CONTEXT, N_ROWS = 40, 9


@pytest.fixture(name="backend")
def _backend(monkeypatch):
    """A maker of TabPFN's fake, with a differentiable pass."""
    monkeypatch.setattr(interp, "_QUERY_BUDGET", 16)
    cls = fakes.install("tabpfn", monkeypatch)

    def make(**params):
        return cls(
            n_estimators=2,
            device="cpu",
            progress=False,
            transforms=("none",),
            outlier_threshold=None,
            **params,
        )

    return make


@pytest.fixture(name="numbers")
def _numbers():
    """Three skewed columns; the target follows the first."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        rng.lognormal(size=(N_CONTEXT + N_ROWS, 3)), columns=["a", "b", "c"]
    )
    y = np.log(X["a"].to_numpy()) + 0.1 * rng.normal(size=len(X))
    return X[:N_CONTEXT], y[:N_CONTEXT], X[N_CONTEXT:], y[N_CONTEXT:]


@pytest.fixture(name="items")
def _items():
    """A number, an image and a text per row; the image sets the target."""
    rng = np.random.default_rng(4)
    n = N_CONTEXT + N_ROWS
    cutouts = [rng.uniform(size=(4, 4, 1)) for _ in range(n)]
    words = np.array(["red", "green", "blue", "dark"])
    X = pd.DataFrame(
        {
            "number": rng.normal(size=n),
            "cutout": cutouts,
            "notes": [" ".join(rng.choice(words, 3)) for _ in range(n)],
        }
    )
    y = np.array([c.mean() for c in cutouts]) * 4 + 0.05 * rng.normal(size=n)
    return X[:N_CONTEXT], y[:N_CONTEXT], X[N_CONTEXT:], y[N_CONTEXT:]


@pytest.fixture(name="images")
def _images(items):
    """The number and the image of each row of ``items``."""
    X_context, y_context, X, y = items
    return (
        X_context.drop(columns="notes"),
        y_context,
        X.drop(columns="notes"),
        y,
    )


def _explain(model, data, **options):
    X_context, y_context, X, y = data
    return interp.explain(model, X, y, X_context, y_context, **options)


def _integrated(model, data):
    """Integrated gradients, checked to sum to what the features add.

    A right Riemann sum of 256 steps is within 2% of the largest gain.
    """
    explanation = _explain(
        model, data, method="integrated_gradients", n_steps=256
    )
    gained = explanation.full_values - explanation.base_values
    np.testing.assert_allclose(
        explanation.values.sum(axis=-1),
        gained,
        atol=0.02 * np.abs(gained).max(),
    )
    return explanation


def test_integrated_gradients_pass_a_pipelines_steps_to_the_raw_columns(
    backend, numbers
):
    model = lazy.make_pipeline(
        preprocessing.FunctionTransformer(np.log1p),
        preprocessing.StandardScaler(),
        backend(),
    ).fit(*numbers[:2])
    explanation = _integrated(model, numbers)
    assert explanation.feature_names == ("a", "b", "c")
    assert explanation.stage is None and explanation.barriers == ()
    assert explanation.importance()["nll"].idxmax() == "a"


def test_a_linear_steps_gradient_is_the_stages_gradient(backend, numbers):
    # Scaling is linear, so gradient times input is the same in the raw
    # columns and in the scaled ones.
    model = lazy.make_pipeline(preprocessing.StandardScaler(), backend())
    model.fit(*numbers[:2])
    raw = _explain(model, numbers, method="gradient")
    staged = _explain(model, numbers, method="gradient", at="standardscaler")
    assert staged.stage == "standardscaler"
    assert staged.feature_names == ("x0", "x1", "x2")
    np.testing.assert_allclose(raw.values, staged.values, atol=1e-9)
    np.testing.assert_allclose(raw.full_values, staged.full_values)


def test_a_barrier_ends_the_attribution_and_is_reported(backend, numbers):
    model = lazy.make_pipeline(
        preprocessing.MinMaxScaler(),
        preprocessing.FunctionTransformer(lambda X: X + 1.0),
        preprocessing.StandardScaler(),
        backend(),
    ).fit(*numbers[:2])
    with pytest.warns(lazy.LazyWarning, match="'functiontransformer', a barr"):
        explanation = _explain(model, numbers, method="gradient")
    assert explanation.stage == "functiontransformer"
    (barrier,) = explanation.barriers
    assert barrier.step == "functiontransformer"
    assert "np.log1p" in barrier.reason
    # A stage named after the barrier warns of nothing; one before it has
    # the barrier in its way.
    staged = _explain(model, numbers, method="gradient", at="standardscaler")
    assert staged.barriers == ()
    with pytest.raises(ValueError, match="does not cross the later step"):
        _explain(model, numbers, method="gradient", at="minmaxscaler")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"at": "scaler"}, r"one of \['standardscaler'\]: at='scaler'"),
        ({"at": "standardscaler", "baseline": [0.0]}, "of shape .3,."),
    ],
)
def test_a_bad_stage_or_baseline_is_refused_by_name(
    backend, numbers, options, message
):
    model = lazy.make_pipeline(preprocessing.StandardScaler(), backend())
    model.fit(*numbers[:2])
    with pytest.raises(ValueError, match=message):
        _explain(model, numbers, method="gradient", **options)
    with pytest.raises(ValueError, match="at names a step of a LazyPipeline"):
        _explain(model[-1], numbers, method="gradient", at="standardscaler")


def _tabularized(backend, **options):
    columns = {"cutout": lazy.image.Pixels(size=2, n_channels=1)}
    columns |= options.pop("more", {})
    return lazy.make_pipeline(lazy.Tabularizer(columns, **options), backend())


def test_gradients_reach_the_pixels_through_the_tabularizer(backend, images):
    model = _tabularized(backend, n_columns=3).fit(*images[:2])
    explanation = _integrated(model, images)
    assert explanation.feature_names == ("cutout", "number")
    pixels = explanation.item_values["cutout"]
    assert pixels.shape == (3, N_ROWS, 4, 4, 1)
    np.testing.assert_allclose(
        pixels.sum(axis=(2, 3, 4)), explanation.values[..., 0]
    )
    assert explanation.importance()["nll"].idxmax() == "cutout"


def test_an_encoder_that_is_a_barrier_gives_its_columns_as_features(
    backend, items
):
    model = _tabularized(
        backend,
        more={"notes": lazy.text.Hashing(n_features=4)},
        n_columns=None,
    ).fit(*items[:2])
    with pytest.warns(lazy.LazyWarning, match=r"tabularizer\['notes'\]"):
        explanation = _explain(model, items, method="gradient")
    assert explanation.feature_names == (
        "cutout",
        *(f"notes__text_hashing_{i}" for i in range(4)),
        "number",
    )
    assert [b.step for b in explanation.barriers] == ["tabularizer['notes']"]
    assert "characters" in explanation.barriers[0].reason
    # At the Tabularizer's output, the features are its columns.
    staged = _explain(model, items, method="gradient", at="tabularizer")
    assert staged.feature_names[-1] == "remainder__number"
    assert staged.item_values is None


def test_images_of_several_sizes_are_attributed_at_their_embedding(
    backend, images
):
    X_context, y_context, X, y = images
    X = X.assign(cutout=[np.ones((3, 5, 1)) * i for i in range(len(X))])
    model = _tabularized(backend, n_columns=None).fit(X_context, y_context)
    with pytest.warns(lazy.LazyWarning, match="do not all have one shape"):
        explanation = interp.explain(
            model, X, y, X_context, y_context, method="gradient"
        )
    assert explanation.feature_names[0] == "cutout__image_pixels_y0x0c0"


@pytest.mark.parametrize("pooling", ["geometric", "linear"])
def test_an_ensembles_gradients_are_those_of_its_pooled_density(
    backend, numbers, pooling
):
    members = [
        lazy.make_pipeline(preprocessing.StandardScaler(), backend()),
        backend(random_state=5),
    ]
    model = lazy.LazyEnsembleModel(members, pooling=pooling)
    model.fit(*numbers[:2])
    explanation = _integrated(model, numbers)
    assert explanation.feature_names == ("a", "b", "c")
    # The pass pools as predict does: its statistics are predict's.
    X_context, y_context, X, y = numbers
    grid = lazy.base.default_grid(
        min(y.min(), y_context.min()), max(y.max(), y_context.max())
    )
    expected = interp._statistics(model, X, y, True, grid)
    np.testing.assert_allclose(explanation.full_values, expected, rtol=1e-4)


def test_an_ensemble_of_one_member_explains_as_the_member(backend, numbers):
    member = lazy.make_pipeline(preprocessing.StandardScaler(), backend())
    model = lazy.LazyEnsembleModel([member], pooling="linear")
    model.fit(*numbers[:2])
    X_context, y_context, X, y = numbers
    grid = model.native_grid_
    pooled = _explain(model, numbers, method="gradient", y_grid=grid)
    alone = _explain(
        model.estimators_[0], numbers, method="gradient", y_grid=grid
    )
    # The pool restricts the member to the pooling grid, which holds all
    # but a sliver of its mass.
    np.testing.assert_allclose(pooled.values, alone.values, atol=5e-3)


def test_an_ensemble_reports_each_members_barriers_and_refuses_quantiles(
    backend, numbers
):
    blocked = lazy.make_pipeline(
        preprocessing.KBinsDiscretizer(4, encode="ordinal"), backend()
    )
    model = lazy.LazyEnsembleModel([blocked, backend()])
    model.fit(*numbers[:2])
    with pytest.warns(lazy.LazyWarning, match="tabpfn__kbinsdiscretizer"):
        explanation = _explain(model, numbers, method="gradient")
    assert explanation.barriers[0].step == "tabpfn__kbinsdiscretizer"
    # The first member's stage columns, then the second's raw columns.
    assert explanation.feature_names == ("x0", "x1", "x2", "a", "b", "c")
    quantile = lazy.LazyEnsembleModel([backend()], pooling="quantile")
    quantile.fit(*numbers[:2])
    with pytest.raises(ValueError, match="pooling='geometric' or 'linear'"):
        _explain(quantile, numbers, method="gradient")


def test_a_removal_method_at_a_stage_is_that_method_on_the_rest(items):
    X_context, y_context, X, y = items
    tab = lazy.Tabularizer(
        {"cutout": lazy.image.Pixels(size=2, n_channels=1)},
        n_columns=2,
        remainder="drop",
    )
    model = lazy.make_pipeline(
        tab, standins.HistogramStandIn(n_estimators=2, progress=False)
    ).fit(X_context, y_context)
    grid = lazy.base.default_grid(
        float(y_context.min()), float(y_context.max())
    )
    staged = interp.explain(
        model, X, y, method="permutation", at="tabularizer", y_grid=grid
    )
    assert staged.stage == "tabularizer"
    assert staged.feature_names == ("embedding__pc0", "embedding__pc1")
    direct = interp.explain(
        model[-1], model[0].transform(X), y, method="permutation", y_grid=grid
    )
    np.testing.assert_allclose(staged.values, direct.values)
    refit = interp.explain(
        model, X, y, X_context, y_context, method="loco", at="tabularizer"
    )
    direct = interp.explain(
        model[-1],
        model[0].transform(X),
        y,
        model[0].transform(X_context),
        y_context,
        method="loco",
    )
    np.testing.assert_allclose(refit.values, direct.values)
    assert refit.feature_names == staged.feature_names


def test_context_gradients_stop_at_the_embeddings_unless_asked(backend, images):
    # The pixel encoder is linear and nothing is reduced, so a context
    # row's gradient times input is the same in its pixels and in its
    # embedding.
    X_context, y_context, X, y = images
    model = _tabularized(backend, n_columns=None).fit(X_context, y_context)
    args = (model, X[:3], y[:3], X_context, y_context)
    at_embeddings = interp.explain_context(*args)
    at_pixels = interp.explain_context(*args, through_encoders=True)
    assert at_embeddings.values.shape == (3, 3, N_CONTEXT)
    np.testing.assert_allclose(
        at_embeddings.values, at_pixels.values, atol=1e-9
    )
    staged = interp.explain_context(*args, at="tabularizer")
    np.testing.assert_allclose(staged.values, at_embeddings.values, atol=1e-9)
    with pytest.raises(ValueError, match="at does not apply to method='loo'"):
        interp.explain_context(*args, method="loo", at="tabularizer")


def test_an_index_map_carries_the_gradient_back_to_the_pixels(backend, images):
    # The quarter turn moves each pixel, so its attribution moves back
    # with it: the image's top-left pixel gets what the encoder's cell at
    # the turned position got.
    X_context, y_context, X, y = images
    columns = lazy.image.Pixels(size=4, n_channels=1)
    plain = lazy.make_pipeline(
        lazy.Tabularizer({"cutout": columns}, n_columns=None), backend()
    ).fit(X_context, y_context)
    turned = lazy.make_pipeline(
        lazy.Tabularizer(
            {"cutout": [lazy.augment.Dihedral("rotate_90"), columns]},
            n_columns=None,
        ),
        backend(),
    ).fit(X_context, y_context)
    explanations = [
        interp.explain(model, X, y, X_context, y_context, method="gradient")
        for model in (plain, turned)
    ]
    assert explanations[1].feature_names == ("cutout", "number")
    # The fake reads the mean of its columns, which a turn leaves as it
    # is, so both models attribute alike, pixel by pixel.
    np.testing.assert_allclose(
        explanations[1].item_values["cutout"],
        explanations[0].item_values["cutout"],
        atol=1e-6,
    )
