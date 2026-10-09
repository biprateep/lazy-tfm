# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Kumo Tabular: the recipe pinned here is upstream's, and so is the answer.

The behavioral contract runs on the fake upstream (tests/lazy/fakes/kumo.py)
in the conformance and bagging suites. Here the pinned recipe is compared
with the one upstream ships, and, with the checkpoint, the backend's answer
with upstream's own default path.
"""

import os
import re

import numpy as np
import pytest

import lazy
from lazy.models import kumo

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)

# The categorical branch of upstream's recipe: lazy hands Kumo numeric
# columns only, so its recipe has none.
_CATEGORICAL = re.compile(r"\n( +)categorical=.*?\n\1\),", flags=re.DOTALL)


def _upstream_recipe():
    sdm = pytest.importorskip("sdm")
    return sdm.models.KumoTabular.default_recipe()


def _auto_recipe():
    sp = pytest.importorskip("sdm.processing")
    return kumo._recipe(
        sp,
        kumo.AUTO_TRANSFORMS,
        feature_shuffle=True,
        outlier_threshold=kumo.AUTO_OUTLIER_THRESHOLD,
    )


def _find(module, kind):
    return [m for m in module.modules() if type(m).__name__ == kind]


def test_the_pinned_features_and_target_are_upstreams_numeric_branch():
    upstream, pinned = _upstream_recipe(), _auto_recipe()
    for role in ("features", "target"):
        expected = _CATEGORICAL.sub("", repr(getattr(upstream, role)))
        assert repr(getattr(pinned, role)) == expected
    ((upstream_select,), (pinned_select,)) = (
        _find(upstream.features, "SelectColumns"),
        _find(pinned.features, "SelectColumns"),
    )
    assert vars(pinned_select) == vars(upstream_select)


def test_the_pinned_output_keeps_the_members_upstream_trims():
    upstream, pinned = _upstream_recipe(), _auto_recipe()
    trims = {
        m.trim_fraction for m in _find(upstream.output, "AverageEstimators")
    }
    assert kumo.AUTO_TRIM_FRACTION in trims
    assert kumo.KumoQuantile.member_trim_fraction == kumo.AUTO_TRIM_FRACTION
    assert _find(pinned.output, "SortQuantiles")
    assert not _find(pinned.output, "AverageEstimators")


def test_an_explicit_recipe_drops_the_clip_and_the_shuffle():
    sp = pytest.importorskip("sdm.processing")
    recipe = kumo._recipe(
        sp, ("identity",), feature_shuffle=False, outlier_threshold=None
    )
    assert not _find(recipe.features, "ClipSigma")
    assert not _find(recipe.features, "ShuffleColumns")
    assert _find(recipe.features, "FlipSign")


def test_members_cycle_through_their_transforms():
    assert kumo._cycle(("identity", "power", "identity")) == (
        "identity",
        "power",
    )
    with pytest.raises(ValueError, match="cycles its transforms"):
        kumo._cycle(("power", "power", "identity"))
    with pytest.raises(ValueError, match="no transform 'yeo'"):
        kumo._transform(None, "yeo")


def test_every_version_is_a_size_with_a_pinned_checkpoint():
    assert sorted(kumo.SIZES) == lazy.list_versions("kumo")
    for version, size in kumo.SIZES.items():
        checkpoint = lazy.get_checkpoint("kumo", version)
        assert checkpoint.filename == f"{size}/regressor.pt"
        assert kumo.KumoQuantile._pinned_recipe(version)["size"] == size


@needs_checkpoint
def test_the_default_path_is_upstreams_on_a_cpu():
    """The trim is lazy's, in float64; upstream's runs in float32."""
    sdm = pytest.importorskip("sdm")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(260, 4))
    y = 2.0 + np.sin(2.0 * X[:, 0]) + 0.1 * rng.normal(size=260)
    model = lazy.KumoQuantile(
        version="v1-small", n_estimators=5, device="cpu", progress=False
    ).fit(X[:200], y[:200])
    ours = model.predict_distribution(X[200:]).locs
    network = sdm.models.KumoTabular(
        task="regression", size="small", device="cpu"
    )
    offset, scale = y[:200].mean(), y[:200].std()
    (group,) = model.member_groups_
    theirs = (
        network(
            x_context=torch.tensor(X[:200]),
            y_context=torch.tensor(
                ((y[:200] - offset) / scale)[:, None]
            ).float(),
            x_query=torch.tensor(X[200:]),
            num_estimators=5,
            generator=torch.Generator().manual_seed(group.seed),
        )
        .numerical.double()
        .numpy()
    )
    np.testing.assert_allclose(ours, theirs * scale + offset, atol=1e-5)
