# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LimiX's member preprocessing: upstream's seeds, fitted on context only."""

import random

import numpy as np
import pytest

from lazy.models import _limix_preprocess
from lazy.models import _limix_source
from lazy.models import _transforms


def _has_limix() -> bool:
    try:
        _limix_source.locate()
    except ImportError:
        return False
    return True


needs_limix = pytest.mark.skipif(
    not _has_limix(), reason="needs LimiX's source ($LAZY_LIMIX_SRC)"
)


@pytest.fixture(scope="module")
def features():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(300, 5))
    values[::7, 2] = np.nan
    values[:, 4] = 3.0  # constant: carries nothing
    return values


def test_member_seeds_are_upstreams_draws():
    """Upstream: random.seed(seed); draws of randint(0, 10000), ten each."""
    state = random.getstate()
    try:
        random.seed(5)
        draws = [random.randint(0, 10_000) for _ in range(30)]
    finally:
        random.setstate(state)
    assert _limix_preprocess.member_seeds(5, 3) == [
        (draws[10 * i + 1], draws[10 * i + 3]) for i in range(3)
    ]


def test_member_seeds_leave_the_global_generator_alone():
    state = random.getstate()
    _limix_preprocess.member_seeds(5, 3)
    assert random.getstate() == state


def test_the_recipe_and_the_native_names_are_pipelines():
    assert set(_limix_preprocess.AUTO_TOKENS) <= set(
        _limix_preprocess.PIPELINES
    )
    vocabulary = set(_transforms.BASE_TRANSFORMS) - {"quantile_rtdl"}
    vocabulary |= {name + "+original" for name in vocabulary}
    assert set(_limix_preprocess.NATIVE_TRANSFORMS) == vocabulary


def test_an_unknown_token_is_refused():
    with pytest.raises(ValueError, match="unknown LimiX pipeline"):
        _limix_preprocess.MemberPipeline("nope", (1, 2), shuffle=True)


@needs_limix
@pytest.mark.parametrize("token", sorted(_limix_preprocess.PIPELINES))
def test_a_row_is_transformed_alone(token, features):
    pipeline = _limix_preprocess.MemberPipeline(
        token, (11, 12), shuffle=True
    ).fit(features[:200])
    together = pipeline.transform(features)
    apart = np.vstack(
        [pipeline.transform(features[:200]), pipeline.transform(features[200:])]
    )
    np.testing.assert_array_equal(together, apart)
    assert together.shape[1] == pipeline.n_features_out_


@needs_limix
def test_constant_and_missing_columns_are_dropped(features):
    values = features.copy()
    values[:200, 1] = np.nan  # missing throughout the context
    pipeline = _limix_preprocess.MemberPipeline(
        "none", (1, 2), shuffle=False
    ).fit(values[:200])
    assert pipeline.keep_.tolist() == [True, False, True, True, False]
    np.testing.assert_array_equal(
        pipeline.transform(values[200:]), values[200:, [0, 2, 3]]
    )


@needs_limix
def test_a_context_with_nothing_to_learn_from_is_refused():
    with pytest.raises(ValueError, match="constant or missing"):
        _limix_preprocess.MemberPipeline("none", (1, 2), shuffle=True).fit(
            np.ones((10, 3))
        )


@needs_limix
def test_the_shuffle_is_seeded(features):
    def columns(seed):
        pipeline = _limix_preprocess.MemberPipeline(
            "none", (1, seed), shuffle=True
        ).fit(features[:200, :4])
        return pipeline.transform(np.eye(4))

    np.testing.assert_array_equal(columns(3), columns(3))
    assert not all(
        np.array_equal(columns(3), columns(seed)) for seed in range(4, 10)
    )
