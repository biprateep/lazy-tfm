# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LimiX's member preprocessing: upstream's seeds, fitted on context only."""

import ast
import importlib
import json
import random

import numpy as np
import pytest

from lazy.models import _limix_preprocess
from lazy.models import _limix_source
from lazy.models import _transforms


def _has_limix() -> bool:
    try:
        _limix_source.locate(download=False)  # a skip check never downloads
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
    assert set(_limix_preprocess.RECIPE.block) <= set(
        _limix_preprocess.PIPELINES
    )
    assert set(_limix_preprocess.NATIVE_TRANSFORMS) == {
        "none",
        "none+original",
        "quantile_uniform",
    }
    assert set(_limix_preprocess.NATIVE_TRANSFORMS) <= set(
        _limix_preprocess.PIPELINES
    )


@needs_limix
def test_the_pinned_recipe_is_upstreams_config():
    root = _limix_source.locate().root
    config = root / "config" / "reg_default_noretrieval_v2.json"
    if not config.exists():
        pytest.skip("this LimiX install has no config/")
    pipelines = json.loads(config.read_text())["pipelines"]
    recipe = _limix_preprocess.RECIPE
    assert len(pipelines) == len(recipe.block)
    for token, pipeline in zip(recipe.block, pipelines, strict=True):
        assert pipeline["RebalanceFeatureDistribution"] == dict(
            recipe.pipelines[token]
        )
        assert pipeline["FeatureShuffler"] == {"mode": "shuffle"}
        assert set(pipeline) <= {
            "RebalanceFeatureDistribution",
            "CategoricalFeatureEncoder",
            "FeatureShuffler",
            "retrieval_config",
        }
        assert not pipeline["retrieval_config"]["use_retrieval"]


def _upstream_predictor_defaults(root):
    """LimiXPredictor's keyword defaults and ``self.x = constant`` lines."""
    source = (root / "inference" / "v2_0" / "predictor.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == "LimiXPredictor":
            init = next(
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            )
            break
    arguments = init.args.args[-len(init.args.defaults) :]
    values = {
        argument.arg: ast.literal_eval(default)
        for argument, default in zip(arguments, init.args.defaults, strict=True)
        if isinstance(default, ast.Constant)
    }
    for statement in ast.walk(init):
        if (
            isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Attribute)
            and isinstance(statement.value, ast.Constant)
        ):
            values[f"self.{statement.targets[0].attr}"] = statement.value.value
    return values


@needs_limix
def test_the_pinned_predictor_settings_are_upstreams():
    defaults = _upstream_predictor_defaults(_limix_source.locate().root)
    recipe = _limix_preprocess.RECIPE
    assert defaults["softmax_temperature"] == recipe.softmax_temperature
    assert defaults["self.preprocess_num"] == recipe.seeds_per_member


def test_whole_blocks_of_the_recipe_are_upstreams():
    block = _limix_preprocess.RECIPE.block
    for blocks in (1, 2, 4):
        assert _limix_preprocess.auto_tokens(8 * blocks) == block * blocks


@pytest.mark.parametrize("n_members", [*range(1, 8), 9, 13, 21, 39])
def test_a_partial_block_keeps_the_recipes_mix(n_members):
    tokens = _limix_preprocess.auto_tokens(n_members)
    blocks, rest = divmod(n_members, 8)
    assert len(tokens) == n_members
    assert tokens[: 8 * blocks] == _limix_preprocess.RECIPE.block * blocks
    remainder = tokens[8 * blocks :]
    n_quantile = remainder.count("auto_quantile")
    assert n_quantile == (rest + 1) // 2
    assert remainder == ("auto_quantile",) * n_quantile + ("auto_power",) * (
        rest - n_quantile
    )


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
@pytest.mark.parametrize("n_rows", [3, 37, 300])
@pytest.mark.parametrize("name", sorted(_limix_preprocess.NATIVE_TRANSFORMS))
def test_a_native_transform_is_the_vocabularys(name, n_rows):
    """LimiX's own step equals the scaffold followed by the "none" step."""
    rng = np.random.default_rng(n_rows)
    values = rng.normal(size=(n_rows + 20, 4))
    values[:, 1] = np.exp(2 * values[:, 1])
    values[::3, 0] = np.nan
    values[:, 3] = 2.0  # constant: dropped either way
    context, queries = values[:n_rows], values[n_rows:]
    native = _limix_preprocess.MemberPipeline(
        _limix_preprocess.NATIVE_TRANSFORMS[name], (1, 2), shuffle=True
    ).fit(context)
    (spec,) = _transforms.parse(name)
    scaffold = _transforms.ScaffoldTransform(spec, 0).fit(context)
    plain = _limix_preprocess.MemberPipeline("none", (1, 2), shuffle=True).fit(
        scaffold.transform(context)
    )
    np.testing.assert_array_equal(
        native.transform(queries), plain.transform(scaffold.transform(queries))
    )


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
def test_a_column_constant_apart_from_missing_values_is_dropped():
    """Else a query's input to the network depends on its chunk.

    The network fills a missing value with the context mean, so the column is
    constant on the context; upstream's ``ValidFeatureEncoder`` decides which
    columns are valid over the context and queries together and rescales
    them by how many are, so a query (and the cached context) would see other
    queries in the chunk.
    """
    torch = pytest.importorskip("torch")
    _limix_source.load()
    encoders = importlib.import_module("lazy.models._limix_ext.model.encoders")
    rng = np.random.default_rng(0)
    context = np.c_[rng.normal(size=50), np.full(50, 5.0)]
    context[7, 1] = np.nan
    queries = np.array([[0.1, 5.0], [0.2, 7.0]])
    pipeline = _limix_preprocess.MemberPipeline(
        "none", (1, 2), shuffle=False
    ).fit(context)
    group = 4  # the network's features per group, padded with zeros
    network_input = encoders.PreprocessPipeline(
        num_features=group,
        nan_handling_enabled=True,
        normalize_on_train_only=True,
        normalize_x=True,
        remove_outliers=False,
        normalize_by_used_features=True,
    )

    def encode(rows):
        x = pipeline.transform(np.concatenate([context, rows]))
        x = np.pad(x, ((0, 0), (0, group - x.shape[1])))
        data = torch.tensor(x, dtype=torch.float32).reshape(1, len(x), 1, -1)
        out = network_input({"data": data, "eval_pos": len(context)})
        return out["data"][0, :, 0].numpy()

    alone, together = encode(queries[:1]), encode(queries)
    np.testing.assert_array_equal(alone, together[: len(alone)])
    np.testing.assert_array_equal(encode(queries[:0]), together[: len(context)])
    assert pipeline.keep_.tolist() == [True, False]


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
