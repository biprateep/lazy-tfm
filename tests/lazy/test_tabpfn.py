# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""What TabPFN is handed: lazy's own pins, not upstream's defaults.

Most tests here read the settings lazy builds and need no checkpoint. Those
marked ``needs_checkpoint`` load one, on the CPU, and run only with
``LAZY_RUN_CHECKPOINT_TESTS=1``.
"""

import dataclasses
import json
import os

import numpy as np
import pytest

import lazy
from lazy.models import _hub
from lazy.models import _members
from lazy.models import _transforms
from lazy.models import tabpfn

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)

VERSIONS = lazy.list_versions("tabpfn")


@pytest.fixture(autouse=True)
def _needs_tabpfn():
    pytest.importorskip("tabpfn")


def _group(transforms=None, *, feature_shuffle=True, bag_rows=10, n=10):
    """The first member group lazy plans for these settings."""
    return _members.plan(
        n_estimators=4,
        transforms=_transforms.parse(transforms or "auto"),
        native_transforms=tabpfn.TabPFNBarDistribution.native_transforms,
        feature_shuffle=feature_shuffle,
        bag_rows=bag_rows,
        n_rows=n,
        n_features=2,
        random_state=0,
        supports_native_bagging=True,
    )[0]


def _without_softmax(recipe):
    return {k: v for k, v in recipe.items() if k != "SOFTMAX_TEMPERATURE"}


def test_every_version_has_a_pinned_recipe():
    assert set(tabpfn._AUTO_RECIPES) == set(VERSIONS)


@pytest.mark.parametrize("version", VERSIONS)
def test_auto_hands_over_the_pinned_recipe(version):
    """Every field is lazy's, none left to the checkpoint or the package."""
    recipe = tabpfn._AUTO_RECIPES[version]
    settings = tabpfn._upstream_settings(
        version, _group(), recipe["OUTLIER_REMOVAL_STD"]
    )
    pinned = {k: v for k, v in settings.items() if k in recipe}
    assert pinned == _without_softmax(recipe)
    assert settings["SUBSAMPLE_SAMPLES"] is None


def test_the_temperature_is_an_argument_not_a_config_field():
    """Upstream rejects a temperature named both ways."""
    settings = tabpfn._upstream_settings("v3", _group(), None)
    assert "SOFTMAX_TEMPERATURE" not in settings
    assert "N_ESTIMATORS" not in settings


@pytest.mark.parametrize(
    ("version", "temperature", "threshold"),
    [
        ("v2", 0.9, None),
        ("v2.5", 0.9, None),
        ("v2.6", 0.9, None),
        ("v3", 0.9, None),
        ("v3.5", 1.0, 12.0),
        ("v3.5-fast", 1.0, 12.0),
    ],
)
def test_auto_temperature_and_clip_are_each_versions_own(
    version, temperature, threshold
):
    est = tabpfn.TabPFNBarDistribution(version=version)
    assert est._auto_softmax_temperature() == temperature
    assert est._auto_outlier_threshold() == threshold


@pytest.mark.parametrize("threshold", [None, 3.0])
def test_the_clip_is_the_resolved_outlier_threshold(threshold):
    settings = tabpfn._upstream_settings("v3.5", _group(), threshold)
    assert settings["OUTLIER_REMOVAL_STD"] == threshold


def test_no_feature_shuffle_turns_upstreams_shift_off():
    settings = tabpfn._upstream_settings(
        "v3.5", _group(feature_shuffle=False), None
    )
    assert settings["FEATURE_SHIFT_METHOD"] is None
    shuffled = tabpfn._upstream_settings("v3.5", _group(), None)
    assert shuffled["FEATURE_SHIFT_METHOD"] == "shuffle"


def test_bags_are_handed_over_as_each_members_rows():
    group = _group(bag_rows=6, n=10)
    settings = tabpfn._upstream_settings("v3", group, None)
    rows = settings["SUBSAMPLE_SAMPLES"]
    assert len(rows) == group.n_members
    assert all(len(r) == 6 for r in rows)


def _upstream_fields(config):
    """An upstream ``InferenceConfig`` in the table's terms."""
    fields = dataclasses.asdict(config)
    fields["PREPROCESS_TRANSFORMS"] = tuple(fields["PREPROCESS_TRANSFORMS"])
    fields["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = tuple(
        fields["REGRESSION_Y_PREPROCESS_TRANSFORMS"]
    )
    fields["OUTLIER_REMOVAL_STD"] = config.get_resolved_outlier_removal_std(
        "regressor"
    )
    return fields


@pytest.mark.parametrize("version", ["v2", "v2.5"])
def test_the_oldest_recipes_are_what_upstream_ships_for_them(version):
    """v2 and v2.5 store no config: their recipe lives in tabpfn's code.

    A failure here means an upgraded tabpfn changed it; lazy keeps the old
    one, so predictions do not move, but the table's provenance is stale.
    """
    constants = pytest.importorskip("tabpfn.constants")
    inference_config = pytest.importorskip("tabpfn.inference_config")
    shipped = _upstream_fields(
        inference_config.InferenceConfig.get_default(
            "regression", constants.ModelVersion(version)
        )
    )
    recipe = tabpfn._AUTO_RECIPES[version]
    assert {k: shipped[k] for k in recipe} == recipe


@needs_checkpoint
@pytest.mark.parametrize("version", VERSIONS)
def test_the_pinned_recipes_are_what_the_checkpoints_store(version):
    upstream = pytest.importorskip("tabpfn")
    path = tabpfn.path_for_tabpfn(
        _hub.get_checkpoint("tabpfn", version).download()
    )
    regressor = upstream.TabPFNRegressor(model_path=path, device="cpu")
    stored = _upstream_fields(regressor.get_inference_config())
    recipe = tabpfn._AUTO_RECIPES[version]
    assert {k: stored[k] for k in recipe} == recipe


def _fit(X, y, **params):
    return tabpfn.TabPFNBarDistribution(
        n_estimators=2, device="cpu", progress=False, **params
    ).fit(X, y)


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=150)
    return X[:120], y[:120], X[120:]


@needs_checkpoint
def test_the_fitted_regressor_ran_the_pinned_recipe(small):
    X, y, _ = small
    est = _fit(X, y, version="v3")
    config = est.regressor_.inference_config_
    recipe = tabpfn._AUTO_RECIPES["v3"]
    assert {k: _upstream_fields(config)[k] for k in recipe} == recipe
    assert est.regressor_.softmax_temperature_ == 0.9
    assert est.provenance_["softmax_temperature"] == 0.9
    (recorded,) = est.provenance_["inference_config"]
    assert recorded["PREPROCESS_TRANSFORMS"] == recipe["PREPROCESS_TRANSFORMS"]
    json.dumps(est.provenance_)


@needs_checkpoint
def test_an_explicit_temperature_reaches_upstream(small):
    X, y, _ = small
    est = _fit(X, y, version="v3.5", softmax_temperature=0.5)
    assert est.regressor_.softmax_temperature_ == 0.5
    assert est.provenance_["softmax_temperature"] == 0.5


@needs_checkpoint
@pytest.mark.parametrize(
    ("outlier_threshold", "expected"),
    [("auto", 12.0), (None, None), (3.0, 3.0)],
)
def test_the_clip_reaches_upstream(small, outlier_threshold, expected):
    X, y, _ = small
    est = _fit(X, y, version="v3.5", outlier_threshold=outlier_threshold)
    config = est.regressor_.inference_config_
    assert config.get_resolved_outlier_removal_std("regressor") == expected
    assert est.clippers_ == [None], "native, so lazy does not clip as well"
    assert est.provenance_["outlier_threshold"] == expected


def test_no_column_is_taken_for_a_category():
    settings = tabpfn._upstream_settings("v3.5", _group(), None)
    assert settings["MIN_UNIQUE_FOR_NUMERICAL_FEATURES"] == 1


@needs_checkpoint
@pytest.mark.parametrize("version", ["v2", "v3.5"])
def test_a_three_valued_column_stays_numeric(small, version):
    """Upstream's own rule would encode it: < 4 values in > 100 rows."""
    modalities = pytest.importorskip("tabpfn.preprocessing.datamodel")
    X, y, _ = small
    X = X.copy()
    X[:, 1] = np.arange(len(X)) % 3
    est = _fit(X, y, version=version)
    schema = est.regressor_.inferred_feature_schema_
    categorical = modalities.FeatureModality.CATEGORICAL
    assert schema.indices_for(categorical) == []
    numerical = schema.indices_for(modalities.FeatureModality.NUMERICAL)
    assert len(numerical) == X.shape[1]


@pytest.mark.parametrize(
    ("mixed_precision", "expected"), [(True, "autocast"), (False, "float32")]
)
def test_only_mixed_precision_autocasts(mixed_precision, expected):
    torch = pytest.importorskip("torch")
    est = tabpfn.TabPFNBarDistribution()
    est.mixed_precision_ = mixed_precision
    precision = est._inference_precision()
    assert precision == (
        "autocast" if expected == "autocast" else torch.float32
    )


@needs_checkpoint
@pytest.mark.parametrize("mixed_precision", [True, False])
def test_the_cpu_runs_in_float32(small, mixed_precision):
    """Upstream would autocast to bfloat16 on a CPU that has it fast."""
    torch = pytest.importorskip("torch")
    X, y, _ = small
    est = _fit(X, y, version="v3", mixed_precision=mixed_precision)
    assert est.regressor_.use_autocast_ is False
    assert est.regressor_.forced_inference_dtype_ is torch.float32
    assert est.provenance_["mixed_precision"] is False
    assert est.provenance_["inference_precision"] == "float32"


@pytest.mark.parametrize(
    ("version", "limit"),
    [
        ("v2", 500),
        ("v2.5", 500),
        ("v2.6", 680),
        ("v3", 500),
        ("v3.5", 768),
        ("v3.5-fast", 768),
    ],
)
def test_an_explicit_recipe_runs_without_the_extras(version, limit):
    """The named transform, the version's feature limit, nothing optional."""
    settings = tabpfn._upstream_settings(version, _group("none"), None)
    (config,) = settings["PREPROCESS_TRANSFORMS"]
    assert config["name"] == "none"
    assert config["categorical_name"] == "numeric"
    assert config["global_transformer_name"] is None
    assert config["append_original"] is False
    assert config["max_features_per_estimator"] == limit
    assert settings["FINGERPRINT_FEATURE"] is False
    assert settings["POLYNOMIAL_FEATURES"] == "no"
    assert settings["REGRESSION_Y_PREPROCESS_TRANSFORMS"] == (None,)
    assert settings["OUTLIER_REMOVAL_STD"] is None


@needs_checkpoint
@pytest.mark.parametrize("outlier_threshold", ["auto", None, 3.0])
def test_an_explicit_recipe_reaches_upstream_bare(small, outlier_threshold):
    X, y, _ = small
    est = _fit(
        X,
        y,
        version="v3.5",
        transforms="none",
        outlier_threshold=outlier_threshold,
    )
    expected = 3.0 if outlier_threshold == 3.0 else None
    for member in est.regressor_.ensemble_configs_:
        assert member.preprocess_config.name == "none"
        assert member.preprocess_config.global_transformer_name is None
        assert member.add_fingerprint_feature is False
        assert member.polynomial_features == "no"
        assert member.target_transform is None
        assert member.outlier_removal_std == expected
