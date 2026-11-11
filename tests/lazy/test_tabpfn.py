# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""What TabPFN is handed: lazy's own pins, not upstream's defaults.

Most tests here read the settings lazy builds and need no checkpoint. Those
marked ``checkpoint`` load one, on the CPU, and run only with
``LAZY_RUN_CHECKPOINT_TESTS=1``.
"""

import dataclasses
import inspect
import json
import pickle
import threading
import types
import warnings

import fakes
from fakes import tabpfn as fake_tabpfn
import numpy as np
import pytest
import threads

import lazy
from lazy.models import _device
from lazy.models import _hub
from lazy.models import _members
from lazy.models import _transforms
from lazy.models import tabpfn

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
        supports_native_bagging=(
            tabpfn.TabPFNBarDistribution.supports_native_bagging
        ),
    )[0]


def _settings(version, group, outlier_threshold):
    """What a group of the four-member ensemble of _group hands TabPFN."""
    return tabpfn._upstream_settings(version, group, outlier_threshold, 4)


def _without_softmax(recipe):
    return {k: v for k, v in recipe.items() if k != "SOFTMAX_TEMPERATURE"}


def test_every_version_has_a_pinned_recipe():
    assert set(tabpfn._RECIPES) == set(VERSIONS)


@pytest.mark.parametrize("version", VERSIONS)
def test_auto_hands_over_the_pinned_recipe(version):
    """Every field is lazy's, none left to the checkpoint or the package."""
    recipe = tabpfn._RECIPES[version]
    settings = _settings(version, _group(), recipe["OUTLIER_REMOVAL_STD"])
    pinned = {k: v for k, v in settings.items() if k in recipe}
    assert pinned == _without_softmax(recipe)
    assert settings["SUBSAMPLE_SAMPLES"] is None


def test_the_temperature_is_an_argument_not_a_config_field():
    """Upstream rejects a temperature named both ways."""
    settings = _settings("v3", _group(), None)
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
    settings = _settings("v3.5", _group(), threshold)
    assert settings["OUTLIER_REMOVAL_STD"] == threshold


def test_no_feature_shuffle_turns_upstreams_shift_off():
    settings = _settings("v3.5", _group(feature_shuffle=False), None)
    assert settings["FEATURE_SHIFT_METHOD"] is None
    shuffled = _settings("v3.5", _group(), None)
    assert shuffled["FEATURE_SHIFT_METHOD"] == "shuffle"


def test_a_bagged_member_is_a_regressor_of_its_own_on_its_bag():
    """Upstream's own subsampling would standardise y on the whole context."""
    group = _group(bag_rows=6, n=10)
    assert group.n_members == 1 and len(group.rows) == 6
    settings = _settings("v3", group, None)
    assert settings["SUBSAMPLE_SAMPLES"] is None
    assert not tabpfn.TabPFNBarDistribution.supports_native_bagging


@pytest.mark.parametrize("version", VERSIONS)
def test_bagged_members_keep_the_recipes_mix(version):
    """Member i runs the preprocessor and target transform upstream gives i."""
    recipe = tabpfn._RECIPES[version]
    groups = _members.plan(
        n_estimators=4,
        transforms=None,
        native_transforms=tabpfn.TabPFNBarDistribution.native_transforms,
        feature_shuffle=True,
        bag_rows=6,
        n_rows=10,
        n_features=2,
        random_state=0,
        supports_native_bagging=False,
    )
    slots = [
        (
            _settings(version, group, None)["PREPROCESS_TRANSFORMS"],
            _settings(version, group, None)[
                "REGRESSION_Y_PREPROCESS_TRANSFORMS"
            ],
        )
        for group in groups
    ]
    # Upstream's own assignment, on placeholders for its configs.
    ensemble = pytest.importorskip("tabpfn.preprocessing.ensemble")
    n_preprocessors = len(recipe["PREPROCESS_TRANSFORMS"])
    upstream = ensemble.generate_regression_ensemble_configs(
        num_estimators=4,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=list(range(n_preprocessors)),
        target_transforms=list(recipe["REGRESSION_Y_PREPROCESS_TRANSFORMS"]),
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )
    expected = [
        (
            recipe["PREPROCESS_TRANSFORMS"][config.preprocess_config],
            config.target_transform,
        )
        for config in upstream
    ]
    assert slots == [((p,), (t,)) for p, t in expected]


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
    recipe = tabpfn._RECIPES[version]
    assert {k: shipped[k] for k in recipe} == recipe


@pytest.mark.checkpoint
@pytest.mark.parametrize("version", VERSIONS)
def test_the_pinned_recipes_are_what_the_checkpoints_store(version):
    upstream = pytest.importorskip("tabpfn")
    path = tabpfn.path_for_tabpfn(
        _hub.get_checkpoint("tabpfn", version).download()
    )
    regressor = upstream.TabPFNRegressor(model_path=path, device="cpu")
    stored = _upstream_fields(regressor.get_inference_config())
    recipe = tabpfn._RECIPES[version]
    assert {k: stored[k] for k in recipe} == recipe


def test_concurrent_fits_share_one_network(monkeypatch):
    """Four fits in four threads, all inside upstream's fit at once."""
    recorder = fakes.Recorder()
    cls = fakes.install("tabpfn", monkeypatch, recorder)
    together = threading.Barrier(4, timeout=threads.TIMEOUT)
    fit = fake_tabpfn.Regressor.fit

    def fit_together(self, X, y):
        together.wait()
        return fit(self, X, y)

    monkeypatch.setattr(fake_tabpfn.Regressor, "fit", fit_together)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0]
    fitted = []

    def run():
        fitted.append(
            cls(n_estimators=1, device="cpu", progress=False).fit(X, y)
        )

    assert threads.run(*[run] * 4), "deadlock"
    assert len(fitted) == 4
    assert len(recorder.loads) == 1
    (network,) = fitted[0].regressor_.models_
    assert all(est.regressor_.models_[0] is network for est in fitted)


def test_a_fitted_regressor_names_the_checkpoint_not_the_spec(monkeypatch):
    """A pickle holds a path, as before; a spec class is not stable."""
    recorder = fakes.Recorder()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    est = fakes.install("tabpfn", monkeypatch, recorder)(
        n_estimators=1, device="cpu", progress=False
    ).fit(X, X[:, 0])
    assert len(recorder.loads) == 1  # the fit ran on the shared network
    assert est.regressor_.model_path == tabpfn.path_for_tabpfn(est.checkpoint_)


@pytest.mark.parametrize("kv_cache", [True, False])
def test_no_regressor_argument_is_left_to_upstream(monkeypatch, kv_cache):
    """A default the installed tabpfn picks could change an answer."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    est = fakes.install("tabpfn", monkeypatch)(
        n_estimators=1, kv_cache=kv_cache, device="cpu", progress=False
    ).fit(X, X[:, 0])
    accepted = inspect.signature(fake_tabpfn.Regressor.upstream).parameters
    assert set(est.regressor_.kwargs) == set(accepted)


def test_an_argument_a_later_tabpfn_adds_is_pinned(monkeypatch):
    """Tabpfn 9.2 adds enable_torch_compile; its default is not left to it."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    estimator = fakes.install("tabpfn", monkeypatch)
    signature = inspect.signature(fake_tabpfn.Regressor.upstream)
    if "enable_torch_compile" not in signature.parameters:
        compile_arg = inspect.Parameter(
            "enable_torch_compile",
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
        )
        signature = signature.replace(
            parameters=[*signature.parameters.values(), compile_arg]
        )

    def later_upstream(**kwargs):
        signature.bind(**kwargs)

    later_upstream.__signature__ = signature
    monkeypatch.setattr(
        fake_tabpfn.Regressor, "upstream", staticmethod(later_upstream)
    )
    monkeypatch.setattr(fake_tabpfn.Regressor, "__signature__", signature)
    est = estimator(n_estimators=1, device="cpu", progress=False)
    est.fit(X, X[:, 0])
    assert est.regressor_.kwargs["enable_torch_compile"] is False


def test_an_unpickled_models_regressors_run_on_its_device(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(_device, "resolve_device", lambda device: "cpu")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    est = fakes.install("tabpfn", monkeypatch)(
        n_estimators=2, bag_size=20, device="auto", progress=False
    ).fit(X, X[:, 0])
    # As if fitted on a GPU, under mixed precision.
    est.device_, est.mixed_precision_ = "cuda:7", True
    for regressor in est.handles_:
        regressor.inference_precision = "autocast"
        regressor.to("cuda:7")
    restored = pickle.loads(pickle.dumps(est))
    for regressor in restored.handles_:
        assert regressor.device == "cpu"
        assert regressor.inference_precision is torch.float32
        # Read by to(), so set before it.
        assert regressor.use_autocast_ is False


def test_a_model_fitted_on_a_gpu_unpickles_without_one(monkeypatch):
    """Torch loads a CUDA tensor without CUDA only when mapped elsewhere."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    est = fakes.install("tabpfn", monkeypatch)(
        n_estimators=2, bag_size=20, device="cpu", progress=False
    ).fit(X, X[:, 0])
    expected = est.predict_proba(X)
    # torch tags every tensor as on a GPU, as on the machine that fitted it,
    # and this machine has none.
    gpu = (0, lambda storage: "cuda:0", lambda storage, location: None)
    registry = [gpu, *torch.serialization._package_registry]
    monkeypatch.setattr(torch.serialization, "_package_registry", registry)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    restored = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(restored.predict_proba(X), expected)


def test_a_pickle_leaves_the_network_and_the_live_model_alone(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    recorder = fakes.Recorder()
    est = fakes.install("tabpfn", monkeypatch, recorder)(
        n_estimators=2, bag_size=20, device="cpu", progress=False
    ).fit(X, X[:, 0])
    expected = est.predict_proba(X)
    network = est.handles_[0].models_[0]
    for regressor in est.handles_:
        regressor.device = "cuda:7"  # As the fake's to() records a GPU.
    restored = pickle.loads(pickle.dumps(est))
    for regressor in est.handles_:
        assert regressor.device == "cuda:7"
        assert regressor.models_ == [network]
        assert regressor.executor_.models == [network]
    np.testing.assert_array_equal(est.predict_proba(X), expected)
    # Not a copy from the pickle: the network the fit shared.
    for regressor in restored.handles_:
        assert regressor.models_ == [network]
        assert regressor.executor_.models == [network]
    assert len(recorder.loads) == 1


def test_a_pickle_of_whole_regressors_loads_with_their_networks(monkeypatch):
    """The format of earlier versions: handles_ a list of regressors."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    est = fakes.install("tabpfn", monkeypatch)(
        n_estimators=1, device="cpu", progress=False
    ).fit(X, X[:, 0])
    expected = est.predict_proba(X)
    state = {**est.__getstate__(), "handles_": est.handles_}
    state["regressor_"] = est.handles_[0]
    restored = object.__new__(type(est))
    restored.__setstate__(pickle.loads(pickle.dumps(state)))
    (regressor,) = restored.handles_
    assert restored.regressor_ is regressor
    # The pickled copy, not the network this process shares.
    assert regressor.models_[0] is not est.handles_[0].models_[0]
    np.testing.assert_array_equal(restored.predict_proba(X), expected)


def test_bucket_masses_returns_float64_borders():
    """float32 borders cannot resolve 0.01 of spread about 1e4."""
    torch = pytest.importorskip("torch")

    znorm = torch.linspace(-3.0, 3.0, 5001)
    regressor = types.SimpleNamespace(
        znorm_space_bardist_=types.SimpleNamespace(borders=znorm),
        y_train_mean_=1e4,
        y_train_std_=0.01,
    )
    output = {
        # What upstream's raw_space_bardist_ holds: the same, in float32.
        "criterion": types.SimpleNamespace(
            borders=(znorm.double() * 0.01 + 1e4).float()
        ),
        "logits": torch.zeros((2, 5000)),
    }
    borders, masses = tabpfn.bucket_masses(output, regressor)
    np.testing.assert_array_equal(borders, tabpfn._bucket_borders(regressor))
    assert np.all(np.diff(borders) > 0)
    assert masses.shape == (2, 5000)
    with pytest.warns(RuntimeWarning, match="zero width"):
        tabpfn.bucket_masses(output)


def _no_download(checkpoint, **kwargs):
    del checkpoint, kwargs  # Unused.
    raise AssertionError("the checkpoint was fetched")


def test_an_fp8_cache_is_refused_on_mps_before_the_checkpoint_loads(
    monkeypatch,
):
    """PyTorch has no float8 on MPS."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    est = fakes.install("tabpfn", monkeypatch)(device="mps", kv_cache="fp8")
    monkeypatch.setattr(_device, "resolve_device", lambda device: "mps")
    monkeypatch.setattr(_hub.Checkpoint, "download", _no_download)
    with pytest.raises(ValueError, match="kv_cache='fp8' is not available"):
        est.fit(X, X[:, 0])


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


@pytest.mark.checkpoint
def test_the_fitted_regressor_ran_the_pinned_recipe(small):
    X, y, _ = small
    est = _fit(X, y, version="v3")
    config = est.regressor_.inference_config_
    recipe = tabpfn._RECIPES["v3"]
    assert {k: _upstream_fields(config)[k] for k in recipe} == recipe
    assert est.regressor_.softmax_temperature_ == 0.9
    assert est.provenance_["softmax_temperature"] == 0.9
    (recorded,) = est.provenance_["inference_config"]
    assert recorded["PREPROCESS_TRANSFORMS"] == recipe["PREPROCESS_TRANSFORMS"]
    json.dumps(est.provenance_)


@pytest.mark.checkpoint
def test_the_fast_checkpoint_runs_on_a_cpu_without_the_warning(
    small, monkeypatch
):
    """TabPFN-3.5-fast is the TabPFN to use without a GPU."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    X, y, X_test = small
    with warnings.catch_warnings():
        warnings.simplefilter("error", lazy.PerformanceWarning)
        model = lazy.LazyModel(
            "tabpfn", version="v3.5-fast", progress=False
        ).fit(X, y)
    est = model.estimator_
    assert est.device_ == "cpu"
    assert (
        est.provenance_["filename"] == "tabpfn-v3.5-fast-20260909.safetensors"
    )
    # The shared defaults, not the checkpoint's own four members.
    assert est.regressor_.n_estimators_ == 8
    assert est.provenance_["softmax_temperature"] == 1.0
    assert est.provenance_["outlier_threshold"] == 12.0
    pdfs = model.predict_proba(X_test, lazy.datasets.DC1_GRID)
    assert pdfs.shape == (len(X_test), lazy.datasets.DC1_GRID.n_bins)
    assert np.all(np.isfinite(pdfs))


@pytest.mark.checkpoint
def test_an_explicit_temperature_reaches_upstream(small):
    X, y, _ = small
    est = _fit(X, y, version="v3.5", softmax_temperature=0.5)
    assert est.regressor_.softmax_temperature_ == 0.5
    assert est.provenance_["softmax_temperature"] == 0.5


@pytest.mark.checkpoint
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
    settings = _settings("v3.5", _group(), None)
    assert settings["MIN_UNIQUE_FOR_NUMERICAL_FEATURES"] == 1


@pytest.mark.checkpoint
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


@pytest.mark.checkpoint
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
    settings = _settings(version, _group("none"), None)
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


@pytest.mark.checkpoint
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


def test_only_the_identity_is_left_to_tabpfn():
    """Its other transforms differ from lazy's, so lazy applies its own."""
    assert tabpfn.TabPFNBarDistribution.native_transforms == {
        "none": ("none", False),
        "none+original": ("none", True),
    }


def _preprocessors(n_rows):
    steps = pytest.importorskip(
        "tabpfn.preprocessing.steps.reshape_feature_distribution_step"
    )
    return steps.get_all_reshape_feature_distribution_preprocessors(
        n_rows, random_state=0
    )


def _features(n_rows, *, missing=False):
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [rng.normal(size=n_rows), np.exp(rng.normal(size=n_rows))]
    )
    if missing:
        X[rng.random(n_rows) < 0.1, 0] = np.nan
    return X


def _lazy(name, X):
    spec = _transforms.TransformSpec(name)
    return _transforms.ScaffoldTransform(spec, seed=0).fit(X).transform(X)


def test_tabpfns_identity_is_lazys():
    X = _features(300, missing=True)
    upstream = _preprocessors(300)["none"].fit(X)
    np.testing.assert_array_equal(upstream.transform(X), _lazy("none", X))


@pytest.mark.parametrize(
    ("name", "upstream_name", "missing"),
    [
        ("power", "power", True),
        ("quantile", "quantile_norm", False),
        ("robust", "robust", False),
    ],
)
def test_tabpfns_other_transforms_are_not_lazys(name, upstream_name, missing):
    """Why they are scaffolded: each differs on a small problem already.

    TabPFN's power transform is lazy's (to 1e-15) on complete data, but
    fills a missing value with the column mean, where lazy leaves it to
    the model. Its quantile_uni is lazy's quantile_uniform below 100,000
    rows; above, it caps the quantiles at 20,000.
    """
    X = _features(300, missing=missing)
    theirs = _preprocessors(300)[upstream_name].fit(X).transform(X)
    ours = _lazy(name, X)
    assert not np.allclose(theirs, ours, equal_nan=True)
