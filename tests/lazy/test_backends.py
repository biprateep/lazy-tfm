# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Backend wiring that can be checked without running a foundation model.

Skipped wholesale when the optional backend is not installed, so the default
CI environment (core + dev only) stays fast.

One test here does run a backbone, because it backs a correctness claim that
cannot be reasoned about from the outside. It loads a multi-gigabyte checkpoint,
so it only runs when ``LAZY_RUN_CHECKPOINT_TESTS=1`` is set.
"""

import builtins
import collections
import inspect
import os
import pathlib
import subprocess
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import lazy
from lazy.models import _icl_stream
from lazy.models import _members
from lazy.models import _transforms
from lazy.models import tabfm
from lazy.models import tabicl
from lazy.models import tabpfn

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)


@pytest.fixture
def tiny():
    generator = np.random.default_rng(0)
    z = generator.uniform(0.2, 1.8, 40)
    X = pd.DataFrame(
        {"a": z + generator.normal(0, 0.1, 40), "b": generator.normal(size=40)}
    )
    return X, z


class TestTabFM:
    @pytest.fixture(autouse=True)
    def _needs_tabfm(self):
        pytest.importorskip("tabfm")

    def test_inference_path_is_resolved_at_fit(self, tiny):
        X, z = tiny
        est = lazy.get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2).fit(
            X, z
        )
        assert est.inference_ in ("stream", "predict_proba")

    def test_an_unknown_cache_mode_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="kv_cache must be one of"):
            lazy.get_estimator(
                "tabfm", n_coarse_bins=2, n_fine_bins=2, kv_cache="fast"
            ).fit(X, z)

    def test_the_cache_uses_the_streaming_decoder_when_it_exists(self, tiny):
        X, z = tiny
        est = lazy.get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2)
        if _icl_stream.streaming_available():
            assert est.fit(X, z).inference_ == "stream"
            assert est.kv_cache_ is True

    def test_falling_back_to_the_slow_path_warns(self, tiny, monkeypatch):
        """A silent 26x slowdown is the worst kind: both paths agree.

        Without the KV-cache API every chunk of query rows re-encodes the whole
        context, so the only symptom is a prediction that takes a day instead
        of an hour. The fallback must announce itself.
        """
        X, z = tiny
        monkeypatch.setattr(
            tabfm, "streaming_available", lambda: False, raising=False
        )
        monkeypatch.setattr(_icl_stream, "streaming_available", lambda: False)

        est = lazy.get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2)
        with pytest.warns(tabfm.TabFMPerformanceWarning, match="KV-cache"):
            est.fit(X, z)
        assert est.inference_ == "predict_proba"

    def test_choosing_the_slow_path_deliberately_does_not_warn(self, tiny):
        """`kv_cache=False` is a decision, not an accident."""
        X, z = tiny
        est = lazy.get_estimator(
            "tabfm", n_coarse_bins=2, n_fine_bins=2, kv_cache=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", tabfm.TabFMPerformanceWarning)
            est.fit(X, z)
        assert est.inference_ == "predict_proba"

    def test_fit_records_its_provenance_without_downloading(self, tiny):
        X, z = tiny
        est = lazy.get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2).fit(
            X, z
        )
        assert est.provenance_["backend"] == "tabfm"
        assert est.provenance_["version"] == "v1.0"
        assert est.provenance_["revision"]
        assert est.name_ == "tabfm:v1.0"

    def test_an_unknown_version_is_rejected_at_fit(self, tiny):
        """Not at the first prediction, an hour later."""
        X, z = tiny
        with pytest.raises(ValueError, match="unknown version"):
            lazy.get_estimator(
                "tabfm", version="v9", n_coarse_bins=2, n_fine_bins=2
            ).fit(X, z)

    def test_more_than_ten_classes_per_level_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="at most 10 classes"):
            lazy.get_estimator("tabfm", n_coarse_bins=11).fit(X, z)

    def test_context_smaller_than_explicit_bin_counts_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="fewer than the"):
            lazy.get_estimator("tabfm", n_coarse_bins=10, n_fine_bins=10).fit(
                X.iloc[:10], z[:10]
            )

    def test_a_small_context_fits_under_auto_bins(self, tiny):
        X, z = tiny
        est = lazy.get_estimator("tabfm").fit(X.iloc[:10], z[:10])
        assert est.provenance_["bins"] == "auto equal-mass 2x2"

    @pytest.mark.parametrize("name", ["member_batch_size", "query_block_rows"])
    @pytest.mark.parametrize("value", [0, -1, 2.0, True, None])
    def test_memory_knobs_must_be_positive_ints(self, tiny, name, value):
        X, z = tiny
        with pytest.raises(ValueError, match=name):
            lazy.get_estimator(
                "tabfm", n_coarse_bins=2, n_fine_bins=2, **{name: value}
            ).fit(X, z)

    def test_chunk_size_is_the_only_query_chunking_knob(self):
        assert "decode_chunk_rows" not in tabfm.TabFMHistogram().get_params()

    @pytest.mark.parametrize(
        ("chunk_size", "block_rows", "passes"),
        [(7, 20, [7, 7, 7, 7, 7, 5]), (7, 3, [7] * 5 + [5]), (0, 20, [40])],
    )
    def test_the_cached_path_decodes_chunk_size_rows_a_pass(
        self, chunk_size, block_rows, passes
    ):
        """Blocks of views hold whole chunks; chunk_size=0 is one pass."""
        chunk, block = _icl_stream._pass_sizes(40, chunk_size, block_rows)
        sizes = [
            min(r + chunk, min(b + block, 40) - b) - r
            for b in range(0, 40, block)
            for r in range(0, min(b + block, 40) - b, chunk)
        ]
        assert sizes == passes

    def test_the_cached_path_is_handed_chunk_size(self, monkeypatch):
        seen = {}

        class Fitted:
            classes_ = np.array([0, 1])

            def fit(self, X, y):
                return self

        def logits(classifier, model, targets, **kwargs):
            seen.update(kwargs)
            return {
                "query": {"mean_logits": np.zeros((len(targets["query"]), 2))}
            }

        monkeypatch.setattr(_icl_stream, "classification_logits", logits)
        est = tabfm.TabFMHistogram(
            chunk_size=7, query_block_rows=21, member_batch_size=3
        )
        est.inference_ = "stream"
        est.softmax_temperature_ = 0.9
        monkeypatch.setattr(
            est, "_classifier", lambda model, group, seed: Fitted()
        )
        X_query = pd.DataFrame({"a": np.linspace(0.0, 1.0, 10)})
        est._class_probabilities(
            None,
            None,
            X_query.iloc[:4],
            np.array([0, 1, 0, 1]),
            2,
            X_query,
            0,
        )
        assert seen["chunk_size"] == 7
        assert seen["query_block_rows"] == 21
        assert seen["member_batch_size"] == 3

    def test_bagged_members_keep_lazys_bags_and_their_rows(self, tiny):
        """The plan's rows are the rows TabFM's members are given."""
        X, z = tiny
        est = tabfm.TabFMHistogram(
            n_coarse_bins=2, n_fine_bins=2, bag_size=30, n_estimators=3
        ).fit(X, z)
        assert est.provenance_["bag_rows"] == 30
        assert "bag_rows_drawn_by" not in est.provenance_
        (group,) = est.member_groups_
        bags = _members.draw_bags(3, 30, len(z), 0)
        for rows, bag in zip(group.member_rows, bags, strict=True):
            np.testing.assert_array_equal(rows, bag)

    def test_the_slow_path_hands_upstream_bounded_chunks(self, monkeypatch):
        """`chunk_size` must reach `predict_proba`, not just the docstring."""
        sizes = []

        class Recording:
            classes_ = np.array([0, 1])

            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                sizes.append(len(X))
                return np.column_stack([X["a"], 1.0 - X["a"]])

        est = tabfm.TabFMHistogram(chunk_size=7)
        est.inference_ = "predict_proba"
        monkeypatch.setattr(
            est, "_classifier", lambda model, group, seed: Recording()
        )
        X_query = pd.DataFrame(
            {"a": np.linspace(0.0, 1.0, 40)}, index=np.arange(100, 140)
        )

        probs = est._class_probabilities(
            None,
            None,
            X_query.iloc[:4],
            np.array([0, 1, 0, 1]),
            2,
            X_query,
            seed=0,
        )

        assert sizes == [7, 7, 7, 7, 7, 5]
        assert np.array_equal(probs[:, 0], X_query["a"].to_numpy())

    @pytest.mark.parametrize(
        "z",
        [
            np.r_[np.zeros(100), np.linspace(0.01, 1.0, 100)],
            np.arange(300) % 5.0,
            np.full(200, 0.5),
            np.full(200, 1e12),
        ],
        ids=["zero-inflated", "integer-valued", "constant", "constant-large"],
    )
    def test_tied_targets_give_valid_bins(self, z, monkeypatch):
        """Ties collapse quantiles into empty bins, which must stay valid."""
        est = tabfm.TabFMHistogram(n_coarse_bins=10, n_fine_bins=10)
        est.support_ = (float(z.min()), float(z.max()))
        est.y_context_ = z
        est._resolve_bins(z.size)
        edges, coarse, fine = est._edges(z, 0.0)
        assert edges.size == 101
        assert np.all(np.diff(edges) >= 0)
        assert all(f.size == 11 and np.all(np.diff(f) >= 0) for f in fine)
        # Every target falls in a bin of non-zero width.
        bins = tabfm._bin_labels(edges, z)
        assert np.all(np.diff(edges)[bins] > 0)
        _stub_classifier(est, monkeypatch)
        handle = {"X": tabfm._frame(z[:, None]), "y": z}
        handle["group"] = types.SimpleNamespace(seed=0, member_rows=None)
        probs, edges, prior = est._hierarchy(
            None, handle, tabfm._frame(z[:7, None]), 0.0
        )
        assert probs.shape == (7, 100)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs[:, np.diff(edges) == 0] == 0.0)
        lazy.distributions.HistogramDistribution(edges, probs)

    def test_targets_outside_the_constructor_grid_are_clipped(self):
        """A redshift grid under a metallicity target must not crash fit."""
        generator = np.random.default_rng(0)
        X = generator.normal(size=(300, 2))
        z = -1.0 + 0.3 * X[:, 0]
        est = tabfm.TabFMHistogram(
            n_estimators=1, y_grid=np.linspace(0.0, 3.0, 301)
        )
        with pytest.warns(UserWarning, match="outside the y_grid range"):
            est.fit(X, z)
        assert est.y_context_.min() == est.support_[0] == -0.005
        edges, _, _ = est._edges(est.y_context_, 0.0)
        assert edges[0] == -0.005 and np.all(np.diff(edges) >= 0)

    @needs_checkpoint
    @pytest.mark.parametrize("kind", ["zero-inflated", "integer", "outside"])
    def test_awkward_targets_predict_valid_densities(self, kind):
        generator = np.random.default_rng(0)
        X = generator.normal(size=(130, 2))
        options = {}
        if kind == "zero-inflated":
            z = np.where(X[:, 0] < 0.3, 0.0, np.abs(X[:, 1]))
        elif kind == "integer":
            z = np.clip(np.round(X[:, 0] + 2.0), 0.0, 4.0)
        else:
            z = -1.0 + 0.3 * X[:, 0]
            options["y_grid"] = np.linspace(0.0, 3.0, 31)
        est = tabfm.TabFMHistogram(
            n_coarse_bins=4,
            n_fine_bins=4,
            n_estimators=1,
            device="cpu",
            progress=False,
            **options,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est.fit(X[:120], z[:120])
        dist = est.predict_distribution(X[120:])
        masses = dist.masses
        assert np.isfinite(masses).all()
        np.testing.assert_allclose(masses.sum(axis=1), 1.0)

    @needs_checkpoint
    @pytest.mark.parametrize("bins", ["auto", 10])
    def test_cuda_rounding_stays_within_the_measured_bounds(self, bins):
        """bfloat16 on CUDA: neither chunking nor the cache is exact there.

        Each path rounds differently, and the differences are those between
        any two bfloat16 runs (each as far from a float32 run): about 3% of
        a row's peak for a typical row, but up to a quarter of the peak in
        single bins, at any bin count. Measured over 8 seeds at 5 x 5, 7 x 7
        and 10 x 10: a median row's largest difference of at most 4.5% of
        its peak, and at most 26% of the peak anywhere. In float32 the paths
        agree to float rounding.
        """
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        generator = np.random.default_rng(0)
        X = generator.normal(size=(360, 3))
        z = -1.0 + 0.3 * (X[:, 0] + 0.3 * generator.normal(size=360))
        grid = lazy.Grid.linear(-2.5, 0.5, 301)

        def run(**options):
            est = tabfm.TabFMHistogram(
                n_estimators=1,
                device="cuda",
                progress=False,
                n_coarse_bins=bins,
                n_fine_bins=bins,
                **options,
            ).fit(X[:300], z[:300])
            return est.predict_proba(X[300:], grid)

        assert not tabfm.TabFMHistogram.exact_chunking
        for precision, typical, worst in (
            (True, 0.06, 0.35),
            (False, 1e-4, 1e-4),
        ):
            whole = run(kv_cache=False, chunk_size=0, mixed_precision=precision)
            for options in (
                {"kv_cache": False, "chunk_size": 7},
                {"kv_cache": True},
            ):
                other = run(mixed_precision=precision, **options)
                difference = np.abs(other - whole)
                rows = difference.max(axis=1) / whole.max(axis=1)
                assert np.median(rows) <= typical
                assert difference.max() <= worst * whole.max()

    def test_mixed_precision_picks_the_weights_dtype(self, monkeypatch):
        """bfloat16 only under mixed precision on CUDA; float32 otherwise."""
        upstream = pytest.importorskip("tabfm")
        torch = pytest.importorskip("torch")

        loads = []

        def load(**kwargs):
            loads.append(kwargs["dtype"])
            return object()

        monkeypatch.setattr(upstream.tabfm_v1_0_0_pytorch, "load", load)
        checkpoint = types.SimpleNamespace(download=lambda: "/nowhere")
        monkeypatch.setattr(
            tabfm._hub, "get_checkpoint", lambda *args: checkpoint
        )
        # What fit resolves on a CPU, set by hand: the hub is stubbed.
        est = tabfm.TabFMHistogram()
        est.device_ = "cpu"
        est.mixed_precision_ = False
        first = est._backbone()
        assert loads == [None]
        assert est._backbone() is first  # cached
        est.mixed_precision_ = True  # what fit resolves on CUDA
        est._backbone()
        assert loads == [None, torch.bfloat16]

    def test_a_cpu_fit_never_uses_mixed_precision(self, tiny):
        X, z = tiny
        est = tabfm.TabFMHistogram(
            n_coarse_bins=2, n_fine_bins=2, device="cpu"
        ).fit(X, z)
        assert est.mixed_precision_ is False
        assert est.provenance_["mixed_precision"] is False

    def test_a_constant_target_fits_with_a_valid_native_grid(self):
        generator = np.random.default_rng(0)
        X = generator.normal(size=(200, 2))
        est = tabfm.TabFMHistogram(n_estimators=1).fit(X, np.full(200, 0.5))
        assert est.native_grid_.y_min == 0.5
        assert 0.5 < est.native_grid_.y_max < 0.5 + 1e-5

    def test_repeated_transforms_keep_their_weights(self, tiny):
        """Upstream cycles norm_methods, so repeats in it are members."""
        X, z = tiny
        est = lazy.get_estimator(
            "tabfm",
            n_coarse_bins=2,
            n_fine_bins=2,
            n_estimators=6,
            transforms=("power", "power", "none"),
        ).fit(X, z)
        classifier = est._classifier(
            model=None, group=est.member_groups_[0], seed=0
        )
        assert (
            classifier.get_params()["norm_methods"]
            == [
                "power",
                "power",
                "none",
            ]
            * 2
        )

    def test_the_classifier_only_receives_supported_keywords(self, tiny):
        """Cache knobs exist on repository builds, not on the PyPI release."""
        upstream = pytest.importorskip("tabfm")
        X, z = tiny
        est = lazy.get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2).fit(
            X, z
        )
        accepted = set(
            inspect.signature(upstream.TabFMClassifier.__init__).parameters
        )
        # _classifier needs a model object only to hand on; None is never
        # touched.
        classifier = est._classifier(
            model=None, group=est.member_groups_[0], seed=0
        )
        assert set(classifier.get_params()) <= accepted

    def test_no_classifier_argument_is_left_to_upstream(
        self, tiny, monkeypatch
    ):
        """A default the installed tabfm picks could change an answer."""
        upstream = pytest.importorskip("tabfm")
        X, z = tiny
        est = lazy.get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2).fit(
            X, z
        )
        accepted = set(
            inspect.signature(upstream.TabFMClassifier.__init__).parameters
        )
        assert set(_classifier_kwargs(est, monkeypatch)) == accepted - {"self"}

    def test_the_auto_recipe_is_tabfms_own_written_out(self, tiny, monkeypatch):
        """Pinned in lazy, and today equal to what upstream defaults to."""
        upstream = pytest.importorskip("tabfm")
        X, z = tiny
        est = lazy.get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2).fit(
            X, z
        )
        seen = _classifier_kwargs(est, monkeypatch)
        assert seen["norm_methods"] == ["none", "power"]
        assert seen["feat_shuffle_method"] == "random"
        assert seen["class_shift"] is True
        assert seen["average_logits"] is True
        assert seen["outlier_threshold"] == 4.0
        assert seen["max_num_features"] == 500
        assert seen["softmax_temperature"] == 0.9
        assert seen["binary_calibration_method"] is None
        assert seen["multiclass_calibration_method"] is None
        assert seen["enable_nnls"] is False
        assert seen["n_feature_crosses"] == seen["n_svd_features"] == 0
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(
                upstream.TabFMClassifier.__init__
            ).parameters.items()
        }
        recipe = set(tabfm._PINNED_CLASSIFIER_ARGS) | {
            "feat_shuffle_method",
            "outlier_threshold",
            "max_num_features",
            "softmax_temperature",
        }
        assert {k: seen[k] for k in recipe} == {k: defaults[k] for k in recipe}
        assert defaults["norm_methods"] is None  # upstream's ["none", "power"]

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            ({}, 4.0),
            ({"outlier_threshold": None}, np.inf),
            ({"outlier_threshold": 2.5}, 2.5),
            ({"transforms": "none"}, np.inf),
            ({"transforms": "none", "outlier_threshold": 3.0}, 3.0),
        ],
    )
    def test_outlier_threshold_drives_tabfms_own_clip(
        self, tiny, monkeypatch, params, expected
    ):
        """Native, never scaffolded on top; None turns TabFM's clip off."""
        X, z = tiny
        est = lazy.get_estimator(
            "tabfm", n_coarse_bins=2, n_fine_bins=2, **params
        ).fit(X, z)
        assert est.clippers_ == [None] * len(est.member_groups_)
        seen = _classifier_kwargs(est, monkeypatch)
        assert seen["outlier_threshold"] == expected
        assert est.provenance_["outlier_threshold"] == (
            None if expected == np.inf else expected
        )

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            ({}, 500),
            ({"feature_shuffle": False}, None),
            ({"transforms": "none"}, None),
        ],
    )
    def test_only_the_auto_recipe_subsamples_features(
        self, tiny, monkeypatch, params, expected
    ):
        X, z = tiny
        est = lazy.get_estimator(
            "tabfm", n_coarse_bins=2, n_fine_bins=2, **params
        ).fit(X, z)
        seen = _classifier_kwargs(est, monkeypatch)
        assert seen["max_num_features"] == expected

    def test_without_feature_shuffle_wide_tables_keep_their_order(self):
        """Above 500 columns upstream subsamples, and shuffles as it does."""
        upstream = pytest.importorskip("tabfm.src.classifier_and_regressor")
        generator = np.random.default_rng(0)
        X = generator.normal(size=(40, 520))
        y = generator.integers(0, 3, 40)

        def patterns(max_num_features):
            ensemble = upstream.EnsembleGenerator(
                n_estimators=3,
                norm_methods=["none"],
                feat_shuffle_method="none",
                max_num_features=max_num_features,
                random_state=0,
            ).fit(X, y)
            return [c[0] for c in ensemble.ensemble_configs_["none"]]

        assert not any(np.array_equal(p, np.arange(520)) for p in patterns(500))
        assert all(np.array_equal(p, np.arange(520)) for p in patterns(None))

    @pytest.mark.parametrize("name", ["none", "power"])
    def test_native_norm_methods_are_lazys_transforms(self, name):
        """A name maps onto TabFM's own method only if that is the same."""
        upstream = pytest.importorskip("tabfm.src.classifier_and_regressor")
        generator = np.random.default_rng(0)
        X = generator.lognormal(size=(300, 3))
        assert tabfm.TabFMHistogram.native_transforms[name] == name
        pipeline = upstream.PreprocessingPipeline(name, np.inf, 0).fit(X)
        # TabFM standardises first; the method itself sees what follows.
        scaled = pipeline.standard_scaler_.transform(X)
        native = (
            scaled
            if pipeline.normalizer_ is None
            else pipeline.normalizer_.transform(scaled)
        )
        ours = _transforms.ScaffoldTransform(_transforms.parse(name)[0], 0)
        np.testing.assert_allclose(
            native, ours.fit(scaled).transform(scaled), rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize("name", ["quantile", "quantile_rtdl", "robust"])
    def test_other_transforms_are_scaffolded(self, tiny, monkeypatch, name):
        X, z = tiny
        est = lazy.get_estimator(
            "tabfm", n_coarse_bins=2, n_fine_bins=2, transforms=name
        ).fit(X, z)
        assert est.member_groups_[0].scaffold.name == name
        assert est.transformers_[0] is not None
        seen = _classifier_kwargs(est, monkeypatch)
        assert seen["norm_methods"] == ["none"] * est.n_estimators

    def test_an_infinite_threshold_is_tabfms_clip_turned_off(self):
        """The off switch is exact: the clip becomes the identity."""
        upstream = pytest.importorskip("tabfm.src.classifier_and_regressor")
        generator = np.random.default_rng(0)
        X = generator.lognormal(size=(200, 3))
        X[0, 0] = 1e4
        clip = upstream.OutlierRemover(threshold=np.inf).fit(X)
        assert np.array_equal(clip.transform(X), X)
        # And a finite one is lazy's SoftClip, the uniform definition.
        np.testing.assert_array_equal(
            upstream.OutlierRemover(threshold=4.0).fit(X).transform(X),
            _transforms.SoftClip(4.0).fit(X).transform(X),
        )


def _classifier_kwargs(est, monkeypatch, group=None):
    """The keywords ``est`` hands TabFMClassifier, through a recording fake.

    The fake keeps the real signature, so the build's optional cache
    keywords are offered exactly as to the real class.
    """
    upstream = pytest.importorskip("tabfm")
    seen = {}

    class Recording:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    Recording.__init__.__signature__ = inspect.signature(
        upstream.TabFMClassifier.__init__
    )
    monkeypatch.setattr(upstream, "TabFMClassifier", Recording)
    est._classifier(
        model=None,
        group=est.member_groups_[0] if group is None else group,
        seed=0,
    )
    return seen


def _stub_classifier(est, monkeypatch):
    """Replaces TabFM's classifier by one returning its context's class mix."""

    class Frequencies:
        def fit(self, X, y):
            self.classes_, counts = np.unique(y, return_counts=True)
            self.frequencies_ = counts / counts.sum()
            return self

        def predict_proba(self, X):
            return np.tile(self.frequencies_, (len(X), 1))

    est.inference_ = "predict_proba"
    monkeypatch.setattr(
        est, "_classifier", lambda model, group, seed: Frequencies()
    )


class _MemberRegressor:
    """TabICLRegressor without a backbone: upstream's real member plan.

    Its "quantiles" are the share of its members that use the power
    transform, so a weighted average of regressors shows the ensemble's mix.
    """

    created: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.random_state = kwargs["random_state"]
        _MemberRegressor.created.append(self)

    def fit(self, X, y):
        preprocessing = pytest.importorskip("tabicl._sklearn.preprocessing")
        self.X_dtype = X.dtype
        self.ensemble_generator_ = preprocessing.EnsembleGenerator(
            classification=False,
            n_estimators=self.kwargs["n_estimators"],
            norm_methods=self.kwargs["norm_methods"],
            feat_shuffle_method=self.kwargs["feat_shuffle_method"],
            outlier_threshold=self.kwargs["outlier_threshold"],
            random_state=self.kwargs["random_state"],
        ).fit(X, y)
        return self

    def predict(self, X, output_type):
        assert output_type == "raw_quantiles"
        configs = self.ensemble_generator_.ensemble_configs_
        power = len(configs.get("power", []))
        share = power / sum(len(c) for c in configs.values())
        return np.zeros((len(X), 1)) + share + np.array([-1.0, 0.0, 1.0])


class TestTabICL:
    @pytest.fixture(autouse=True)
    def _needs_tabicl(self):
        pytest.importorskip("tabicl")

    @pytest.fixture
    def members(self, monkeypatch):
        """TabICLQuantile on _MemberRegressor, with no checkpoint to load."""
        upstream = pytest.importorskip("tabicl")
        monkeypatch.setattr(upstream, "TabICLRegressor", _MemberRegressor)
        monkeypatch.setattr(_MemberRegressor, "created", [])

        def load(est):
            est.checkpoint_ = pathlib.Path("tabicl-regressor.ckpt")
            est.provenance_ = {}

        monkeypatch.setattr(tabicl.TabICLQuantile, "_load_checkpoint", load)

        def fit(n_features=5, **params):
            generator = np.random.default_rng(0)
            X = generator.normal(size=(60, n_features))
            z = generator.uniform(0.2, 1.8, 60)
            est = tabicl.TabICLQuantile(device="cpu", progress=False, **params)
            return est.fit(X, z), X

        return fit

    def _power_share(self, est, X):
        offset, scale = est.handles_[0].offset, est.handles_[0].scale
        median = est.predict_distribution(X[:2]).locs[:, 1]
        return (median - offset) / scale

    @pytest.mark.parametrize(
        ("n_estimators", "n_features", "feature_shuffle", "weights"),
        [
            (8, 5, True, (8,)),  # 5 Latin-square shuffles cover 8 members
            (8, 3, True, (6, 2)),  # 3 shuffles x 2 methods, then 2 more
            (13, 3, True, (6, 6, 1)),
            (1, 3, True, (1,)),
            (8, 3, False, (4, 4)),  # one shuffle: each method once, weighted
            (3, 3, False, (2, 1)),
            (2, 3, False, (2,)),  # one of each fits one regressor
            (5, 1, True, (3, 2)),  # one column: one shuffle
        ],
    )
    def test_exactly_n_estimators_members_run(
        self, members, n_estimators, n_features, feature_shuffle, weights
    ):
        """Upstream alone ran min(n, shuffles x methods) of them."""
        est, X = members(
            n_features=n_features,
            n_estimators=n_estimators,
            feature_shuffle=feature_shuffle,
        )
        (handle,) = est.handles_
        assert handle.weights == weights
        n_power = n_estimators // 2
        assert handle.members_by_method() == {
            "none": n_estimators - n_power,
            **({"power": n_power} if n_power else {}),
        }
        np.testing.assert_allclose(
            self._power_share(est, X), n_power / n_estimators
        )
        (group,) = est.provenance_["tabicl"]["groups"]
        ran = [r["members"] for r in group["regressors"]]
        assert sum(sum(counts.values()) for counts in ran) == n_estimators

    def test_extra_regressors_have_seeds_of_their_own(self, members):
        est, _ = members(n_estimators=13, n_features=3)
        seeds = [r.random_state for r in est.handles_[0].regressors]
        assert seeds[0] == _members.member_seed(0, 0) and len(set(seeds)) == 3
        assert all(0 <= seed < 2**31 for seed in seeds)
        assert seeds == [
            r["random_state"]
            for r in est.provenance_["tabicl"]["groups"][0]["regressors"]
        ]

    def test_one_regressor_is_exposed_as_regressor_(self, members):
        est, _ = members(n_estimators=4, n_features=4)
        assert est.regressor_ is est.handles_[0].regressors[0]
        est, _ = members(n_estimators=8, n_features=3)
        assert not hasattr(est, "regressor_")

    def test_no_regressor_argument_is_left_to_upstream(self, members):
        """A default the installed tabicl picks could change an answer."""
        upstream = pytest.importorskip("tabicl._sklearn.regressor")
        accepted = set(
            inspect.signature(upstream.TabICLRegressor.__init__).parameters
        ) - {"self"}
        members(n_estimators=8, n_features=3)
        for regressor in _MemberRegressor.created:
            assert set(regressor.kwargs) == accepted

    def test_the_auto_recipe_is_passed_explicitly(self, members):
        members()
        (regressor,) = _MemberRegressor.created
        assert regressor.kwargs["norm_methods"] == ["none", "power"]
        assert regressor.kwargs["feat_shuffle_method"] == "latin"
        assert regressor.kwargs["outlier_threshold"] == 4.0
        assert regressor.kwargs["use_fa3"] is False
        assert regressor.kwargs["allow_auto_download"] is False
        assert tabicl.TabICLQuantile.auto_tokens == tabicl.AUTO_NORM_METHODS

    @pytest.mark.parametrize(
        ("params", "threshold"),
        [
            ({}, 4.0),
            ({"outlier_threshold": 2.5}, 2.5),
            ({"outlier_threshold": None}, np.inf),
            ({"transforms": "none"}, np.inf),
            ({"transforms": "power", "outlier_threshold": 3.0}, 3.0),
        ],
    )
    def test_outlier_threshold_is_passed_to_upstream(
        self, members, params, threshold
    ):
        est, _ = members(**params)
        for regressor in _MemberRegressor.created:
            assert regressor.kwargs["outlier_threshold"] == threshold
        assert est.clippers_ == [None] * len(est.handles_)  # never twice
        expected = None if threshold == np.inf else threshold
        assert est.provenance_["outlier_threshold"] == expected
        assert est.provenance_["tabicl"]["outlier_threshold"] == expected

    def test_an_infinite_threshold_leaves_every_value_unclipped(self):
        """What the model sees is then the z-score and nothing else."""
        preprocessing = pytest.importorskip("tabicl._sklearn.preprocessing")
        generator = np.random.default_rng(0)
        X = generator.normal(size=(200, 3))
        X[5, 0] = 1e3
        queries = np.array([[50.0, -40.0, 0.0]])
        off = preprocessing.PreprocessingPipeline("none", np.inf, 0).fit(X)
        on = preprocessing.PreprocessingPipeline("none", 4.0, 0).fit(X)
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
        np.testing.assert_array_equal(
            off.X_transformed_, np.clip((X - mean) / std, -100, 100)
        )
        np.testing.assert_array_equal(
            off.transform(queries), (queries - mean) / std
        )
        assert on.X_transformed_[5, 0] < 0.5 * off.X_transformed_[5, 0]

    @pytest.mark.parametrize("mixed_precision", [True, False])
    def test_a_cpu_runs_in_float32(self, members, mixed_precision):
        est, _ = members(mixed_precision=mixed_precision)
        (regressor,) = _MemberRegressor.created
        assert regressor.kwargs["use_amp"] is False
        assert est.provenance_["tabicl"]["use_amp"] is False

    def test_mixed_precision_on_cuda_is_always_on(self, monkeypatch):
        """Upstream's own "auto" waits for 1,024 rows or 60 features."""
        est = tabicl.TabICLQuantile()
        est.checkpoint_ = pathlib.Path("tabicl-regressor.ckpt")
        est.device_ = "cuda"
        est.outlier_threshold_ = 4.0
        for flag in (True, False):
            est.mixed_precision_ = flag
            regressor = est._regressor(2, ["none", "power"], "latin", 0)
            assert regressor.use_amp is flag
            assert regressor.use_fa3 is False

    def test_a_softmax_temperature_is_refused(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="softmax_temperature must be"):
            tabicl.TabICLQuantile(softmax_temperature=0.9).fit(X, z)

    def test_features_reach_upstream_in_float64(self, members):
        members()
        (regressor,) = _MemberRegressor.created
        assert regressor.X_dtype == np.float64

    def test_members_are_averaged_in_a_fixed_order(self, monkeypatch):
        """Upstream orders them by a set, which PYTHONHASHSEED reorders."""
        preprocessing = pytest.importorskip("tabicl._sklearn.preprocessing")
        generator = np.random.default_rng(0)
        X, y = generator.normal(size=(40, 3)), generator.normal(size=40)
        regressor = types.SimpleNamespace(
            ensemble_generator_=preprocessing.EnsembleGenerator(
                classification=False,
                n_estimators=4,
                norm_methods=["none", "power"],
                random_state=0,
            ).fit(X, y)
        )
        configs = dict(regressor.ensemble_generator_.ensemble_configs_)
        regressor.ensemble_generator_.ensemble_configs_ = (
            collections.OrderedDict(reversed(configs.items()))
        )
        tabicl._order_members(regressor, ["none", "power"])
        ordered = regressor.ensemble_generator_.ensemble_configs_
        assert list(ordered) == ["none", "power"]
        assert dict(ordered) == configs

    def test_only_none_is_tabicls_own_transform(self):
        assert tabicl.TabICLQuantile.native_transforms == {"none": "none"}

    @pytest.mark.filterwarnings("ignore:n_quantiles")
    @pytest.mark.parametrize(
        "name", ["power", "quantile", "quantile_rtdl", "robust"]
    )
    def test_tabicls_norm_methods_are_not_the_uniform_transforms(self, name):
        """Why lazy applies them: the model would see something else."""
        preprocessing = pytest.importorskip("tabicl._sklearn.preprocessing")
        generator = np.random.default_rng(1)
        X = generator.normal(size=(300, 3))
        X[:, 1] = np.exp(X[:, 1])
        X[:, 2] = generator.standard_t(3, size=300)
        spec = _transforms.parse(name)[0]
        scaffold = _transforms.ScaffoldTransform(spec, 7).fit(X)
        ours = (
            preprocessing.PreprocessingPipeline("none", np.inf, 7)
            .fit(scaffold.transform(X))
            .X_transformed_
        )
        theirs = (
            preprocessing.PreprocessingPipeline(name, np.inf, 7)
            .fit(X)
            .X_transformed_
        )
        assert np.abs(ours - theirs).max() > 1e-6

    def test_none_is_tabicls_none(self):
        preprocessing = pytest.importorskip("tabicl._sklearn.preprocessing")
        X = np.random.default_rng(1).lognormal(size=(300, 3))
        scaffold = _transforms.ScaffoldTransform(
            _transforms.parse("none")[0], 7
        ).fit(X)
        np.testing.assert_array_equal(
            preprocessing.PreprocessingPipeline("none", np.inf, 7)
            .fit(scaffold.transform(X))
            .X_transformed_,
            preprocessing.PreprocessingPipeline("none", np.inf, 7)
            .fit(X)
            .X_transformed_,
        )

    @needs_checkpoint
    def test_exactly_n_members_run_on_the_backbone(self):
        generator = np.random.default_rng(0)
        X = generator.normal(size=(120, 3))
        z = 1.0 + 0.1 * X[:, 0] + 0.02 * generator.normal(size=120)
        est = tabicl.TabICLQuantile(device="cpu", progress=False).fit(X, z)
        (handle,) = est.handles_
        assert handle.weights == (6, 2)
        assert [
            sum(
                len(c) for c in r.ensemble_generator_.ensemble_configs_.values()
            )
            for r in handle.regressors
        ] == [6, 2]
        assert np.isfinite(est.predict_distribution(X[:5]).locs).all()

    @needs_checkpoint
    def test_an_explicit_recipe_feeds_the_outlier_unclipped(self):
        generator = np.random.default_rng(0)
        X = generator.normal(size=(120, 3))
        X[7, 0] = 40.0
        z = 1.0 + 0.1 * X[:, 1] + 0.02 * generator.normal(size=120)

        def seen(**params):
            est = tabicl.TabICLQuantile(
                n_estimators=2, device="cpu", progress=False, **params
            ).fit(X, z)
            generator = est.regressor_.ensemble_generator_
            assert generator.X_.dtype == np.float64
            return generator.preprocessors_["none"].X_transformed_[7, 0]

        zscore = (40.0 - X[:, 0].mean()) / (X[:, 0].std() + 1e-6)
        assert seen(transforms="none") == pytest.approx(zscore, rel=1e-12)
        assert seen(outlier_threshold=None) == pytest.approx(zscore, rel=1e-12)
        assert seen() < 0.5 * zscore  # the auto recipe clips it at 4 sigma

    @needs_checkpoint
    def test_the_answer_does_not_depend_on_string_hashing(self, tmp_path):
        """Upstream alone differs between processes in the last digits."""
        script = (
            "import sys, numpy as np\n"
            "from lazy.models import tabicl\n"
            "g = np.random.default_rng(0)\n"
            "X = g.normal(size=(120, 5)); z = X[:, 0] + g.normal(size=120)\n"
            "est = tabicl.TabICLQuantile(device='cpu', progress=False)\n"
            "locs = est.fit(X, z).predict_distribution(X[:9]).locs\n"
            "np.save(sys.argv[1], locs)\n"
        )
        answers = []
        for seed in ("0", "1"):
            path = tmp_path / f"{seed}.npy"
            env = {**os.environ, "PYTHONHASHSEED": seed}
            subprocess.run(
                [sys.executable, "-c", script, str(path)], check=True, env=env
            )
            answers.append(np.load(path))
        np.testing.assert_array_equal(*answers)

    def test_negative_chunk_size_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="chunk_size"):
            lazy.get_estimator("tabicl", chunk_size=-1).fit(X, z)

    @needs_checkpoint
    def test_chunking_the_query_rows_is_bit_identical(self):
        """Same claim as for TabFM, and the reason chunking is the default."""
        generator = np.random.default_rng(0)
        z = generator.uniform(0.2, 1.8, 260)
        X = pd.DataFrame(
            {
                "a": z + generator.normal(0, 0.1, 260),
                "b": generator.normal(size=260),
                "c": z**2,
            }
        )
        X_ctx, X_q = X.iloc[:200], X.iloc[200:].reset_index(drop=True)

        def run(chunk_size):
            model = lazy.LazyModel(
                "tabicl", n_estimators=2, device="cpu", chunk_size=chunk_size
            )
            return model.fit(X_ctx, z[:200]).predict_proba(X_q)

        assert np.array_equal(run(0), run(7))

    def test_the_native_grid_leaves_room_to_extrapolate(self):
        """Padded by a quarter of the range, at the same bin width."""
        est = tabicl.TabICLQuantile()
        est._support = (0.05, 1.65)
        grid = est._native_grid()
        assert grid.y_min == pytest.approx(-0.35)
        assert grid.y_max == pytest.approx(2.05)
        assert grid.n_bins == 1500
        np.testing.assert_allclose(grid.widths, 1.6 / 1000)

    @pytest.mark.parametrize("value", [0.0, 0.5, -1e4])
    def test_a_constant_target_is_padded_on_its_own_scale(self, value):
        est = tabicl.TabICLQuantile()
        est._support = (value, value)
        grid = est._native_grid()
        pad = 0.01 * max(abs(value), 1.0)
        assert grid.y_min == pytest.approx(value - pad)
        assert grid.y_max == pytest.approx(value + pad)

    def test_mass_outside_the_native_grid_warns(self, monkeypatch):
        levels = tabicl.quantile_levels(99)
        inside = np.linspace(0.1, 0.9, 99)
        leaking = inside - 0.5  # 3/16 of it below the grid, at -0.25.
        dist = lazy.distributions.QuantileDistribution(
            levels, np.stack([inside, leaking])
        )
        est = tabicl.TabICLQuantile()
        est._support = (0.0, 1.0)
        est.native_grid_ = est._native_grid()
        outside = tabicl._mass_outside(dist, est.native_grid_)
        assert outside[0] == 0.0
        assert outside[1] == pytest.approx(3 / 16, abs=0.011)

        monkeypatch.setattr(est, "_chunks", lambda X: iter([dist]))
        X = pd.DataFrame({"a": [0.0, 1.0]})
        with pytest.warns(UserWarning, match="1 of 2 rows put more than 1%"):
            est._predict_pdf(X, est.native_grid_)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            est._predict_pdf(X, lazy.Grid.linear(-1.0, 1.0, 10))

    def test_targets_are_standardised_in_float64(self):
        """float32 cannot resolve 0.01 of spread about 1e4; float64 can."""
        y = 1e4 + 0.01 * np.random.default_rng(0).normal(size=100)
        offset, scale = tabicl._target_scaling(y)
        standardised = ((y - offset) / scale).astype(np.float32)
        assert np.unique(standardised).size == y.size
        np.testing.assert_allclose(
            standardised.astype(np.float64) * scale + offset, y, atol=1e-8
        )
        assert tabicl._target_scaling(np.full(5, 3.0)) == (3.0, 1.0)

    @needs_checkpoint
    def test_a_large_offset_keeps_its_resolution(self):
        generator = np.random.default_rng(0)
        X = generator.normal(size=(120, 3))
        noise = 0.3 * generator.normal(size=120)

        def median_error(offset):
            y = offset + 0.01 * (X[:, 0] + noise)
            est = tabicl.TabICLQuantile(
                n_estimators=1, device="cpu", progress=False
            ).fit(X[:100], y[:100])
            median = est.predict_distribution(X[100:]).ppf([0.5])[:, 0]
            return median - offset

        np.testing.assert_allclose(
            median_error(1e4), median_error(0.0), atol=1e-5
        )

    def test_a_plan_cycling_its_transforms_runs_as_planned(self):
        assert tabicl._norm_methods(("power", "none", "power")) == [
            "power",
            "none",
        ]
        assert tabicl._norm_methods(("none",) * 4) == ["none"]

    def test_repeated_transforms_are_rejected(self):
        """Repeats would be identical members upstream, not more of them."""
        with pytest.raises(ValueError, match="once per cycle"):
            tabicl._norm_methods(("power", "power", "none"))

    def test_quantile_levels_are_interior_and_symmetric(self):
        levels = tabicl.quantile_levels(999)
        assert levels.size == 999
        assert 0.0 < levels[0] < levels[-1] < 1.0
        assert levels[0] + levels[-1] == pytest.approx(1.0)


class TestTabPFN:
    @pytest.fixture(autouse=True)
    def _needs_tabpfn(self):
        pytest.importorskip("tabpfn")

    def test_negative_chunk_size_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="chunk_size"):
            lazy.get_estimator("tabpfn", chunk_size=-1).fit(X, z)

    def test_an_unknown_cache_mode_is_rejected(self, tiny):
        """Only full precision and upstream's two quantised modes exist."""
        X, z = tiny
        with pytest.raises(ValueError, match="kv_cache must be one of"):
            lazy.get_estimator("tabpfn", kv_cache="int4").fit(X, z)

    def test_the_cache_is_exact_unless_quantisation_is_asked_for(self):
        assert tabpfn._cache_options(True) == {
            "fit_mode": "fit_with_cache",
            "kv_cache_precision": "auto",
        }
        assert tabpfn._cache_options("int8")["kv_cache_precision"] == "int8"
        assert tabpfn._cache_options(False) == {"fit_mode": "fit_preprocessors"}

    @pytest.mark.parametrize("version", ["v2", "v2.5", "v2.6"])
    def test_a_quantised_cache_is_rejected_where_upstream_has_none(
        self, tiny, version
    ):
        """Upstream would fall back to full precision behind a warning."""
        X, z = tiny
        with pytest.raises(ValueError, match="no quantised key/value cache"):
            lazy.get_estimator("tabpfn", version=version, kv_cache="int8").fit(
                X, z
            )

    def test_an_unknown_version_is_rejected_before_anything_is_downloaded(
        self, tiny
    ):
        X, z = tiny
        with pytest.raises(ValueError, match=r"unknown version 'v9'"):
            lazy.get_estimator("tabpfn", version="v9").fit(X, z)

    def test_a_safetensors_checkpoint_keeps_a_readable_suffix(self, tmp_path):
        """The suffix has to survive resolving the HF cache's symlink.

        The HF cache symlinks snapshots to extension-less blobs, and TabPFN
        reads the format off the *resolved* path.
        """
        blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshots" / "abc"
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        blob = blobs / "0123456789abcdef0123"
        blob.write_bytes(b"weights")
        link = snapshot / "model.safetensors"
        link.symlink_to(blob)

        readable = tabpfn.path_for_tabpfn(link)
        assert readable.resolve().suffix == ".safetensors"
        assert readable.read_bytes() == b"weights"
        assert readable.stat().st_ino == blob.stat().st_ino, (
            "hardlinked, not copied"
        )

    def test_a_path_that_already_survives_resolving_is_left_alone(
        self, tmp_path
    ):
        real = tmp_path / "model.safetensors"
        real.write_bytes(b"weights")
        assert tabpfn.path_for_tabpfn(real) is real

    def test_repeated_transforms_keep_their_weights(self):
        """Upstream shares members equally among the configs it is given."""
        ensemble = pytest.importorskip("tabpfn.preprocessing.ensemble")
        recipe = ("none+original", "none+original", "none")
        (group,) = _members.plan(
            n_estimators=12,
            transforms=_transforms.parse(recipe),
            native_transforms=tabpfn.TabPFNBarDistribution.native_transforms,
            feature_shuffle=True,
            bag_rows=10,
            n_rows=10,
            n_features=2,
            random_state=0,
            supports_native_bagging=True,
        )
        est = tabpfn.TabPFNBarDistribution()
        est.outlier_threshold_ = None
        settings = est._inference_config(group)
        configs = settings["PREPROCESS_TRANSFORMS"]
        assert [c.append_original for c in configs] == [True, True, False]
        members = ensemble.generate_regression_ensemble_configs(
            num_estimators=12,
            add_fingerprint_feature=False,
            polynomial_features="no",
            feature_shift_decoder=None,
            preprocessor_configs=configs,
            target_transforms=settings["REGRESSION_Y_PREPROCESS_TRANSFORMS"],
            random_state=0,
            num_models=1,
            outlier_removal_std=None,
        )
        original = [m.preprocess_config.append_original for m in members]
        assert original.count(True) == 2 * original.count(False) == 8

    def test_bucket_masses_reads_the_bar_distribution(self):
        """Softmaxed logits and the borders as redshifts, with no reindexing."""
        torch = pytest.importorskip("torch")

        criterion = types.SimpleNamespace(
            borders=torch.tensor([0.0, 1.0, 2.0, 4.0])
        )
        logits = torch.log(torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))
        borders, masses = tabpfn.bucket_masses(
            {"criterion": criterion, "logits": logits}
        )

        assert borders.tolist() == [0.0, 1.0, 2.0, 4.0]
        assert masses == pytest.approx(
            np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]])
        )

    def test_a_constant_target_reads_the_one_bucket_upstream_keeps(self):
        """Upstream skips the model then, and has no raw-space borders."""
        torch = pytest.importorskip("torch")

        regressor = types.SimpleNamespace(
            is_constant_target_=True,
            znorm_space_bardist_=types.SimpleNamespace(
                borders=torch.tensor([0.25, 0.75])
            ),
        )
        assert tabpfn._bucket_borders(regressor).tolist() == [0.25, 0.75]

    def test_borders_are_rebuilt_in_float64(self):
        """float32 borders cannot resolve 0.01 of spread about 1e4."""
        torch = pytest.importorskip("torch")

        regressor = types.SimpleNamespace(
            znorm_space_bardist_=types.SimpleNamespace(
                borders=torch.linspace(-3.0, 3.0, 5001)
            ),
            y_train_mean_=1e4,
            y_train_std_=0.01,
        )
        borders = tabpfn._bucket_borders(regressor)
        assert borders.dtype == np.float64
        assert np.all(np.diff(borders) > 0)
        assert borders[0] == pytest.approx(1e4 - 0.03, abs=1e-9)

    @needs_checkpoint
    @pytest.mark.parametrize("offset", [0.0, 1e4])
    def test_the_borders_are_upstream_buckets_uncollapsed(self, offset):
        """The same buckets as upstream's, but none of them collapsed."""
        generator = np.random.default_rng(0)
        X = generator.normal(size=(80, 3))
        y = offset + 0.01 * (X[:, 0] + 0.3 * generator.normal(size=80))
        est = tabpfn.TabPFNBarDistribution(
            n_estimators=1, device="cpu", progress=False
        ).fit(X[:60], y[:60])
        raw = est.regressor_.raw_space_bardist_.borders.cpu().numpy()
        assert est.borders_.size == raw.size
        np.testing.assert_allclose(est.borders_, raw, rtol=1e-6, atol=1e-8)
        assert np.all(np.diff(est.borders_) > 0)
        dist = est.predict_distribution(X[60:])
        np.testing.assert_array_equal(dist.bins, est.borders_)

    @needs_checkpoint
    def test_a_constant_target_predicts_one_narrow_bucket(self):
        generator = np.random.default_rng(0)
        X = generator.normal(size=(60, 3))
        est = tabpfn.TabPFNBarDistribution(
            n_estimators=1, device="cpu", progress=False
        ).fit(X[:50], np.full(50, 1e4))
        assert est.n_buckets_ == 1
        assert est.native_grid_.y_min < 1e4 < est.native_grid_.y_max
        dist = est.predict_distribution(X[50:])
        np.testing.assert_allclose(dist.masses.sum(axis=1), 1.0)
        np.testing.assert_allclose(dist.ppf([0.5])[:, 0], 1e4)

    def test_a_logit_count_that_misses_the_buckets_is_reported(self):
        """A mismatch here would otherwise reach `rebin` as a bare error."""
        torch = pytest.importorskip("torch")

        criterion = types.SimpleNamespace(borders=torch.tensor([0.0, 1.0, 2.0]))
        with pytest.raises(RuntimeError, match="bar distribution"):
            tabpfn.bucket_masses(
                {"criterion": criterion, "logits": torch.zeros((2, 5))}
            )

    @needs_checkpoint
    def test_chunking_the_query_rows_is_bit_identical(self):
        """Same claim as for the other two, and why chunking is the default."""
        generator = np.random.default_rng(0)
        z = generator.uniform(0.2, 1.8, 260)
        X = pd.DataFrame(
            {
                "a": z + generator.normal(0, 0.1, 260),
                "b": generator.normal(size=260),
                "c": z**2,
            }
        )
        X_ctx, X_q = X.iloc[:200], X.iloc[200:].reset_index(drop=True)

        def run(chunk_size):
            model = lazy.LazyModel(
                "tabpfn", n_estimators=2, device="cpu", chunk_size=chunk_size
            )
            return model.fit(X_ctx, z[:200]).predict_proba(X_q)

        assert np.array_equal(run(0), run(7))

    @needs_checkpoint
    @pytest.mark.parametrize("version", lazy.list_versions("tabpfn"))
    def test_every_pinned_version_loads_and_answers(self, version):
        """A wrong revision or filename is only caught by loading the thing.

        Each version is a different architecture and, from v3.5, a different
        checkpoint format, so this is also what keeps `path_for_tabpfn` honest.
        """
        generator = np.random.default_rng(0)
        z = generator.uniform(0.2, 1.8, 160)
        X = pd.DataFrame(
            {
                "a": z + generator.normal(0, 0.05, 160),
                "b": generator.normal(size=160),
            }
        )
        grid = lazy.Grid.linear(0.0, 2.0, 200)

        model = lazy.LazyModel(
            "tabpfn", version=version, n_estimators=2, device="cpu"
        )
        pdfs = model.fit(X.iloc[:120], z[:120]).predict_proba(
            X.iloc[120:], grid
        )

        assert pdfs.shape == (40, grid.n_bins)
        assert np.allclose(np.trapezoid(pdfs, grid.centers, axis=1), 1.0)
        assert model.provenance_["version"] == version
        assert model.name_ == f"tabpfn:{version}"

    @needs_checkpoint
    def test_the_density_matches_the_models_own_quantiles(self):
        """The bar distribution we read is the one upstream reduces to numbers.

        If the borders and the logits were ever mismatched -- a reindexing, a
        normalisation in the wrong space -- the density would still look
        plausible while being wrong. So it is checked against a quantity
        computed entirely inside TabPFN: the median of its own predictive
        distribution, which must land where our CDF crosses one half.
        """
        generator = np.random.default_rng(0)
        z = generator.uniform(0.2, 1.8, 240)
        X = pd.DataFrame(
            {
                "a": z + generator.normal(0, 0.05, 240),
                "b": generator.normal(size=240),
            }
        )
        X_ctx, X_q = X.iloc[:200], X.iloc[200:].reset_index(drop=True)

        grid = lazy.Grid.linear(0.0, 2.0, 2000)
        model = lazy.get_estimator("tabpfn", n_estimators=2, device="cpu").fit(
            X_ctx, z[:200]
        )
        ours = model.predict(X_q, method="median", y_grid=grid)
        theirs = model.regressor_.predict(
            X_q.to_numpy(dtype=np.float64), output_type="median"
        )

        assert ours == pytest.approx(theirs, abs=2 * grid.widths.max())


@needs_checkpoint
def test_chunking_the_query_rows_changes_only_rounding(monkeypatch):
    """The claim that makes bounded memory free: chunking changes nothing.

    TabFM's in-context stage builds its keys and values from the context rows
    alone, so a query row's answer cannot depend on which other query rows
    share its chunk. If that ever stopped holding, `chunk_size` would silently
    change results instead of only peak memory, so it is asserted rather than
    assumed. In float32 on a CPU the kernels round differently for different
    batch shapes (about 2e-6 of the peak density), so the comparison is to
    float rounding, not bit for bit (``exact_chunking`` is False). The rows
    each upstream call receives are recorded too, so the comparison cannot
    pass by never chunking at all.
    """
    upstream = pytest.importorskip("tabfm")
    predict_proba = upstream.TabFMClassifier.predict_proba
    sizes = []

    def recording(self, X, *args, **kwargs):
        sizes.append(len(X))
        return predict_proba(self, X, *args, **kwargs)

    monkeypatch.setattr(upstream.TabFMClassifier, "predict_proba", recording)

    generator = np.random.default_rng(0)
    z = generator.uniform(0.2, 1.8, 240)
    X = pd.DataFrame(
        {
            "a": z + generator.normal(0, 0.1, 240),
            "b": generator.normal(size=240),
        }
    )
    X_ctx, z_ctx = X.iloc[:200], z[:200]
    X_q = X.iloc[200:].reset_index(drop=True)

    def run(chunk_size):
        model = lazy.LazyModel(
            "tabfm",
            n_coarse_bins=2,
            n_fine_bins=2,
            n_estimators=1,
            device="cpu",
            kv_cache=False,
            chunk_size=chunk_size,
        )
        sizes.clear()
        pdfs = model.fit(X_ctx, z_ctx).predict_proba(X_q)
        return pdfs, list(sizes)

    whole, whole_sizes = run(0)
    chunked, chunked_sizes = run(7)

    assert set(whole_sizes) == {40}
    assert max(chunked_sizes) == 7
    assert len(chunked_sizes) == 6 * len(whole_sizes)
    np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-5 * whole.max())


class TestMissingBackend:
    """A forgotten extra must name the extra, not fail somewhere unrelated.

    `sys.modules[name] = None` makes `import name` raise ImportError, which is
    how an absent optional dependency looks from inside the estimator.
    """

    @pytest.fixture
    def without(self, monkeypatch):
        def hide(name):
            monkeypatch.setitem(sys.modules, name, None)

        return hide

    @pytest.fixture
    def tiny(self):
        generator = np.random.default_rng(0)
        z = generator.uniform(0.2, 1.8, 40)
        X = pd.DataFrame({"a": z, "b": generator.normal(size=40)})
        return X, z

    def test_tabicl_names_its_extra(self, without, tiny):
        without("tabicl")
        X, z = tiny
        with pytest.raises(ImportError, match=r"lazy-tfm\[tabicl\]"):
            lazy.LazyModel("tabicl").fit(X, z)

    def test_tabpfn_names_its_extra(self, without, tiny):
        without("tabpfn")
        X, z = tiny
        with pytest.raises(ImportError, match=r"lazy-tfm\[tabpfn\]"):
            lazy.LazyModel("tabpfn").fit(X, z)

    def test_tabfm_names_its_extra(self, without, tiny):
        without("tabfm")
        X, z = tiny
        with pytest.raises(ImportError, match=r"lazy-tfm\[tabfm\]"):
            lazy.LazyModel("tabfm").fit(X, z)

    @pytest.mark.parametrize("name", ["tabicl", "tabpfn", "tabfm"])
    def test_a_backend_missing_a_dependency_says_so(
        self, name, tiny, monkeypatch
    ):
        """Not "pip install lazy-tfm[x]" when x is installed but broken."""
        real_import = builtins.__import__

        def importing(module, *args, **kwargs):
            if module == name:
                raise ModuleNotFoundError(
                    "No module named 'torch'", name="torch"
                )
            return real_import(module, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", importing)
        X, z = tiny
        with pytest.raises(ModuleNotFoundError, match="torch") as raised:
            lazy.LazyModel(name).fit(X, z)
        assert "lazy-tfm" not in str(raised.value)

    def test_the_missing_backend_is_reported_before_parameter_problems(
        self, without, tiny
    ):
        """With 40 context rows the default 10x10 hierarchy is also too big.

        The install problem is the one to report: sending someone off to
        change n_coarse_bins when the real fix is `pip install` wastes their
        time.
        """
        without("tabfm")
        X, z = tiny
        with pytest.raises(ImportError, match=r"lazy-tfm\[tabfm\]"):
            lazy.LazyModel("tabfm").fit(X, z)
