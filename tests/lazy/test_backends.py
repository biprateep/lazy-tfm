# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Backend wiring that can be checked without running a foundation model.

Skipped wholesale when the optional backend is not installed, so the default
CI environment (core + dev only) stays fast.

One test here does run a backbone, because it backs a correctness claim that
cannot be reasoned about from the outside. It loads a multi-gigabyte checkpoint,
so it only runs when ``LAZY_RUN_CHECKPOINT_TESTS=1`` is set.
"""

import inspect
import os
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
        with pytest.raises(KeyError, match="unknown version"):
            lazy.get_estimator(
                "tabfm", version="v9", n_coarse_bins=2, n_fine_bins=2
            ).fit(X, z)

    def test_more_than_ten_classes_per_level_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="at most 10 classes"):
            lazy.get_estimator("tabfm", n_coarse_bins=11).fit(X, z)

    def test_context_smaller_than_the_bin_count_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="fewer than the"):
            lazy.get_estimator("tabfm").fit(X.iloc[:10], z[:10])

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
            est, "_classifier", lambda model, group, seed, rows: Recording()
        )
        X_query = pd.DataFrame(
            {"a": np.linspace(0.0, 1.0, 40)}, index=np.arange(100, 140)
        )

        handle = {"bag_fraction": None, "group": None}
        probs = est._class_probabilities(
            None,
            handle,
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
        est = tabfm.TabFMHistogram()
        est.support_ = (float(z.min()), float(z.max()))
        edges, coarse, fine = est._edges(z, 0.0)
        assert edges.size == 101
        assert np.all(np.diff(edges) >= 0)
        assert all(f.size == 11 and np.all(np.diff(f) >= 0) for f in fine)
        # Every target falls in a bin of non-zero width.
        bins = tabfm._bin_labels(edges, z)
        assert np.all(np.diff(edges)[bins] > 0)
        _stub_classifier(est, monkeypatch)
        handle = {"X": tabfm._frame(z[:, None]), "z": z, "bag_fraction": None}
        handle["group"] = types.SimpleNamespace(seed=0)
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
            n_estimators=1, z_grid=np.linspace(0.0, 3.0, 301)
        )
        with pytest.warns(UserWarning, match="outside the z_grid range"):
            est.fit(X, z)
        assert est.z_context_.min() == est.support_[0] == -0.005
        edges, _, _ = est._edges(est.z_context_, 0.0)
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
            options["z_grid"] = np.linspace(0.0, 3.0, 31)
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
    def test_cuda_rounding_stays_within_the_declared_tolerance(self):
        """bfloat16 on CUDA: neither chunking nor the cache is exact there."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        generator = np.random.default_rng(0)
        X = generator.normal(size=(360, 3))
        z = -1.0 + 0.3 * (X[:, 0] + 0.3 * generator.normal(size=360))
        grid = lazy.Grid.linear(-2.5, 0.5, 301)
        backbone = None

        def run(**options):
            nonlocal backbone
            est = tabfm.TabFMHistogram(
                n_estimators=1, device="cuda", progress=False, **options
            ).fit(X[:300], z[:300])
            if backbone is not None:
                est._backbone_cache = backbone
            pdfs = est.predict_proba(X[300:], grid)
            backbone = est._backbone_cache
            return pdfs

        whole = run(kv_cache=False, chunk_size=0)
        rtol = tabfm.TabFMHistogram.kv_cache_rtol
        assert not tabfm.TabFMHistogram.exact_chunking
        for other in (
            run(kv_cache=False, chunk_size=7),
            run(kv_cache=True),
        ):
            np.testing.assert_allclose(
                other, whole, rtol=rtol, atol=rtol * whole.max()
            )

    def test_a_constant_target_fits_with_a_valid_native_grid(self):
        generator = np.random.default_rng(0)
        X = generator.normal(size=(200, 2))
        est = tabfm.TabFMHistogram(n_estimators=1).fit(X, np.full(200, 0.5))
        assert est.native_grid_.z_min == 0.5
        assert 0.5 < est.native_grid_.z_max < 0.5 + 1e-5

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
            model=None, group=est.member_groups_[0], seed=0, max_rows=None
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
            model=None, group=est.member_groups_[0], seed=0, max_rows=None
        )
        assert set(classifier.get_params()) <= accepted


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
        est, "_classifier", lambda model, group, seed, rows: Frequencies()
    )


class TestTabICL:
    @pytest.fixture(autouse=True)
    def _needs_tabicl(self):
        pytest.importorskip("tabicl")

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
        assert grid.z_min == pytest.approx(-0.35)
        assert grid.z_max == pytest.approx(2.05)
        assert grid.n_bins == 1500
        np.testing.assert_allclose(grid.widths, 1.6 / 1000)

    @pytest.mark.parametrize("value", [0.0, 0.5, -1e4])
    def test_a_constant_target_is_padded_on_its_own_scale(self, value):
        est = tabicl.TabICLQuantile()
        est._support = (value, value)
        grid = est._native_grid()
        pad = 0.01 * max(abs(value), 1.0)
        assert grid.z_min == pytest.approx(value - pad)
        assert grid.z_max == pytest.approx(value + pad)

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

    def test_an_unknown_version_is_rejected_before_anything_is_downloaded(
        self, tiny
    ):
        X, z = tiny
        with pytest.raises(KeyError, match=r"unknown version 'v9'"):
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
        (group,) = _members.plan(
            n_estimators=12,
            transforms=_transforms.parse(("power", "power", "none")),
            native_transforms=tabpfn.TabPFNBarDistribution.native_transforms,
            feature_shuffle=True,
            bag_rows=10,
            n_rows=10,
            n_features=2,
            random_state=0,
            supports_native_bagging=True,
        )
        est = tabpfn.TabPFNBarDistribution()
        configs = est._inference_config(group)["PREPROCESS_TRANSFORMS"]
        assert [c.name for c in configs] == ["power", "power", "none"]
        members = ensemble.generate_regression_ensemble_configs(
            num_estimators=12,
            add_fingerprint_feature=False,
            polynomial_features="no",
            feature_shift_decoder=None,
            preprocessor_configs=configs,
            target_transforms=[None, None],
            random_state=0,
            num_models=1,
            outlier_removal_std=None,
        )
        names = [m.preprocess_config.name for m in members]
        assert names.count("power") == 2 * names.count("none") == 8

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
        assert est.native_grid_.z_min < 1e4 < est.native_grid_.z_max
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
        ours = model.predict(X_q, method="z_median", z_grid=grid)
        theirs = model.regressor_.predict(
            X_q.to_numpy(dtype=np.float64), output_type="median"
        )

        assert ours == pytest.approx(theirs, abs=2 * grid.widths.max())


@needs_checkpoint
def test_chunking_the_query_rows_is_bit_identical(monkeypatch):
    """The claim that makes bounded memory free: chunking changes nothing.

    TabFM's in-context stage builds its keys and values from the context rows
    alone, so a query row's answer cannot depend on which other query rows
    share its chunk. If that ever stopped holding, `chunk_size` would silently
    change results instead of only peak memory, so it is asserted rather than
    assumed. The rows each upstream call receives are recorded too, so the
    comparison cannot pass by never chunking at all.
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
    assert np.array_equal(whole, chunked)


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
