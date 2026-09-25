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
            X_query,
            seed=0,
        )

        assert sizes == [7, 7, 7, 7, 7, 5]
        assert np.array_equal(probs[:, 0], X_query["a"].to_numpy())

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
