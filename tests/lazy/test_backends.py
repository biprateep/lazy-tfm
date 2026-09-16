"""Backend wiring that can be checked without running a foundation model.

Skipped wholesale when the optional backend is not installed, so the default
CI environment (core + dev only) stays fast.

One test here does run a backbone, because it backs a correctness claim that
cannot be reasoned about from the outside. It loads a multi-gigabyte checkpoint,
so it only runs when ``LAZY_RUN_CHECKPOINT_TESTS=1`` is set.
"""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from lazy import LazyModel, RedshiftGrid, get_estimator, list_versions

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason="set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint",
)


@pytest.fixture
def tiny():
    generator = np.random.default_rng(0)
    z = generator.uniform(0.2, 1.8, 40)
    X = pd.DataFrame({"a": z + generator.normal(0, 0.1, 40), "b": generator.normal(size=40)})
    return X, z


class TestTabFM:
    @pytest.fixture(autouse=True)
    def _needs_tabfm(self):
        pytest.importorskip("tabfm")

    def test_inference_path_is_resolved_at_fit(self, tiny):
        X, z = tiny
        est = get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2).fit(X, z)
        assert est.inference_ in ("stream", "predict_proba")

    def test_unknown_inference_mode_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="inference must be"):
            get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2, inference="fast").fit(X, z)

    def test_forcing_stream_without_the_kv_cache_api_explains_itself(self, tiny):
        """The PyPI release of tabfm has no prefill/decode; say so, don't TypeError."""
        from lazy.models._icl_stream import streaming_available

        X, z = tiny
        est = get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2, inference="stream")
        if streaming_available():
            assert est.fit(X, z).inference_ == "stream"
        else:
            with pytest.raises(RuntimeError, match="KV-cache API"):
                est.fit(X, z)

    def test_fit_records_its_provenance_without_downloading(self, tiny):
        X, z = tiny
        est = get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2).fit(X, z)
        assert est.provenance_["backend"] == "tabfm"
        assert est.provenance_["version"] == "v1.0"
        assert est.provenance_["revision"]
        assert est.name_ == "tabfm:v1.0"

    def test_an_unknown_version_is_rejected_at_fit(self, tiny):
        """Not at the first prediction, an hour later."""
        X, z = tiny
        with pytest.raises(KeyError, match="unknown version"):
            get_estimator("tabfm", version="v9", n_coarse_bins=2, n_fine_bins=2).fit(X, z)

    def test_more_than_ten_classes_per_level_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="at most 10 classes"):
            get_estimator("tabfm", n_coarse_bins=11).fit(X, z)

    def test_context_smaller_than_the_bin_count_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="fewer than the"):
            get_estimator("tabfm").fit(X.iloc[:10], z[:10])

    def test_the_classifier_only_receives_supported_keywords(self, tiny):
        """Cache knobs exist on repository builds and not on the PyPI release."""
        import inspect

        from tabfm import TabFMClassifier

        X, z = tiny
        est = get_estimator("tabfm", n_coarse_bins=2, n_fine_bins=2).fit(X, z)
        accepted = set(inspect.signature(TabFMClassifier.__init__).parameters)
        # _classifier needs a model object only to hand on; None is never touched.
        classifier = est._classifier(model=None, seed=0)
        assert set(classifier.get_params()) <= accepted


class TestTabICL:
    @pytest.fixture(autouse=True)
    def _needs_tabicl(self):
        pytest.importorskip("tabicl")

    def test_negative_chunk_size_is_rejected(self, tiny):
        X, z = tiny
        with pytest.raises(ValueError, match="chunk_size"):
            get_estimator("tabicl", chunk_size=-1).fit(X, z)

    @needs_checkpoint
    def test_chunking_the_query_rows_is_bit_identical(self):
        """Same claim as for TabFM, and the reason chunking is the default."""
        generator = np.random.default_rng(0)
        z = generator.uniform(0.2, 1.8, 260)
        X = pd.DataFrame({"a": z + generator.normal(0, 0.1, 260), "b": generator.normal(size=260), "c": z**2})
        X_ctx, X_q = X.iloc[:200], X.iloc[200:].reset_index(drop=True)

        def run(chunk_size):
            model = LazyModel("tabicl", n_estimators=2, device="cpu", chunk_size=chunk_size)
            return model.fit(X_ctx, z[:200]).predict_proba(X_q)

        assert np.array_equal(run(0), run(7))

    def test_quantile_levels_are_interior_and_symmetric(self):
        from lazy.models.tabicl import quantile_levels

        levels = quantile_levels(999)
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
            get_estimator("tabpfn", chunk_size=-1).fit(X, z)

    def test_the_batched_fit_mode_is_rejected(self, tiny):
        """Upstream has it, but it belongs to `predict_batched`, not `predict`."""
        X, z = tiny
        with pytest.raises(ValueError, match="fit_mode must be one of"):
            get_estimator("tabpfn", fit_mode="batched").fit(X, z)

    def test_an_unknown_version_is_rejected_before_anything_is_downloaded(self, tiny):
        X, z = tiny
        with pytest.raises(KeyError, match=r"unknown version 'v9'"):
            get_estimator("tabpfn", version="v9").fit(X, z)

    def test_a_safetensors_checkpoint_keeps_a_readable_suffix(self, tmp_path):
        """The HF cache symlinks snapshots to extension-less blobs; TabPFN reads
        the format off the *resolved* path, so the suffix has to survive it."""
        from lazy.models.tabpfn import path_for_tabpfn

        blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshots" / "abc"
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        blob = blobs / "0123456789abcdef0123"
        blob.write_bytes(b"weights")
        link = snapshot / "model.safetensors"
        link.symlink_to(blob)

        readable = path_for_tabpfn(link)
        assert readable.resolve().suffix == ".safetensors"
        assert readable.read_bytes() == b"weights"
        assert readable.stat().st_ino == blob.stat().st_ino, "hardlinked, not copied"

    def test_a_path_that_already_survives_resolving_is_left_alone(self, tmp_path):
        from lazy.models.tabpfn import path_for_tabpfn

        real = tmp_path / "model.safetensors"
        real.write_bytes(b"weights")
        assert path_for_tabpfn(real) is real

    def test_bucket_masses_reads_the_bar_distribution(self):
        """Softmaxed logits, and the borders as redshifts, with no reindexing."""
        torch = pytest.importorskip("torch")

        from lazy.models.tabpfn import bucket_masses

        criterion = types.SimpleNamespace(borders=torch.tensor([0.0, 1.0, 2.0, 4.0]))
        logits = torch.log(torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))
        borders, masses = bucket_masses({"criterion": criterion, "logits": logits})

        assert borders.tolist() == [0.0, 1.0, 2.0, 4.0]
        assert masses == pytest.approx(np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))

    def test_a_logit_count_that_misses_the_buckets_is_reported(self):
        """A shape mismatch here would otherwise land in `rebin` as a bare ValueError."""
        torch = pytest.importorskip("torch")

        from lazy.models.tabpfn import bucket_masses

        criterion = types.SimpleNamespace(borders=torch.tensor([0.0, 1.0, 2.0]))
        with pytest.raises(RuntimeError, match="bar distribution"):
            bucket_masses({"criterion": criterion, "logits": torch.zeros((2, 5))})

    @needs_checkpoint
    def test_chunking_the_query_rows_is_bit_identical(self):
        """Same claim as for the other two, and the reason chunking is the default."""
        generator = np.random.default_rng(0)
        z = generator.uniform(0.2, 1.8, 260)
        X = pd.DataFrame({"a": z + generator.normal(0, 0.1, 260), "b": generator.normal(size=260), "c": z**2})
        X_ctx, X_q = X.iloc[:200], X.iloc[200:].reset_index(drop=True)

        def run(chunk_size):
            model = LazyModel("tabpfn", n_estimators=2, device="cpu", chunk_size=chunk_size)
            return model.fit(X_ctx, z[:200]).predict_proba(X_q)

        assert np.array_equal(run(0), run(7))

    @needs_checkpoint
    @pytest.mark.parametrize("version", list_versions("tabpfn"))
    def test_every_pinned_version_loads_and_answers(self, version):
        """A wrong revision or filename is only caught by loading the thing.

        Each version is a different architecture and, from v3.5, a different
        checkpoint format, so this is also what keeps `path_for_tabpfn` honest.
        """
        generator = np.random.default_rng(0)
        z = generator.uniform(0.2, 1.8, 160)
        X = pd.DataFrame({"a": z + generator.normal(0, 0.05, 160), "b": generator.normal(size=160)})
        grid = RedshiftGrid.linear(0.0, 2.0, 200)

        model = LazyModel("tabpfn", version=version, n_estimators=2, device="cpu")
        pdfs = model.fit(X.iloc[:120], z[:120]).predict_proba(X.iloc[120:], grid)

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
        X = pd.DataFrame({"a": z + generator.normal(0, 0.05, 240), "b": generator.normal(size=240)})
        X_ctx, X_q = X.iloc[:200], X.iloc[200:].reset_index(drop=True)

        grid = RedshiftGrid.linear(0.0, 2.0, 2000)
        model = get_estimator("tabpfn", n_estimators=2, device="cpu").fit(X_ctx, z[:200])
        ours = model.predict(X_q, method="z_median", z_grid=grid)
        theirs = model.regressor_.predict(X_q.to_numpy(dtype=np.float64), output_type="median")

        assert ours == pytest.approx(theirs, abs=2 * grid.widths.max())


@needs_checkpoint
def test_chunking_the_query_rows_is_bit_identical():
    """The claim that makes bounded memory free: chunking changes nothing.

    TabFM's in-context stage builds its keys and values from the context rows
    alone, so a query row's answer cannot depend on which other query rows share
    its chunk. If that ever stopped holding, `chunk_size` would silently change
    results instead of only peak memory, so it is asserted rather than assumed.
    """
    pytest.importorskip("tabfm")

    generator = np.random.default_rng(0)
    z = generator.uniform(0.2, 1.8, 240)
    X = pd.DataFrame({"a": z + generator.normal(0, 0.1, 240), "b": generator.normal(size=240)})
    X_ctx, z_ctx = X.iloc[:200], z[:200]
    X_q = X.iloc[200:].reset_index(drop=True)

    def run(chunk_size):
        model = LazyModel(
            "tabfm",
            n_coarse_bins=2,
            n_fine_bins=2,
            n_estimators=1,
            device="cpu",
            inference="predict_proba",
            chunk_size=chunk_size,
        )
        return model.fit(X_ctx, z_ctx).predict_proba(X_q)

    assert np.array_equal(run(0), run(7))


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
        with pytest.raises(ImportError, match=r"lazy-photoz\[tabicl\]"):
            LazyModel("tabicl").fit(X, z)

    def test_tabpfn_names_its_extra(self, without, tiny):
        without("tabpfn")
        X, z = tiny
        with pytest.raises(ImportError, match=r"lazy-photoz\[tabpfn\]"):
            LazyModel("tabpfn").fit(X, z)

    def test_tabfm_names_its_extra(self, without, tiny):
        without("tabfm")
        X, z = tiny
        with pytest.raises(ImportError, match=r"lazy-photoz\[tabfm\]"):
            LazyModel("tabfm").fit(X, z)

    def test_the_missing_backend_is_reported_before_parameter_problems(self, without, tiny):
        """With 40 context rows the default 10x10 hierarchy is also too big.

        The install problem is the one to report: sending someone off to change
        n_coarse_bins when the real fix is `pip install` wastes their time.
        """
        without("tabfm")
        X, z = tiny
        with pytest.raises(ImportError, match=r"lazy-photoz\[tabfm\]"):
            LazyModel("tabfm").fit(X, z)
