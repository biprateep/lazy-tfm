"""Backend wiring that can be checked without running a foundation model.

Skipped wholesale when the optional backend is not installed, so the default
CI environment (core + dev only) stays fast.

One test here does run a backbone, because it backs a correctness claim that
cannot be reasoned about from the outside. It loads a multi-gigabyte checkpoint,
so it only runs when ``LAZY_RUN_CHECKPOINT_TESTS=1`` is set.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

from lazy import LazyModel, get_estimator

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
