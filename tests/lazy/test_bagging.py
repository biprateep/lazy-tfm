# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Bagging means the same on every backend: lazy's bags, fitted per bag.

Each backend's upstream model is replaced by a recording fake, so these run
on a CPU with nothing downloaded: what is checked is which rows, transforms
and targets each member is handed, not what a network makes of them. Every
feature column is ``row + 1000 * column``, so a row's smallest value names
it whatever order its columns arrive in.
"""

import collections
import os
import pathlib
import types

import numpy as np
import pytest

from lazy.models import _hub
from lazy.models import _icl_stream
from lazy.models import _limix_preprocess
from lazy.models import _limix_stream
from lazy.models import _members
from lazy.models import _transforms
from lazy.models import limix
from lazy.models import tabfm
from lazy.models import tabicl
from lazy.models import tabpfn

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)

N_ROWS, N_MEMBERS, BAG_ROWS, SEED = 60, 4, 25, 3

BACKENDS = ("tabpfn", "tabicl", "tabfm", "limix")


@pytest.fixture
def data():
    """Features that name their rows, and targets, as a tuple (X, y)."""
    rows = np.arange(N_ROWS, dtype=float)[:, None]
    X = rows + 1000.0 * np.arange(3)
    y = np.random.default_rng(0).uniform(0.1, 1.5, N_ROWS)
    return X, y


def _row_ids(features):
    """The rows some prepared features came from, in their order."""
    return np.asarray(features, dtype=float).min(axis=1).round().astype(int)


def _bags(n_rows=N_ROWS):
    return _members.draw_bags(N_MEMBERS, BAG_ROWS, n_rows, SEED)


def _member_of(seed):
    """The member a group seed belongs to: a group's first member's."""
    seeds = {_members.member_seed(SEED, i): i for i in range(N_MEMBERS)}
    return seeds[seed]


class _Recorder:
    """What each fake upstream model was handed, keyed by member."""

    def __init__(self):
        self.calls = collections.defaultdict(list)
        self.loads = []

    def add(self, key, **seen):
        self.calls[key].append(types.SimpleNamespace(**seen))


def _provenance(est):
    """The provenance a real checkpoint load records, for a faked load."""
    return _hub.get_checkpoint(est.backend, est.version).provenance(
        device=est.device_
    )


# -- the fakes -----------------------------------------------------------------


def _fake_tabpfn(monkeypatch, recorder, *, limit=None):
    """TabPFNRegressor without a network, validating sizes as upstream does.

    Its bucket borders are fixed in standardised units and stretched by the
    targets it is given, as upstream's are. It loads its "network" through
    upstream's loader, as the real regressor does, and the loader counts
    the networks it builds in ``recorder.loads``.
    """
    pytest.importorskip("tabpfn")
    import tabpfn as upstream  # noqa: PLC0415 - optional extra.
    from tabpfn import base  # noqa: PLC0415 - optional extra.
    from tabpfn import validation  # noqa: PLC0415 - optional extra.
    import torch  # noqa: PLC0415 - optional extra.

    def load(**kwargs):
        recorder.loads.append(kwargs)
        return [object()], None, torch.nn.Module(), None

    monkeypatch.setattr(base, "initialize_tabpfn_model", load)

    class Regressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            self.models_, *_ = base.initialize_tabpfn_model(
                model_path=self.kwargs["model_path"],
                which="regressor",
                fit_mode=self.kwargs["fit_mode"],
                softmax_temperature_override=None,
                n_estimators_override=self.kwargs["n_estimators"],
            )
            self.inference_config_ = types.SimpleNamespace(
                MAX_NUMBER_OF_SAMPLES=limit or 10**9, MAX_CPU_SAMPLES=10**9
            )
            validation.validate_dataset_size(
                X,
                y,
                max_num_samples=limit or 10**9,
                max_num_features=500,
                devices=(torch.device("cpu"),),
                ignore_pretraining_limits=self.kwargs[
                    "ignore_pretraining_limits"
                ],
                max_cpu_samples=10**9,
            )
            recorder.add(
                self.kwargs["random_state"],
                rows=_row_ids(X),
                y=np.array(y),
                config=self.kwargs["inference_config"],
            )
            self.znorm_space_bardist_ = types.SimpleNamespace(
                borders=torch.linspace(-3.0, 3.0, 11, dtype=torch.float64)
            )
            self.y_train_mean_ = float(np.mean(y))
            self.y_train_std_ = float(np.std(y))
            return self

        def predict(self, X, output_type):
            del output_type  # Unused: always the full output.
            return {
                "criterion": self.znorm_space_bardist_,
                "logits": torch.zeros(len(X), 10),
            }

    monkeypatch.setattr(upstream, "TabPFNRegressor", Regressor)

    def load(est):
        est.checkpoint_ = pathlib.Path("tabpfn-v3.ckpt")
        est.provenance_ = _provenance(est)

    monkeypatch.setattr(tabpfn.TabPFNBarDistribution, "_load_checkpoint", load)
    return tabpfn.TabPFNBarDistribution


def _fake_tabicl(monkeypatch, recorder):
    """TabICLRegressor without a backbone, with upstream's member plan."""
    upstream = pytest.importorskip("tabicl")
    preprocessing = pytest.importorskip("tabicl._sklearn.preprocessing")

    class Regressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.random_state = kwargs["random_state"]

        def fit(self, X, y):
            recorder.add(self.random_state, rows=_row_ids(X), y=np.array(y))
            self.ensemble_generator_ = preprocessing.EnsembleGenerator(
                classification=False,
                n_estimators=self.kwargs["n_estimators"],
                norm_methods=self.kwargs["norm_methods"],
                feat_shuffle_method=self.kwargs["feat_shuffle_method"],
                outlier_threshold=self.kwargs["outlier_threshold"],
                random_state=self.random_state,
            ).fit(X, y)
            return self

        def predict(self, X, output_type):
            del output_type  # Unused: always the raw quantiles.
            return np.zeros((len(X), 1)) + np.array([-1.0, 0.0, 1.0])

    monkeypatch.setattr(upstream, "TabICLRegressor", Regressor)

    def load(est):
        est.checkpoint_ = pathlib.Path("tabicl-regressor.ckpt")
        est.provenance_ = _provenance(est)

    monkeypatch.setattr(tabicl.TabICLQuantile, "_load_checkpoint", load)
    return tabicl.TabICLQuantile


def _fake_limix(monkeypatch, recorder):
    """LimiX without its source or network: pipelines that record."""

    class Pipeline:
        def __init__(self, token, seeds, *, shuffle):
            self.token, self.seeds, self.shuffle = token, seeds, shuffle

        def fit(self, context):
            self.n_features_out_ = context.shape[1]
            return self

        def transform(self, features):
            return features

    class Member(_limix_stream.Member):
        def __init__(self, x, y, seed):
            super().__init__(x, y, seed)
            recorder.add(seed, rows=_row_ids(x), y=np.array(y))

    monkeypatch.setattr(_limix_preprocess, "MemberPipeline", Pipeline)
    monkeypatch.setattr(_limix_stream, "Member", Member)
    monkeypatch.setattr(
        limix.LimiXBarDistribution,
        "_import_backend",
        lambda est: types.ModuleType("limix"),
    )

    def load(est):
        est.checkpoint_ = pathlib.Path("limix.ckpt")
        est.provenance_ = _provenance(est)
        est.borders_ = np.linspace(-3.0, 3.0, 11)
        est.n_buckets_ = 10

    monkeypatch.setattr(limix.LimiXBarDistribution, "_load_checkpoint", load)
    monkeypatch.setattr(
        limix.LimiXBarDistribution,
        "_member_probabilities",
        lambda est, entry, X: np.full((len(X), est.n_buckets_), 0.1),
    )
    return limix.LimiXBarDistribution


def _fake_tabfm(monkeypatch, recorder):
    """TabFM's real classifier and member views, without the backbone.

    Records, for every member run on either path, the rows of its view and
    the scaling its standardisation was refitted with.
    """
    upstream = pytest.importorskip("tabfm")
    monkeypatch.setattr(_icl_stream, "streaming_available", lambda: True)
    monkeypatch.setattr(
        tabfm.TabFMHistogram,
        "_backbone",
        lambda est: types.SimpleNamespace(max_classes=10),
    )

    def logits(classifier, model, targets, **kwargs):
        del model, kwargs  # Unused: nothing runs.
        generator = classifier.ensemble_generator_
        (pipeline,) = generator.preprocessors_.values()
        recorder.add(
            "tabfm",
            rows=_row_ids(generator.X_),
            n_members=classifier.n_estimators,
            mean=pipeline.standard_scaler_.mean_,
            features=np.asarray(generator.X_),
        )
        n_rows = len(targets["query"])
        return {
            "query": {"mean_logits": np.zeros((n_rows, classifier.n_classes_))}
        }

    def internal(classifier, X):
        frame = {"query": X}
        return logits(classifier, None, frame)["query"]["mean_logits"][None]

    monkeypatch.setattr(_icl_stream, "classification_logits", logits)
    monkeypatch.setattr(
        upstream.TabFMClassifier, "_predict_proba_internal", internal
    )
    return tabfm.TabFMHistogram


_FAKES = {
    "tabpfn": _fake_tabpfn,
    "tabicl": _fake_tabicl,
    "tabfm": _fake_tabfm,
    "limix": _fake_limix,
}

_SMALL = {
    "tabfm": {"n_coarse_bins": 2, "n_fine_bins": 2, "n_dither": 1},
}


def _fit(name, monkeypatch, X, y, **params):
    """A bagged fit (and, for TabFM, a prediction) on the fake backend."""
    recorder = _Recorder()
    cls = _FAKES[name](monkeypatch, recorder)
    settings = {
        "n_estimators": N_MEMBERS,
        "bag_size": BAG_ROWS,
        "random_state": SEED,
        "device": "cpu",
        "progress": False,
        "kv_cache": name != "limix",
        **_SMALL.get(name, {}),
        **params,
    }
    est = cls(**settings).fit(X, y)
    # TabFM fits its classifiers when it predicts.
    est.predict_distribution(X[:3])
    return est, recorder


def _member_calls(recorder):
    """The rows each member was handed, from a seed-keyed recorder."""
    return {
        _member_of(seed): [call.rows for call in calls]
        for seed, calls in recorder.calls.items()
    }


# -- every member sees the same rows on every backend ------------------------


@pytest.mark.parametrize("name", ["tabpfn", "tabicl", "limix"])
def test_each_member_is_handed_exactly_its_bag(name, monkeypatch, data):
    X, y = data
    _, recorder = _fit(name, monkeypatch, X, y)
    seen = _member_calls(recorder)
    assert sorted(seen) == list(range(N_MEMBERS))
    for member, bag in enumerate(_bags()):
        (rows,) = seen[member]
        np.testing.assert_array_equal(rows, bag)


@pytest.mark.parametrize("kv_cache", [True, False])
def test_tabfm_members_are_handed_their_bags_at_both_levels(
    monkeypatch, data, kv_cache
):
    """Coarse: bag i. Fine level j: the rows of bag i in coarse bin j."""
    X, y = data
    est, recorder = _fit("tabfm", monkeypatch, X, y, kv_cache=kv_cache)
    assert est.inference_ == ("stream" if kv_cache else "predict_proba")
    calls = recorder.calls["tabfm"]
    assert all(call.n_members == 1 for call in calls)
    bags = _bags()
    _, coarse_edges, _ = est._edges(est.y_context_, 0.0)
    coarse = tabfm._bin_labels(coarse_edges, est.y_context_)
    expected = [bag for bag in bags]
    for j in range(est.n_coarse_bins):
        level = np.flatnonzero(coarse == j)
        expected.extend(
            np.intersect1d(bag, level)
            for bag in bags
            if np.intersect1d(bag, level).size
        )
    assert len(calls) == len(expected)
    for call, rows in zip(calls, expected, strict=True):
        np.testing.assert_array_equal(call.rows, rows)


@pytest.mark.parametrize("name", BACKENDS)
def test_the_bags_are_the_plans_on_every_backend(name, monkeypatch, data):
    """And the plan's are draw_bags(m, k, n, seed), the paper's draw."""
    X, y = data
    est, _ = _fit(name, monkeypatch, X, y)
    planned = {}
    for group in est.member_groups_:
        rows = group.member_rows or (group.rows,)
        for member, member_rows in zip(group.members, rows, strict=True):
            planned[member.index] = member_rows
    for member, bag in enumerate(_bags()):
        np.testing.assert_array_equal(planned[member], bag)


# -- what is fitted on each member's bag -------------------------------------


@pytest.mark.parametrize("name", BACKENDS)
def test_a_scaffold_is_fitted_on_each_members_bag(name, monkeypatch, data):
    X, y = data
    est, _ = _fit(name, monkeypatch, X, y, transforms="robust")
    bags = _bags()
    assert len(est.member_groups_) == N_MEMBERS
    for group, fitted in zip(
        est.member_groups_, est.transformers_, strict=True
    ):
        (member,) = group.members
        np.testing.assert_array_equal(group.rows, bags[member.index])
        np.testing.assert_allclose(
            fitted._transformer.center_, np.median(X[group.rows], axis=0)
        )


def test_a_scaffolded_clip_is_fitted_on_each_members_bag(monkeypatch, data):
    """LimiX has no clip of its own, so lazy's SoftClip runs per bag."""
    X, y = data
    est, _ = _fit("limix", monkeypatch, X, y, outlier_threshold=2.0)
    for group, clipper in zip(est.member_groups_, est.clippers_, strict=True):
        reference = _transforms.SoftClip(2.0).fit(X[group.rows])
        probe = np.linspace(-5e3, 5e3, 7)[:, None] + np.zeros(3)
        np.testing.assert_array_equal(
            clipper.transform(probe), reference.transform(probe)
        )


def test_tabfm_members_refit_their_preprocessing_on_their_rows(
    monkeypatch, data
):
    """Standardisation, norm method and clip: on the member's own rows."""
    X, y = data
    _, recorder = _fit("tabfm", monkeypatch, X, y)
    for call in recorder.calls["tabfm"]:
        np.testing.assert_allclose(call.mean, call.features.mean(axis=0))
        np.testing.assert_array_equal(
            call.rows, np.unique(call.rows)
        )  # its own rows, each once


@pytest.mark.parametrize("name", ["tabicl", "limix"])
def test_the_target_is_standardised_on_each_members_bag(
    name, monkeypatch, data
):
    X, y = data
    _, recorder = _fit(name, monkeypatch, X, y)
    ddof = 1 if name == "limix" else 0
    for calls in recorder.calls.values():
        (call,) = calls
        np.testing.assert_allclose(call.y.mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(call.y.std(ddof=ddof), 1.0, rtol=1e-6)


def test_tabpfn_standardises_each_bag_itself(monkeypatch, data):
    """Each regressor is given its bag's raw targets, so scales on them."""
    X, y = data
    est, recorder = _fit("tabpfn", monkeypatch, X, y)
    for seed, (call,) in recorder.calls.items():
        bag = _bags()[_member_of(seed)]
        np.testing.assert_array_equal(call.y, y[bag])
        assert call.config["SUBSAMPLE_SAMPLES"] is None
    for handle in est.handles_:
        np.testing.assert_allclose(
            tabpfn._bucket_borders(handle),
            np.linspace(-3.0, 3.0, 11) * handle.y_train_std_
            + handle.y_train_mean_,
        )


# -- native grids under bagging ----------------------------------------------


@pytest.mark.parametrize("name", ["tabpfn", "limix"])
def test_the_native_grid_covers_every_members_buckets(name, monkeypatch, data):
    X, y = data
    est, _ = _fit(name, monkeypatch, X, y)
    edges = est.native_grid_.edges
    for handle in est.handles_:
        buckets = (
            tabpfn._bucket_borders(handle)
            if name == "tabpfn"
            else limix._bucket_edges(
                est.borders_, handle["mean"], handle["std"]
            )
        )
        assert np.isin(buckets, edges).all()


def test_tabicls_native_grid_spans_every_context_target(monkeypatch, data):
    """Not only the targets some bag drew."""
    X, y = data
    est, _ = _fit("tabicl", monkeypatch, X, y, bag_size=10)
    assert est._support == (y.min(), y.max())
    assert est.native_grid_.y_min < y.min() < y.max() < est.native_grid_.y_max


def test_tabfm_groups_share_the_whole_contexts_bins(monkeypatch, data):
    X, y = data
    est, _ = _fit("tabfm", monkeypatch, X, y, transforms=("none", "robust"))
    dist = est.predict_distribution(X[:2])
    np.testing.assert_array_equal(dist.bins, est.native_grid_.edges)


# -- TabPFN's memory and size limits -------------------------------------------


def test_tabpfns_bagged_members_share_one_network(monkeypatch, data):
    """One regressor per bag, but the network is loaded once, not per bag.

    Eight copies of TabPFN-3.5 made a bagged fit need more GPU memory than
    an unbagged one on four times the rows.
    """
    pytest.importorskip("tabpfn")
    from tabpfn import base  # noqa: PLC0415 - optional extra.

    X, y = data
    est, recorder = _fit("tabpfn", monkeypatch, X, y)
    assert len(est.handles_) == N_MEMBERS
    assert len(recorder.loads) == 1
    (network,) = est.handles_[0].models_
    assert all(handle.models_[0] is network for handle in est.handles_)
    # The loader is upstream's own again once the fit is done.
    assert base.initialize_tabpfn_model.__name__ == "load"
    assert "_networks" not in vars(est)
    est.fit(X, y)
    assert len(recorder.loads) == 1  # a refit reuses it (lazy._weights)
    assert est.handles_[0].models_[0] is network


# -- TabPFN's size limits ------------------------------------------------------


def test_tabpfn_takes_a_large_context_in_small_bags(monkeypatch, data):
    """Only the bag meets upstream's limit, so bagging is the way in."""
    X, y = data
    recorder = _Recorder()
    cls = _fake_tabpfn(monkeypatch, recorder, limit=BAG_ROWS)
    est = cls(
        n_estimators=N_MEMBERS,
        bag_size=BAG_ROWS,
        device="cpu",
        random_state=SEED,
        progress=False,
    ).fit(X, y)
    assert len(est.handles_) == N_MEMBERS
    assert all(
        len(call.rows) == BAG_ROWS for (call,) in recorder.calls.values()
    )


@pytest.mark.parametrize(
    ("bag_size", "seen"),
    [(None, "the context has 60 rows"), (40, "each member's bag has 40")],
)
def test_tabpfn_says_bagging_below_its_limit_is_the_remedy(
    monkeypatch, data, bag_size, seen
):
    X, y = data
    cls = _fake_tabpfn(monkeypatch, _Recorder(), limit=BAG_ROWS)
    est = cls(n_estimators=2, bag_size=bag_size, device="cpu", progress=False)
    with pytest.raises(ValueError, match="officially supported") as raised:
        est.fit(X, y)
    message = str(raised.value)
    assert seen in message
    assert f"bag_size={BAG_ROWS} (or fewer) with n_estimators >= 3" in message


@needs_checkpoint
def test_tabpfn_bags_a_context_beyond_its_cpu_limit():
    """Upstream refuses over 5,000 rows on a CPU; bags of 1,000 run."""
    pytest.importorskip("tabpfn")
    generator = np.random.default_rng(0)
    X = generator.normal(size=(6_000, 3))
    y = X[:, 0] + generator.normal(0.0, 0.1, 6_000)
    params = {
        "version": "v3.5-fast",
        "n_estimators": 2,
        "device": "cpu",
        "progress": False,
    }
    with pytest.raises(RuntimeError, match="Bag the context below the limit"):
        tabpfn.TabPFNBarDistribution(**params, bag_size=5_500).fit(X, y)
    est = tabpfn.TabPFNBarDistribution(**params, bag_size=1_000).fit(X, y)
    networks = {id(m) for handle in est.handles_ for m in handle.models_}
    assert len(est.handles_) == 2 and len(networks) == 1
    densities = est.predict_proba(X[:4])
    masses = densities * est.native_grid_.widths
    np.testing.assert_allclose(masses.sum(axis=1), 1.0, rtol=1e-6)


# -- seeds ---------------------------------------------------------------------


def test_member_seeds_of_neighbouring_seeds_do_not_overlap():
    """random_state + i gave member i at r the seed of member i - 1 at r + 1."""
    seeds = {
        r: {_members.member_seed(r, i) for i in range(64)} for r in range(32)
    }
    assert all(len(s) == 64 for s in seeds.values())
    for r in range(31):
        assert not seeds[r] & seeds[r + 1]


@pytest.mark.parametrize("bag_rows", [10, 20])
@pytest.mark.parametrize("native", [True, False])
def test_every_group_is_seeded_by_its_first_member(bag_rows, native):
    groups = _members.plan(
        n_estimators=6,
        transforms=_transforms.parse(("none", "robust")),
        native_transforms={"none": "none"},
        feature_shuffle=True,
        bag_rows=bag_rows,
        n_rows=20,
        n_features=3,
        random_state=SEED,
        supports_native_bagging=native,
    )
    for group in groups:
        assert group.seed == _members.member_seed(SEED, group.members[0].index)


def test_no_seed_draws_a_fresh_ensemble_each_fit(monkeypatch, data):
    X, y = data
    cls = _fake_tabicl(monkeypatch, _Recorder())
    params = {"n_estimators": 2, "random_state": None, "device": "cpu"}
    first = cls(**params).fit(X, y)
    second = cls(**params).fit(X, y)
    assert first.random_state_ != second.random_state_
    assert [g.seed for g in first.member_groups_] != [
        g.seed for g in second.member_groups_
    ]
