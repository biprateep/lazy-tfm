# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""How TabFM chooses its bins: sized from the context, or the user's grid.

Everything but the last test runs a stand-in classifier, so the bin layout
and the probability bookkeeping are checked without a checkpoint. The last
loads TabFM on the CPU, and runs only with ``LAZY_RUN_CHECKPOINT_TESTS=1``.
"""

import os
import warnings

import numpy as np
import pytest

import lazy
from lazy import datasets
from lazy.models import tabfm

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)

#: A grid of 25 bins over the targets of _problem.
GRID_25 = lazy.Grid.linear(0.0, 1.75, 25)

#: The bin counts that take the bins from the constructor's grid.
GRID = {"n_coarse_bins": "grid", "n_fine_bins": "grid"}


@pytest.fixture(autouse=True)
def _needs_tabfm():
    pytest.importorskip("tabfm")


class _Nearest:
    """A stand-in classifier, from the distance to each class's mean.

    The softmax of minus the distance, so its answers depend on the context
    and the query, and a class absent from the context never appears.
    """

    seen: list = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        _Nearest.seen.append(self.classes_)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        distance = ((X[:, None, :] - self.means_[None]) ** 2).sum(axis=-1)
        logits = distance.min(axis=1, keepdims=True) - distance
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)


def _problem(n_context, n_query=40, seed=0):
    """Photometry-like features of a redshift-like target in (0.05, 1.6)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.05, 1.6, n_context + n_query)
    X = np.column_stack(
        [np.sin(2.0 * z + k) + rng.normal(0.0, 0.05, z.size) for k in range(3)]
    )
    return X[:n_context], z[:n_context], X[n_context:]


def _fit(monkeypatch, X, z, **params):
    """A TabFMHistogram on the stand-in classifier, fitted."""
    _Nearest.seen = []
    est = tabfm.TabFMHistogram(
        n_estimators=1, kv_cache=False, device="cpu", progress=False, **params
    )
    monkeypatch.setattr(est, "_classifier", lambda *args: _Nearest())
    monkeypatch.setattr(est, "_backbone", lambda: None)
    return est.fit(X, z)


@pytest.mark.parametrize(
    ("n_context", "expected"),
    [
        (1, 2),
        (19, 2),
        (20, 2),
        (45, 3),
        (80, 4),
        (277, 7),
        (499, 9),
        (500, 10),
        (10_000, 10),
    ],
)
def test_auto_gives_about_five_rows_per_bin(n_context, expected):
    assert tabfm.auto_bin_count(n_context) == expected


@pytest.mark.parametrize("n_context", [1, 4, 30])
def test_auto_fits_contexts_of_any_size(monkeypatch, n_context):
    X, z, X_query = _problem(n_context)
    est = _fit(monkeypatch, X, z)
    assert est.bins_ == "auto equal-mass 2x2"
    masses = est.predict_distribution(X_query).masses
    np.testing.assert_allclose(masses.sum(axis=1), 1.0)


@pytest.mark.parametrize("y_grid", [None, datasets.DC1_GRID])
@pytest.mark.parametrize("n_context", [500, 1_000, 10_000])
def test_auto_is_the_old_ten_by_ten_from_500_rows(
    monkeypatch, n_context, y_grid
):
    """From 500 rows "auto" is 10 x 10, bit for bit.

    With DC1's 200-bin grid in the constructor too, which "auto" never
    matches.
    """
    X, z, X_query = _problem(n_context)
    auto = _fit(monkeypatch, X, z, y_grid=y_grid)
    fixed = _fit(
        monkeypatch, X, z, y_grid=y_grid, n_coarse_bins=10, n_fine_bins=10
    )
    assert auto.bin_grid_ is None
    assert (auto.n_coarse_bins_, auto.n_fine_bins_) == (10, 10)
    assert auto.native_grid_ == fixed.native_grid_
    np.testing.assert_array_equal(
        auto.native_grid_.edges, fixed.native_grid_.edges
    )
    np.testing.assert_array_equal(
        auto.predict_proba(X_query), fixed.predict_proba(X_query)
    )


def test_provenance_records_the_bins(monkeypatch):
    X, z, _ = _problem(277)
    assert _fit(monkeypatch, X, z).provenance_["bins"] == (
        "auto equal-mass 7x7"
    )
    explicit = _fit(monkeypatch, X, z, n_coarse_bins=5, n_fine_bins=4)
    assert explicit.provenance_["bins"] == "equal-mass 5x4"
    mixed = _fit(monkeypatch, X, z, n_fine_bins=3)
    assert mixed.provenance_["bins"] == "equal-mass 7x3"
    gridded = _fit(monkeypatch, X, z, y_grid=GRID_25)
    assert gridded.provenance_["bins"] == "auto equal-mass 7x7"
    matched = _fit(monkeypatch, X, z, y_grid=GRID_25, **GRID)
    assert matched.provenance_["bins"] == "grid 25 bins in 3 groups"
    assert matched.provenance_["n_dither"] == 1


@pytest.mark.parametrize(
    ("n_bins", "sizes"),
    [(2, [2]), (10, [10]), (11, [6, 5]), (25, [9, 8, 8]), (100, [10] * 10)],
)
def test_grid_groups_are_the_fewest_balanced_runs(n_bins, sizes):
    bounds = tabfm.grid_groups(n_bins)
    assert np.diff(bounds).tolist() == sizes


def test_a_matched_grid_s_bins_are_the_classes(monkeypatch):
    X, z, X_query = _problem(400)
    est = _fit(monkeypatch, X, z, y_grid=GRID_25, **GRID)
    assert est.bin_grid_ == GRID_25
    assert (est.n_coarse_bins_, est.n_fine_bins_) == (3, 9)
    edges, coarse, fine = est._edges(est.y_context_, 0.0)
    np.testing.assert_array_equal(edges, GRID_25.edges)
    np.testing.assert_array_equal(coarse, GRID_25.edges[[0, 9, 17, 25]])
    assert [f.size - 1 for f in fine] == [9, 8, 8]
    dist = est.predict_distribution(X_query)
    # The coarse classifier, then each group's, over the grid's own bins.
    assert [c.tolist() for c in _Nearest.seen] == [
        [0, 1, 2],
        list(range(9)),
        list(range(8)),
        [0, 1, 2, 3, 4, 5],  # targets stop at 1.6, inside bin 22 of 25
    ]
    np.testing.assert_array_equal(dist.bins, GRID_25.edges)
    np.testing.assert_allclose(dist.masses.sum(axis=1), 1.0)
    # On the grid itself the densities are the masses over the widths.
    assert est.native_grid_ is est.bin_grid_
    np.testing.assert_allclose(
        est.predict_proba(X_query),
        GRID_25.normalize(dist.masses / GRID_25.widths),
        rtol=1e-12,
    )
    on_histogram = lazy.Grid.from_edges(
        GRID_25.edges, normalization="histogram"
    )
    np.testing.assert_allclose(
        est.predict_proba(X_query, on_histogram) * on_histogram.widths,
        dist.masses,
        rtol=1e-12,
        atol=1e-15,
    )


def test_a_matched_grid_s_empty_bins_get_zero(monkeypatch):
    """No rows: zero; a group whose rows share one bin gives it all."""
    X, z, X_query = _problem(300)
    grid = lazy.Grid.linear(0.0, 2.0, 20)
    # Group 0 (bins 0-9, up to 1.0) loses bins 2-4; group 1 (bins 10-19)
    # keeps one bin's worth of rows, all in bin 11.
    keep = ((z < 0.2) | (z >= 0.5)) & (z < 1.0)
    z_used = np.r_[z[keep], np.full(5, 1.15)]
    X_used = np.r_[X[keep], X[:5]]
    est = _fit(monkeypatch, X_used, z_used, y_grid=grid, **GRID)
    masses = est.predict_distribution(X_query).masses
    assert np.all(masses[:, 2:5] == 0.0)
    assert np.all(masses[:, 10] == 0.0)
    assert np.all(masses[:, 12:] == 0.0)
    assert np.all(masses[:, 11] > 0.0)
    np.testing.assert_allclose(masses.sum(axis=1), 1.0)
    prior = est.bin_prior_
    assert prior[11] == pytest.approx(5 / z_used.size)
    np.testing.assert_allclose(prior.sum(), 1.0)


def test_a_grid_of_ten_bins_or_fewer_is_one_classifier(monkeypatch):
    X, z, X_query = _problem(200)
    grid = lazy.Grid.linear(0.0, 1.75, 7)
    est = _fit(monkeypatch, X, z, y_grid=grid, **GRID)
    assert est.bins_ == "grid 7 bins in 1 groups"
    est.predict_distribution(X_query)
    # One class at the coarse level runs no classifier.
    assert [c.size for c in _Nearest.seen] == [7]


def test_targets_outside_a_matched_grid_go_to_its_end_bins(monkeypatch):
    X, z, X_query = _problem(300)
    grid = lazy.Grid.linear(0.3, 1.3, 20)
    with pytest.warns(UserWarning, match="outside the y_grid range"):
        est = _fit(monkeypatch, X, z, y_grid=grid, **GRID)
    est.predict_distribution(X_query)
    prior = est.bin_prior_
    assert prior[0] == pytest.approx(np.mean(z < 0.35))
    assert prior[-1] == pytest.approx(np.mean(z >= 1.25))


def test_a_matched_grid_refuses_dithering(monkeypatch):
    X, z, _ = _problem(300)
    with pytest.raises(ValueError, match="n_dither > 1"):
        _fit(monkeypatch, X, z, y_grid=GRID_25, n_dither=3, **GRID)


def test_dithering_still_works_with_an_unmatched_grid(monkeypatch):
    X, z, X_query = _problem(300)
    for options in (
        {"y_grid": datasets.DC1_GRID},
        {"y_grid": GRID_25},
        {"y_grid": GRID_25, "n_coarse_bins": 4, "n_fine_bins": 4},
    ):
        est = _fit(monkeypatch, X, z, n_dither=3, **options)
        assert est.bin_grid_ is None
        native = est.native_grid_
        pdfs = est.predict_proba(X_query, native)
        np.testing.assert_allclose((pdfs * native.widths).sum(axis=1), 1.0)


def test_explicit_bin_counts_are_equal_mass_with_a_grid(monkeypatch):
    X, z, X_query = _problem(300)
    est = _fit(
        monkeypatch, X, z, y_grid=GRID_25, n_coarse_bins=4, n_fine_bins=4
    )
    assert est.bins_ == "equal-mass 4x4"
    assert est.bin_grid_ is None
    assert est.native_grid_.n_bins == 16
    assert est.predict_distribution(X_query).masses.shape == (40, 16)


def test_auto_never_matches_the_grid(monkeypatch):
    """Matching is opt-in: "auto" stays equal-mass whatever the grid."""
    X, z, X_query = _problem(300)
    est = _fit(monkeypatch, X, z, y_grid=GRID_25)
    assert est.bins_ == "auto equal-mass 7x7"
    assert est.bin_grid_ is None
    assert est.native_grid_ != GRID_25
    assert est.predict_distribution(X_query).masses.shape == (40, 49)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"n_coarse_bins": "grid", "y_grid": GRID_25}, "must both be 'grid'"),
        (
            {"n_coarse_bins": "grid", "n_fine_bins": 5, "y_grid": GRID_25},
            "must both be 'grid'",
        ),
        ({"n_fine_bins": "grid", "y_grid": GRID_25}, "must both be 'grid'"),
        ({**GRID}, "need a grid there"),
        ({**GRID, "y_grid": "native"}, "need a grid there"),
        ({**GRID, "y_grid": datasets.DC1_GRID}, "y_grid has 200"),
    ],
)
def test_grid_bins_are_refused_where_they_cannot_be_built(
    monkeypatch, options, match
):
    X, z, _ = _problem(300)
    with pytest.raises(ValueError, match=match):
        _fit(monkeypatch, X, z, **options)


def test_a_grid_of_100_bins_is_ten_groups_of_ten(monkeypatch):
    X, z, _ = _problem(300)
    grid = lazy.Grid.linear(0.0, 1.75, 100)
    est = _fit(monkeypatch, X, z, y_grid=grid, **GRID)
    assert (est.n_coarse_bins_, est.n_fine_bins_) == (10, 10)
    assert est.bins_ == "grid 100 bins in 10 groups"


def test_a_call_time_grid_does_not_change_the_bins(monkeypatch):
    X, z, X_query = _problem(300)
    est = _fit(monkeypatch, X, z)
    before = est.predict_distribution(X_query)
    pdfs = est.predict_proba(X_query, GRID_25)
    after = est.predict_distribution(X_query)
    assert est.bins_ == "auto equal-mass 7x7"
    np.testing.assert_array_equal(before.bins, after.bins)
    np.testing.assert_array_equal(before.masses, after.masses)
    np.testing.assert_allclose(pdfs, before.on_grid(GRID_25), rtol=1e-12)


def test_an_array_of_centres_is_matched_like_a_grid(monkeypatch):
    X, z, _ = _problem(300)
    est = _fit(monkeypatch, X, z, y_grid=GRID_25.centers, **GRID)
    assert est.bin_grid_ == GRID_25


@pytest.mark.parametrize("value", ["fine", 1, 11, 2.0, True, None])
def test_bin_counts_are_auto_grid_or_ints_from_two_to_ten(monkeypatch, value):
    X, z, _ = _problem(300)
    with pytest.raises(ValueError, match="n_fine_bins"):
        _fit(monkeypatch, X, z, n_fine_bins=value)


@needs_checkpoint
def test_the_checkpoint_gives_auto_and_ten_by_ten_alike():
    """The real classifier, on the CPU: 500 rows are 10 x 10 either way."""
    X, z, X_query = _problem(500, n_query=20)
    runs = []
    for options in ({}, {"n_coarse_bins": 10, "n_fine_bins": 10}):
        est = tabfm.TabFMHistogram(
            n_estimators=1, device="cpu", progress=False, **options
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            runs.append(est.fit(X, z).predict_proba(X_query))
    np.testing.assert_array_equal(*runs)


@needs_checkpoint
def test_the_checkpoint_predicts_on_a_matched_grid():
    """Real class labels with gaps: the absent bins come back as zero."""
    X, z, X_query = _problem(200, n_query=20)
    grid = lazy.Grid.linear(0.0, 2.0, 40)
    est = tabfm.TabFMHistogram(
        n_estimators=1, device="cpu", progress=False, y_grid=grid, **GRID
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dist = est.fit(X, z).predict_distribution(X_query)
    np.testing.assert_array_equal(dist.bins, grid.edges)
    np.testing.assert_allclose(dist.masses.sum(axis=1), 1.0)
    assert np.all(dist.masses[:, grid.centers > 1.6] == 0.0)
