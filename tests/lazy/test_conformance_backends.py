# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The uniform features work on every registered backend (tier B).

Loads each backend's checkpoint, so it runs only with
LAZY_RUN_CHECKPOINT_TESTS=1. Parametrised over the registry: a newly
registered backend is tested here automatically, and fails at once if it has
no entry in backend_settings.BACKEND_SETTINGS.

Runs on CPU: bit-identity (chunking, the cache) is a property of the maths,
and a GPU's kernel choice and mixed precision change the last digits with
the batch shape.
"""

import os
import warnings

import backend_settings
import numpy as np
import pytest

import lazy
from lazy.models import _transforms

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)

NAMES = sorted(lazy.ESTIMATORS)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4)
    z = rng.uniform(0.2, 1.6, 240)
    X = np.column_stack(
        [np.sin(z * k) + 0.2 * z + rng.normal(0, 0.05, z.size) for k in (1, 2)]
    )
    return X[:200], z[:200], X[200:]


def _skip_without(name):
    """Skips unless the backend's model code can be imported."""
    try:
        lazy.ESTIMATORS[name]()._import_backend()
    except ImportError as error:
        pytest.skip(f"{name} is not installed: {error}")


def _model(name, **params):
    settings = {
        **backend_settings.BACKEND_SETTINGS[name],
        "progress": False,
        "device": "cpu",
    }
    return lazy.ESTIMATORS[name](**{**settings, **params})


def _fit(name, data, **params):
    X, z, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", lazy.PerformanceWarning)
        return _model(name, **params).fit(X, z)


@pytest.mark.parametrize("name", NAMES)
def test_every_backend_has_checkpoint_test_settings(name):
    assert name in backend_settings.BACKEND_SETTINGS


@needs_checkpoint
@pytest.mark.parametrize("name", NAMES)
def test_the_cache_changes_nothing(name, data):
    _skip_without(name)
    _, _, X_test = data
    cached = _fit(name, data, kv_cache=True).predict_proba(
        X_test, lazy.datasets.DC1_GRID
    )
    uncached = _fit(name, data, kv_cache=False).predict_proba(
        X_test, lazy.datasets.DC1_GRID
    )
    rtol = lazy.ESTIMATORS[name].kv_cache_rtol
    np.testing.assert_allclose(
        cached, uncached, rtol=rtol, atol=rtol * uncached.max()
    )


@needs_checkpoint
@pytest.mark.parametrize("name", NAMES)
def test_chunking_changes_nothing(name, data):
    _skip_without(name)
    _, _, X_test = data
    whole = _fit(name, data, chunk_size=0, kv_cache=False)
    chunked = _fit(name, data, chunk_size=7, kv_cache=False)
    whole_pdfs = whole.predict_proba(X_test, lazy.datasets.DC1_GRID)
    chunked_pdfs = chunked.predict_proba(X_test, lazy.datasets.DC1_GRID)
    if lazy.ESTIMATORS[name].exact_chunking:
        np.testing.assert_array_equal(whole_pdfs, chunked_pdfs)
    else:  # independent of the chunk, but batched differently: rounding
        np.testing.assert_allclose(
            whole_pdfs, chunked_pdfs, rtol=1e-4, atol=1e-4 * whole_pdfs.max()
        )


@needs_checkpoint
@pytest.mark.parametrize("name", NAMES)
def test_the_native_grid_is_the_default_and_integrates_to_one(name, data):
    _skip_without(name)
    _, _, X_test = data
    model = _fit(name, data)
    assert model.grid_ == model.native_grid_
    assert model.grid_.normalization == "histogram"
    pdfs = model.predict_proba(X_test)
    np.testing.assert_allclose(pdfs @ model.grid_.widths, 1.0)


@needs_checkpoint
@pytest.mark.parametrize("name", NAMES)
def test_quantiles_invert_the_native_distribution(name, data):
    _skip_without(name)
    _, _, X_test = data
    model = _fit(name, data)
    levels = np.array([0.16, 0.5, 0.84])
    quantiles = model.predict_quantiles(X_test, levels)
    dist = model.predict_distribution(X_test)
    assert (np.diff(quantiles, axis=1) >= 0).all()
    for row in range(len(X_test)):
        np.testing.assert_allclose(
            dist[row].cdf(quantiles[row])[0], levels, atol=1e-6
        )


@needs_checkpoint
@pytest.mark.parametrize("name", NAMES)
def test_a_bag_as_large_as_the_context_is_no_bag(name, data):
    _skip_without(name)
    _, _, X_test = data
    np.testing.assert_array_equal(
        _fit(name, data).predict_proba(X_test, lazy.datasets.DC1_GRID),
        _fit(name, data, bag_size=10**6).predict_proba(
            X_test, lazy.datasets.DC1_GRID
        ),
    )


@needs_checkpoint
@pytest.mark.parametrize("name", NAMES)
def test_bagging_runs_and_the_seed_controls_it(name, data):
    _skip_without(name)
    _, _, X_test = data
    first = _fit(name, data, bag_size=0.5).predict_proba(
        X_test, lazy.datasets.DC1_GRID
    )
    again = _fit(name, data, bag_size=0.5).predict_proba(
        X_test, lazy.datasets.DC1_GRID
    )
    other = _fit(name, data, bag_size=0.5, random_state=7).predict_proba(
        X_test, lazy.datasets.DC1_GRID
    )
    np.testing.assert_array_equal(first, again)
    assert np.isfinite(first).all() and not np.array_equal(first, other)


@needs_checkpoint
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("transforms", [*_transforms.BASE_TRANSFORMS, "limix"])
def test_every_transform_runs(name, transforms, data):
    _skip_without(name)
    _, _, X_test = data
    pdfs = _fit(name, data, transforms=transforms).predict_proba(
        X_test, lazy.datasets.DC1_GRID
    )
    assert np.isfinite(pdfs).all()


@needs_checkpoint
@pytest.mark.parametrize("name", NAMES)
def test_the_auto_recipe_is_the_default(name, data):
    _skip_without(name)
    _, _, X_test = data
    np.testing.assert_array_equal(
        _fit(name, data).predict_proba(X_test, lazy.datasets.DC1_GRID),
        _fit(name, data, transforms="auto").predict_proba(
            X_test, lazy.datasets.DC1_GRID
        ),
    )


@needs_checkpoint
@pytest.mark.parametrize("name", NAMES)
def test_unshuffled_columns_and_missing_values_run(name, data):
    _skip_without(name)
    X, z, X_test = data
    X = X.copy()
    X[::5, 1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", lazy.PerformanceWarning)
        model = _model(name, feature_shuffle=False).fit(X, z)
    assert np.isfinite(
        model.predict_proba(X_test, lazy.datasets.DC1_GRID)
    ).all()
