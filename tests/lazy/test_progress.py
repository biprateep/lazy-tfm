# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The progress bars: the helper, the parameter, and what each backend reports.

The helper tests need nothing. The per-backend ones load a checkpoint, so they
run only with ``LAZY_RUN_CHECKPOINT_TESTS=1``, like the rest of
``test_backends.py``.
"""

import os

import backend_settings
import numpy as np
import pandas as pd
import pytest
from sklearn import base as sklearn_base

import lazy
from lazy.models import _progress

needs_checkpoint = pytest.mark.skipif(
    os.environ.get("LAZY_RUN_CHECKPOINT_TESTS") != "1",
    reason=(
        "set LAZY_RUN_CHECKPOINT_TESTS=1 to run tests that load a checkpoint"
    ),
)


@pytest.mark.parametrize("value", ["auto", True, False])
def test_the_three_modes_are_accepted(value):
    _progress.check_progress(value)


@pytest.mark.parametrize("value", [1, 0, "yes", None, "AUTO"])
def test_anything_else_is_rejected(value):
    """``1`` in particular: it equals ``True`` and must still be refused."""
    with pytest.raises(ValueError, match="progress must be one of"):
        _progress.check_progress(value)


def test_forced_on_writes_to_stderr_not_stdout(capsys):
    with _progress.bar(True, total=4, desc="demo", unit="row") as progress:
        progress.update(4)
    captured = capsys.readouterr()
    assert not captured.out
    assert "demo" in captured.err and "4/4" in captured.err


def test_auto_stays_quiet_when_output_is_not_a_terminal(capsys):
    """Pytest captures stderr, which is exactly the batch-job log case."""
    with _progress.bar("auto", total=4, desc="demo", unit="row") as progress:
        progress.update(4)
    assert not capsys.readouterr().err


def test_off_is_off(capsys):
    with _progress.bar(False, total=4, desc="demo", unit="row") as progress:
        progress.update(4)
    assert not capsys.readouterr().err


@pytest.mark.parametrize("name", sorted(lazy.ESTIMATORS))
def test_every_backend_takes_the_parameter_and_clones_it(name):
    estimator = lazy.ESTIMATORS[name](progress=False)
    assert estimator.get_params()["progress"] is False
    assert sklearn_base.clone(estimator).progress is False
    assert lazy.ESTIMATORS[name]().progress == "auto"


# -- what each backend actually reports ---------------------------------------

#: Small enough for CPU, with the settings each backend's own tests use.
SETTINGS = backend_settings.BACKEND_SETTINGS

#: What the finished bar must say: the unit it counts in and the final tally.
#: TabPFN and TabICL count rows, TabFM counts in-context stages --
#: n_dither * (1 + n_coarse_bins) = 2 * 3 here.
EXPECTED = {
    "tabpfn": ("40/40", "row", "buckets="),
    "tabicl": ("40/40", "row", "quantiles="),
    "tabfm": ("6/6", "stage", "level=fine 2/2"),
}


@pytest.fixture
def data():
    generator = np.random.default_rng(0)
    z = generator.uniform(0.2, 1.8, 240)
    X = pd.DataFrame(
        {
            "a": z + generator.normal(0, 0.1, 240),
            "b": generator.normal(size=240),
        }
    )
    return X.iloc[:200], z[:200], X.iloc[200:].reset_index(drop=True)


@needs_checkpoint
@pytest.mark.parametrize("name", sorted(lazy.ESTIMATORS))
def test_the_bar_reports_the_backends_own_unit_and_changes_nothing(
    name, data, capsys
):
    pytest.importorskip(name)
    X_ctx, z_ctx, X_q = data

    def run(progress):
        model = lazy.LazyModel(
            name, device="cpu", progress=progress, **SETTINGS[name]
        )
        return model.fit(X_ctx, z_ctx).predict_proba(X_q)

    quiet = run(False)
    capsys.readouterr()
    shown = run(True)
    err = capsys.readouterr().err

    assert np.array_equal(quiet, shown)
    tally, unit, postfix = EXPECTED[name]
    assert tally in err and unit in err and postfix in err
    assert "context=" in err
