# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Every registered backend arrives with everything a backend needs.

Registering a class puts it in LazyModel, the ensembles and the conformance
suites, but a backend needs more than the registry can check: a pinned commit
and recipe for every version, a pip extra, a docs page, golden densities, and
the settings and fakes the other test suites run it with. One test per
registered backend lists everything it still lacks. The docs' tables of the
models are written by docs/conf.py from the checkpoints and the registry, so
what is checked for them is the fields they are written from.
"""

import importlib
import pathlib
import re
import tomllib

import backend_settings
import fakes
import golden_data
import pytest
import standins
import test_golden
import test_weights_cache

import lazy

ROOT = pathlib.Path(__file__).parents[2]
MODELS_DOCS = ROOT / "docs" / "models"
GPU_NEEDS = ("yes", "recommended", "optional")
# The size docs/conf.py quotes from a checkpoint's size_note.
SIZE_NOTE = re.compile(r"~?[\d.,]+ [kMGT]?B\b")


def _extras():
    """The pip extras pyproject.toml declares, by name."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return pyproject["project"]["optional-dependencies"]


def _docs_toctree():
    """The pages the models overview's toctree lists."""
    index = (MODELS_DOCS / "index.md").read_text()
    (block,) = re.findall(r"```\{toctree\}\n(.*?)```", index, flags=re.DOTALL)
    return {
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.startswith(":")
    }


def _checkpoint_gaps(name, cls):
    """What the backend's pinned versions lack."""
    if name not in lazy.DEFAULT_VERSIONS:
        return [f"no default version: add {name!r} to _hub.DEFAULT_VERSIONS"]
    found = []
    for version in lazy.list_versions(name):
        revision = lazy.get_checkpoint(name, version).revision or ""
        if not re.fullmatch("[0-9a-f]{40}", revision):
            found.append(
                f"checkpoint {name}:{version} pins no full commit hash "
                f"(revision={revision!r} in _hub.CHECKPOINTS)"
            )
        found.extend(_table_gaps(lazy.get_checkpoint(name, version)))
        try:
            recipe = cls._pinned_recipe(version)
        except KeyError:
            found.append(
                f"no recipe pinned for version {version!r} "
                f"({cls.__name__}._pinned_recipe)"
            )
        else:
            if not recipe:
                found.append(f"the recipe pinned for {version!r} is empty")
    return found


def _table_gaps(checkpoint):
    """What a checkpoint lacks for the docs' tables of the models."""
    found = [
        f"checkpoint {checkpoint.key} has no {field} (_hub.CHECKPOINTS)"
        for field in (
            "display_name",
            "parameters",
            "size_bytes",
            "license_name",
            "cpu",
        )
        if not getattr(checkpoint, field)
    ]
    if checkpoint.gpu not in GPU_NEEDS:
        found.append(
            f"checkpoint {checkpoint.key} has gpu={checkpoint.gpu!r}, not one "
            f"of {GPU_NEEDS} (_hub.CHECKPOINTS)"
        )
    if not SIZE_NOTE.match(checkpoint.size_note):
        found.append(
            f"checkpoint {checkpoint.key}'s size_note does not start with "
            "its size, such as '~100 MB' (_hub.CHECKPOINTS)"
        )
    return found


def _docs_gaps(name, cls):
    """What the documentation lacks."""
    found = []
    if not (MODELS_DOCS / f"{name}.md").is_file():
        found.append(f"no docs page docs/models/{name}.md")
    if name not in _docs_toctree():
        found.append(f"docs/models/index.md's toctree does not list {name}")
    if not getattr(cls, "method_note", ""):
        found.append(
            f"no method_note on {cls.__name__} for the table in "
            "docs/guide/interface.md"
        )
    return found


def _test_gaps(name, cls):
    """What the test suites lack."""
    found = []
    if name not in golden_data.RECORDED_PARAMS:
        found.append(
            "no golden settings in tests/lazy/golden_data.RECORDED_PARAMS"
        )
    if not (golden_data.GOLDEN_DIR / f"{name}.npz").is_file():
        found.append(
            f"no golden file tests/lazy/golden/{name}.npz (record it with "
            f"tests/lazy/record_golden.py {name})"
        )
    if name not in test_golden.RTOL:
        found.append("no golden tolerance in tests/lazy/test_golden.RTOL")
    if name not in backend_settings.BACKEND_SETTINGS:
        found.append(
            "no CPU settings in tests/lazy/backend_settings.BACKEND_SETTINGS"
        )
    fake = fakes.FAKES_DIR / f"{name}.py"
    if not fake.is_file():
        found.append(
            f"no fake upstream tests/lazy/fakes/{name}.py (see "
            "tests/lazy/fakes/__init__.py)"
        )
    else:
        module = importlib.import_module(f"fakes.{name}")
        found.extend(
            f"tests/lazy/fakes/{name}.py defines no {attribute}"
            for attribute in ("install", "SETTINGS")
            if not hasattr(module, attribute)
        )
    real = [param.values[0] for param in test_weights_cache._REAL]
    if cls not in real:
        found.append(
            "no real-weights entry in tests/lazy/test_weights_cache._REAL"
        )
    return found


def _missing(name):
    """Everything the backend registered as ``name`` lacks, one line each."""
    cls = lazy.ESTIMATORS[name]
    found = _checkpoint_gaps(name, cls)
    if cls.extra not in _extras():
        found.append(f"no pip extra {cls.extra!r} in pyproject.toml")
    found.extend(_docs_gaps(name, cls))
    found.extend(_test_gaps(name, cls))
    return found


@pytest.mark.parametrize("name", sorted(lazy.ESTIMATORS))
def test_every_registered_backend_is_complete(name):
    missing = _missing(name)
    assert not missing, f"backend {name!r} lacks:\n  - " + "\n  - ".join(
        missing
    )


def test_a_new_backend_is_told_everything_it_lacks(monkeypatch):
    monkeypatch.setitem(lazy.ESTIMATORS, "newcomer", standins.HistogramStandIn)
    missing = "\n".join(_missing("newcomer"))
    for gap in (
        "DEFAULT_VERSIONS",
        "pip extra",
        "docs/models/newcomer.md",
        "toctree",
        "method_note",
        "RECORDED_PARAMS",
        "golden/newcomer.npz",
        "RTOL",
        "BACKEND_SETTINGS",
        "fakes/newcomer.py",
        "_REAL",
    ):
        assert gap in missing


def test_a_version_without_a_recipe_is_reported(monkeypatch):
    cls = lazy.ESTIMATORS["tabicl"]
    checkpoint = lazy.get_checkpoint("tabicl")
    monkeypatch.setitem(
        lazy.CHECKPOINTS,
        "tabicl:v9",
        type(checkpoint)(**{**vars(checkpoint), "version": "v9"}),
    )
    assert _checkpoint_gaps("tabicl", cls) == [
        "no recipe pinned for version 'v9' (TabICLQuantile._pinned_recipe)"
    ]


def test_a_checkpoint_without_its_table_fields_is_reported(monkeypatch):
    cls = lazy.ESTIMATORS["tabicl"]
    checkpoint = lazy.get_checkpoint("tabicl")
    bare = {
        **vars(checkpoint),
        "display_name": "",
        "parameters": 0,
        "gpu": "some",
        "size_note": "About 100 MB.",
    }
    monkeypatch.setitem(lazy.CHECKPOINTS, "tabicl:v2", type(checkpoint)(**bare))
    missing = "\n".join(_checkpoint_gaps("tabicl", cls))
    for gap in ("display_name", "parameters", "gpu='some'", "size_note"):
        assert gap in missing
    assert "size_bytes" not in missing
