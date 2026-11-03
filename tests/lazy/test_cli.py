# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""``lazy setup``: TabFM's repository build, LimiX's source and Kumo's sdm."""

from importlib import util
import pathlib
import shutil
import subprocess
import sys

import pytest

from lazy import _cli
from lazy.models import _icl_stream
from lazy.models import _limix_source
from lazy.models import kumo
from lazy.models import tabfm


@pytest.fixture
def backends(monkeypatch, tmp_path):
    """The extras installed, the TabFM release, no LimiX source, no sdm."""
    monkeypatch.chdir(tmp_path)  # outside any uv project
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))  # not uv's
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.setattr(util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda name: None)  # no uv on PATH
    monkeypatch.setattr(_icl_stream, "streaming_available", lambda: False)
    runs = []

    def run(command, check=False):
        del check  # Unused.
        runs.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    fetched = []

    def locate(*, download=True):
        if not download:
            raise ImportError("not cached")
        fetched.append(True)
        return _limix_source.Source(
            tmp_path, _limix_source.LIMIX_COMMIT, _limix_source.CACHE_ORIGIN
        )

    monkeypatch.setattr(_limix_source, "locate", locate)
    monkeypatch.setitem(sys.modules, "sdm", None)  # no Kumo package
    return monkeypatch, runs, fetched


def test_setup_installs_the_build_and_fetches_the_source(backends, capsys):
    _, runs, fetched = backends
    assert _cli.main(["setup"]) == 0
    install, check = runs
    assert install == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        tabfm.REPOSITORY_BUILD,
    ]
    assert check[:2] == [sys.executable, "-c"]  # a fresh interpreter checks
    assert fetched == [True]
    assert "uv add" not in capsys.readouterr().out


def test_a_dry_run_changes_nothing(backends):
    _, runs, fetched = backends
    assert _cli.main(["setup", "--dry-run"]) == 0
    assert not runs
    assert not fetched


def test_complete_backends_are_left_alone(backends, capsys):
    monkeypatch, runs, fetched = backends
    monkeypatch.setattr(_icl_stream, "streaming_available", lambda: True)
    monkeypatch.setattr(
        _limix_source,
        "locate",
        lambda download=True: _limix_source.Source(
            pathlib.Path("/src"), None, "$LAZY_LIMIX_SRC"
        ),
    )
    assert _cli.main(["setup"]) == 0
    assert not runs
    out = capsys.readouterr().out
    assert f"{pathlib.Path('/src')} ($LAZY_LIMIX_SRC)" in out


def test_missing_extras_are_skipped(backends, capsys):
    monkeypatch, runs, fetched = backends
    monkeypatch.setattr(util, "find_spec", lambda name: None)
    assert _cli.main(["setup"]) == 0
    assert not runs
    assert not fetched
    out = capsys.readouterr().out
    assert "'lazy-tfm[tabfm]'" in out
    assert "'lazy-tfm[limix]'" in out


def test_without_pip_uv_installs_into_this_interpreter(backends, tmp_path):
    monkeypatch, runs, _ = backends
    monkeypatch.setattr(
        util,
        "find_spec",
        lambda name: None if name == "pip" else object(),
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/uv")
    assert _cli.main(["setup"]) == 0
    assert runs[0][:5] == [
        "/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
    ]
    assert "--reinstall" in runs[0]


def test_in_a_uv_project_setup_prints_the_uv_add_instead(
    backends, tmp_path, capsys
):
    monkeypatch, runs, fetched = backends
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "mine"\n')
    (tmp_path / "uv.lock").write_text("")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv"))
    assert _cli.main(["setup"]) == 1  # TabFM is not complete yet
    assert runs == []  # an install uv sync would undo is not made
    assert fetched == [True]  # LimiX is still set up
    out = capsys.readouterr().out
    assert f"uv add '{tabfm.REPOSITORY_BUILD}'" in out
    assert _cli.main(["setup", "--dry-run"]) == 1  # as the real run
    assert runs == []
    assert (
        "TabFM: not installing the repository build" in capsys.readouterr().out
    )


def test_a_failed_install_fails_the_command(backends):
    monkeypatch, runs, _ = backends
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, check=False: subprocess.CompletedProcess(command, 1),
    )
    assert _cli.main(["setup"]) == 1


def test_unnamed_setup_neither_installs_nor_mentions_kumo(backends, capsys):
    _, runs, _ = backends
    assert _cli.main(["setup"]) == 0
    assert all(kumo._VALIDATED_RELEASE not in run for run in runs)
    assert "Kumo" not in capsys.readouterr().out


def test_naming_kumo_warns_and_installs_a_missing_package(backends):
    _, runs, fetched = backends
    with pytest.warns(FutureWarning, match="removed in 0.7.0"):
        assert _cli.main(["setup", "kumo"]) == 0
    install, check = runs  # Kumo's alone: TabFM and LimiX were not named.
    assert install[-1] == kumo._VALIDATED_RELEASE
    assert check[:2] == [sys.executable, "-c"]
    assert not fetched


def test_naming_kumo_warns_and_leaves_an_importable_package(backends, capsys):
    monkeypatch, runs, _ = backends
    monkeypatch.setattr(kumo.KumoQuantile, "_import_backend", lambda self: None)
    with pytest.warns(FutureWarning, match=r"'lazy-tfm\[kumo\]' now installs"):
        assert _cli.main(["setup", "kumo"]) == 0
    assert not runs
    assert "structured-data-models is installed" in capsys.readouterr().out


def test_naming_kumo_without_its_dependencies_fails(backends, capsys):
    monkeypatch, runs, _ = backends
    monkeypatch.setattr(
        util, "find_spec", lambda name: None if name == "pyarrow" else object()
    )
    with pytest.warns(FutureWarning):
        assert _cli.main(["setup", "kumo"]) == 1
    assert not runs
    assert "pip install 'lazy-tfm[kumo]'" in capsys.readouterr().out


def test_an_unknown_backend_is_an_error(backends):
    with pytest.raises(SystemExit):
        _cli.main(["setup", "kumoo"])
