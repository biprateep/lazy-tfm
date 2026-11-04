# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The ``lazy`` command: setup, models and download."""

from importlib import util
import pathlib
import shutil
import subprocess
import sys

import pytest

import lazy
from lazy import _cli
from lazy.models import _hub
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


def test_models_lists_every_version_and_marks_the_defaults(monkeypatch, capsys):
    monkeypatch.setattr(_hub, "is_cached", lambda name, version: False)
    assert _cli.main(["models"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows[0] == [
        "MODEL",
        "VERSION",
        "DEFAULT",
        "CACHED",
        "COMMERCIAL",
    ]
    expected = []
    for name in lazy.list_estimators():
        expected.extend([name, version] for version in lazy.list_versions(name))
    assert [row[:2] for row in rows[1:]] == expected
    defaults = {row[0]: row[1] for row in rows[1:] if row[2] == "yes"}
    assert defaults == lazy.DEFAULT_VERSIONS


def test_models_reports_which_licences_allow_commercial_use(
    monkeypatch, capsys
):
    monkeypatch.setattr(_hub, "is_cached", lambda name, version: False)
    assert _cli.main(["models"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    assert {(row[0], row[1]): row[4] == "yes" for row in rows} == {
        (spec.backend, spec.version): spec.commercial_use
        for spec in _hub.CHECKPOINTS.values()
    }


def test_models_reports_which_checkpoints_are_cached(monkeypatch, capsys):
    monkeypatch.setattr(
        _hub,
        "is_cached",
        lambda name, version: (name, version) == ("tabpfn", "v2"),
    )
    assert _cli.main(["models"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    assert {(row[0], row[1]) for row in rows if row[3] == "yes"} == {
        ("tabpfn", "v2")
    }


def test_models_neither_raises_nor_downloads_without_the_backend_package(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        _hub.Checkpoint,
        "download",
        lambda self: pytest.fail("models must not download"),
    )
    monkeypatch.setattr(_hub.Checkpoint, "_fetch", lambda self, **kwargs: 1 / 0)
    assert _cli.main(["models"]) == 0
    assert "no" in capsys.readouterr().out


def test_download_fetches_each_named_checkpoint(monkeypatch, capsys):
    calls = []

    def download(name, version=None):
        calls.append((name, version))
        return pathlib.Path("/cache") / name

    monkeypatch.setattr(_hub, "download_checkpoint", download)
    assert _cli.main(["download", "tabpfn:v2.5", "tabicl"]) == 0
    assert calls == [("tabpfn:v2.5", None), ("tabicl", None)]
    assert capsys.readouterr().out.splitlines() == [
        str(pathlib.Path("/cache/tabpfn:v2.5")),
        str(pathlib.Path("/cache/tabicl")),
    ]


def test_download_of_an_unknown_name_exits_2_naming_the_choices(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        _hub,
        "download_checkpoint",
        lambda name, version=None: pytest.fail("nothing is downloaded"),
    )
    with pytest.raises(SystemExit) as exit_info:
        _cli.main(["download", "tabicl", "tabpfnn"])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "tabpfnn" in err
    assert all(name in err for name in lazy.list_estimators())


def test_download_with_no_checkpoint_named_exits_2():
    with pytest.raises(SystemExit) as exit_info:
        _cli.main(["download"])
    assert exit_info.value.code == 2


def test_download_passes_the_licence_notice_to_the_user(monkeypatch):
    def download(self, *, local_files_only=False):
        _hub._warnings.warn(self.license_note or "notice", _hub.LicenseWarning)
        return pathlib.Path("/cache")

    monkeypatch.setattr(_hub.Checkpoint, "download", download)
    with pytest.warns(_hub.LicenseWarning):
        assert _cli.main(["download", "tabpfn:v2.5"]) == 0


def test_download_of_an_unknown_version_exits_2_naming_the_versions(capsys):
    with pytest.raises(SystemExit) as exit_info:
        _cli.main(["download", "tabicl:v9"])
    assert exit_info.value.code == 2
    assert "known: ['v2']" in capsys.readouterr().err
