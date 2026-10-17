# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Installing from a repository, and when a uv project forbids it."""

import pathlib
import shlex
import subprocess
import sys

import pytest

from lazy.models import _install

REQUIREMENT = "pkg @ git+https://example.org/pkg@abc"


def _project(root: pathlib.Path) -> pathlib.Path:
    """A fake uv project at ``root``, with its environment directory."""
    (root / ".venv").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "other"\n')
    (root / "uv.lock").write_text("")
    return root


@pytest.fixture
def where(monkeypatch, tmp_path):
    """Sets the working directory and the interpreter's ``sys.prefix``."""
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "conda").mkdir()

    def at(cwd, prefix):
        monkeypatch.chdir(cwd)
        monkeypatch.setattr(sys, "prefix", str(prefix))

    return at


def test_another_environment_run_inside_a_uv_project_is_not_the_projects(
    where, tmp_path
):
    project = _project(tmp_path / "project")
    where(project, tmp_path / "conda")
    assert _install.uv_project() is None
    assert not _install.in_uv_project()


def test_the_projects_own_environment_is(where, tmp_path):
    project = _project(tmp_path / "project")
    (project / "src").mkdir()
    where(project / "src", project / ".venv")
    assert _install.uv_project() == project
    assert _install.in_uv_project()


def test_the_projects_environment_run_from_elsewhere_is(where, tmp_path):
    project = _project(tmp_path / "project")
    where(tmp_path / "elsewhere", project / ".venv")
    assert _install.uv_project() == project


def test_a_uv_project_environment_setting_moves_it(
    where, tmp_path, monkeypatch
):
    project = _project(tmp_path / "project")
    (project / "envs" / "py312").mkdir(parents=True)
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "envs/py312")
    where(project, project / ".venv")
    assert _install.uv_project() is None
    where(project, project / "envs" / "py312")
    assert _install.uv_project() == project
    where(tmp_path / "elsewhere", project / "envs" / "py312")
    assert _install.uv_project() == project
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "conda"))
    where(project, tmp_path / "conda")
    assert _install.uv_project() == project


def test_a_workspace_member_belongs_to_the_locked_root(where, tmp_path):
    project = _project(tmp_path / "project")
    member = project / "packages" / "member"
    member.mkdir(parents=True)
    (member / "pyproject.toml").write_text('[project]\nname = "member"\n')
    where(member, project / ".venv")
    assert _install.uv_project() == project


def test_a_lock_without_a_pyproject_is_no_project(where, tmp_path):
    (tmp_path / "elsewhere" / "uv.lock").write_text("")
    (tmp_path / "elsewhere" / ".venv").mkdir()
    where(tmp_path / "elsewhere", tmp_path / "elsewhere" / ".venv")
    assert _install.uv_project() is None


@pytest.fixture
def runs(monkeypatch):
    """Records the commands run instead of running them."""
    made = []

    def run(command, check=False):
        del check  # Unused.
        made.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return made


def _install_it(dry_run=False):
    return _install.install(
        "Pkg", REQUIREMENT, what="the build", check="pass", dry_run=dry_run
    )


def test_inside_a_uv_project_another_environment_is_installed_into(
    where, tmp_path, runs, capsys
):
    project = _project(tmp_path / "project")
    where(project, tmp_path / "conda")
    assert _install_it()
    install, check = runs
    assert install[-1] == REQUIREMENT
    assert check[:2] == [sys.executable, "-c"]
    assert "uv add" not in capsys.readouterr().out


def test_from_elsewhere_the_uv_add_names_the_project(
    where, tmp_path, runs, capsys
):
    project = _project(tmp_path / "project")
    where(tmp_path / "elsewhere", project / ".venv")
    assert not _install_it()
    assert not runs
    out = capsys.readouterr().out
    add = shlex.join(["uv", "add", "--project", str(project), REQUIREMENT])
    assert add in out
