# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Installing a package from its repository, for ``lazy setup``.

PyPI refuses a git URL in an extra, so a backend whose package (or whose
fast build of it) is on GitHub alone has ``lazy setup`` install it from there
(:meth:`lazy.models.ContextEnsembleEstimator.setup`). :func:`install`
does it the same way for every such backend: with pip when this interpreter
has it, else with uv, leaving the dependencies to the extra; when this
interpreter runs a uv project's environment it prints the ``uv add`` that
keeps the package instead, since ``uv sync`` would remove it again (in a
checkout of this repository, the ``uv sync`` whose sources hold it).

Typical usage example:

  complete = _install.install(
      "TabFM",
      "tabfm[pytorch] @ git+https://github.com/google-research/tabfm@<commit>",
      what="the repository build",
      check="from tabfm import TabFM",
      dry_run=False,
  )
"""

from __future__ import annotations

from importlib import util
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tomllib

__all__ = [
    "CHECKOUT_SYNC",
    "in_uv_project",
    "install",
    "install_command",
    "uv_project",
]

#: What gives a checkout of this repository every backend: the extras, and
#: the dependency group holding what PyPI cannot (pyproject.toml).
CHECKOUT_SYNC = ("uv", "sync", "--extra", "all", "--group", "backends")


def install_command(requirement: str) -> list[str] | None:
    """The command replacing an installed package with ``requirement``.

    pip when this interpreter has it, else uv, whose environments have no
    pip; both reinstall even though a build and a release may share a
    version number, and leave the dependencies alone.

    Args:
        requirement: A PEP 508 requirement, such as a ``name @ git+URL``.

    Returns:
        The command, or ``None`` when neither installer is available.
    """
    if util.find_spec("pip") is not None:
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            requirement,
        ]
    uv = shutil.which("uv")
    if uv is None:
        return None
    return [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall",
        "--no-deps",
        requirement,
    ]


def _nearest_project(path: pathlib.Path) -> pathlib.Path | None:
    """The nearest directory at or above ``path`` holding a uv project.

    Args:
        path: Where to start looking.

    Returns:
        The first of ``path`` and its parents with both a pyproject.toml and
        a uv.lock, or None when there is none.
    """
    for directory in (path, *path.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "uv.lock"
        ).is_file():
            return directory
    return None


def uv_project() -> pathlib.Path | None:
    """The uv project whose environment this interpreter runs, if any.

    A project is the nearest directory with a pyproject.toml and a uv.lock
    beside it, and its environment is the ``.venv`` there, or
    ``$UV_PROJECT_ENVIRONMENT`` when that is set (relative to the project).
    The project is looked for above the working directory and above
    ``sys.prefix``, so that ``.venv/bin/lazy setup`` run from elsewhere is
    still recognized, and it counts only when its environment is
    ``sys.prefix``: a pip or conda environment run from inside a uv
    project's directory is not the project's, and ``uv sync`` never touches
    it.

    Returns:
        The project's directory, or None when this interpreter runs no uv
        project's environment.
    """
    prefix = pathlib.Path(sys.prefix).resolve()
    environment = os.environ.get("UV_PROJECT_ENVIRONMENT") or ".venv"
    for start in (pathlib.Path.cwd(), prefix):
        root = _nearest_project(start)
        if root is not None and (root / environment).resolve() == prefix:
            return root
    return None


def _is_checkout(project: pathlib.Path) -> bool:
    """Whether a uv project is a checkout of this repository, lazy-tfm.

    Args:
        project: The project's directory.

    Returns:
        Whether its pyproject.toml names the project ``lazy-tfm``.
    """
    try:
        with open(project / "pyproject.toml", "rb") as f:
            name = tomllib.load(f).get("project", {}).get("name")
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return name == "lazy-tfm"


def in_uv_project() -> bool:
    """Whether this interpreter runs a uv project's environment.

    See :func:`uv_project` for the rule.
    """
    return uv_project() is not None


def install(
    label: str,
    requirement: str,
    *,
    what: str,
    check: str,
    missing: str = "",
    dry_run: bool,
) -> bool:
    """Installs ``requirement`` into this interpreter, saying what it does.

    Args:
        label: The model's name, which starts every line printed.
        requirement: A PEP 508 requirement, such as a ``name @ git+URL``.
        what: What is installed, for the messages (``"the repository
            build"``).
        check: Python code that a fresh interpreter runs after the install,
            exiting with a nonzero status if the install did not take; the
            running process has already imported whatever was there before.
        missing: What the check found missing, for its message; empty for
            ``what``.
        dry_run: Say what would be installed, and install nothing.

    Returns:
        Whether ``requirement`` is installed, or would be after a dry run;
        False in a uv project's environment (:func:`uv_project`), where it
        prints the ``uv add`` to run instead, or in a checkout of this
        repository the ``uv sync`` (:data:`CHECKOUT_SYNC`). A dry run there
        is False as well, since the run it previews would not complete the
        install either: the user has to.
    """
    project = uv_project()
    if project is not None:
        # uv sync and uv run would undo the install from uv.lock, so the
        # package has to be a requirement of the project itself.
        if _is_checkout(project):
            # lazy-tfm's own requirements are published, and PyPI refuses a
            # direct URL in them; the checkout's uv sources hold the build.
            advised = list(CHECKOUT_SYNC)
            advice = "In this checkout, sync the development environment:"
        else:
            advised = ["uv", "add", requirement]
            advice = "Add it to the project, which completes the setup:"
        if not pathlib.Path.cwd().is_relative_to(project):
            advised[2:2] = ["--project", str(project)]
        print(
            f"{label}: not installing {what}, since this environment "
            f"belongs to the uv project at {project} and uv sync would "
            f"remove it again. {advice}\n  {shlex.join(advised)}"
        )
        return False
    command = install_command(requirement)
    if command is None:
        print(
            f"{label}: found neither pip nor uv to install {what} with. "
            "Install it by hand:\n"
            f"  pip install --force-reinstall --no-deps "
            f"{shlex.quote(requirement)}"
        )
        return False
    print(f"{label}: installing {what}:\n  {shlex.join(command)}")
    if dry_run:
        return True
    if subprocess.run(command, check=False).returncode != 0:
        print(f"{label}: the install failed; see the output above.")
        return False
    if subprocess.run([sys.executable, "-c", check], check=False).returncode:
        print(f"{label}: installed, but {missing or what} is still missing.")
        return False
    print(f"{label}: {what} is installed.")
    return True
