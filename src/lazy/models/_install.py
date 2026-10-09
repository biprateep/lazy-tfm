# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Installing a package from its repository, for ``lazy setup``.

PyPI refuses a git URL in an extra, so a backend whose package (or whose
fast build of it) is on GitHub alone has ``lazy setup`` install it from there
(:meth:`lazy.models.ContextEnsembleEstimator.setup`). :func:`install`
does it the same way for every such backend: with pip when this interpreter
has it, else with uv, leaving the dependencies to the extra; in a uv project
it prints the ``uv add`` that keeps the package instead, since ``uv sync``
would remove it again.

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
import pathlib
import shlex
import shutil
import subprocess
import sys

__all__ = ["in_uv_project", "install", "install_command"]


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


def in_uv_project() -> bool:
    """Whether the working directory is inside a project uv has locked."""
    here = pathlib.Path.cwd()
    return any((d / "uv.lock").is_file() for d in (here, *here.parents))


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
        False in a uv project, where it prints the ``uv add`` to run instead.
    """
    if in_uv_project():
        # uv sync and uv run would undo the install from uv.lock, so the
        # package has to be a requirement of the project itself.
        print(
            f"{label}: in a uv project, add {what} to the project, so that "
            "uv sync keeps it:\n"
            f"  uv add {shlex.quote(requirement)}"
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
