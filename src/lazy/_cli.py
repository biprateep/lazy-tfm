# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The ``lazy`` command: finishing the installs PyPI cannot do.

``lazy setup`` completes the backends whose extras are installed:

* TabFM: the extra brings the PyPI release, which has no KV-cache API, and
  PyPI refuses a git URL in an extra, so this installs the repository build
  (:data:`lazy.models.tabfm.REPOSITORY_BUILD`) over it, with pip or uv.
* LimiX-2: its source is not on PyPI, so this downloads it into the cache
  (:func:`lazy.models._limix_source.fetch`), which a model otherwise does on
  first use. Running it here warms the cache before going offline.

A backend whose extra is not installed is skipped, and one already complete
is left alone, so the command is safe to run again.

Typical usage example:

  pip install 'lazy-tfm[all]'
  lazy setup
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import util
import pathlib
import shlex
import shutil
import subprocess
import sys

from lazy.models import _icl_stream
from lazy.models import _limix_source
from lazy.models import tabfm

# What `lazy setup` runs in a fresh interpreter to check the TabFM it just
# installed: this process has already imported the old one.
_CHECK_TABFM = (
    "from lazy.models import _icl_stream; "
    "raise SystemExit(not _icl_stream.streaming_available())"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the ``lazy`` command.

    Args:
        argv: The arguments after the program name; ``None`` for
            ``sys.argv[1:]``.

    Returns:
        The exit status: 0 when every installed backend is complete.
    """
    parser = argparse.ArgumentParser(
        prog="lazy", description="Tools for the lazy-tfm package."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser(
        "setup",
        help="finish installing the backends PyPI cannot install fully",
        description=(
            "Install TabFM's repository build and download LimiX's source, "
            "for whichever of the two backends' extras are installed."
        ),
    )
    setup.add_argument(
        "--dry-run",
        action="store_true",
        help="say what would be done, and change nothing",
    )
    args = parser.parse_args(argv)
    complete = [
        _setup_tabfm(dry_run=args.dry_run),
        _setup_limix(dry_run=args.dry_run),
    ]
    return 0 if all(complete) else 1


def _setup_tabfm(*, dry_run: bool) -> bool:
    """Installs TabFM's repository build over the release, if need be."""
    if util.find_spec("tabfm") is None:
        print("TabFM: not installed, skipped ('lazy-tfm[tabfm]' adds it).")
        return True
    if _icl_stream.streaming_available():
        print("TabFM: the repository build is installed.")
        return True
    if _in_uv_project():
        # uv sync and uv run would put the release back from uv.lock, so the
        # build has to be a requirement of the project itself.
        print(
            "TabFM: in a uv project, add the repository build to the project, "
            "so that uv sync keeps it:\n"
            f"  uv add {shlex.quote(tabfm.REPOSITORY_BUILD)}"
        )
        return False
    command = _install_command(tabfm.REPOSITORY_BUILD)
    if command is None:
        print(
            "TabFM: found neither pip nor uv to install the repository build "
            "with. Install it by hand:\n"
            "  pip install --force-reinstall --no-deps "
            f"{shlex.quote(tabfm.REPOSITORY_BUILD)}"
        )
        return False
    print(f"TabFM: installing the repository build:\n  {shlex.join(command)}")
    if dry_run:
        return True
    if subprocess.run(command, check=False).returncode != 0:
        print("TabFM: the install failed; see the output above.")
        return False
    check = subprocess.run([sys.executable, "-c", _CHECK_TABFM], check=False)
    if check.returncode != 0:
        print("TabFM: installed, but the KV-cache API is still missing.")
        return False
    print("TabFM: the repository build is installed.")
    return True


def _setup_limix(*, dry_run: bool) -> bool:
    """Downloads LimiX's source into the cache, if nothing else has it."""
    if any(util.find_spec(name) is None for name in ("einops", "kditransform")):
        print("LimiX-2: not installed, skipped ('lazy-tfm[limix]' adds it).")
        return True
    try:
        source = _limix_source.locate(download=False)
    except ImportError:
        print(f"LimiX-2: downloading its source from {_limix_source.ARCHIVE}")
        if dry_run:
            return True
        try:
            source = _limix_source.locate()
        except ImportError as error:
            print(f"LimiX-2: {error}")
            return False
    print(f"LimiX-2: source at {source.root} ({source.origin}).")
    return True


def _install_command(requirement: str) -> list[str] | None:
    """The command replacing an installed package with ``requirement``.

    pip when this interpreter has it, else uv, whose environments have no
    pip; both reinstall even though the build and the release share a
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


def _in_uv_project() -> bool:
    """Whether the working directory is inside a project uv has locked."""
    here = pathlib.Path.cwd()
    return any((d / "uv.lock").is_file() for d in (here, *here.parents))


if __name__ == "__main__":
    sys.exit(main())
