# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The cache directory, kept apart so importing it loads nothing heavy."""

from collections.abc import Iterator
import contextlib
import os
import pathlib
import sys
import tempfile
from typing import BinaryIO


def data_home(data_home: str | pathlib.Path | None = None) -> pathlib.Path:
    """Returns the directory cached catalogs live in, creating it if needed.

    Resolution order: the ``data_home`` argument, then ``$LAZY_DATA_HOME``,
    then ``$XDG_CACHE_HOME/lazy-tfm``, then the platform's user cache:
    ``lazy-tfm`` under ``%LOCALAPPDATA%`` on Windows, ``~/.cache/lazy-tfm``
    elsewhere.

    Nothing about the location is baked in, so a shared scratch filesystem or
    a node-local disk is one environment variable away and the same code runs
    unchanged on a laptop and on a cluster.

    Args:
        data_home: An explicit cache directory, overriding the environment.

    Returns:
        The cache directory, with ``~`` expanded.
    """
    explicit = data_home or os.environ.get("LAZY_DATA_HOME")
    root = pathlib.Path(explicit) if explicit else default_home("lazy-tfm")
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_home(name: str) -> pathlib.Path:
    """The default cache directory ``name``, under the user's cache."""
    if sys.platform == "win32":
        user_cache = os.environ.get("LOCALAPPDATA") or (
            pathlib.Path.home() / "AppData" / "Local"
        )
    else:
        user_cache = pathlib.Path.home() / ".cache"
    cache = os.environ.get("XDG_CACHE_HOME") or user_cache
    return pathlib.Path(cache).expanduser() / name


def umask() -> int:
    """The process's file-mode creation mask.

    Read from ``/proc`` where there is one, as setting the mask to read it
    back would change it for every thread in the meantime.
    """
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except OSError:
        pass
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


@contextlib.contextmanager
def atomic_file(path: str | os.PathLike[str]) -> Iterator[BinaryIO]:
    """A file to write whole, which then replaces ``path`` in one move.

    The file is written beside ``path`` under a name of its own, so that
    processes writing the same file into a shared cache never write to, or
    delete, one another's. Once the block ends, the file gets the mode any
    new file would get and moves into place; if the block raises, it is
    deleted. On Windows, which refuses to replace a file that another
    process holds open, a copy already in place is kept.

    Args:
        path: Where the file belongs; its folder must exist.

    Yields:
        The file, open for writing in binary mode.

    Raises:
        FileNotFoundError: If the folder of ``path`` does not exist.
    """
    folder, name = os.path.split(os.fspath(path))
    handle, partial = tempfile.mkstemp(
        dir=folder or None, prefix=f".{name}.", suffix=".part"
    )
    try:
        with os.fdopen(handle, "wb") as file:
            yield file
        # mkstemp makes the file private (0600); give it the mode any new
        # file would get, so that others can read a shared cache.
        os.chmod(partial, 0o666 & ~umask())
        try:
            os.replace(partial, path)
        except PermissionError:
            if not os.path.exists(path):
                raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(partial)
