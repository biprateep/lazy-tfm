# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The cache directory, kept apart so importing it loads nothing heavy."""

import os
import pathlib


def data_home(data_home: str | pathlib.Path | None = None) -> pathlib.Path:
    """Returns the directory cached catalogs live in, creating it if needed.

    Resolution order: the ``data_home`` argument, then ``$LAZY_DATA_HOME``,
    then ``$XDG_CACHE_HOME/lazy-tfm``, then ``~/.cache/lazy-tfm``.

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
    cache = os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache"
    return pathlib.Path(cache).expanduser() / name
