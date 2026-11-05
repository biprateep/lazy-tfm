# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The ``lazy`` command: finishing the installs PyPI cannot do.

``lazy setup`` runs every registered backend's ``setup`` hook
(:meth:`lazy.models.ContextEnsembleEstimator.setup`), or those of the
backends it names, which does what PyPI cannot for a backend whose extra is
installed: install a repository build that PyPI refuses in an extra, say, or
download source code that is not on PyPI, which a model would otherwise do on
first use, so that running it warms the cache before going offline. A backend
whose extra is not installed is skipped, and one already complete is left
alone, so the command is safe to run again.

``lazy models`` lists the pinned checkpoints and whether each is cached, with
no network. ``lazy download NAME[:VERSION]`` fetches the weights of each
checkpoint named, as :func:`lazy.download_checkpoint` does.

Typical usage example:

  pip install 'lazy-tfm[all]'
  lazy setup
  lazy models
  lazy download tabpfn:v2.5 tabicl
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from lazy.models import _ensemble
from lazy.models import _hub
from lazy.models import registry


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the ``lazy`` command.

    Args:
        argv: The arguments after the program name; ``None`` for
            ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success; 1 when a backend is incomplete after
        ``setup``; 2 for an unknown checkpoint, as ``argparse`` reports it.
    """
    parser = argparse.ArgumentParser(
        prog="lazy", description="Tools for the lazy-tfm package."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser(
        "setup",
        help="finish installing the backends PyPI cannot install fully",
        description=(
            "Do what PyPI cannot for every backend whose extra is installed: "
            "install a repository build, or download source code."
        ),
    )
    setup.add_argument(
        "--dry-run",
        action="store_true",
        help="say what would be done, and change nothing",
    )
    backends = {
        name: cls
        for name, cls in sorted(registry.ESTIMATORS.items())
        if issubclass(cls, _ensemble.ContextEnsembleEstimator)
    }
    setup.add_argument(
        "backends",
        nargs="*",
        metavar="BACKEND",
        help="set up only these backends; all of them when none is named",
    )
    commands.add_parser(
        "models",
        help="list the pinned checkpoints and whether each is cached",
        description=(
            "List every backend, its versions, the default version and "
            "whether the weights are in the local cache. Uses no network."
        ),
    )
    download = commands.add_parser(
        "download",
        help="download the weights of checkpoints into the local cache",
        description=(
            "Download the weights of each checkpoint named, so that a "
            "machine with no network can load them. Prints the licence "
            "notice of a checkpoint that has one."
        ),
    )
    download.add_argument(
        "checkpoints",
        nargs="+",
        metavar="NAME[:VERSION]",
        help="a backend, with its default version, or a backend and version",
    )
    args = parser.parse_args(argv)
    if args.command == "models":
        return _models()
    if args.command == "download":
        return _download(parser, args.checkpoints)
    unknown = sorted(set(args.backends) - set(registry.ESTIMATORS))
    if unknown:
        parser.error(
            f"unknown backends {unknown}; known: {sorted(registry.ESTIMATORS)}"
        )
    complete = [
        cls.setup(dry_run=args.dry_run, requested=name in args.backends)
        for name, cls in backends.items()
        if not args.backends or name in args.backends
    ]
    return 0 if all(complete) else 1


def _models() -> int:
    """Prints a table of every pinned checkpoint.

    The columns are the default version, the cache status and whether the
    license allows commercial use.

    Returns:
        0.
    """
    rows = [("MODEL", "VERSION", "DEFAULT", "CACHED", "COMMERCIAL")]
    for name in registry.list_estimators():
        for version in _hub.list_versions(name):
            rows.append(
                (
                    name,
                    version,
                    "yes"
                    if _hub.DEFAULT_VERSIONS.get(name) == version
                    else "no",
                    "yes" if _hub.is_cached(name, version) else "no",
                    "yes"
                    if _hub.get_checkpoint(name, version).commercial_use
                    else "no",
                )
            )
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    for row in rows:
        print("  ".join(map(str.ljust, row, widths)).rstrip())
    return 0


def _download(parser: argparse.ArgumentParser, specs: Sequence[str]) -> int:
    """Downloads each checkpoint named, after checking that all exist.

    Args:
        parser: The parser, which reports an unknown checkpoint and exits 2.
        specs: Checkpoint keys, ``NAME`` or ``NAME:VERSION``, as
            :func:`lazy.download_checkpoint` takes them.

    Returns:
        0.
    """
    for spec in specs:
        try:
            _hub.get_checkpoint(spec)
        except _hub.UnknownNameError as error:
            parser.error(str(error))
    for spec in specs:
        print(_hub.download_checkpoint(spec))
    return 0
