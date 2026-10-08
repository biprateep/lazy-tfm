# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The ``lazy`` command: finishing the installs PyPI cannot do.

``lazy setup`` runs every registered backend's ``setup`` hook
(:meth:`lazy.models.ContextEnsembleEstimator.setup`), which does what PyPI
cannot for a backend whose extra is installed: install a repository build
that PyPI refuses in an extra, say, or download source code that is not on
PyPI, which a model would otherwise do on first use, so that running it
warms the cache before going offline. A backend whose extra is not installed
is skipped, and one already complete is left alone, so the command is safe to
run again.

Typical usage example:

  pip install 'lazy-tfm[all]'
  lazy setup
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from lazy.models import _ensemble
from lazy.models import registry


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
            "Do what PyPI cannot for every backend whose extra is installed: "
            "install a repository build, or download source code."
        ),
    )
    setup.add_argument(
        "--dry-run",
        action="store_true",
        help="say what would be done, and change nothing",
    )
    args = parser.parse_args(argv)
    complete = [
        cls.setup(dry_run=args.dry_run)
        for _, cls in sorted(registry.ESTIMATORS.items())
        if issubclass(cls, _ensemble.ContextEnsembleEstimator)
    ]
    return 0 if all(complete) else 1


if __name__ == "__main__":
    sys.exit(main())
