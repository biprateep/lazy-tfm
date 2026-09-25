# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""A warning for when another distribution also installs ``lazy``.

This package is published as ``lazy-tfm`` but imported as ``lazy``, and an
unrelated PyPI package, ``lazy`` (lazy attributes for Python objects), also
installs a top-level ``lazy`` directory. Two distributions cannot share an
import name in one environment: whichever was installed last overwrote the
other's files, and uninstalling either deletes files the other needs.
Renaming at import (``import lazy as lz``) does not help, since there is only
one ``lazy`` on disk.

When this package's code is the one that runs, it can see the other
distribution's metadata and say so; :func:`warn_if_shared` does that once, at
import. When the other package was installed last, this code never runs, and
the symptom is an immediate ``AttributeError`` on ``lazy.LazyModel``; the
installation docs cover that case.

Typical usage example:

  _namespace.warn_if_shared()
"""

from importlib import metadata
import warnings

__all__ = ["OTHER_DISTRIBUTION", "ImportNameWarning", "warn_if_shared"]

#: The unrelated PyPI distribution that also installs a ``lazy`` package.
OTHER_DISTRIBUTION = "lazy"


class ImportNameWarning(UserWarning):
    """Another installed distribution also provides the ``lazy`` package."""


def warn_if_shared() -> bool:
    """Warns if the unrelated ``lazy`` distribution is installed here too.

    Returns:
        Whether it is installed (and the warning was issued).
    """
    try:
        other = metadata.distribution(OTHER_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return False
    warnings.warn(
        f"The PyPI package {OTHER_DISTRIBUTION!r} {other.version} (lazy "
        "attributes for Python objects) is installed in this environment. It "
        "also installs a top-level 'lazy' package, so it and lazy-tfm "
        "overwrite each other's files; aliasing the import (import lazy as "
        "lz) does not help. Install lazy-tfm in an environment without it: "
        f"pip uninstall {OTHER_DISTRIBUTION} lazy-tfm, then pip install "
        "lazy-tfm.",
        ImportNameWarning,
        stacklevel=3,
    )
    return True
