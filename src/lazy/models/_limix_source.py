# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Finding LimiX's source and loading the parts of it the backend runs.

LimiX is not on PyPI: its code is a research repository that installs, when it
is installed at all, as the top-level packages ``model``, ``inference``,
``utils`` and ``config`` -- names that collide with anything else so generic.
So nothing here imports it by those names. Two self-contained subtrees are
loaded from wherever the source lives, under private aliases:

* ``model/v2_0`` (the network) as ``lazy.models._limix_ext.model``;
* ``inference/v2_0`` (its preprocessing steps) as
  ``lazy.models._limix_ext.inference``.

No ``sys.path`` edits, no copies of LimiX's code in this package. The source
is found, in order, from ``$LAZY_LIMIX_SRC``, ``$LIMIX_SRC``, or an installed
``LimiX`` distribution (``pip install "LimiX @ git+<REPOSITORY>@<commit>"``).
Its commit is checked against :data:`LIMIX_COMMIT`, the one the backend was
validated on, and a :class:`LimiXSourceWarning` says so when they differ.

Typical usage example:

  limix = _limix_source.load()
  network, config = limix.loading.load_from_checkpoint(state)
"""

from __future__ import annotations

import dataclasses
import importlib
from importlib import metadata
from importlib import util
import json
import os
import pathlib
import subprocess
import sys
import types
import warnings

__all__ = [
    "ENV_VARS",
    "LIMIX_COMMIT",
    "REPOSITORY",
    "LimiXSourceWarning",
    "Source",
    "load",
    "locate",
    "missing_dependency",
]

#: The LimiX commit the backend was validated against.
LIMIX_COMMIT = "516bf396333feb3198cf7aff8a6c10421f218e24"
#: Where LimiX's source is published.
REPOSITORY = "https://github.com/limix-ldm-ai/LimiX"
#: Environment variables naming a LimiX checkout, in the order they are read.
ENV_VARS = ("LAZY_LIMIX_SRC", "LIMIX_SRC")

_ALIAS = "lazy.models._limix_ext"
_SUBTREES = {"model": ("model", "v2_0"), "inference": ("inference", "v2_0")}
# Loaded subtrees, keyed by source root: loading twice would give two copies
# of every class, which breaks isinstance checks inside LimiX's own code.
_loaded: dict[pathlib.Path, types.SimpleNamespace] = {}

_INSTALL_HINT = (
    "LimiXBarDistribution needs LimiX's source, which is not on PyPI. "
    f'Install it with pip install "LimiX @ git+{REPOSITORY}@{LIMIX_COMMIT}" '
    "and its dependencies with pip install 'lazy-tfm[limix]', or point "
    "$LAZY_LIMIX_SRC at a checkout."
)

# The top-level modules LimiX's two subtrees import that are not in the
# standard library or lazy's own requirements: the limix extra's, and triton,
# which comes with Linux torch wheels.
_DEPENDENCIES = frozenset({"torch", "einops", "kditransform", "nvtx", "triton"})


class LimiXSourceWarning(UserWarning):
    """The LimiX source is not the commit the backend was validated on."""


@dataclasses.dataclass(frozen=True)
class Source:
    """Where LimiX's source was found.

    Attributes:
        root: The directory holding ``model/`` and ``inference/``.
        commit: Its git commit, or None when it cannot be told.
        origin: How it was found: an environment variable's name, or
            ``"installed LimiX <version>"``.
    """

    root: pathlib.Path
    commit: str | None
    origin: str


def locate() -> Source:
    """Finds LimiX's source, without importing any of it.

    Returns:
        Where it is, and at which commit.

    Raises:
        ImportError: If no source is found, or an environment variable names
            a directory without LimiX's v2 code.
    """
    for name in ENV_VARS:
        value = os.environ.get(name)
        if value:
            root = pathlib.Path(value).expanduser().resolve()
            if not _has_subtrees(root):
                raise ImportError(
                    f"${name}={value} does not hold LimiX's model/v2_0 and "
                    f"inference/v2_0. {_INSTALL_HINT}"
                )
            return Source(root, _git_commit(root), f"${name}")
    try:
        distribution = metadata.distribution("LimiX")
    except metadata.PackageNotFoundError:
        raise ImportError(_INSTALL_HINT) from None
    root = pathlib.Path(str(distribution.locate_file("")))
    if not _has_subtrees(root):
        raise ImportError(
            f"the installed LimiX at {root} lacks model/v2_0 or "
            f"inference/v2_0. {_INSTALL_HINT}"
        )
    return Source(
        root,
        _installed_commit(distribution),
        f"installed LimiX {distribution.version}",
    )


def load() -> types.SimpleNamespace:
    """Loads LimiX's network and preprocessing code under private aliases.

    Returns:
        A namespace with ``loading`` (``model.v2_0.loading``), ``preprocess``
        (``inference.v2_0.preprocess``) and ``source`` (a :class:`Source`).

    Raises:
        ImportError: If the source cannot be found, or one of its own
            dependencies (``torch``, ``einops``, ``kditransform``, ``nvtx``,
            ``triton``) is missing.
    """
    source = locate()
    if source.root in _loaded:
        return _loaded[source.root]
    if source.commit != LIMIX_COMMIT:
        warnings.warn(
            f"LimiX source at {source.root} ({source.origin}) is at commit "
            f"{source.commit or 'unknown'}; lazy was validated on "
            f"{LIMIX_COMMIT}. Results may differ.",
            LimiXSourceWarning,
            stacklevel=2,
        )
    if _ALIAS not in sys.modules:
        parent = types.ModuleType(_ALIAS)
        parent.__path__ = []
        sys.modules[_ALIAS] = parent
    try:
        for name, parts in _SUBTREES.items():
            _load_package(f"{_ALIAS}.{name}", source.root.joinpath(*parts))
        namespace = types.SimpleNamespace(
            loading=importlib.import_module(f"{_ALIAS}.model.loading"),
            preprocess=importlib.import_module(
                f"{_ALIAS}.inference.preprocess"
            ),
            source=source,
        )
    except ModuleNotFoundError as error:
        if (error.name or "").partition(".")[0] not in _DEPENDENCIES:
            raise
        raise missing_dependency(error.name or "") from error
    _loaded[source.root] = namespace
    return namespace


def missing_dependency(name: str) -> ImportError:
    """The error for one of LimiX's dependencies missing, with the fix.

    Args:
        name: The module that failed to import, such as ``"einops"``.

    Returns:
        An :class:`ImportError` naming it and how to install LimiX's
        dependencies and source.
    """
    top = name.partition(".")[0]
    message = (
        f"LimiXBarDistribution needs {top!r}, one of LimiX's dependencies, "
        "which is not installed. Install them with pip install "
        "'lazy-tfm[limix]', and LimiX's source, which is not on PyPI, with "
        f'pip install "LimiX @ git+{REPOSITORY}@{LIMIX_COMMIT}".'
    )
    if top == "triton":
        message += (
            " triton is published for Linux only (it comes with Linux torch "
            "wheels), and LimiX's network imports it, so the backend runs "
            "on Linux."
        )
    return ImportError(message, name=name)


def _has_subtrees(root: pathlib.Path) -> bool:
    """Whether ``root`` holds both subtrees the backend loads."""
    return all(
        root.joinpath(*parts, "__init__.py").is_file()
        for parts in _SUBTREES.values()
    )


def _load_package(alias: str, directory: pathlib.Path) -> None:
    """Imports ``directory`` as the package ``alias``."""
    if alias in sys.modules:
        return
    spec = util.spec_from_file_location(
        alias,
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {directory} as {alias}")
    module = util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[alias]
        raise


def _git_commit(root: pathlib.Path) -> str | None:
    """The checkout's HEAD commit, or None outside a git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _installed_commit(distribution: metadata.Distribution) -> str | None:
    """The commit a ``pip install git+...`` recorded, or None."""
    text = distribution.read_text("direct_url.json")
    if not text:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    commit = record.get("vcs_info", {}).get("commit_id")
    return commit if isinstance(commit, str) else None
