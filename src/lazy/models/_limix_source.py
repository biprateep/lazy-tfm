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
is found, in order, from ``$LAZY_LIMIX_SRC``, ``$LIMIX_SRC``, an installed
``LimiX`` distribution (``pip install "LimiX @ git+<REPOSITORY>@<commit>"``),
or :func:`cache_dir`. When none of them has it, the archive of
:data:`LIMIX_COMMIT` is downloaded from GitHub into :func:`cache_dir`, once,
like a checkpoint: nothing is installed, so the generic names never reach the
environment. The commit is checked against :data:`LIMIX_COMMIT`, the one the
backend was validated on, and a :class:`LimiXSourceWarning` says so when they
differ.

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
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
import urllib.request
import warnings

from lazy import datasets

__all__ = [
    "ENV_VARS",
    "LIMIX_COMMIT",
    "REPOSITORY",
    "LimiXSourceWarning",
    "Source",
    "cache_dir",
    "fetch",
    "load",
    "locate",
    "missing_dependency",
]

# Warnings skip every frame inside this package, so that they point at the
# user's call however deep in the package (or LazyModel) they are raised.
_PACKAGE_PREFIX = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
)

#: The LimiX commit the backend was validated against.
LIMIX_COMMIT = "516bf396333feb3198cf7aff8a6c10421f218e24"
#: Where LimiX's source is published.
REPOSITORY = "https://github.com/limix-ldm-ai/LimiX"
#: Environment variables naming a LimiX checkout, in the order they are read.
ENV_VARS = ("LAZY_LIMIX_SRC", "LIMIX_SRC")
#: GitHub's archive of :data:`LIMIX_COMMIT`, which :func:`fetch` downloads.
ARCHIVE = f"{REPOSITORY}/archive/{LIMIX_COMMIT}.tar.gz"
#: The :attr:`Source.origin` of the source in :func:`cache_dir`.
CACHE_ORIGIN = "lazy's cache"

_ALIAS = "lazy.models._limix_ext"
_SUBTREES = {"model": ("model", "v2_0"), "inference": ("inference", "v2_0")}
# Loaded subtrees, keyed by source root: loading twice would give two copies
# of every class, which breaks isinstance checks inside LimiX's own code. The
# first source loaded is the one the process keeps.
_loaded: dict[pathlib.Path, types.SimpleNamespace] = {}
# Roots an environment variable named after the first load, already warned of.
_ignored: set[pathlib.Path] = set()

_INSTALL_HINT = (
    "LimiXBarDistribution needs LimiX's source, which is not on PyPI. lazy "
    f"downloads it from {ARCHIVE} on first use; on a machine without a "
    'network, run lazy.download_checkpoint("limix") on one with a network '
    "first, or point $LAZY_LIMIX_SRC at a checkout. Its dependencies come "
    "with pip install 'lazy-tfm[limix]'."
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
        origin: How it was found: an environment variable's name,
            ``"installed LimiX <version>"``, or :data:`CACHE_ORIGIN`.
    """

    root: pathlib.Path
    commit: str | None
    origin: str

    @property
    def package(self) -> str:
        """The ``"package"`` label of a provenance: where LimiX came from.

        ``"LimiX <version>"`` for an installed LimiX, ``"LimiX (source
        checkout at <commit>)"`` for a checkout an environment variable
        names, and ``"LimiX (source archive at <commit>)"`` for the cache.
        """
        if self.origin.startswith("installed "):
            return self.origin.removeprefix("installed ")
        kind = "archive" if self.origin == CACHE_ORIGIN else "checkout"
        return f"LimiX (source {kind} at {self.commit or 'unknown commit'})"


def locate(*, download: bool = True) -> Source:
    """Finds LimiX's source, without importing any of it.

    Args:
        download: Download the source into :func:`cache_dir` when nothing
            else has it. Off, or with ``$HF_HUB_OFFLINE`` set, a source that
            is not already there is an error.

    Returns:
        Where it is, and at which commit.

    Raises:
        ImportError: If no source is found or can be downloaded, or an
            environment variable names a directory without LimiX's v2 code.
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
        root = cache_dir()
        if not _has_subtrees(root):
            if not download or _offline():
                raise ImportError(
                    f"LimiX's source is not in {root}, and downloading it is "
                    f"off. {_INSTALL_HINT}"
                ) from None
            fetch()
        return Source(root, LIMIX_COMMIT, CACHE_ORIGIN)
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


def cache_dir() -> pathlib.Path:
    """Where :func:`fetch` unpacks LimiX's source.

    ``limix/<commit>`` under :func:`lazy.datasets.data_home`, so
    ``$LAZY_DATA_HOME`` moves it with the catalogues.

    Returns:
        The directory, which holds the source once :func:`fetch` has run.
    """
    return datasets.data_home() / "limix" / LIMIX_COMMIT


def fetch() -> pathlib.Path:
    """Downloads LimiX's source at :data:`LIMIX_COMMIT` into the cache.

    A no-op when it is already there. The archive is unpacked next to
    :func:`cache_dir` and moved into place whole, so an interrupted download
    never leaves a half-written source behind.

    Returns:
        :func:`cache_dir`.

    Raises:
        ImportError: If the download fails, or the archive does not hold
            LimiX's v2 code.
    """
    root = cache_dir()
    if _has_subtrees(root):
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as scratch:
        archive = pathlib.Path(scratch) / "LimiX.tar.gz"
        try:
            with (
                urllib.request.urlopen(ARCHIVE, timeout=60) as response,
                archive.open("wb") as f,
            ):
                shutil.copyfileobj(response, f)
        except OSError as error:  # urllib's URLError is an OSError.
            raise ImportError(
                f"could not download LimiX's source from {ARCHIVE}: {error}. "
                f"{_INSTALL_HINT}"
            ) from error
        unpacked = pathlib.Path(scratch) / "unpacked"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        # GitHub's archive holds one directory, LimiX-<commit>.
        tops = list(unpacked.iterdir())
        if len(tops) != 1 or not _has_subtrees(tops[0]):
            raise ImportError(
                f"the archive {ARCHIVE} does not hold LimiX's model/v2_0 and "
                "inference/v2_0."
            )
        try:
            tops[0].rename(root)
        except OSError:
            if not _has_subtrees(root):  # Not another process's download.
                raise
    return root


def _offline() -> bool:
    """Whether ``$HF_HUB_OFFLINE`` asks for no downloads, as for weights."""
    value = os.environ.get("HF_HUB_OFFLINE", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load() -> types.SimpleNamespace:
    """Loads LimiX's network and preprocessing code under private aliases.

    The first successful load is kept for the rest of the process: later
    calls return it without looking for the source again, and a
    :class:`LimiXSourceWarning` says so if ``$LAZY_LIMIX_SRC`` or
    ``$LIMIX_SRC`` has since come to name another checkout.

    Returns:
        A namespace with ``loading`` (``model.v2_0.loading``), ``preprocess``
        (``inference.v2_0.preprocess``) and ``source`` (a :class:`Source`).

    Raises:
        ImportError: If the source cannot be found, or one of its own
            dependencies (``torch``, ``einops``, ``kditransform``, ``nvtx``,
            ``triton``) is missing.
    """
    if _loaded:
        loaded = next(iter(_loaded.values()))
        _warn_if_redirected(loaded.source)
        return loaded
    source = locate()
    if source.commit != LIMIX_COMMIT:
        warnings.warn(
            f"LimiX source at {source.root} ({source.origin}) is at commit "
            f"{source.commit or 'unknown'}; lazy was validated on "
            f"{LIMIX_COMMIT}. Results may differ.",
            LimiXSourceWarning,
            skip_file_prefixes=(_PACKAGE_PREFIX,),
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


def _warn_if_redirected(source: Source) -> None:
    """Warns, once per root, when the environment names another source."""
    for name in ENV_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        root = pathlib.Path(value).expanduser().resolve()
        if root != source.root and root not in _ignored:
            _ignored.add(root)
            warnings.warn(
                f"${name}={value} names another LimiX source than the one "
                f"this process loaded, {source.root} ({source.origin}); "
                "LimiX's code cannot be loaded twice, so it keeps using that "
                "one. Restart Python to switch.",
                LimiXSourceWarning,
                skip_file_prefixes=(_PACKAGE_PREFIX,),
            )
        return


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
        "'lazy-tfm[limix]'."
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
