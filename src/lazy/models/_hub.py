# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Pretrained checkpoints: where they come from, and where they are cached.

Every backend in :mod:`lazy.models` runs a pretrained tabular foundation model
whose weights are not shipped with this package. They are fetched from the
Hugging Face Hub the first time an estimator predicts, and reused from the
local HF cache (``$HF_HOME``, by default ``~/.cache/huggingface/hub``) on every
later call, in this or any other project.

Revisions are pinned. A foundation model's weights are part of the method, so
an unpinned checkpoint would silently change published numbers.

A backbone is not one model but a family of them, so a checkpoint is named by
*backend and version* -- ``"tabpfn:v3.5"``, not ``"tabpfn"`` -- in the version
string upstream itself uses. Every estimator takes that version as an ordinary
parameter, defaulting to the one in :data:`DEFAULT_VERSIONS`, and records what
it actually loaded in its ``provenance_`` (:meth:`Checkpoint.provenance`).
Adding a version is one entry here and nothing else.

:func:`download_checkpoint` pre-fetches on a machine with a network connection
before running somewhere without one, and :func:`is_cached` checks without
triggering a download. Both, with :func:`get_checkpoint` and
:func:`list_versions`, are part of the public API and are re-exported from the
top-level :mod:`lazy`.
"""

from __future__ import annotations

import dataclasses
from importlib import metadata
import pathlib

__all__ = [
    "CHECKPOINTS",
    "DEFAULT_VERSIONS",
    "Checkpoint",
    "download_checkpoint",
    "get_checkpoint",
    "is_cached",
    "list_versions",
]


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """A pinned set of pretrained weights on the Hugging Face Hub.

    Attributes:
        backend: Which backend loads these weights; a name in
            :data:`lazy.ESTIMATORS`.
        version: The model version, in the backbone's own vocabulary --
            ``"v2.5"``, ``"v3"``, ``"v3.5-fast"`` are the strings TabPFN itself
            uses.
        repo_id: The Hugging Face Hub repository holding the weights.
        package: The distribution providing the backend. Its *installed*
            version is half of "which model was this": the weights are the
            other half.
        allow_patterns: ``snapshot_download`` patterns, for repositories
            holding several models.
        filename: A single file to fetch instead of a filtered snapshot.
        revision: The pinned git commit of ``repo_id``. ``None`` would track
            the repository's default branch; no registered checkpoint does,
            because even a dated filename can be replaced upstream.
        license_note: Anything a user must know before downloading. Surfaced
            in the docs.
        size_note: The download size, and any limit the model declares.
    """

    backend: str
    version: str
    repo_id: str
    package: str
    allow_patterns: tuple[str, ...] | None = None
    filename: str | None = None
    revision: str | None = None
    license_note: str = ""
    size_note: str = ""

    @property
    def key(self) -> str:
        """The ``"<backend>:<version>"`` key this checkpoint is filed under.

        Examples:
            >>> get_checkpoint("tabpfn").key
            'tabpfn:v3.5'
        """
        return f"{self.backend}:{self.version}"

    def download(self, *, local_files_only: bool = False) -> pathlib.Path:
        """Fetches (or locates) the weights and returns the path they live at.

        A pinned revision cannot change, so the local cache is tried first
        and the Hub is contacted only when the weights are not there: a
        cached checkpoint loads with no network traffic at all.

        Args:
            local_files_only: Only look in the local Hugging Face cache, never
                on the network.

        Returns:
            The checkpoint file, or the snapshot directory when the
            checkpoint is a filtered snapshot rather than one file.
        """
        if self.revision is not None and not local_files_only:
            try:
                return self._fetch(local_files_only=True)
            except FileNotFoundError:
                # Not cached (huggingface_hub's LocalEntryNotFoundError is a
                # FileNotFoundError): fall through to the network.
                pass
        return self._fetch(local_files_only=local_files_only)

    def _fetch(self, *, local_files_only: bool) -> pathlib.Path:
        """One ``huggingface_hub`` download call; see :meth:`download`."""
        import huggingface_hub  # noqa: PLC0415 - kept out of `import lazy`.

        if self.filename is not None:
            return pathlib.Path(
                huggingface_hub.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    revision=self.revision,
                    local_files_only=local_files_only,
                )
            )
        return pathlib.Path(
            huggingface_hub.snapshot_download(
                repo_id=self.repo_id,
                revision=self.revision,
                allow_patterns=list(self.allow_patterns)
                if self.allow_patterns
                else None,
                local_files_only=local_files_only,
            )
        )

    def provenance(self, *, device: str | None = None) -> dict[str, str | None]:
        """Returns which weights, and which code, produced a set of numbers.

        Everything needed to identify the model, and nothing that is merely a
        setting -- ``n_estimators``, ``chunk_size`` and the rest are in
        ``get_params()``. There are no local paths in it either, so the same
        model on two machines gives the same record, and it survives being
        written to JSON next to a results table.

        Args:
            device: The torch device the model ran on, if known.

        Returns:
            A JSON-serialisable record with the keys ``backend``, ``version``,
            ``repo_id``, ``filename``, ``revision``, ``package`` (the
            backend's distribution and installed version), ``lazy`` (this
            package's) and ``device``.

        Examples:
            >>> record = get_checkpoint("tabpfn", "v2.5").provenance(
            ...     device="cpu"
            ... )
            >>> record["backend"], record["version"], record["revision"][:7]
            ('tabpfn', 'v2.5', '6c45f3a')
        """
        return {
            "backend": self.backend,
            "version": self.version,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "revision": self.revision,
            "package": _package_version(self.package),
            "lazy": _package_version("lazy-tfm"),
            "device": device,
        }


def _package_version(distribution: str) -> str:
    """Returns ``"tabpfn 9.0.0"``, or a stand-in when it is not installed."""
    try:
        return f"{distribution} {metadata.version(distribution)}"
    except (
        metadata.PackageNotFoundError
    ):  # pragma: no cover - only outside an install
        return f"{distribution} (not installed)"


#: Every pinned checkpoint, keyed ``"<backend>:<version>"``. Built from the
#: records themselves so a key can never disagree with what it points at.
CHECKPOINTS: dict[str, Checkpoint] = {
    checkpoint.key: checkpoint
    for checkpoint in (
        Checkpoint(
            backend="tabfm",
            version="v1.0",
            repo_id="google/tabfm-1.0.0-pytorch",
            package="tabfm",
            revision="77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83",
            allow_patterns=(
                "classification/**",
                "config.json",
                "LICENSE",
                "README.md",
            ),
            license_note=(
                "TabFM weights are released by Google under a non-commercial "
                "licence; read it at "
                "https://huggingface.co/google/tabfm-1.0.0-pytorch before use."
            ),
            size_note="~6.6 GB for the classification checkpoint.",
        ),
        Checkpoint(
            backend="limix",
            version="v2",
            repo_id="stable-ai/LimiX-2",
            package="LimiX",
            filename="LimiX-2.ckpt",
            revision="de07b679e74a41b50b9de18251a8fa245e537440",
            license_note=(
                "LimiX-2 weights are released by Stable AI under the StableAI "
                "LimiX Non-Commercial License 1.0, which also requires "
                'distributions, derivatives and publications to display "Built '
                'with StableAI LimiX"; LimiX\'s code is under the Apache-2.0-'
                "based Stable AI Technology Co., Ltd. License 1.0. See "
                "https://huggingface.co/stable-ai/LimiX-2."
            ),
            size_note="~1.6 GB. Pretrained for at most about 20,000 rows.",
        ),
        Checkpoint(
            backend="tabicl",
            version="v2",
            repo_id="jingang/TabICL",
            package="tabicl",
            filename="tabicl-regressor-v2-20260212.ckpt",
            revision="4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
            license_note=(
                "TabICL is released under BSD-3-Clause; see "
                "https://huggingface.co/jingang/TabICL."
            ),
            size_note="~100 MB.",
        ),
        # The TabPFN family. Every filename below names "the default" of its
        # release, which is a moving name -- the repository is free to replace
        # the file a later release calls that -- so the revision is what
        # fixes the weights.
        Checkpoint(
            backend="tabpfn",
            version="v2",
            repo_id="Prior-Labs/TabPFN-v2-reg",
            package="tabpfn",
            filename="tabpfn-v2-regressor.ckpt",
            revision="4972a65a1b30806315c6f92499959ffbfc69a673",
            license_note=(
                "TabPFN v2 weights are released by Prior Labs under the Prior "
                "Labs License v1.1, which is Apache-2.0 with an added "
                "attribution clause -- the only TabPFN version here that is "
                "not non-commercial. See "
                "https://huggingface.co/Prior-Labs/TabPFN-v2-reg."
            ),
            size_note="~45 MB. Pretrained for at most 10,000 context rows.",
        ),
        Checkpoint(
            backend="tabpfn",
            version="v2.5",
            repo_id="Prior-Labs/tabpfn_2_5",
            package="tabpfn",
            filename="tabpfn-v2.5-regressor-v2.5_default.ckpt",
            revision="6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b",
            license_note=(
                "TabPFN-2.5 weights are released by Prior Labs under the "
                "TABPFN-2.5 Non-Commercial License; read it at "
                "https://huggingface.co/Prior-Labs/tabpfn_2_5 before use."
            ),
            size_note="~41 MB. Pretrained for at most 50,000 context rows.",
        ),
        Checkpoint(
            backend="tabpfn",
            version="v2.6",
            repo_id="Prior-Labs/tabpfn_2_6",
            package="tabpfn",
            filename="tabpfn-v2.6-regressor-v2.6_default.ckpt",
            revision="24148873a9a5992b429833d07363690dcffd02a8",
            license_note=(
                "TabPFN-2.6 weights are released by Prior Labs under the "
                "TABPFN-2.6 Non-Commercial License; read it at "
                "https://huggingface.co/Prior-Labs/tabpfn_2_6 before use."
            ),
            size_note="~52 MB.",
        ),
        Checkpoint(
            backend="tabpfn",
            version="v3",
            repo_id="Prior-Labs/tabpfn_3",
            package="tabpfn",
            filename="tabpfn-v3-regressor-v3_default.ckpt",
            revision="24a16a89d245878b846555110985634aa2e656d7",
            license_note=(
                "TabPFN-3 weights are released by Prior Labs under the "
                "TABPFN-3 Non-Commercial License; read it at "
                "https://huggingface.co/Prior-Labs/tabpfn_3 before use."
            ),
            size_note="~230 MB.",
        ),
        # From v3.5 on one checkpoint carries both a classification and a
        # regression head, so these two files are not regressor-specific.
        Checkpoint(
            backend="tabpfn",
            version="v3.5",
            repo_id="Prior-Labs/tabpfn_3_5",
            package="tabpfn",
            filename="tabpfn-v3.5-20260909.safetensors",
            revision="06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
            license_note=(
                "TabPFN-3.5 weights are released by Prior Labs under the "
                "TABPFN-3.5 Non-Commercial License; read it at "
                "https://huggingface.co/Prior-Labs/tabpfn_3_5 before use."
            ),
            size_note="~880 MB.",
        ),
        Checkpoint(
            backend="tabpfn",
            version="v3.5-fast",
            repo_id="Prior-Labs/tabpfn_3_5",
            package="tabpfn",
            filename="tabpfn-v3.5-fast-20260909.safetensors",
            revision="06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
            license_note=(
                "TabPFN-3.5 weights are released by Prior Labs under the "
                "TABPFN-3.5 Non-Commercial License; read it at "
                "https://huggingface.co/Prior-Labs/tabpfn_3_5 before use."
            ),
            size_note=(
                "~335 MB. A separate, faster model, not a smaller copy of v3.5."
            ),
        ),
    )
}

#: The version each backend loads when none is asked for. Changing one of these
#: changes what every unversioned call returns, so they move only deliberately.
DEFAULT_VERSIONS: dict[str, str] = {
    "limix": "v2",
    "tabfm": "v1.0",
    "tabicl": "v2",
    "tabpfn": "v3.5",
}


def get_checkpoint(name: str, version: str | None = None) -> Checkpoint:
    """Returns the pinned checkpoint for a backend, at a version.

    Args:
        name: A backend name, which takes that backend's default version, or
            a full ``"backend:version"`` key.
        version: The model version; an explicit one wins over one spelled
            into ``name``. ``None`` for the default in
            :data:`DEFAULT_VERSIONS`.

    Returns:
        The registered checkpoint.

    Raises:
        KeyError: If the backend is unknown, or has no pinned checkpoint at
            that version.

    Examples:
        >>> get_checkpoint("tabpfn").version
        'v3.5'
        >>> get_checkpoint("tabpfn:v2.5").repo_id
        'Prior-Labs/tabpfn_2_5'
    """
    if ":" in name:
        name, _, spelled = name.partition(":")
        version = version if version is not None else spelled
    if name not in DEFAULT_VERSIONS:
        raise KeyError(
            f"unknown backend {name!r}; known: {sorted(DEFAULT_VERSIONS)}"
        )
    if version is None:
        version = DEFAULT_VERSIONS[name]
    try:
        return CHECKPOINTS[f"{name}:{version}"]
    except KeyError:
        raise KeyError(
            f"unknown version {version!r} for {name!r}; "
            f"known: {list_versions(name)}"
        ) from None


def list_versions(name: str) -> list[str]:
    """Returns the model versions ``name`` has pinned weights for, sorted.

    Args:
        name: A backend name; one that is not registered has no versions.

    Returns:
        The version strings, sorted.

    Examples:
        >>> list_versions("tabpfn")
        ['v2', 'v2.5', 'v2.6', 'v3', 'v3.5', 'v3.5-fast']
        >>> list_versions("tabicl")
        ['v2']
    """
    return sorted(
        spec.version for spec in CHECKPOINTS.values() if spec.backend == name
    )


def download_checkpoint(name: str, version: str | None = None) -> pathlib.Path:
    """Fetches the pretrained weights for ``name``, or returns the cached path.

    Safe to call repeatedly: the Hugging Face cache makes every call after the
    first a no-op. Use it to warm the cache on a login node before submitting
    a job to a compute node with no outbound network.

    Args:
        name: A backend name or a ``"backend:version"`` key, as for
            :func:`get_checkpoint`.
        version: The model version, or ``None`` for the default.

    Returns:
        The local path of the checkpoint file or snapshot directory.

    Raises:
        KeyError: If there is no pinned checkpoint for ``name`` at
            ``version``.

    Examples:
        >>> sorted(CHECKPOINTS)  # doctest: +NORMALIZE_WHITESPACE
        ['limix:v2', 'tabfm:v1.0', 'tabicl:v2', 'tabpfn:v2', 'tabpfn:v2.5',
         'tabpfn:v2.6', 'tabpfn:v3', 'tabpfn:v3.5', 'tabpfn:v3.5-fast']
    """
    return get_checkpoint(name, version).download()


def is_cached(name: str, version: str | None = None) -> bool:
    """Returns whether ``name``'s weights are already in the local cache.

    Never triggers a download, so this is the cheap way to decide whether to
    warn a user that they are about to pull several gigabytes.

    Args:
        name: A backend name or a ``"backend:version"`` key, as for
            :func:`get_checkpoint`.
        version: The model version, or ``None`` for the default.

    Returns:
        ``True`` if the weights can be loaded without a network connection.
    """
    try:
        get_checkpoint(name, version).download(local_files_only=True)
    except Exception:  # noqa: BLE001 - any failure to load offline is a miss.
        return False
    return True
