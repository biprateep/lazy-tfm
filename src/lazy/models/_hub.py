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
Adding a version is one entry here and the backend's recipe for it
(``_pinned_recipe``); tests/lazy/test_backend_completeness.py reports a
version without one.

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
import warnings

__all__ = [
    "CHECKPOINTS",
    "DEFAULT_VERSIONS",
    "PROVENANCE_KEYS",
    "Checkpoint",
    "download_checkpoint",
    "get_checkpoint",
    "is_cached",
    "list_versions",
]

#: The keys of every :meth:`Checkpoint.provenance` record, in its order.
PROVENANCE_KEYS: tuple[str, ...] = (
    "backend",
    "version",
    "repo_id",
    "filename",
    "revision",
    "package",
    "lazy",
    "device",
)


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
            The docs quote its first word, such as ``"~880 MB"``.
        display_name: The model's name at this version, as its authors
            write it (``"TabPFN-3.5"``).
        parameters: The number of parameters, counted over the
            checkpoint's tensors.
        size_bytes: The download size in bytes, at ``revision``.
        license_name: The license of the weights, in a few words.
        gpu: Whether the model needs a GPU: ``"yes"``, ``"recommended"``
            or ``"optional"``.
        cpu: Whether it runs at a usable speed on a CPU: ``"yes"``,
            ``"no"``, or up to how many context rows (``"≤1,000 rows"``),
            with any condition in parentheses.

    The last six fields write the documentation's table of the models
    (docs/models/index.md), so they are kept with the weights they describe.
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
    display_name: str = ""
    parameters: int = 0
    size_bytes: int = 0
    license_name: str = ""
    gpu: str = ""
    cpu: str = ""

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

        A single file appears in the cache only once it is whole, but a
        snapshot directory appears with its first file, so an interrupted
        download leaves one that looks cached. A snapshot is taken from the
        cache only once it is known to be complete (:meth:`_complete`);
        otherwise the Hub completes it, and confirms it against the
        repository's file list. Without a network connection an
        unconfirmed snapshot is returned with a warning.

        Args:
            local_files_only: Only look in the local Hugging Face cache, never
                on the network.

        Returns:
            The checkpoint file, or the snapshot directory when the
            checkpoint is a filtered snapshot rather than one file.
        """
        if self.revision is None:
            return self._fetch(local_files_only=local_files_only)
        try:
            cached: pathlib.Path | None = self._fetch(local_files_only=True)
        except FileNotFoundError:
            # Not cached (huggingface_hub's LocalEntryNotFoundError is a
            # FileNotFoundError): fall through to the network.
            if local_files_only:
                raise
            cached = None
        if cached is not None and self._complete(cached):
            return cached
        if cached is not None and local_files_only:
            self._warn_unconfirmed(cached)
            return cached
        try:
            path = self._fetch(local_files_only=False)
        except Exception:
            if cached is None:
                raise
            # The Hub cannot be reached: what is cached may yet be whole.
            self._warn_unconfirmed(cached)
            return cached
        if self.filename is None and not self._confirm(path):
            self._warn_unconfirmed(path)
        return path

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
                allow_patterns=self._patterns(),
                local_files_only=local_files_only,
            )
        )

    def _complete(self, path: pathlib.Path) -> bool:
        """Whether the cached weights at ``path`` are all there.

        A single file always is. A snapshot is when a completed download
        marked it (:meth:`_confirm`), or when the files it should hold add
        up to :attr:`size_bytes`, which tells a snapshot cached before the
        marker existed, offline.

        Args:
            path: What :meth:`_fetch` returned from the cache.

        Returns:
            Whether the checkpoint can be loaded from ``path``.
        """
        if self.filename is not None:
            return True
        marker = self._marker(path)
        try:
            if marker.read_text() == self._marker_text():
                return True
        except OSError:
            pass
        return bool(self.size_bytes) and self._bytes(path) == self.size_bytes

    def _confirm(self, path: pathlib.Path) -> bool:
        """Checks a snapshot against the repository, and marks it if whole.

        Args:
            path: The snapshot directory, just fetched.

        Returns:
            Whether every file the repository holds under
            :attr:`allow_patterns`, at :attr:`revision`, is in ``path``;
            ``False`` too when the Hub cannot be reached to say.
        """
        import huggingface_hub  # noqa: PLC0415 - kept out of `import lazy`.
        from huggingface_hub import utils  # noqa: PLC0415

        try:
            listed = huggingface_hub.HfApi().list_repo_files(
                self.repo_id, revision=self.revision
            )
        except Exception:  # noqa: BLE001 - offline, or the Hub is down.
            return False
        expected = utils.filter_repo_objects(
            listed, allow_patterns=self._patterns()
        )
        if not all((path / name).is_file() for name in expected):
            return False
        marker = self._marker(path)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(self._marker_text())
        except OSError:
            # A read-only cache: the snapshot is whole all the same, and
            # the next download confirms it again.
            pass
        return True

    def _warn_unconfirmed(self, path: pathlib.Path) -> None:
        """Warns that a cached snapshot may be missing files."""
        warnings.warn(
            f"the cached snapshot of {self.repo_id} at {path} could not be "
            "confirmed complete, and may be what an interrupted download "
            "left behind. Run lazy.download_checkpoint("
            f"{self.key!r}) with a network connection to complete it.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _patterns(self) -> list[str] | None:
        """:attr:`allow_patterns` as ``huggingface_hub`` takes them."""
        return list(self.allow_patterns) if self.allow_patterns else None

    def _marker(self, path: pathlib.Path) -> pathlib.Path:
        """Where a snapshot's completed download is recorded.

        Beside the snapshots in the repository's cache entry
        (``<entry>/snapshots/<revision>``), so deleting the model from the
        cache takes it too.
        """
        return path.parent.parent / "lazy-complete" / path.name

    def _marker_text(self) -> str:
        """What a marker holds: the patterns the snapshot was fetched with."""
        return "\n".join(self._patterns() or ["*"])

    def _bytes(self, path: pathlib.Path) -> int:
        """The bytes in the files of ``path`` the patterns select."""
        from huggingface_hub import utils  # noqa: PLC0415

        files = [p for p in path.rglob("*") if p.is_file()]
        names = {p.relative_to(path).as_posix(): p for p in files}
        selected = utils.filter_repo_objects(
            names, allow_patterns=self._patterns()
        )
        return sum(names[name].stat().st_size for name in selected)

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
            package's) and ``device``: :data:`PROVENANCE_KEYS`.

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


# One license covers every Kumo Tabular size.
_KUMO_LICENSE = (
    "Kumo Tabular weights are released by NVIDIA under the OpenMDW License "
    "1.1, which permits commercial use; see "
    "https://huggingface.co/nvidia/Kumo-Tabular."
)

#: Every pinned checkpoint, keyed ``"<backend>:<version>"``. Built from the
#: records themselves so a key can never disagree with what it points at.
CHECKPOINTS: dict[str, Checkpoint] = {
    checkpoint.key: checkpoint
    for checkpoint in (
        # Kumo Tabular's three sizes live in one repository, tagged v1.0.1 at
        # this commit; each has a classifier and a regressor.
        Checkpoint(
            backend="kumo",
            version="v1-large",
            repo_id="nvidia/Kumo-Tabular",
            package="structured-data-models",
            filename="large/regressor.pt",
            revision="3c3e10bbdb590ace29e7026847f92db3c603096d",
            license_note=_KUMO_LICENSE,
            size_note="~860 MB. Pretrained on tables of up to 60,000 rows.",
            display_name="Kumo Tabular Large",
            parameters=215_683_543,
            size_bytes=862_909_259,
            license_name="OpenMDW-1.1",
            gpu="recommended",
            cpu="no",
        ),
        Checkpoint(
            backend="kumo",
            version="v1-medium",
            repo_id="nvidia/Kumo-Tabular",
            package="structured-data-models",
            filename="medium/regressor.pt",
            revision="3c3e10bbdb590ace29e7026847f92db3c603096d",
            license_note=_KUMO_LICENSE,
            size_note="~250 MB. Pretrained on tables of up to 60,000 rows.",
            display_name="Kumo Tabular Medium",
            parameters=62_492_439,
            size_bytes=250_144_651,
            license_name="OpenMDW-1.1",
            gpu="recommended",
            cpu="no",
        ),
        Checkpoint(
            backend="kumo",
            version="v1-small",
            repo_id="nvidia/Kumo-Tabular",
            package="structured-data-models",
            filename="small/regressor.pt",
            revision="3c3e10bbdb590ace29e7026847f92db3c603096d",
            license_note=_KUMO_LICENSE,
            size_note="~110 MB. Pretrained on tables of up to 60,000 rows.",
            display_name="Kumo Tabular Small",
            parameters=28_466_343,
            size_bytes=113_968_523,
            license_name="OpenMDW-1.1",
            gpu="optional",
            cpu="yes",
        ),
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
            display_name="TabFM v1.0",
            parameters=1_639_444_522,
            size_bytes=6_557_901_837,
            license_name="Google, non-commercial",
            gpu="yes",
            cpu="no",
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
                "with StableAI LimiX\"; LimiX's code is under the Apache-2.0-"
                "based Stable AI Technology Co., Ltd. License 1.0. See "
                "https://huggingface.co/stable-ai/LimiX-2."
            ),
            size_note="~1.6 GB. Pretrained for at most about 20,000 rows.",
            display_name="LimiX-2",
            parameters=406_232_101,
            size_bytes=1_625_774_719,
            license_name="StableAI LimiX, non-commercial",
            gpu="yes",
            cpu="no (and Linux only)",
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
            display_name="TabICLv2",
            parameters=28_544_991,
            size_bytes=114_324_594,
            license_name="BSD-3-Clause",
            gpu="optional",
            cpu="yes",
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
            display_name="TabPFN-2",
            parameters=11_091_865,
            size_bytes=44_390_977,
            license_name="Apache-2.0 + attribution",
            gpu="optional",
            cpu="≤1,000 rows",
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
            display_name="TabPFN-2.5",
            parameters=10_196_761,
            size_bytes=40_831_995,
            license_name="Prior Labs, non-commercial",
            gpu="optional",
            cpu="≤1,000 rows",
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
            display_name="TabPFN-2.6",
            parameters=12_864_793,
            size_bytes=51_576_286,
            license_name="Prior Labs, non-commercial",
            gpu="optional",
            cpu="≤1,000 rows",
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
            display_name="TabPFN-3",
            parameters=58_279_945,
            size_bytes=233_289_807,
            license_name="Prior Labs, non-commercial",
            gpu="recommended",
            cpu="no",
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
            display_name="TabPFN-3.5",
            parameters=218_987_657,
            size_bytes=876_027_932,
            license_name="Prior Labs, non-commercial",
            gpu="recommended",
            cpu="no",
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
            display_name="TabPFN-3.5-fast",
            parameters=83_534_985,
            size_bytes=334_181_708,
            license_name="Prior Labs, non-commercial",
            gpu="optional",
            cpu="≤5,000 rows",
        ),
    )
}

#: The version each backend loads when none is asked for. Changing one of these
#: changes what every unversioned call returns, so they move only deliberately.
DEFAULT_VERSIONS: dict[str, str] = {
    "kumo": "v1-large",
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
    a job to a compute node with no outbound network. It also fetches
    whatever else the backend needs beside its weights, such as source code
    that is not on PyPI (the backend's ``prefetch``).

    Args:
        name: A backend name or a ``"backend:version"`` key, as for
            :func:`get_checkpoint`.
        version: The model version, or ``None`` for the default.

    Returns:
        The local path of the checkpoint file or snapshot directory.

    Raises:
        KeyError: If there is no pinned checkpoint for ``name`` at
            ``version``.
        ImportError: If what the backend needs beside its weights cannot be
            found or downloaded.

    Examples:
        >>> sorted(CHECKPOINTS)  # doctest: +NORMALIZE_WHITESPACE
        ['kumo:v1-large', 'kumo:v1-medium', 'kumo:v1-small', 'limix:v2',
         'tabfm:v1.0', 'tabicl:v2', 'tabpfn:v2', 'tabpfn:v2.5', 'tabpfn:v2.6',
         'tabpfn:v3', 'tabpfn:v3.5', 'tabpfn:v3.5-fast']
    """
    checkpoint = get_checkpoint(name, version)
    _prefetch(checkpoint, download=True)
    return checkpoint.download()


def is_cached(name: str, version: str | None = None) -> bool:
    """Returns whether ``name``'s weights are already in the local cache.

    Never triggers a download, so this is the cheap way to decide whether to
    warn a user that they are about to pull several gigabytes.

    Args:
        name: A backend name or a ``"backend:version"`` key, as for
            :func:`get_checkpoint`.
        version: The model version, or ``None`` for the default.

    Returns:
        ``True`` if the weights, and whatever else the backend fetches
        beside them, can be loaded without a network connection. A snapshot
        an interrupted download left behind is not cached.
    """
    try:
        checkpoint = get_checkpoint(name, version)
        _prefetch(checkpoint, download=False)
        # This module's own class: the local lookup, without the warning
        # download() gives an unconfirmed snapshot.
        path = checkpoint._fetch(local_files_only=True)  # noqa: SLF001
    except Exception:  # noqa: BLE001 - any failure to load offline is a miss.
        return False
    return checkpoint._complete(path)  # noqa: SLF001


def _prefetch(checkpoint: Checkpoint, *, download: bool) -> None:
    """Runs the ``prefetch`` hook of the backend ``checkpoint`` belongs to.

    Args:
        checkpoint: The checkpoint whose weights are being fetched.
        download: Whether the hook may download, or only checks.
    """
    # Imported here: the registry imports every backend, and they import
    # this module.
    from lazy.models import _ensemble  # noqa: PLC0415
    from lazy.models import registry  # noqa: PLC0415

    cls = registry.ESTIMATORS.get(checkpoint.backend)
    if cls is not None and issubclass(cls, _ensemble.ContextEnsembleEstimator):
        cls.prefetch(checkpoint.version, download=download)
