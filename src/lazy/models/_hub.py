"""Pretrained checkpoints: where they come from, and where they are cached.

Every backend in :mod:`lazy.models` runs a pretrained tabular foundation model
whose weights are not shipped with this package. They are fetched from the
Hugging Face Hub the first time an estimator predicts, and reused from the
local HF cache (``$HF_HOME``, by default ``~/.cache/huggingface/hub``) on every
later call, in this or any other project.

Revisions are pinned. A foundation model's weights are part of the method, so
an unpinned checkpoint would silently change published numbers.

A backbone is not one model but a family of them, so a checkpoint is named by
*backend and version* -- ``"tabpfn:v3"``, not ``"tabpfn"`` -- in the version
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

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from pathlib import Path

__all__ = [
    "CHECKPOINTS",
    "DEFAULT_VERSIONS",
    "Checkpoint",
    "download_checkpoint",
    "get_checkpoint",
    "is_cached",
    "list_versions",
]


@dataclass(frozen=True)
class Checkpoint:
    """A pinned set of pretrained weights on the Hugging Face Hub."""

    #: Which backend loads these weights; a name in :data:`lazy.ESTIMATORS`.
    backend: str
    #: The model version, in the backbone's own vocabulary -- ``"v2.5"``,
    #: ``"v3"``, ``"v3.5-fast"`` are the strings TabPFN itself uses.
    version: str
    repo_id: str
    #: The distribution providing the backend. Its *installed* version is half
    #: of "which model was this": the weights are the other half.
    package: str
    #: ``snapshot_download`` patterns, for repositories holding several models.
    allow_patterns: tuple[str, ...] | None = None
    #: A single file to fetch instead of a filtered snapshot.
    filename: str | None = None
    #: Pinned git revision; ``None`` means the repository has no history worth
    #: pinning because ``filename`` already names an immutable artifact.
    revision: str | None = None
    #: Anything a user must know before downloading. Surfaced in the docs.
    license_note: str = ""
    size_note: str = ""

    @property
    def key(self) -> str:
        """``"<backend>:<version>"``: the key this checkpoint is registered under.

        >>> get_checkpoint("tabpfn").key
        'tabpfn:v3'
        """
        return f"{self.backend}:{self.version}"

    def download(self, *, local_files_only: bool = False) -> Path:
        """Fetch (or locate) the weights and return the path they live at."""
        from huggingface_hub import hf_hub_download, snapshot_download

        if self.filename is not None:
            return Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    revision=self.revision,
                    local_files_only=local_files_only,
                )
            )
        return Path(
            snapshot_download(
                repo_id=self.repo_id,
                revision=self.revision,
                allow_patterns=list(self.allow_patterns)
                if self.allow_patterns
                else None,
                local_files_only=local_files_only,
            )
        )

    def provenance(self, *, device: str | None = None) -> dict[str, str | None]:
        """Which weights, and which code, produced a set of numbers.

        Everything needed to identify the model, and nothing that is merely a
        setting -- ``n_estimators``, ``chunk_size`` and the rest are in
        ``get_params()``. There are no local paths in it either, so the same
        model on two machines gives the same record, and it survives being
        written to JSON next to a results table.

        >>> record = get_checkpoint("tabpfn", "v2.5").provenance(device="cpu")
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
            "lazy": _package_version("lazy-photoz"),
            "device": device,
        }


def _package_version(distribution: str) -> str:
    """``"tabpfn 9.0.0"``, or a stand-in when the distribution is not installed."""
    try:
        return f"{distribution} {_installed_version(distribution)}"
    except PackageNotFoundError:  # pragma: no cover - only outside an install
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
                "TabFM weights are released by Google under a non-commercial licence; "
                "read it at https://huggingface.co/google/tabfm-1.0.0-pytorch before use."
            ),
            size_note="~6.6 GB for the classification checkpoint.",
        ),
        Checkpoint(
            backend="tabicl",
            version="v2",
            repo_id="jingang/TabICL",
            package="tabicl",
            filename="tabicl-regressor-v2-20260212.ckpt",
            license_note=(
                "TabICL is released under BSD-3-Clause; see https://huggingface.co/jingang/TabICL."
            ),
            size_note="~100 MB.",
        ),
        # The TabPFN family. Every filename below names "the default" of its
        # release, which is a moving name -- the repository is free to replace
        # the file a later release calls that -- so unlike TabICL's dated
        # filename each one needs the revision as well.
        Checkpoint(
            backend="tabpfn",
            version="v2",
            repo_id="Prior-Labs/TabPFN-v2-reg",
            package="tabpfn",
            filename="tabpfn-v2-regressor.ckpt",
            revision="4972a65a1b30806315c6f92499959ffbfc69a673",
            license_note=(
                "TabPFN v2 weights are released by Prior Labs under the Prior Labs License "
                "v1.1, which is Apache-2.0 with an added attribution clause -- the only "
                "TabPFN version here that is not non-commercial. See "
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
                "TabPFN-2.5 weights are released by Prior Labs under the TABPFN-2.5 "
                "Non-Commercial License; read it at "
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
                "TabPFN-2.6 weights are released by Prior Labs under the TABPFN-2.6 "
                "Non-Commercial License; read it at "
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
                "TabPFN-3 weights are released by Prior Labs under the TABPFN-3 "
                "Non-Commercial License; read it at "
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
                "TabPFN-3.5 weights are released by Prior Labs under the TABPFN-3.5 "
                "Non-Commercial License; read it at "
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
                "TabPFN-3.5 weights are released by Prior Labs under the TABPFN-3.5 "
                "Non-Commercial License; read it at "
                "https://huggingface.co/Prior-Labs/tabpfn_3_5 before use."
            ),
            size_note="~335 MB. A separate, faster model, not a smaller copy of v3.5.",
        ),
    )
}

#: The version each backend loads when none is asked for. Changing one of these
#: changes what every unversioned call returns, so they move only deliberately.
DEFAULT_VERSIONS: dict[str, str] = {
    "tabfm": "v1.0",
    "tabicl": "v2",
    "tabpfn": "v3",
}


def get_checkpoint(name: str, version: str | None = None) -> Checkpoint:
    """The pinned checkpoint for a backend, at a version.

    ``name`` is a backend name, which takes that backend's default version, or
    a full ``"backend:version"`` key. An explicit ``version`` wins over one
    spelled into ``name``.

    >>> get_checkpoint("tabpfn").version
    'v3'
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
            f"unknown version {version!r} for {name!r}; known: {list_versions(name)}"
        ) from None


def list_versions(name: str) -> list[str]:
    """The model versions ``name`` has pinned weights for, sorted.

    >>> list_versions("tabpfn")
    ['v2', 'v2.5', 'v2.6', 'v3', 'v3.5', 'v3.5-fast']
    >>> list_versions("tabicl")
    ['v2']
    """
    return sorted(
        spec.version for spec in CHECKPOINTS.values() if spec.backend == name
    )


def download_checkpoint(name: str, version: str | None = None) -> Path:
    """Fetch the pretrained weights for ``name``, or return the cached path.

    Safe to call repeatedly: the Hugging Face cache makes every call after the
    first a no-op. Use it to warm the cache on a login node before submitting a
    job to a compute node with no outbound network.

    >>> sorted(CHECKPOINTS)  # doctest: +NORMALIZE_WHITESPACE
    ['tabfm:v1.0', 'tabicl:v2', 'tabpfn:v2', 'tabpfn:v2.5', 'tabpfn:v2.6',
     'tabpfn:v3', 'tabpfn:v3.5', 'tabpfn:v3.5-fast']
    """
    return get_checkpoint(name, version).download()


def is_cached(name: str, version: str | None = None) -> bool:
    """Whether ``name``'s weights are already in the local cache.

    Never triggers a download, so this is the cheap way to decide whether to
    warn a user that they are about to pull several gigabytes.
    """
    try:
        get_checkpoint(name, version).download(local_files_only=True)
    except Exception:  # noqa: BLE001 - any hub failure means "not usable offline"
        return False
    return True
