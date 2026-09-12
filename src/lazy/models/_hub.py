"""Pretrained checkpoints: where they come from, and where they are cached.

Every backend in :mod:`lazy.models` runs a pretrained tabular foundation model
whose weights are not shipped with this package. They are fetched from the
Hugging Face Hub the first time an estimator predicts, and reused from the
local HF cache (``$HF_HOME``, by default ``~/.cache/huggingface/hub``) on every
later call, in this or any other project.

Revisions are pinned. A foundation model's weights are part of the method, so
an unpinned checkpoint would silently change published numbers.

Call :func:`download_checkpoint` to pre-fetch on a machine with a network
connection before running somewhere without one, and :func:`is_cached` to check
without triggering a download. These are the only functions here that are part
of the public API; they are re-exported from the top-level :mod:`lazy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["CHECKPOINTS", "Checkpoint", "download_checkpoint", "is_cached"]


@dataclass(frozen=True)
class Checkpoint:
    """A pinned set of pretrained weights on the Hugging Face Hub."""

    repo_id: str
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
                allow_patterns=list(self.allow_patterns) if self.allow_patterns else None,
                local_files_only=local_files_only,
            )
        )


#: Checkpoints keyed by the estimator name they belong to (see
#: :func:`lazy.get_estimator`).
CHECKPOINTS: dict[str, Checkpoint] = {
    "tabfm": Checkpoint(
        repo_id="google/tabfm-1.0.0-pytorch",
        revision="77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83",
        allow_patterns=("classification/**", "config.json", "LICENSE", "README.md"),
        license_note=(
            "TabFM weights are released by Google under a non-commercial licence; "
            "read it at https://huggingface.co/google/tabfm-1.0.0-pytorch before use."
        ),
        size_note="~6.6 GB for the classification checkpoint.",
    ),
    "tabicl": Checkpoint(
        repo_id="jingang/TabICL",
        filename="tabicl-regressor-v2-20260212.ckpt",
        license_note="TabICL is released under BSD-3-Clause; see https://huggingface.co/jingang/TabICL.",
        size_note="~100 MB.",
    ),
}


def download_checkpoint(name: str) -> Path:
    """Fetch the pretrained weights for ``name``, or return the cached path.

    Safe to call repeatedly: the Hugging Face cache makes every call after the
    first a no-op. Use it to warm the cache on a login node before submitting a
    job to a compute node with no outbound network.

    >>> sorted(CHECKPOINTS)
    ['tabfm', 'tabicl']
    """
    return _checkpoint(name).download()


def is_cached(name: str) -> bool:
    """Whether ``name``'s weights are already in the local cache.

    Never triggers a download, so this is the cheap way to decide whether to
    warn a user that they are about to pull several gigabytes.
    """
    try:
        _checkpoint(name).download(local_files_only=True)
    except Exception:  # noqa: BLE001 - any hub failure means "not usable offline"
        return False
    return True


def _checkpoint(name: str) -> Checkpoint:
    try:
        return CHECKPOINTS[name]
    except KeyError:
        raise KeyError(f"unknown checkpoint {name!r}; known: {sorted(CHECKPOINTS)}") from None
