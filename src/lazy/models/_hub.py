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
(``recipes``); tests/lazy/test_backend_completeness.py reports a
version without one.

:func:`download_checkpoint` pre-fetches on a machine with a network connection
before running somewhere without one, and :func:`is_cached` checks without
triggering a download. Both, with :func:`get_checkpoint`,
:func:`list_versions` and :class:`LicenseWarning`, are part of the public API
and are re-exported from the top-level :mod:`lazy`.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from importlib import metadata
import pathlib

from lazy import _warnings

__all__ = [
    "CHECKPOINTS",
    "DEFAULT_ENCODER_MODELS",
    "DEFAULT_VERSIONS",
    "ENCODER_CHECKPOINTS",
    "PROVENANCE_KEYS",
    "Checkpoint",
    "LicenseWarning",
    "UnknownNameError",
    "download_checkpoint",
    "get_checkpoint",
    "is_cached",
    "list_versions",
]

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
"""The keys of every :meth:`Checkpoint.provenance` record, in its order."""

_DOCS = "https://lazy-tfm.readthedocs.io/en/stable"
"""The released documentation, whose page for each backend lists its
references.
"""


class LicenseWarning(_warnings.LazyWarning):
    """The license of weights that lazy is about to download.

    Issued before every download that finds nothing cached, so it shows
    again if that download fails. It names the license, which using the
    weights accepts, and the model to cite as well as LAZY.
    """


class UnknownNameError(KeyError, ValueError):
    """A backend name or model version that lazy does not know.

    It is both a ``KeyError`` and a ``ValueError``, so code that caught
    either for a lookup keeps working. Its message names the valid choices.
    """

    def __str__(self) -> str:
        """Returns the message without the quotes ``KeyError`` adds."""
        # BaseException's, which handles any number of arguments.
        return ValueError.__str__(self)


def unknown_name(
    kind: str,
    value: object,
    known: Sequence[str],
    backend: str | None = None,
    argument: str | None = None,
) -> UnknownNameError:
    """Returns the error for an unknown backend name or model version.

    Args:
        kind: What ``value`` should have been, ``"backend"`` or ``"version"``.
        value: The name that was rejected.
        known: The valid names, in the order to show them.
        backend: For a version, the backend it was asked of.
        argument: The argument that held ``value``, when the caller has one.

    Returns:
        The error, for the caller to raise.
    """
    of = ""
    if backend is not None:
        of = f" for {backend!r}"
    elif argument is not None:
        of = f" for {argument}"
    return UnknownNameError(
        f"unknown {kind} {value!r}{of}; known: {list(known)}"
    )


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """A pinned set of pretrained weights on the Hugging Face Hub.

    Attributes:
        backend: Which backend loads these weights, a name in
            :data:`lazy.ESTIMATORS`; for an encoder, its modality.
        version: The model version, in the backbone's own vocabulary --
            ``"v2.5"``, ``"v3"``, ``"v3.5-fast"`` are the strings TabPFN itself
            uses; for an encoder, the name of its model.
        repo_id: The Hugging Face Hub repository holding the weights.
        package: The distribution providing the backend. Its *installed*
            version is half of "which model was this": the weights are the
            other half.
        commercial_use: Whether the license of the weights allows commercial
            use. Read the license itself before relying on it.
        allow_patterns: ``snapshot_download`` patterns, for repositories
            holding several models.
        filename: A single file to fetch instead of a filtered snapshot.
        revision: The pinned git commit of ``repo_id``. ``None`` would track
            the repository's default branch; no registered checkpoint does,
            because even a dated filename can be replaced upstream.
        license_note: Anything a user must know before downloading. The
            first download issues it as a :class:`LicenseWarning`, and the
            docs quote it.
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
        page: The page of the documentation that lists the references of
            the weights, relative to its root; ``""`` for the page of the
            backend, ``models/<backend>.html``.
        extra: The extra of lazy-tfm that installs ``package``, when it is
            not the one its backend or modality names; ``""`` for that one.

    The six fields from ``display_name`` to ``cpu`` write the
    documentation's table of the models (docs/models/index.md), so they
    are kept with the weights they describe.
    """

    backend: str
    version: str
    repo_id: str
    package: str
    commercial_use: bool
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
    page: str = ""
    extra: str = ""

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

        Raises:
            FileNotFoundError: If the weights are not cached and
                ``local_files_only`` is true or the Hub cannot be reached.
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
        if cached is None:
            _warnings.warn(
                f"Downloading the {self.display_name} weights. "
                f"{self.license_note} Using them accepts that license. If "
                f"you publish results from them, cite {self.display_name} "
                f"as well as LAZY; {_DOCS}/"
                f"{self.page or f'models/{self.backend}.html'} lists "
                "its references.",
                LicenseWarning,
            )
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
        _warnings.warn(
            f"the cached snapshot of {self.repo_id} at {path} could not be "
            "confirmed complete, and may be what an interrupted download "
            "left behind. Run lazy.download_checkpoint("
            f"{self.key!r}) with a network connection to complete it.",
            RuntimeWarning,
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
            package's) and ``device``: ``PROVENANCE_KEYS``.

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
            commercial_use=True,
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
            commercial_use=True,
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
            commercial_use=True,
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
            commercial_use=False,
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
            commercial_use=False,
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
            cpu="no",
        ),
        Checkpoint(
            backend="tabicl",
            version="v2",
            repo_id="jingang/TabICL",
            package="tabicl",
            commercial_use=True,
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
            commercial_use=True,
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
            commercial_use=False,
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
            commercial_use=False,
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
            commercial_use=False,
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
            commercial_use=False,
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
            commercial_use=False,
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
"""Every pinned checkpoint, keyed ``"<backend>:<version>"``.

Built from the records themselves so a key can never disagree with what it
points at.
"""

DEFAULT_VERSIONS: dict[str, str] = {
    "kumo": "v1-large",
    "limix": "v2",
    "tabfm": "v1.0",
    "tabicl": "v2",
    "tabpfn": "v3.5",
}
"""The version each backend loads when none is asked for.

Changing one of these changes what every unversioned call returns, so they
move only deliberately.
"""


def _encoder(
    modality: str,
    model: str,
    *,
    repo_id: str,
    revision: str,
    files: tuple[str, ...],
    size_bytes: int,
    parameters: int,
    display_name: str,
    license_name: str,
    package: str = "transformers",
    commercial_use: bool = True,
    extra: str = "",
) -> Checkpoint:
    """The pinned weights of one network of a pretrained encoder.

    A network may restrict commercial use, as some backends' weights do;
    its download notice then asks the user to read the license first, at
    the repository's page. A modality's default network permits it. The
    notice points to the encoders page, and the version of the network is
    that of ``package``, the distribution that loads it.
    """
    note = f"{display_name} is released under {license_name}."
    if not commercial_use:
        note = (
            f"{display_name} is released under a license that restricts "
            f"commercial use ({license_name}); read it at "
            f"https://huggingface.co/{repo_id} before use."
        )
    return Checkpoint(
        backend=modality,
        version=model,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=files,
        package=package,
        commercial_use=commercial_use,
        extra=extra,
        license_note=note,
        display_name=display_name,
        parameters=parameters,
        size_bytes=size_bytes,
        license_name=license_name,
        page="guide/encoders.html",
    )


_TEXT_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_IMAGE_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
_T5_FILES = ("config.json", "generation_config.json", "model.safetensors")

ENCODER_CHECKPOINTS: dict[str, Checkpoint] = {
    checkpoint.key: checkpoint
    for checkpoint in (
        _encoder(
            "text",
            "e5-small-v2",
            repo_id="intfloat/e5-small-v2",
            revision="ffb93f3bd4047442299a41ebb6fa998a38507c52",
            files=(*_TEXT_FILES, "vocab.txt"),
            size_bytes=134_410_262,
            parameters=33_360_512,
            display_name="E5-small-v2",
            license_name="MIT",
        ),
        _encoder(
            "text",
            "bge-small-en-v1.5",
            repo_id="BAAI/bge-small-en-v1.5",
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            files=(*_TEXT_FILES, "vocab.txt"),
            size_bytes=134_410_442,
            parameters=33_360_512,
            display_name="BGE-small-en-v1.5",
            license_name="MIT",
        ),
        _encoder(
            "text",
            "all-MiniLM-L6-v2",
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            files=(*_TEXT_FILES, "vocab.txt"),
            size_bytes=91_567_205,
            parameters=22_713_728,
            display_name="all-MiniLM-L6-v2",
            license_name="Apache-2.0",
        ),
        _encoder(
            "text",
            "gte-modernbert-base",
            repo_id="Alibaba-NLP/gte-modernbert-base",
            revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
            files=_TEXT_FILES,
            size_bytes=301_647_541,
            parameters=149_014_272,
            display_name="GTE-ModernBERT-base",
            license_name="Apache-2.0",
        ),
        # The tokenizer and the weights only: the network is loaded without
        # its language-model head, and sentence-transformers' files are not
        # read.
        _encoder(
            "text",
            "Qwen3-Embedding-0.6B",
            repo_id="Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            files=(
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
            size_bytes=1_203_020_554,
            parameters=595_776_512,
            display_name="Qwen3-Embedding-0.6B",
            license_name="Apache-2.0",
        ),
        _encoder(
            "image",
            "dinov2-with-registers-small",
            repo_id="facebook/dinov2-with-registers-small",
            revision="0d9846e56b43a21fa46d7f3f5070f0506a5795a9",
            files=_IMAGE_FILES,
            size_bytes=88_257_637,
            parameters=22_058_112,
            display_name="DINOv2-small with registers",
            license_name="Apache-2.0",
        ),
        _encoder(
            "image",
            "dinov2-large",
            repo_id="facebook/dinov2-large",
            revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
            files=_IMAGE_FILES,
            size_bytes=1_217_523_873,
            parameters=304_368_640,
            display_name="DINOv2-large",
            license_name="Apache-2.0",
        ),
        # The repository holds the text tower too: the parameters and the
        # download count both towers, and only the vision tower is loaded.
        _encoder(
            "image",
            "siglip2-base-patch16-224",
            repo_id="google/siglip2-base-patch16-224",
            revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
            files=_IMAGE_FILES,
            size_bytes=1_500_801_551,
            parameters=375_187_970,
            display_name="SigLIP 2 base",
            license_name="Apache-2.0",
        ),
        # The same: both towers. The weights are a PyTorch pickle, which
        # transformers loads with torch.load(weights_only=True).
        _encoder(
            "image",
            "clip-vit-base-patch16",
            repo_id="openai/clip-vit-base-patch16",
            revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
            files=(
                "config.json",
                "pytorch_model.bin",
                "preprocessor_config.json",
            ),
            size_bytes=598_645_443,
            parameters=149_621_011,
            display_name="CLIP ViT-B/16",
            license_name="MIT",
        ),
        # A timm network, which transformers wraps. The weights are a PyTorch
        # pickle, loaded with torch.load(weights_only=True); timm runs them.
        _encoder(
            "image",
            "zoobot-encoder-convnext_nano",
            repo_id="mwalmsley/zoobot-encoder-convnext_nano",
            revision="494bae25811233bb6cec2389322b09866bfdbe08",
            files=("config.json", "pytorch_model.bin"),
            size_bytes=59_865_437,
            parameters=14_952_560,
            display_name="Zoobot ConvNeXt-Nano",
            license_name="Apache-2.0",
        ),
        _encoder(
            "time",
            "chronos-t5-tiny",
            repo_id="amazon/chronos-t5-tiny",
            revision="29d808298f1a62493e7b9a5e08529d0d930fa189",
            files=_T5_FILES,
            size_bytes=33_589_724,
            parameters=8_394_496,
            display_name="Chronos-T5-tiny",
            license_name="Apache-2.0",
            package="chronos-forecasting",
        ),
        _encoder(
            "time",
            "chronos-t5-mini",
            repo_id="amazon/chronos-t5-mini",
            revision="bd6a4fde8403b8469acd0abd52852b29dbe61c7b",
            files=_T5_FILES,
            size_bytes=81_836_527,
            parameters=20_456_192,
            display_name="Chronos-T5-mini",
            license_name="Apache-2.0",
            package="chronos-forecasting",
        ),
        _encoder(
            "time",
            "chronos-t5-small",
            repo_id="amazon/chronos-t5-small",
            revision="a971ba21945c4f1796b17a91fe69214b5f4ad472",
            files=_T5_FILES,
            size_bytes=184_633_735,
            parameters=46_154_240,
            display_name="Chronos-T5-small",
            license_name="Apache-2.0",
            package="chronos-forecasting",
        ),
        _encoder(
            "time",
            "chronos-bolt-small",
            repo_id="amazon/chronos-bolt-small",
            revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8",
            files=("config.json", "model.safetensors"),
            size_bytes=190_889_945,
            parameters=47_718_016,
            display_name="Chronos-Bolt-small",
            license_name="Apache-2.0",
            package="chronos-forecasting",
        ),
        _encoder(
            "time",
            "chronos-2",
            repo_id="amazon/chronos-2",
            revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
            files=("config.json", "model.safetensors"),
            size_bytes=477_931_539,
            parameters=119_477_664,
            display_name="Chronos-2",
            license_name="Apache-2.0",
            package="chronos-forecasting",
        ),
        # Weights in transformers' format; the original's torch_model.ckpt
        # is not downloaded. The file holds the forecast head too (4.9 M
        # parameters), which is not loaded.
        _encoder(
            "time",
            "timesfm-2.0-500m",
            repo_id="google/timesfm-2.0-500m-pytorch",
            revision="dc2443792ce5516872b89b37cf1bc058c3bf0c10",
            files=("config.json", "model.safetensors"),
            size_bytes=1_995_407_668,
            parameters=498_828_960,
            display_name="TimesFM-2.0-500M",
            license_name="Apache-2.0",
        ),
        # The port to transformers of google/timesfm-2.5-200m-pytorch, which
        # Google publishes under Apache-2.0, as the original. The file holds
        # the forecast heads too (33 M parameters), which are not loaded.
        _encoder(
            "time",
            "timesfm-2.5-200m",
            repo_id="google/timesfm-2.5-200m-transformers",
            revision="5a9806b9b291fad9233b5249d88263f1846304d3",
            files=("config.json", "model.safetensors"),
            size_bytes=925_188_362,
            parameters=231_289_280,
            display_name="TimesFM-2.5-200M",
            license_name="Apache-2.0",
        ),
        # The timesfm package, under Apache-2.0, loads the network; the
        # file holds the forecast head too (0.7 M parameters), which the
        # embedding does not use. The license comes with the weights.
        _encoder(
            "time",
            "timesfm-3.0",
            repo_id="google/timesfm-3.0-pytorch",
            revision="43046b85ec22d584a13f8098c2ed39c889e129c2",
            files=("LICENSE", "config.json", "model.safetensors"),
            size_bytes=1_322_907_367,
            parameters=330_710_976,
            display_name="TimesFM-3.0",
            license_name="TimesFM, non-commercial",
            package="timesfm",
            commercial_use=False,
        ),
        # A PyTorch Lightning checkpoint, which tirex-ts loads with
        # torch.load(weights_only=True). The NXAI Community License covers
        # tirex-ts too.
        _encoder(
            "time",
            "tirex",
            repo_id="NX-AI/TiRex",
            revision="43ebaaff068c1390785c387b3e5178df3f4c92ca",
            files=("LICENSE", "model.ckpt"),
            size_bytes=141_237_624,
            parameters=35_291_200,
            display_name="TiRex",
            license_name="NXAI Community, restricted commercial use",
            package="tirex-ts",
            commercial_use=False,
            # tirex-ts's own code is under that license, so neither the
            # encoders extra nor all installs it.
            extra="tirex",
        ),
    )
}
"""The pinned weights of every pretrained encoder, keyed
``"<modality>:<model>"``. The models' docs tables do not list them.
"""

DEFAULT_ENCODER_MODELS: dict[str, str] = {
    "image": "dinov2-with-registers-small",
    "text": "e5-small-v2",
    "time": "chronos-t5-tiny",
}
"""The model each modality's pretrained encoder loads when none is asked for.
"""


def tables(name: str) -> tuple[dict[str, Checkpoint], dict[str, str]]:
    """The checkpoints and the defaults that ``name`` is looked up in.

    Args:
        name: A backend name, or a modality of the encoders.

    Returns:
        :data:`ENCODER_CHECKPOINTS` and :data:`DEFAULT_ENCODER_MODELS` for a
        modality, else :data:`CHECKPOINTS` and :data:`DEFAULT_VERSIONS`.
    """
    if name in DEFAULT_ENCODER_MODELS:
        return ENCODER_CHECKPOINTS, DEFAULT_ENCODER_MODELS
    return CHECKPOINTS, DEFAULT_VERSIONS


def get_checkpoint(name: str, version: str | None = None) -> Checkpoint:
    """Returns the pinned checkpoint for a backend, at a version.

    An encoder's weights are found the same way: ``name`` is its modality
    (``"text"``, ``"image"`` or ``"time"``), the version is its model, and
    the defaults are :data:`lazy.encoders.DEFAULT_MODELS`.

    Args:
        name: A backend name, which takes that backend's default version, or
            a full ``"backend:version"`` key.
        version: The model version; an explicit one wins over one spelled
            into ``name``. ``None`` for the default in
            :data:`DEFAULT_VERSIONS`.

    Returns:
        The registered checkpoint.

    Raises:
        UnknownNameError: If the backend is unknown, or has no pinned
            checkpoint at that version.

    Examples:
        >>> get_checkpoint("tabpfn").version
        'v3.5'
        >>> get_checkpoint("tabpfn:v2.5").repo_id
        'Prior-Labs/tabpfn_2_5'
    """
    if ":" in name:
        name, _, spelled = name.partition(":")
        version = version if version is not None else spelled
    checkpoints, defaults = tables(name)
    if name not in defaults:
        known = sorted([*DEFAULT_VERSIONS, *DEFAULT_ENCODER_MODELS])
        raise unknown_name("backend", name, known)
    if version is None:
        version = defaults[name]
    try:
        return checkpoints[f"{name}:{version}"]
    except KeyError:
        raise unknown_name(
            "version", version, list_versions(name), name
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
    checkpoints, _ = tables(name)
    return sorted(
        spec.version for spec in checkpoints.values() if spec.backend == name
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
        UnknownNameError: If there is no pinned checkpoint for ``name`` at
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
        an interrupted download left behind is not cached. ``False`` also
        for a backend or version with no pinned checkpoint, with a
        ``FutureWarning``: from lazy 0.7.0 that raises
        :class:`UnknownNameError`.
    """
    try:
        checkpoint = get_checkpoint(name, version)
    except UnknownNameError as error:
        _warnings.warn(
            f"{error}. lazy.is_cached returns False for it; from lazy 0.7.0 "
            "it raises lazy.UnknownNameError.",
            FutureWarning,
        )
        return False
    try:
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
