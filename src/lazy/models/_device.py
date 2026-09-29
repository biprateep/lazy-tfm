# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The ``device`` parameter shared by the foundation-model backends."""

from __future__ import annotations

import re

__all__ = ["resolve_device"]

#: The device strings accepted besides ``"auto"``, case-insensitively.
_DEVICE_PATTERN = re.compile(r"cpu|mps|cuda(:\d+)?")


def resolve_device(device: object = "auto") -> str:
    """Turns a ``device`` parameter into a concrete torch device string.

    ``"auto"`` picks CUDA when it is usable and falls back to CPU otherwise.
    Any other value names a device outright, so ``"cuda:1"`` works, and an
    explicit ``"cuda"`` (or ``"mps"``) on a machine without one raises rather
    than silently running a foundation model on the CPU for hours.

    Args:
        device: ``"auto"``, a torch device string such as ``"cpu"``,
            ``"cuda"``, ``"cuda:1"`` or ``"mps"`` (in any case), or a
            ``torch.device``.

    Returns:
        The torch device string to run on, in lower case.

    Raises:
        ValueError: If ``device`` is not one of those.
        RuntimeError: If the requested accelerator is unavailable.
    """
    if device is None or isinstance(device, bool):
        raise ValueError(
            f"device must be 'auto', 'cpu', 'cuda', 'cuda:<index>', 'mps' or "
            f"a torch.device, not {device!r}"
        )
    # A torch.device prints as its string form ("cuda:0"); str() also lets
    # this module check it without importing torch.
    name = str(device).strip().lower()
    if name == "auto":
        return _auto()
    if not _DEVICE_PATTERN.fullmatch(name):
        raise ValueError(
            f"unknown device {device!r}: use 'auto', 'cpu', 'cuda', "
            "'cuda:<index>', 'mps' or a torch.device"
        )
    if name.startswith("cuda"):
        _check_cuda(name, device)
    elif name == "mps":
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

        if not torch.backends.mps.is_available():
            raise RuntimeError(
                f"device={device!r} requested but MPS is unavailable"
            )
    return name


def _auto() -> str:
    """CUDA if usable, else CPU."""
    try:
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.
    except ImportError:  # pragma: no cover - torch is a backend extra
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _check_cuda(name: str, device: object) -> None:
    """Raises unless the CUDA device ``name`` exists."""
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but CUDA is unavailable"
        )
    _, _, index = name.partition(":")
    count = torch.cuda.device_count()
    if index and int(index) >= count:
        raise RuntimeError(
            f"device={device!r} requested but only {count} CUDA device(s) "
            "are visible"
        )
