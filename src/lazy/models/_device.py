# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The ``device`` parameter shared by the foundation-model backends."""

from __future__ import annotations

import functools
import re

__all__ = [
    "free_memory",
    "is_auto",
    "mps_matmul_into_views_works",
    "release_memory",
    "resolve_device",
]

_DEVICE_PATTERN = re.compile(r"cpu|mps|cuda(:\d+)?")
"""The device strings accepted besides ``"auto"``, case-insensitively."""


def resolve_device(device: object = "auto") -> str:
    """Turns a ``device`` parameter into a concrete torch device string.

    ``"auto"`` picks CUDA when it is usable, then Apple's MPS, and falls back
    to CPU otherwise.
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
    if is_auto(device):
        return _auto()
    name = str(device).strip().lower()
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


def is_auto(device: object) -> bool:
    """Whether a ``device`` parameter is ``"auto"``, in any case or spacing.

    Args:
        device: A ``device`` parameter, as :func:`resolve_device` takes it.

    Returns:
        True if :func:`resolve_device` picks the device for ``device``.
    """
    return str(device).strip().lower() == "auto"


def release_memory(device: object) -> None:
    """Returns the memory PyTorch caches unused on a GPU to the system.

    On Apple silicon the GPU shares memory with the system, so memory
    PyTorch keeps cached slows the whole machine. Does nothing on a CPU.

    Args:
        device: A torch device, or its string form, such as ``"mps"``.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    kind = torch.device(str(device)).type
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def free_memory(device: object) -> int | None:
    """Free bytes on a GPU, or None on a CPU.

    On MPS, the share of system memory Metal recommends for the GPU, less
    what PyTorch already holds there.

    Args:
        device: A torch device, or its string form, such as ``"cuda:0"``.

    Returns:
        The free bytes, or None when ``device`` is a CPU.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    gpu = torch.device(str(device))
    if gpu.type == "cuda":
        free, _ = torch.cuda.mem_get_info(gpu)
        return int(free)
    if gpu.type == "mps":
        return max(
            0,
            torch.mps.recommended_max_memory()
            - torch.mps.driver_allocated_memory(),
        )
    return None


@functools.cache
def mps_matmul_into_views_works() -> bool:
    """Whether MPS writes a batched matmul into a strided ``out`` correctly.

    On some Apple silicon setups (seen on macOS 26.0.1 with torch 2.13 to
    2.15), ``torch.matmul(a, b, out=view)`` on MPS silently writes wrong
    values when ``view`` is a transposed slice of a larger tensor, as
    upstream code does to save memory. This checks it once per process on
    a small product, which takes about a millisecond once Metal has
    started; the CPU computes the reference.

    Returns:
        True when the MPS product matches the CPU's to float32 rounding.
    """
    import torch  # noqa: PLC0415 - torch is an optional, heavy extra.

    # Smaller products happen to come out right on the affected setups.
    n_groups, n_rows, n_inner, n_out = 4, 64, 32, 32
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(n_groups, n_rows, n_inner, generator=generator)
    b = torch.randn(n_groups, n_inner, n_out, generator=generator)
    buffer = torch.empty(n_rows, 1 + n_groups, n_out, device="mps")
    view = buffer[:, 1:, :]
    torch.matmul(a.to("mps"), b.to("mps"), out=view.transpose(0, 1))
    expected = torch.matmul(a, b).transpose(0, 1)
    return torch.allclose(view.cpu(), expected, rtol=1e-4, atol=1e-4)


def _auto() -> str:
    """CUDA if usable, else MPS if usable, else CPU."""
    try:
        import torch  # noqa: PLC0415 - torch is an optional, heavy extra.
    except ImportError:  # pragma: no cover - torch is a backend extra
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


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
