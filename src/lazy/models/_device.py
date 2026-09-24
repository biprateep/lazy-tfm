# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Resolving the ``device`` parameter shared by the foundation-model backends."""

from __future__ import annotations

__all__ = ["resolve_device"]


def resolve_device(device: str = "auto") -> str:
    """Turn ``"auto"`` into a concrete torch device string.

    ``"auto"`` picks CUDA when it is usable and falls back to CPU otherwise.
    Any other value is passed through untouched, so ``"cuda:1"`` works, and an
    explicit ``"cuda"`` on a machine without one raises rather than silently
    running a foundation model on the CPU for hours.
    """
    if device != "auto":
        if device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"device={device!r} requested but CUDA is unavailable"
                )
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a backend extra
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
