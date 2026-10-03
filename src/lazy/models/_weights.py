# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Loaded networks, shared by every fit in the process.

A fit reads its checkpoint from disk and builds the network on the device,
which on a GPU is most of what a small fit costs. Nothing a fit learns is
stored in the network -- the in-context state, key/value caches included,
lives in the fitted model -- so a second fit with the same weights, on the
same device and in the same precision, can run on the network the first one
loaded. A grid search, a cross-validation or a sweep over ``n_estimators``
then loads each network once.

A network is filed under everything that changes what is loaded: the
backend, the version, the checkpoint file and its pinned revision, the
device, the precision of the weights, and the loader itself (a stand-in
loader in a test is a different network). Each is held once, and
:func:`clear_model_cache` lets go of them all. A fitted model keeps a
reference to the network it ran on, so the memory comes back only once those
models are gone too.

The cache is on by default. ``$LAZY_MODEL_CACHE=0`` turns it off for a
process, and :func:`set_model_cache` at run time; off, every fit loads its
own network, as it did before the cache existed.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
import gc
import os
import sys
import threading
from typing import Any, TypeVar

from lazy.models import _hub

__all__ = [
    "ENV_VAR",
    "clear_model_cache",
    "key",
    "model_cache_enabled",
    "network",
    "set_model_cache",
]

#: The environment variable that turns the cache off, with ``0``, ``false``,
#: ``no`` or ``off``.
ENV_VAR = "LAZY_MODEL_CACHE"

_OFF = frozenset({"0", "false", "no", "off"})

_T = TypeVar("_T")

# The process-wide store. It is mutable global state by design: sharing
# across estimators is its purpose. Every access holds _LOCK.
_NETWORKS: dict[tuple[Hashable, ...], Any] = {}
_LOCK = threading.Lock()
# The switch: None defers to $LAZY_MODEL_CACHE, and set_model_cache
# overrides it.
_SWITCH: dict[str, bool | None] = {"enabled": None}


def model_cache_enabled() -> bool:
    """Returns whether fits share loaded networks.

    Returns:
        The value :func:`set_model_cache` last set, or else whether
        ``$LAZY_MODEL_CACHE`` leaves the cache on.
    """
    enabled = _SWITCH["enabled"]
    if enabled is not None:
        return enabled
    return os.environ.get(ENV_VAR, "1").strip().lower() not in _OFF


def set_model_cache(enabled: bool) -> None:
    """Turns sharing loaded networks across fits on or off.

    Turning it off also lets go of every network held, as
    :func:`clear_model_cache` does.

    Args:
        enabled: Whether later fits reuse a network an earlier fit loaded.
    """
    _SWITCH["enabled"] = bool(enabled)
    if not enabled:
        clear_model_cache()


def clear_model_cache() -> int:
    """Lets go of every loaded network the package holds.

    A fitted model keeps the network it ran on, so the memory is returned
    only once no fitted model refers to it either. The copy of the last
    checkpoint TabPFN keeps in memory goes too, and freed GPU memory is
    handed back to the driver.

    Returns:
        How many networks were dropped.
    """
    with _LOCK:
        dropped = len(_NETWORKS)
        _NETWORKS.clear()
    _clear_upstream_caches()
    if dropped:
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    return dropped


def _clear_upstream_caches() -> None:
    """Empties the checkpoint caches TabPFN keeps of its own.

    TabPFN holds the last checkpoint it read in memory, beside the network
    built from it, and, when ``$TABPFN_MODEL_CACHE_SIZE`` asks, the built
    networks too. Both are private, so each is emptied only if it is there.
    """
    model_loading = sys.modules.get("tabpfn.model_loading")
    if model_loading is None:
        return
    raw = getattr(model_loading, "_load_checkpoint_cached", None)
    if raw is not None and hasattr(raw, "cache_clear"):
        raw.cache_clear()
    built = getattr(model_loading, "clear_built_model_cache", None)
    if built is not None:
        built()


def key(
    backend: str,
    version: str,
    checkpoint: os.PathLike[str] | str,
    device: Any,
    precision: Any,
    *extra: Hashable,
) -> tuple[Hashable, ...]:
    """The key a loaded network is filed under.

    Args:
        backend: The backend's name, as in :data:`lazy.ESTIMATORS`.
        version: The model version.
        checkpoint: The checkpoint file or snapshot directory loaded.
        device: The device the network ends up on.
        precision: The dtype the weights end up in, or whatever setting
            decides it.
        *extra: Anything else that changes what the loader returns, the
            loader itself included.

    Returns:
        A hashable key; its revision is the one pinned for that version.
    """
    spec = _hub.CHECKPOINTS.get(f"{backend}:{version}")
    revision = None if spec is None else spec.revision
    return (
        backend,
        version,
        os.fspath(checkpoint),
        revision,
        str(device),
        str(precision),
        *extra,
    )


def network(cache_key: tuple[Hashable, ...], load: Callable[[], _T]) -> _T:
    """Returns the network filed under ``cache_key``, loading it if need be.

    The lock is held while loading, so two threads asking for the same
    network load it once.

    Args:
        cache_key: What :func:`key` returned for the network.
        load: Loads the network; called only on a miss, or every time when
            the cache is off.

    Returns:
        The network, or what ``load`` returns with it.
    """
    if not model_cache_enabled():
        return load()
    with _LOCK:
        if cache_key not in _NETWORKS:
            _NETWORKS[cache_key] = load()
        return _NETWORKS[cache_key]


def cached_keys() -> list[tuple[Hashable, ...]]:
    """Returns the keys of the networks held now, for tests and debugging."""
    with _LOCK:
        return list(_NETWORKS)
