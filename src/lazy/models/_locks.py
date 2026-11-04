# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""One lock per loaded network, for networks that keep state between calls.

Every fit with the same checkpoint and device runs on one network, shared
through :mod:`lazy.models._weights`. Some upstream networks keep the context
of their last call on themselves (TabICL's and Kumo Tabular's key/value
caches), so two threads using one such network at once would answer from
each other's context. Whatever hands such a network a context holds the
network's lock until it has taken the network's answer and cleared it.

An estimator that rebuilds state on first use, as an unpickled estimator
rebuilds the caches its pickle left out, is locked the same way, by the
estimator itself, so that it does not hold up the other models on its network.

Typical usage example:

  with _locks.network_lock(network):
      network._cache = cache
      out = network.predict(queries)
      network._cache = None
"""

from __future__ import annotations

import threading
from typing import Any
import weakref

__all__ = ["network_lock"]

# Mutable global state by design: the locks belong to the objects they guard,
# and go with them. Every access holds _GUARD.
_LOCKS: weakref.WeakKeyDictionary[Any, threading.Lock] = (
    weakref.WeakKeyDictionary()
)
_GUARD = threading.Lock()


def network_lock(network: Any) -> threading.Lock:
    """Returns the lock of a loaded network, the same one for every caller.

    Any other object whose state threads share can be locked the same way,
    such as an estimator that rebuilds its caches on first use.

    Args:
        network: The loaded network, or another such object; it must be
            hashable by identity and accept weak references, as a
            ``torch.nn.Module`` or an estimator does.

    Returns:
        The object's lock, created on first use and dropped with the
        object. It is not reentrant.
    """
    with _GUARD:
        return _LOCKS.setdefault(network, threading.Lock())
