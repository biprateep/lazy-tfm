# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Runs test bodies in threads with a time limit, so a deadlock fails fast."""

from collections.abc import Callable
import threading
import time
from typing import Any

import pytest

from lazy.models import _locks

TIMEOUT = 20.0
"""Seconds a test waits on a barrier or event shared between its threads.
:func:`run` waits twice as long, so a broken barrier is reported as the error it
raises rather than as a thread that never finished.
"""


def run(*targets: Callable[[], object], timeout: float = 2 * TIMEOUT) -> bool:
    """Runs each target in its own daemon thread and waits for all of them.

    Args:
        *targets: Functions that take no arguments.
        timeout: Seconds to wait for all the threads together.

    Returns:
        Whether every thread finished in time.

    Raises:
        BaseException: The first exception a target raised, including a
            pytest outcome such as ``pytest.fail``.
    """
    errors = []

    def wrap(target: Callable[[], object]) -> None:
        try:
            target()
        except BaseException as error:  # noqa: BLE001 - re-raised below.
            errors.append(error)

    threads = [
        threading.Thread(target=wrap, args=(target,), daemon=True)
        for target in targets
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if errors:
        raise errors[0]
    return not any(thread.is_alive() for thread in threads)


def watch_contention(
    monkeypatch: pytest.MonkeyPatch, watched: Callable[[object], bool]
) -> threading.Event:
    """Notes when a thread finds a watched object's lock already held.

    Patches :func:`lazy.models._locks.network_lock` for the test's duration.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        watched: Says which objects' locks to watch.

    Returns:
        An event set when a thread finds a watched lock held.
    """
    contended = threading.Event()
    lock_of = _locks.network_lock

    class WatchedLock:
        def __init__(self, obj: object) -> None:
            self.lock = lock_of(obj)
            self.watched = watched(obj)

        def __enter__(self) -> bool:
            if not self.lock.acquire(blocking=False):
                if self.watched:
                    contended.set()
                self.lock.acquire()
            return True

        def __exit__(self, *exc: object) -> None:
            self.lock.release()

    monkeypatch.setattr(_locks, "network_lock", WatchedLock)
    return contended


def held_until(
    event: threading.Event, function: Callable[..., Any]
) -> Callable[..., Any]:
    """``function``, whose first call waits for ``event`` and fails without.

    Args:
        event: What the first call waits for, at most :data:`TIMEOUT`.
        function: The function to wrap.

    Returns:
        The wrapped function.
    """
    first = threading.Lock()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Only the first call acquires it, and none releases it.
        if first.acquire(blocking=False):
            assert event.wait(TIMEOUT), "no other thread found the lock held"
        return function(*args, **kwargs)

    return wrapped
