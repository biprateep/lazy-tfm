# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Runs test bodies in threads with a time limit, so a deadlock fails fast."""

from collections.abc import Callable
import threading
import time

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
