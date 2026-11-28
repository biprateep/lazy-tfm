# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Warnings that point at the user's line, not at lazy's own code."""

import inspect
import os
from typing import Any
import warnings

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class LazyWarning(UserWarning):
    """The base of lazy's own warning classes.

    Deprecations stay ``FutureWarning``, and notices about the runtime state,
    such as an incomplete cached snapshot, stay ``RuntimeWarning``.
    """


def warn(message: str, category: type[Warning]) -> None:
    """Issues a warning attributed to the first caller outside lazy.

    Args:
        message: The warning's text.
        category: Its class.
    """
    # On Python 3.12, skip_file_prefixes does not skip the "<string>" frame
    # of the __init__ that dataclasses generates (the warning then points at
    # "<string>"), so count the frames to pass. A "<string>" frame of any
    # other function, such as exec, is the caller's own code.
    frame = inspect.currentframe()
    level = 1
    while (
        frame is not None
        and frame.f_back is not None
        and (
            frame.f_code.co_filename.startswith(_PACKAGE_DIR)
            or (
                frame.f_code.co_filename == "<string>"
                and frame.f_code.co_name == "__init__"
            )
        )
    ):
        frame = frame.f_back
        level += 1
    warnings.warn(message, category, stacklevel=level)


class _Default(str):
    """A default that equals its string and is not any string a caller passes.

    ``is`` tells it from the same string passed on purpose.
    """


POINT_DEFAULT = _Default("mode")
"""The default of the point-estimate argument, until it becomes the median."""
# Any: the Normalization Literal does not accept a str subclass.
NORMALIZATION_DEFAULT: Any = _Default("trapezoid")
"""The default of ``Grid.normalization``, until it becomes the histogram."""


def resolve_default(value: str, argument: str, new: str) -> str:
    """Warns if a caller left out ``argument``, and returns the old default.

    Args:
        value: The argument as received.
        argument: Its name, for the message.
        new: The default of lazy 0.8.0.

    Returns:
        ``value`` as a plain string.
    """
    if isinstance(value, _Default):
        warn(
            f"{argument} was not passed: its default changes from "
            f"{str(value)!r} to {new!r} in lazy 0.8.0. Pass "
            f"{argument}={str(value)!r} to keep today's behaviour, or "
            f"{argument}={new!r} to adopt the new one now.",
            FutureWarning,
        )
        return str(value)
    return value
