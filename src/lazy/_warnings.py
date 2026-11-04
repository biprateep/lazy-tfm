# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Warnings that point at the user's line, not at lazy's own code."""

import os
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
    warnings.warn(message, category, skip_file_prefixes=(_PACKAGE_DIR,))
