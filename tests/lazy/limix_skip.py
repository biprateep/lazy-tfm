# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Whether LimiX's real source is at hand, for the tests that need it."""

from lazy.models import _limix_source


def source_present() -> bool:
    """Whether LimiX's source is here, without downloading it.

    Not cached: a test that runs earlier may download the source.

    Returns:
        True when :func:`lazy.models._limix_source.locate` finds it.
    """
    try:
        _limix_source.locate(download=False)
    except ImportError:
        return False
    return True
