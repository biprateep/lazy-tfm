# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Whether LimiX's real source is at hand, for the tests that need it."""

from importlib import util

from lazy.models import _limix_source


def source_present() -> bool:
    """Whether LimiX's source and its dependencies are here.

    Checked without downloading. A source cached by ``lazy setup`` in an
    environment without the ``limix`` extra cannot import, so it counts as
    absent. Not cached: a test that runs earlier may download the source.

    Returns:
        True when :func:`lazy.models._limix_source.locate` finds it and
        every package in ``_limix_source._DEPENDENCIES`` is installed.
    """
    if any(
        util.find_spec(name) is None for name in _limix_source._DEPENDENCIES
    ):
        return False
    try:
        _limix_source.locate(download=False)
    except ImportError:
        return False
    return True
