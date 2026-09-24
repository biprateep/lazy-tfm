# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import lazy


def test_version():
    """Check to see that we can get the package version."""
    assert lazy.__version__ is not None


def test_public_api_is_importable():
    """Every name in __all__ actually resolves."""
    for name in lazy.__all__:
        assert hasattr(lazy, name), name
