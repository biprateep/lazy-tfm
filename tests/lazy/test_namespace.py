# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The warning for another distribution installing ``lazy``."""

from importlib import metadata
import types

import pytest

from lazy import _namespace


def _installed(name):
    assert name == _namespace.OTHER_DISTRIBUTION
    return types.SimpleNamespace(version="2.0")


def _absent(name):
    raise metadata.PackageNotFoundError(name)


def test_the_other_lazy_is_reported(monkeypatch):
    monkeypatch.setattr(_namespace.metadata, "distribution", _installed)
    with pytest.warns(_namespace.ImportNameWarning, match="does not help"):
        assert _namespace.warn_if_shared()


def test_nothing_is_said_without_it(monkeypatch, recwarn):
    monkeypatch.setattr(_namespace.metadata, "distribution", _absent)
    assert not _namespace.warn_if_shared()
    assert not recwarn.list


def test_this_environment_has_no_conflict():
    """The development environment must not carry the other package."""
    with pytest.raises(metadata.PackageNotFoundError):
        metadata.distribution(_namespace.OTHER_DISTRIBUTION)
