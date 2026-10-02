# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import pathlib
import tomllib

import lazy


def test_version():
    """Check to see that we can get the package version."""
    assert lazy.__version__ is not None


def test_public_api_is_importable():
    """Every name in __all__ actually resolves."""
    for name in lazy.__all__:
        assert hasattr(lazy, name), name


def test_the_all_extra_installs_every_other_extra():
    """``lazy-tfm[all]`` is everything: each backend and each extra."""
    pyproject = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text())["project"][
        "optional-dependencies"
    ]
    (requirement,) = extras["all"]
    named = requirement.partition("[")[2].rstrip("]").split(",")
    assert sorted(named) == sorted(set(extras) - {"all"})
