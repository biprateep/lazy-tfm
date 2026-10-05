# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import pathlib
import tomllib

import lazy
from lazy.models import tabfm


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


def test_lazy_setup_installs_the_tabfm_commit_uv_pins():
    """``lazy setup`` and a uv checkout get the same TabFM build."""
    pyproject = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text())
    source = config["tool"]["uv"]["sources"]["tabfm"]
    assert tabfm.REPOSITORY_BUILD.endswith(
        f"git+{source['git']}@{source['rev']}"
    )
    assert config["project"]["scripts"]["lazy"] == "lazy._cli:main"
