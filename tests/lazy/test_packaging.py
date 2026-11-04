# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import pathlib
import subprocess
import sys
import tomllib

from packaging import requirements
import pytest

import lazy
from lazy.models import kumo
from lazy.models import tabfm

PYPROJECT = tomllib.loads(
    (pathlib.Path(__file__).parents[2] / "pyproject.toml").read_text()
)


def test_importing_lazy_loads_the_dataset_modules_on_first_access():
    """``lazy.datasets`` and ``lazy.selection`` load when first accessed."""
    code = (
        "import sys, lazy\n"
        "heavy = {'h5py', 'lazy.datasets', 'lazy.selection', "
        "'sklearn.datasets'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
        "print(lazy.datasets.data_home.__name__, lazy.selection.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split("\n")[:2] == ["[]", "data_home lazy.selection"]


def test_an_unknown_attribute_of_lazy_raises_attribute_error():
    with pytest.raises(AttributeError, match="no_such_name"):
        lazy.no_such_name  # noqa: B018 - the access is the test.


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


def test_the_published_requirements_hold_no_direct_url():
    """PyPI refuses an upload whose metadata has a direct URL in it.

    The wheel's Requires-Dist lines are exactly these requirements; the
    dependency groups and [tool.uv.sources] are never published.
    """
    project = PYPROJECT["project"]
    published = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        published.extend(extra)
    for line in published:
        assert requirements.Requirement(line).url is None, line


def test_the_kumo_extra_installs_the_release_lazy_setup_installs():
    """The first release with Kumo Tabular, which ``lazy setup kumo`` adds."""
    extras = PYPROJECT["project"]["optional-dependencies"]
    assert extras["kumo"] == [kumo._REQUIREMENT]
