# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import pathlib
import tomllib

from packaging import requirements

import lazy
from lazy.models import _install
from lazy.models import kumo
from lazy.models import tabfm

PYPROJECT = tomllib.loads(
    (pathlib.Path(__file__).parents[2] / "pyproject.toml").read_text()
)


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


def test_a_checkout_syncs_the_structured_data_models_lazy_setup_installs():
    """``uv sync --group backends`` and ``lazy setup kumo`` agree."""
    source = PYPROJECT["tool"]["uv"]["sources"]["structured-data-models"]
    assert source["rev"] == kumo.REPOSITORY_COMMIT
    assert kumo.REPOSITORY_BUILD.endswith(
        f"git+{source['git']}@{source['rev']}"
    )
    group = _install.CHECKOUT_SYNC[_install.CHECKOUT_SYNC.index("--group") + 1]
    assert PYPROJECT["dependency-groups"][group] == ["structured-data-models"]


def test_a_plain_sync_installs_no_backend():
    """CI runs ``uv sync``, which must not bring PyTorch with Kumo's package."""
    defaults = PYPROJECT["tool"]["uv"].get("default-groups", ["dev"])
    assert "backends" not in defaults
    for group in defaults:
        for line in PYPROJECT["dependency-groups"][group]:
            assert "structured-data-models" not in line, group
