# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Finding LimiX's source and loading it under private names."""

from importlib import metadata
import pathlib
import sys

import pytest

from lazy.models import _limix_source

_ALIAS = "lazy.models._limix_ext"


def _fake_source(root: pathlib.Path) -> pathlib.Path:
    """A directory shaped like LimiX's source, with stand-in modules."""
    for parts in (("model", "v2_0"), ("inference", "v2_0")):
        package = root.joinpath(*parts)
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
    (root / "model" / "v2_0" / "helper.py").write_text("VALUE = 7\n")
    (root / "model" / "v2_0" / "loading.py").write_text(
        "from .helper import VALUE\n"
    )
    (root / "inference" / "v2_0" / "preprocess.py").write_text("STEP = 1\n")
    return root


@pytest.fixture
def clean_env(monkeypatch):
    for name in _limix_source.ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Loading registers aliases process-wide; isolate each test from them.
    for name in [m for m in sys.modules if m.startswith(_ALIAS)]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(_limix_source, "_loaded", {})
    return monkeypatch


def test_an_environment_variable_names_the_source(clean_env, tmp_path):
    root = _fake_source(tmp_path / "LimiX")
    clean_env.setenv("LAZY_LIMIX_SRC", str(root))
    source = _limix_source.locate()
    assert source.root == root.resolve()
    assert source.origin == "$LAZY_LIMIX_SRC"
    assert source.commit is None  # not a git checkout


def test_lazy_limix_src_wins_over_limix_src(clean_env, tmp_path):
    first = _fake_source(tmp_path / "first")
    second = _fake_source(tmp_path / "second")
    clean_env.setenv("LIMIX_SRC", str(second))
    clean_env.setenv("LAZY_LIMIX_SRC", str(first))
    assert _limix_source.locate().root == first.resolve()


def test_a_directory_without_the_code_is_refused(clean_env, tmp_path):
    clean_env.setenv("LIMIX_SRC", str(tmp_path))
    with pytest.raises(ImportError, match="does not hold LimiX"):
        _limix_source.locate()


def test_no_source_says_how_to_install_it(clean_env):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    clean_env.setattr(_limix_source.metadata, "distribution", missing)
    with pytest.raises(ImportError, match=r"lazy-photoz\[limix\]"):
        _limix_source.locate()


def test_load_uses_private_names_and_warns_on_another_commit(
    clean_env, tmp_path
):
    clean_env.setenv("LAZY_LIMIX_SRC", str(_fake_source(tmp_path / "L")))
    with pytest.warns(_limix_source.LimiXSourceWarning, match="unknown"):
        limix = _limix_source.load()
    assert limix.loading.VALUE == 7  # relative imports resolve inside it
    assert limix.loading.__name__ == f"{_ALIAS}.model.loading"
    assert limix.preprocess.STEP == 1
    assert "model" not in sys.modules or not hasattr(
        sys.modules["model"], "VALUE"
    )
    assert _limix_source.load() is limix  # loaded once
