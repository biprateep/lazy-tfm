# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Finding LimiX's source and loading it under private names."""

from importlib import metadata
import io
import pathlib
import sys
import tarfile
import urllib.error
import warnings

import pytest

from lazy.models import _hub
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


def _archive(tmp_path: pathlib.Path) -> bytes:
    """GitHub's archive of a LimiX commit: one LimiX-<commit> directory."""
    root = _fake_source(tmp_path / "archived" / "LimiX-abc")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(root, arcname=root.name)
    return buffer.getvalue()


def _not_installed(name):
    raise metadata.PackageNotFoundError(name)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in (*_limix_source.ENV_VARS, "HF_HUB_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAZY_DATA_HOME", str(tmp_path / "cache"))
    # Loading registers aliases process-wide; isolate each test from them.
    for name in [m for m in sys.modules if m.startswith(_ALIAS)]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(_limix_source, "_loaded", {})
    monkeypatch.setattr(_limix_source, "_ignored", set())
    return monkeypatch


def test_an_environment_variable_names_the_source(clean_env, tmp_path):
    root = _fake_source(tmp_path / "LimiX")
    clean_env.setenv("LAZY_LIMIX_SRC", str(root))
    source = _limix_source.locate()
    assert source.root == root.resolve()
    assert source.origin == "$LAZY_LIMIX_SRC"
    assert source.commit is None  # not a git checkout


def test_the_package_label_says_where_the_code_came_from(tmp_path):
    checkout = _limix_source.Source(tmp_path, "abc123", "$LAZY_LIMIX_SRC")
    unknown = _limix_source.Source(tmp_path, None, "$LIMIX_SRC")
    installed = _limix_source.Source(tmp_path, "abc123", "installed LimiX 2.0")
    cached = _limix_source.Source(
        tmp_path, "abc123", _limix_source.CACHE_ORIGIN
    )
    assert checkout.package == "LimiX (source checkout at abc123)"
    assert unknown.package == "LimiX (source checkout at unknown commit)"
    assert installed.package == "LimiX 2.0"
    assert cached.package == "LimiX (source archive at abc123)"


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


def test_with_no_source_the_pinned_archive_is_downloaded(clean_env, tmp_path):
    clean_env.setattr(_limix_source.metadata, "distribution", _not_installed)
    urls = []

    def urlopen(url, timeout):
        del timeout  # Unused.
        urls.append(url)
        return io.BytesIO(_archive(tmp_path))

    clean_env.setattr(_limix_source.urllib.request, "urlopen", urlopen)
    source = _limix_source.locate()
    assert urls == [_limix_source.ARCHIVE]
    assert source.root == tmp_path / "cache" / "limix" / (
        _limix_source.LIMIX_COMMIT
    )
    assert source.commit == _limix_source.LIMIX_COMMIT
    assert source.origin == _limix_source.CACHE_ORIGIN
    assert _limix_source.locate().root == source.root
    assert len(urls) == 1  # downloaded once
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the pinned commit: no warning
        assert _limix_source.load().loading.VALUE == 7


@pytest.mark.parametrize("offline", ["argument", "HF_HUB_OFFLINE"])
def test_an_offline_lookup_does_not_download(clean_env, offline):
    clean_env.setattr(_limix_source.metadata, "distribution", _not_installed)
    clean_env.setattr(_limix_source, "fetch", pytest.fail)
    if offline == "HF_HUB_OFFLINE":
        clean_env.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ImportError, match="downloading it is off"):
        _limix_source.locate(download=offline == "HF_HUB_OFFLINE")


def test_a_failed_download_says_how_to_get_the_source(clean_env):
    clean_env.setattr(_limix_source.metadata, "distribution", _not_installed)

    def urlopen(url, timeout):
        raise urllib.error.URLError("no network")

    clean_env.setattr(_limix_source.urllib.request, "urlopen", urlopen)
    with pytest.raises(ImportError, match="LAZY_LIMIX_SRC") as caught:
        _limix_source.locate()
    assert "no network" in str(caught.value)
    assert not _limix_source.cache_dir().exists()


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


@pytest.mark.parametrize("module", ["kditransform", "einops", "triton"])
def test_a_missing_dependency_says_how_to_install_it(
    clean_env, tmp_path, module
):
    root = _fake_source(tmp_path / "L")
    (root / "inference" / "v2_0" / "preprocess.py").write_text(
        f"import {module}\n"
    )
    clean_env.setenv("LAZY_LIMIX_SRC", str(root))
    clean_env.setitem(sys.modules, module, None)  # as if not installed
    with (
        pytest.warns(_limix_source.LimiXSourceWarning),
        pytest.raises(ImportError, match=r"lazy-tfm\[limix\]") as caught,
    ):
        _limix_source.load()
    message = str(caught.value)
    assert repr(module) in message
    assert ("Linux only" in message) == (module == "triton")


def test_other_missing_modules_are_not_reworded(clean_env, tmp_path):
    root = _fake_source(tmp_path / "L")
    (root / "model" / "v2_0" / "loading.py").write_text(
        "import lazy_no_such_module\n"
    )
    clean_env.setenv("LAZY_LIMIX_SRC", str(root))
    with (
        pytest.warns(_limix_source.LimiXSourceWarning),
        pytest.raises(ModuleNotFoundError, match="lazy_no_such_module"),
    ):
        _limix_source.load()


def test_load_finds_the_source_once(clean_env, tmp_path):
    clean_env.setenv("LAZY_LIMIX_SRC", str(_fake_source(tmp_path / "L")))
    with pytest.warns(_limix_source.LimiXSourceWarning):
        first = _limix_source.load()
    calls = []
    clean_env.setattr(_limix_source, "locate", lambda: calls.append(1))
    assert _limix_source.load() is first
    assert not calls


def test_switching_the_source_mid_process_warns(clean_env, tmp_path):
    clean_env.setenv("LAZY_LIMIX_SRC", str(_fake_source(tmp_path / "one")))
    with pytest.warns(_limix_source.LimiXSourceWarning, match="unknown"):
        first = _limix_source.load()
    other = _fake_source(tmp_path / "two")
    clean_env.setenv("LAZY_LIMIX_SRC", str(other))
    with pytest.warns(_limix_source.LimiXSourceWarning, match="another"):
        assert _limix_source.load() is first
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _limix_source.load() is first  # warned once


def test_the_checkpoint_helpers_cover_the_source(clean_env, tmp_path):
    clean_env.setattr(_limix_source.metadata, "distribution", _not_installed)
    clean_env.setattr(
        _hub.Checkpoint, "download", lambda self, **kwargs: tmp_path
    )
    assert not _hub.is_cached("limix")  # the weights alone are not enough
    clean_env.setattr(
        _limix_source.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(_archive(tmp_path)),
    )
    assert _hub.download_checkpoint("limix") == tmp_path
    assert _hub.is_cached("limix")
