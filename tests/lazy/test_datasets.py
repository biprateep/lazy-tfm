"""Where cached catalogues land, and that it is never baked into the package.

The same code has to run on a laptop, a shared login node and a compute node
with a node-local disk, so every part of the path is overridable and none of it
is hardcoded.
"""

import numpy as np
import pytest

from lazy.datasets import Catalog, data_home, fetch_dc1


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real home directory or cache while testing."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LAZY_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    (tmp_path / "home").mkdir()
    return tmp_path


def test_default_is_under_the_users_cache(isolated_home):
    assert data_home() == isolated_home / "home" / ".cache" / "lazy-photoz"


def test_xdg_cache_home_is_honoured(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(isolated_home / "xdg"))
    assert data_home() == isolated_home / "xdg" / "lazy-photoz"


def test_an_empty_xdg_cache_home_falls_back(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert data_home() == isolated_home / "home" / ".cache" / "lazy-photoz"


def test_lazy_data_home_wins_over_xdg(isolated_home, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(isolated_home / "xdg"))
    monkeypatch.setenv("LAZY_DATA_HOME", str(isolated_home / "scratch"))
    assert data_home() == isolated_home / "scratch"


def test_the_argument_wins_over_everything(isolated_home, monkeypatch):
    monkeypatch.setenv("LAZY_DATA_HOME", str(isolated_home / "scratch"))
    assert data_home(isolated_home / "explicit") == isolated_home / "explicit"


def test_the_directory_is_created(isolated_home):
    assert data_home(isolated_home / "made" / "deeply").is_dir()


def test_a_missing_cache_is_reported_rather_than_downloaded(isolated_home):
    """`download_if_missing=False` is what a network-less compute node passes."""
    with pytest.raises(FileNotFoundError, match="network access"):
        fetch_dc1("train", data_home=isolated_home, download_if_missing=False)


def test_unknown_split_names_the_alternatives(isolated_home):
    with pytest.raises(ValueError, match="test"):
        fetch_dc1("validation", data_home=isolated_home, download_if_missing=False)


def test_catalog_repr_summarises_without_dumping_the_frame():
    import pandas as pd

    catalog = Catalog(
        split="train",
        raw=pd.DataFrame({"U": np.zeros(3)}),
        redshift=np.zeros(3),
        object_id=np.arange(3),
    )
    assert repr(catalog) == "Catalog(split='train', n=3, columns=['U'])"
