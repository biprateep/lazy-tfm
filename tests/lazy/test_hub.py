# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""A filtered snapshot counts as cached only once it is known to be whole.

The Hugging Face Hub is replaced by a stand-in over a cache directory in
``tmp_path``, so nothing here reaches the network.
"""

import dataclasses

import huggingface_hub
import pytest

import lazy
from lazy.models import _hub

#: The stand-in repository: path -> contents. ``regression/`` is outside the
#: TabFM checkpoint's patterns.
REPO = {
    "classification/model.safetensors": b"w" * 40,
    "classification/config.json": b"{}",
    "config.json": b'{"a": 1}',
    "LICENSE": b"license",
    "README.md": b"readme",
    "regression/model.safetensors": b"r" * 30,
}

#: The bytes the TabFM patterns select from REPO.
SELECTED_BYTES = sum(
    len(data) for name, data in REPO.items() if not name.startswith("regr")
)


class _Hub:
    """The Hub's download calls and file listing, over ``root``.

    ``online`` False makes the network calls fail as they do offline: the
    listing raises, and an online snapshot download falls back to what is
    cached, as ``huggingface_hub`` does.
    """

    def __init__(self, root, revision):
        self.snapshot = root / "models--google--tabfm" / "snapshots" / revision
        self.online = True
        self.calls = []
        self.listings = 0

    def write(self, *names):
        for name in names:
            path = self.snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(REPO[name])

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs["local_files_only"])
        if not kwargs["local_files_only"] and self.online:
            self.write(
                *huggingface_hub.utils.filter_repo_objects(
                    REPO, allow_patterns=kwargs["allow_patterns"]
                )
            )
        if not self.snapshot.is_dir():
            raise huggingface_hub.errors.LocalEntryNotFoundError("absent")
        return str(self.snapshot)

    def list_repo_files(self, repo_id, revision=None):
        del repo_id, revision  # Unused: one stand-in repository.
        self.listings += 1
        if not self.online:
            raise huggingface_hub.errors.OfflineModeIsEnabled("offline")
        return list(REPO)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    spec = _hub.get_checkpoint("tabfm")
    fake = _Hub(tmp_path, spec.revision)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", fake.snapshot_download
    )
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "list_repo_files",
        lambda api, *args, **kwargs: fake.list_repo_files(*args, **kwargs),
    )
    return fake


def test_an_interrupted_snapshot_is_completed(hub):
    """What a download cut short leaves is neither cached nor kept."""
    hub.write("config.json")
    assert not lazy.is_cached("tabfm")
    path = _hub.get_checkpoint("tabfm").download()
    assert hub.calls == [True, True, False]
    assert (path / "classification" / "model.safetensors").is_file()
    assert lazy.is_cached("tabfm")
    hub.calls.clear()
    assert _hub.get_checkpoint("tabfm").download() == path
    assert hub.calls == [True]  # confirmed once, then cached


def test_a_whole_snapshot_cached_before_the_marker_stays_offline(
    hub, monkeypatch
):
    """Its files add up to the checkpoint's size: no network needed."""
    hub.write(*(name for name in REPO if not name.startswith("regression")))
    spec = dataclasses.replace(
        _hub.get_checkpoint("tabfm"), size_bytes=SELECTED_BYTES
    )
    monkeypatch.setitem(_hub.CHECKPOINTS, spec.key, spec)
    assert lazy.is_cached("tabfm")
    assert spec.download() == hub.snapshot
    assert hub.calls == [True, True]
    assert hub.listings == 0


def test_offline_an_unconfirmed_snapshot_is_used_with_a_warning(hub):
    hub.write("config.json")
    hub.online = False
    spec = _hub.get_checkpoint("tabfm")
    with pytest.warns(RuntimeWarning, match="interrupted download"):
        assert spec.download() == hub.snapshot
    with pytest.warns(RuntimeWarning, match="interrupted download"):
        assert spec.download(local_files_only=True) == hub.snapshot
    assert not lazy.is_cached("tabfm")


def test_an_unreachable_hub_leaves_the_cached_snapshot_in_use(hub, monkeypatch):
    """Some failures reach the caller instead of falling back to the cache."""
    hub.write("config.json")

    def refused(**kwargs):
        if kwargs["local_files_only"]:
            return str(hub.snapshot)
        raise ConnectionError("refused")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refused)
    with pytest.warns(RuntimeWarning, match="interrupted download"):
        assert _hub.get_checkpoint("tabfm").download() == hub.snapshot


def test_an_uncached_snapshot_still_fails_without_the_hub(hub, monkeypatch):
    def refused(**kwargs):
        if kwargs["local_files_only"]:
            raise huggingface_hub.errors.LocalEntryNotFoundError("absent")
        raise ConnectionError("refused")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refused)
    with pytest.raises(ConnectionError):
        _hub.get_checkpoint("tabfm").download()


def test_a_snapshot_short_of_files_is_not_marked(hub, monkeypatch):
    """The Hub answered, but the download left a file out."""
    hub.write("config.json")
    monkeypatch.setattr(hub, "write", lambda *names: None)
    with pytest.warns(RuntimeWarning, match="interrupted download"):
        _hub.get_checkpoint("tabfm").download()
    assert not lazy.is_cached("tabfm")


def test_a_single_file_is_trusted_as_it_is(monkeypatch, tmp_path):
    """hf_hub_download only puts a whole file in place."""
    calls = []
    weights = tmp_path / "weights.ckpt"
    weights.write_bytes(b"w")

    def fetch(**kwargs):
        calls.append(kwargs["local_files_only"])
        return str(weights)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    assert _hub.get_checkpoint("tabicl").download() == weights
    assert calls == [True]
