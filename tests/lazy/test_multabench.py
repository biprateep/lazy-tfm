# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""MulTaBench's regression datasets, loaded through lazy.datasets.

A fake Hub serves a small table and a shard of images in place of each
repository, so the loader runs offline; it needs pyarrow to write the
table. The split is checked against digests of the indices that
``multabench.benchmark.splits.get_split`` 0.2.1 gives.
"""

import hashlib
import os
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from lazy import _multabench
from lazy import datasets

N_ROWS = 12

SPLIT_DIGESTS = {
    (300, "10k"): (
        "0b89b6cc1a2e0fc0",
        "ec7201f8f10267bd",
        "02486b6c38b8fb2f",
        "6f8a1d4a07cfe297",
        "d50b336b85097cfc",
        "3ce2d984c25d2f7a",
    ),
    (31_000, "10k"): (
        "27e3884c855d98e5",
        "30d23bb338e08e06",
        "3334e62024d938ae",
        "745587aeb080cf2b",
        "d022b115c924291f",
        "769564a64e8705fa",
    ),
    (31_000, "full"): (
        "8301fa4ab5a95fce",
        "1cb89f43f3ae3ef7",
        "303d0601bf2564aa",
        "abae6f070ad0e2ff",
        "2826af19d32bed36",
        "7282ea9d16652a2b",
    ),
}
"""The first 16 hex digits of the SHA-256 of each split's training rows,
-1, then its test rows, as int64, from multabench 0.2.1, by (rows, size)."""


def _frame(table):
    """A small table with the dataset's target and image column."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "notes": pd.array(
                [f"item {i}" if i % 5 else None for i in range(N_ROWS)],
                dtype="string",
            ),
            "kind": pd.Categorical(["a", "b"] * (N_ROWS // 2)),
            "count": rng.integers(0, 9, N_ROWS),
            table.target: rng.normal(size=N_ROWS),
        }
    )
    if table.image_column is not None:
        frame[table.image_column] = pd.array(
            [f"images/{i}.png" if i != 3 else None for i in range(N_ROWS)],
            dtype="string",
        )
    return frame


@pytest.fixture(name="hub")
def _fake_hub(monkeypatch, tmp_path):
    """A Hub of one small table and two shards of images per repository.

    Returns:
        The calls to ``hf_hub_download``, as keyword dicts.
    """
    pytest.importorskip("pyarrow")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    calls = []

    def table_of(repo_id):
        name = repo_id.removeprefix("multabench/")
        (table,) = [t for t in _multabench.DATASETS.values() if t.repo == name]
        return table

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        path = pathlib.Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if kwargs["filename"] == "data.parquet":
            _frame(table_of(kwargs["repo_id"])).to_parquet(path)
            return str(path)
        half = int(kwargs["filename"][7:10])
        with zipfile.ZipFile(path, "w") as shard:
            for i in range(half, N_ROWS, 2):
                image = tmp_path / f"{i}.png"
                Image.new("RGB", (4, 4), (i, 0, 0)).save(image)
                shard.write(image, f"images/{i}.png")
        return str(path)

    def list_repo_files(api, repo_id, repo_type, revision):
        assert (repo_type, revision) == ("dataset", table_of(repo_id).revision)
        return ["README.md", "data.parquet", "images-000.zip", "images-001.zip"]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(
        huggingface_hub.HfApi, "list_repo_files", list_repo_files
    )
    monkeypatch.setenv("LAZY_DATA_HOME", str(tmp_path / "home"))
    return calls


def test_a_dataset_loads_at_its_pinned_commit_with_readable_images(hub):
    table = _multabench.DATASETS["khaadi_clothes"]
    dataset = datasets.load_dataset("multabench_khaadi_clothes")
    assert {call["revision"] for call in hub} == {table.revision}
    assert {call["repo_type"] for call in hub} == {"dataset"}
    assert [call["filename"] for call in hub] == [
        "data.parquet",
        "images-000.zip",
        "images-001.zip",
    ]
    shards = pathlib.Path(hub[1]["local_dir"]) / "images-000.zip"
    assert not shards.exists()  # Each shard is deleted once extracted.
    assert dataset.y.dtype == np.float64 and len(dataset) == N_ROWS
    assert table.target not in dataset.X
    paths = dataset.X[table.image_column]
    assert paths.isna().tolist() == [i == 3 for i in range(N_ROWS)]
    with Image.open(paths.iloc[7]) as image:
        assert image.getpixel((0, 0)) == (7, 0, 0)
    assert dataset.modalities == {"notes": "text", table.image_column: "image"}
    assert pathlib.Path(paths.iloc[0]).is_absolute()
    assert dataset.X["notes"].isna().sum() == 3


def test_a_second_load_reads_the_cache_and_downloads_nothing(hub):
    datasets.load_dataset("multabench_khaadi_clothes")
    calls = len(hub)
    datasets.load_dataset("multabench_khaadi_clothes")
    assert len(hub) == calls == 3


def test_without_images_the_shards_are_not_downloaded(hub):
    dataset = datasets.load_dataset("multabench_khaadi_clothes", images=False)
    assert [call["filename"] for call in hub] == ["data.parquet"]
    assert dataset.X["img"].iloc[0] == "images/0.png"
    assert dataset.modalities == {"notes": "text"}


def test_a_split_has_multabench_rows_and_keeps_their_labels(hub):
    train, test = datasets.load_dataset(
        "multabench_book_price", split=True, fold=2
    )
    rows = _multabench.split(N_ROWS, 2, "10k")
    assert train.X.index.tolist() == rows[0].tolist()
    assert test.X.index.tolist() == rows[1].tolist()
    assert train.modalities == test.modalities == {"notes": "text"}
    assert train.modalities is not test.modalities


@pytest.mark.parametrize(("n_rows", "size"), sorted(SPLIT_DIGESTS))
def test_the_split_is_multabenchs(n_rows, size):
    for fold, expected in enumerate(SPLIT_DIGESTS[n_rows, size]):
        train, test = _multabench.split(n_rows, fold, size)
        rows = np.concatenate([train, [-1], test]).astype(np.int64)
        assert hashlib.sha256(rows.tobytes()).hexdigest()[:16] == expected


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"fold": 1}, "pass split=True"),
        ({"split": True, "fold": 6}, "fold must be an integer from 0 to 5"),
        ({"split": True, "fold": True}, "fold must be an integer"),
        ({"split": True, "size": "1k"}, "size must be one of"),
    ],
)
def test_a_bad_option_is_refused_before_any_download(
    monkeypatch, options, message
):
    monkeypatch.setattr(_multabench, "load", _no_download)
    with pytest.raises(ValueError, match=message):
        datasets.load_dataset("multabench_book_price", **options)
    with pytest.raises(TypeError, match="images must be True or False"):
        datasets.load_dataset("multabench_book_price", images="no")


def _no_download(*args, **kwargs):
    """Stands in for the loader where nothing may download."""
    raise AssertionError("the dataset was downloaded")


def test_options_of_one_kind_of_dataset_are_refused_by_another():
    with pytest.raises(TypeError, match="unexpected keywords .'noise'"):
        datasets.load_dataset("multabench_book_price", noise=0.1)
    with pytest.raises(TypeError, match="unexpected keywords .'fold'"):
        datasets.load_dataset("chirp", fold=1)


def test_without_pyarrow_the_extra_is_named(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match=r"lazy-tfm\[multabench\]"):
        datasets.load_dataset("multabench_book_price", data_home=tmp_path)


def test_every_regression_dataset_is_listed_with_its_source():
    table = datasets.list_datasets()
    names = [n for n in table.index if n.startswith("multabench_")]
    assert len(names) == 40
    images = [n for n in names if "image" in table.loc[n, "description"]]
    assert len(images) == 20
    for name in names:
        pinned = _multabench.DATASETS[name.removeprefix("multabench_")]
        assert len(pinned.revision) == 40
        assert table.loc[name, "url"].endswith(f"multabench/{pinned.repo}")
        assert pinned.source in table.loc[name, "license"]


@pytest.mark.skipif(
    os.environ.get("LAZY_RUN_NETWORK_TESTS") != "1",
    reason="set LAZY_RUN_NETWORK_TESTS=1 to run tests that download data",
)
def test_the_real_data_loads_as_multabench_states():
    dataset = datasets.load_dataset("multabench_chocolate_bar_ratings")
    row = datasets.list_datasets().loc["multabench_chocolate_bar_ratings"]
    assert dataset.X.shape == (row["n_rows"], row["n_features"])
    assert set(dataset.modalities.values()) == {"text"}


def test_shards_without_an_images_folder_are_reported(hub, monkeypatch):
    import huggingface_hub  # noqa: PLC0415 - the fixture skips without it.

    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "list_repo_files",
        lambda api, repo_id, repo_type, revision: ["data.parquet"],
    )
    with pytest.raises(OSError, match="no images/ folder in its 0 image"):
        datasets.load_dataset("multabench_khaadi_clothes")


def test_a_relative_cache_still_gives_absolute_image_paths(
    hub, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    dataset = datasets.load_dataset(
        "multabench_khaadi_clothes", data_home="rel"
    )
    assert pathlib.Path(dataset.X["img"].iloc[0]).is_absolute()
