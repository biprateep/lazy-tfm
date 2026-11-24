# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""MulTaBench's regression datasets: their pins, their files and their splits.

MulTaBench (Arazi et al. 2026, arXiv:2605.10616) is a benchmark of tables
with columns of text or images. Each dataset is a repository on the Hugging
Face Hub, ``multabench/<repo>``, that holds a typed ``data.parquet`` and,
for images, ``images-*.zip`` shards. :data:`DATASETS` pins the 40
regression datasets to the commits that the ``multabench`` package 0.2.1
pins, with the target, the image column and the size its
``metadata.json`` states. :func:`lazy.datasets.load_dataset` loads them;
this module holds what is MulTaBench's own.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import pathlib
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn import model_selection

from lazy import _typing
from lazy.models import _ensemble

__all__ = ["DATASETS", "SIZES", "SPLITS", "Table", "check", "load", "split"]

_FOLDS, _REPEATS, _SEED = 3, 2, 42
_MAX_TRAIN, _MAX_TEST = 10_000, 5_000

SPLITS = _FOLDS * _REPEATS
"""MulTaBench's splits: split ``k`` is fold ``k % 3`` of repeat ``k // 3``."""

SIZES = ("10k", "full")
"""The sizes of a split: at most 10,000 training and 5,000 test rows, or all."""


@dataclasses.dataclass(frozen=True)
class Table:
    """One of MulTaBench's regression datasets.

    Attributes:
        repo: The repository under ``multabench/`` on the Hugging Face Hub.
        revision: Its pinned commit.
        target: The target column.
        image_column: The column of image paths; ``None`` for text only.
        n_rows: The number of rows.
        n_features: The number of feature columns, the target excluded.
        source: The page of the data's source, whose licence applies.
    """

    repo: str
    revision: str
    target: str
    image_column: str | None
    n_rows: int
    n_features: int
    source: str

    @property
    def tier(self) -> str:
        """``"core"`` or ``"extended"``, MulTaBench's two sets."""
        return self.repo.split("-")[0]

    @property
    def modality(self) -> str:
        """``"image"`` if the dataset has an image column, else ``"text"``."""
        return "text" if self.image_column is None else "image"


# Read from each repository's metadata.json at its pinned commit; the
# commits are those of multabench 0.2.1's datasets/revisions.json, and the
# sources those of its MULTABENCH_SOURCES.
_TABLES = (
    Table(
        "extended-img-reg-airbnb-nyc",
        "a9c96f0e32ba1d086a79a3b550aad82128e3dfe6",
        "price",
        "listing photo",
        16_780,
        50,
        "https://www.kaggle.com/datasets/dominoweir/inside-airbnb-nyc",
    ),
    Table(
        "extended-text-reg-airbnb-seattle",
        "a8e1cae6ef69b5f220383502b4535942bb22948a",
        "price",
        None,
        3_818,
        65,
        "https://www.kaggle.com/datasets/airbnb/seattle",
    ),
    Table(
        "core-img-reg-amazon-bestseller",
        "25ca481b6b8b0b4f833b548b75b7f49bd77f542a",
        "price",
        "photo_url",
        3_488,
        5,
        "https://www.kaggle.com/datasets/amankumar20d/amazon-best-seller-all-departments-us",
    ),
    Table(
        "core-img-reg-amazon-packages",
        "e7b8971de32a83c3166bfef4d0bac13cc3489cf7",
        "Total Weight in Pounds",
        "Amazon bin",
        46_398,
        3,
        "https://www.kaggle.com/datasets/dhruvildave/amazon-bin-image-dataset",
    ),
    Table(
        "extended-text-reg-anime-planet",
        "41fe7288b71016e3fea85900a230f98b76eb0fc1",
        "Rating Score",
        None,
        14_391,
        14,
        "https://www.kaggle.com/datasets/hernan4444/animeplanet-recommendation-database-2020",
    ),
    Table(
        "core-text-reg-babies-prices",
        "6e7a295a70fd857c2fd205bb93e8c410b0adb2d8",
        "price",
        None,
        5_085,
        12,
        "http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/csv_files/babies_r_us.csv",
    ),
    Table(
        "core-text-reg-book-price",
        "e10d77f2241c8341f6d305e3bcdd9e663cab7c96",
        "Price in USD",
        None,
        4_989,
        8,
        "https://www.openml.org/search?type=data&id=46663",
    ),
    Table(
        "core-text-reg-book-readability",
        "cf2c716a4bea29d02d0d53bc341abee9f64c2276",
        "New Dale Chall Readability Formula",
        None,
        4_724,
        30,
        "https://www.kaggle.com/datasets/verracodeguacas/clear-corpus",
    ),
    Table(
        "extended-text-reg-chocolate-bar-ratings",
        "4756ba1340d0604cb7efef3a5d7cd3a58a8ec2e0",
        "Rating",
        None,
        1_795,
        8,
        "https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings",
    ),
    Table(
        "extended-img-reg-dvm-car",
        "84ceb91ad802748dcb3cb10049ef19c988d2d706",
        "Price",
        "Car Picture",
        100_000,
        16,
        "https://www.kaggle.com/datasets/osamasaifs/dvm-car-with-images",
    ),
    Table(
        "extended-text-reg-fifa22-wages",
        "c238c2ee11db4f3e16d1f244e27e109473713a46",
        "Wage in Euros",
        None,
        19_178,
        28,
        "https://www.openml.org/search?type=data&id=45012",
    ),
    Table(
        "extended-img-reg-flipkart-ratio",
        "b60d49c7704fc07c1ab7ad8f3b04998723c72333",
        "price_ratio",
        "id",
        60_900,
        4,
        "https://www.kaggle.com/datasets/kuchhbhi/stylish-product-image-dataset",
    ),
    Table(
        "extended-img-reg-goias-houses",
        "e88bc6908e79497e0e778abdf6a59fdc9a1dd0a4",
        "price",
        "department pic",
        9_830,
        75,
        "https://www.kaggle.com/datasets/carloseduardogo/apartment-prices-in-goinia-gois-brazil",
    ),
    Table(
        "core-img-reg-hnm-fashion",
        "971dd43f53cb97a3eecf23759559e95ecc94a134",
        "AvgConsumerAge",
        "ProductPic",
        104_072,
        14,
        "https://www.kaggle.com/datasets/odins0n/handm-dataset-128x128",
    ),
    Table(
        "extended-img-reg-kamernet-size",
        "bf64105505d11a35639535ac1ee8ec4a75b52a0c",
        "size",
        "room pic",
        1_638,
        25,
        "https://www.kaggle.com/datasets/amancapy/all-kamernet-rent-listings-w-photos-netherlands",
    ),
    Table(
        "core-img-reg-khaadi-clothes",
        "5b7d8b1a05672a6dc33ea674c552a3814f9af490",
        "Price",
        "img",
        400,
        4,
        "https://www.kaggle.com/datasets/usman8/khaadis-clothes-data-with-images",
    ),
    Table(
        "extended-img-reg-lahaina-auction",
        "8c1dc04993261c06ea0fc44b86cbfa6453208c60",
        "winning_bid",
        "file_name",
        968,
        9,
        "https://www.kaggle.com/datasets/quillen/artists-for-lahaina-2023",
    ),
    Table(
        "core-img-reg-letterboxd-movies",
        "19ddc49ea2df0176bee4fb05de4ef7a822925a3f",
        "rating",
        "poster",
        12_564,
        27,
        "https://www.kaggle.com/datasets/gsimonx37/letterboxd",
    ),
    Table(
        "core-img-reg-mango-mass",
        "6102371bbd0bc534bb0470da8960dd60c5aa1e86",
        "Mass(kg)",
        "Fruit No",
        546,
        3,
        "https://www.kaggle.com/datasets/mypapit/mangomassnet552-dataset",
    ),
    Table(
        "core-text-reg-mercari-marketplace",
        "aa6fabd3d877680dd50f45318ef649e592da5f02",
        "log_price",
        None,
        100_000,
        9,
        "https://www.openml.org/search?type=data&id=46660",
    ),
    Table(
        "core-img-reg-mkphoto-bots",
        "e47d4b7f5bc3c651d940d34a444ed42e225b6940",
        "TrustNZ_IDNNZ",
        "image",
        13_748,
        9,
        "https://www.kaggle.com/datasets/guardeec/mkphoto2023",
    ),
    Table(
        "core-text-reg-montgomery-salaries",
        "9ec74b0d3a3b1d5c031e1eee4612456996e3cf85",
        "current_annual_salary",
        None,
        9_228,
        11,
        "https://www.openml.org/search?type=data&id=42125",
    ),
    Table(
        "core-img-reg-painting-price",
        "ceba4484fc5c5feeac797d10968873432371a187",
        "price",
        "image_url",
        12_369,
        248,
        "https://www.kaggle.com/datasets/denozavrus/paintings-price-prediction",
    ),
    Table(
        "extended-text-reg-ramen-ratings",
        "b3a15685baae18e116096bf39811d42cb3ea2980",
        "Stars",
        None,
        4_105,
        4,
        "https://www.kaggle.com/datasets/ankanhore545/top-ramen-ratings-2022",
    ),
    Table(
        "extended-img-reg-romania-price",
        "b66c2106f3d0094d2c38f0593a7f5f4245f5af48",
        "price",
        "product pic",
        38_294,
        3,
        "https://www.kaggle.com/datasets/furduisorinoctavian/romanian-products-from-emag-and-flanco",
    ),
    Table(
        "core-text-reg-rotten-tomatoes",
        "910b0982f6698f10cacca7f1244973ee4a16340e",
        "RatingValue",
        None,
        7_158,
        15,
        "http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv",
    ),
    Table(
        "extended-img-reg-sao-paulo-houses",
        "eab2214b46806f38c69b972cba5de19145e674ff",
        "price",
        "property pic",
        23_432,
        30,
        "https://www.kaggle.com/datasets/rogerio/image-based-property-price-prediction-in-so-paulo",
    ),
    Table(
        "core-text-reg-scimagojr-impact",
        "6fae2fd821abd8c5b14655f5b3a7acea83bcb307",
        "H index",
        None,
        31_136,
        22,
        "https://www.scimagojr.com/journalrank.php?out=xls",
    ),
    Table(
        "extended-img-reg-socal-houses",
        "68d7bbf750e3459a010e799213f67df40e56ffe2",
        "price",
        "image_id",
        15_474,
        7,
        "https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal",
    ),
    Table(
        "extended-img-reg-tokopedia-weight",
        "87666f959932b08ea31a403a5632429656e8cb99",
        "weight",
        "product_image",
        1_197,
        16,
        "https://www.kaggle.com/datasets/nsmlehq/tokopedia-products-2025",
    ),
    Table(
        "extended-text-reg-used-car-pakistan",
        "0639f624c9ddfe9670b53f762a9a9bd3fc73dd10",
        "Price",
        None,
        72_655,
        9,
        "https://www.kaggle.com/datasets/mustafaimam/used-car-prices-in-pakistan-2021",
    ),
    Table(
        "extended-text-reg-used-car-saudi",
        "14fb452e3cd8df301ffba26984f376fc32b16b0e",
        "Price",
        None,
        8_035,
        12,
        "https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset",
    ),
    Table(
        "core-text-reg-vancouver-salaries",
        "5d23a36a52d46397edaa10ec1e8ce434175b1ac7",
        "remuneration",
        None,
        44_574,
        5,
        "https://opendata.vancouver.ca/api/records/1.0/download/?dataset=employee-remuneration-and-expenses-earning-over-75000&format=csv",
    ),
    Table(
        "core-text-reg-video-games-sales",
        "e96c33cb9b9e49b4fb66bcf782069fd3e46337cf",
        "Global_Sales",
        None,
        16_598,
        5,
        "https://www.kaggle.com/datasets/gregorut/videogamesales",
    ),
    Table(
        "extended-img-reg-watch-tier",
        "4857416f0a5e8fae5ba0c37dbc697808d553a177",
        "tier_premium",
        "watch",
        1_260,
        3,
        "https://www.kaggle.com/datasets/mathewkouch/a-dataset-of-watches",
    ),
    Table(
        "extended-text-reg-wikiliq-prices",
        "abc5016a2224b68b08ec0832d1f57f743a78cff6",
        "Price",
        None,
        12_569,
        12,
        "https://www.kaggle.com/datasets/limtis/wikiliq-dataset",
    ),
    Table(
        "extended-text-reg-wine-polish-market",
        "c317dc41a5d689f1e27cfaddef95d210d005682e",
        "price (PLN)",
        None,
        2_247,
        18,
        "https://www.kaggle.com/datasets/skamlo/wine-price-on-polish-market",
    ),
    Table(
        "extended-text-reg-wine-vivino-spain",
        "3d4a6362c600c25b1a18a08bc8918d62f70b3303",
        "Rating",
        None,
        8_650,
        7,
        "https://www.kaggle.com/datasets/joshuakalobbowles/vivino-wine-data",
    ),
    Table(
        "extended-img-reg-zepto-price",
        "f8d34ce8b5032bc44c85747247823943fdde5dbd",
        "Price",
        "Image",
        13_345,
        8,
        "https://www.kaggle.com/datasets/simpleaditya/zepto-products-dataset",
    ),
    Table(
        "core-text-reg-zomato-restaurants",
        "4b4ea5160a56768d607a7a4cf2253ff78fe865a1",
        "rate",
        None,
        41_665,
        15,
        "https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants",
    ),
)

DATASETS: dict[str, Table] = {
    # The repository's name without its tier and task: "book_price".
    table.repo.split("-reg-")[1].replace("-", "_"): table
    for table in _TABLES
}
"""MulTaBench's 40 regression datasets, by name."""


def check(fold: object, size: object) -> None:
    """Checks a split's options.

    Raises:
        ValueError: If ``fold`` is not an integer from 0 to 5, or ``size``
            is not one of :data:`SIZES`.
    """
    if not _ensemble.is_integer(fold) or not 0 <= fold < SPLITS:
        raise ValueError(
            f"fold must be an integer from 0 to {SPLITS - 1}: {fold=}"
        )
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}: {size=}")


def load(
    table: Table, root: pathlib.Path, *, images: bool
) -> tuple[pd.DataFrame, _typing.FloatArray, dict[str, str]]:
    """Downloads a dataset once and reads it.

    Args:
        table: The dataset.
        root: The directory to cache it under.
        images: Whether to download and extract the images, so that the
            image column holds the paths of their files.

    Returns:
        The features, the target as float64, and the modality of each
        column of text or images. Without the images, the image column
        holds names within the dataset, and is not among the modalities.

    Raises:
        ImportError: If pyarrow, which reads the parquet table, is missing.
        OSError: If a download fails.
    """
    # Absolute, so the image paths hold wherever the frame is used.
    folder = (root / "multabench" / f"{table.repo}@{table.revision}").resolve()
    frame = _read(table, folder)
    y = frame.pop(table.target).to_numpy(dtype=np.float64)
    image_column = table.image_column if images else None
    if image_column is not None:
        _extract_images(table, folder)
        # The paths are relative to the repository, as "images/<file>"; a
        # missing image stays missing.
        frame[image_column] = f"{folder.as_posix()}/" + frame[
            image_column
        ].astype("string")
    found = modalities(frame, image_column)
    if image_column is None and table.image_column is not None:
        found.pop(table.image_column, None)
    return frame, y, found


def modalities(frame: pd.DataFrame, image_column: str | None) -> dict[str, str]:
    """The columns of text or images, as MulTaBench types them.

    The image column is ``"image"``. A column that is not numeric, bool,
    a date or a category is ``"text"``.

    Args:
        frame: The features.
        image_column: The column of image paths, or ``None``.

    Returns:
        Column name -> ``"text"`` or ``"image"``, in the columns' order.
    """
    types = pd.api.types
    found = {}
    for column in frame.columns:
        values = frame[column]
        if column == image_column:
            found[column] = "image"
        elif not (
            types.is_numeric_dtype(values)
            or types.is_bool_dtype(values)
            or types.is_datetime64_any_dtype(values)
            or isinstance(values.dtype, pd.CategoricalDtype)
        ):
            found[column] = "text"
    return found


def split(
    n_rows: int, fold: int, size: str
) -> tuple[_typing.IntArray, _typing.IntArray]:
    """MulTaBench's split of a regression dataset, as row positions.

    Split ``fold`` is fold ``fold % 3`` of repeat ``fold // 3`` of a 3-fold
    split, repeated twice with seed 42. With ``"10k"``, its training and
    test rows are then cut to 10,000 and 5,000 at random, with seed
    ``42 + fold``. This is TabArena's scheme, as ``multabench.get_split``
    applies it.

    Args:
        n_rows: The number of rows.
        fold: The split, from 0 to 5.
        size: ``"10k"`` or ``"full"``.

    Returns:
        The sorted positions of the training rows and of the test rows.
    """
    folds = model_selection.RepeatedKFold(
        n_splits=_FOLDS, n_repeats=_REPEATS, random_state=_SEED
    )
    train, test = list(folds.split(np.zeros(n_rows)))[fold]
    if size == "10k":
        train = _cap(train, _MAX_TRAIN, _SEED + fold)
        test = _cap(test, _MAX_TEST, _SEED + fold)
    return np.sort(train), np.sort(test)


def _cap(rows: _typing.IntArray, size: int, seed: int) -> _typing.IntArray:
    """At most ``size`` of ``rows``, drawn as MulTaBench draws them."""
    if len(rows) <= size:
        return rows
    kept, _ = model_selection.train_test_split(
        rows, train_size=size, random_state=seed
    )
    return np.asarray(kept)


def _read(table: Table, folder: pathlib.Path) -> pd.DataFrame:
    """The table of a dataset, downloaded once; missing text is ``NaN``.

    Raises:
        ImportError: If pyarrow is missing.
    """
    try:
        import pyarrow as pa  # noqa: F401, PLC0415 - only to name the extra.
    except ImportError as error:
        if (error.name or "").partition(".")[0] != "pyarrow":
            raise  # pyarrow is there, but something it imports is not.
        raise ImportError(
            "MulTaBench's tables are parquet files, which pyarrow reads:"
            " pip install 'lazy-tfm[multabench]'"
        ) from error
    path = folder / "data.parquet"
    if not path.is_file():
        _fetch(table, folder, "data.parquet")
    frame = pd.read_parquet(path)
    for column in frame.columns:
        if frame[column].dtype == object:
            # Parquet reads a missing string as None; MulTaBench reads NaN.
            frame[column] = frame[column].where(frame[column].notna(), np.nan)
    return frame


def _extract_images(table: Table, folder: pathlib.Path) -> None:
    """Downloads a dataset's image shards and extracts them, once.

    Each shard is downloaded into a temporary directory beside the images,
    extracted and deleted before the next, so that the extra disk it
    takes is one shard's.
    """
    images = folder / "images"
    if images.is_dir():
        return
    import huggingface_hub  # noqa: PLC0415 - kept out of `import lazy`.

    files = huggingface_hub.HfApi().list_repo_files(
        f"multabench/{table.repo}",
        repo_type="dataset",
        revision=table.revision,
    )
    shards = sorted(f for f in files if fnmatch.fnmatch(f, "images-*.zip"))
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        scratch_path = pathlib.Path(scratch)
        for name in shards:
            shard = _fetch(table, scratch_path, name)
            shutil.unpack_archive(shard, scratch_path / "extracted", "zip")
            shard.unlink()
        extracted = scratch_path / "extracted" / "images"
        if not extracted.is_dir():
            raise OSError(
                f"multabench/{table.repo} at {table.revision} has no"
                f" images/ folder in its {len(shards)} image shards"
            )
        # One rename, so a cut-off extraction never looks complete.
        try:
            extracted.rename(images)
        except OSError:
            if not images.is_dir():  # Not another process's extraction.
                raise


def _fetch(table: Table, folder: pathlib.Path, filename: str) -> pathlib.Path:
    """Downloads one file of a dataset's repository, at its pinned commit."""
    import huggingface_hub  # noqa: PLC0415 - kept out of `import lazy`.

    return pathlib.Path(
        huggingface_hub.hf_hub_download(
            repo_id=f"multabench/{table.repo}",
            filename=filename,
            repo_type="dataset",
            revision=table.revision,
            local_dir=folder,
        )
    )
