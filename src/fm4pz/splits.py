# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""The training and test sets, drawn exactly as the submitted paper drew them.

Representative: the DC1 training file and its nested random subsets, scored on
the whole test file. Biased: the HSC-like selection of
``lazy.datasets.fetch_dc1_biased`` applied to the merged catalogue, its nested
subsets, scored on a representative hold-out. A subset of size ``n`` for seed
``s`` is the first ``n`` rows of one permutation per seed, so the sizes of one
seed are nested. The draws reproduce the submitted paper's frozen index files
bit for bit (``checksum`` gives the same digests).

A biased training set can also take ``k`` galaxies of the representative
calibration sample, added to its rows rather than replacing any: the first
``k`` of one permutation of the calibration sample per seed, drawn as the
submitted paper drew its calibration matrix.

Typical usage example:

  data = splits.load("biased", n_train=5_000, seed=2)
  data.train.features("mag-color"), data.train.redshift
  mixed = splits.load("biased", n_train=5_000, seed=0, n_calibration=500)
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib

from lazy import datasets
import numpy as np
import numpy.typing as npt

from fm4pz import config

__all__ = ["Split", "calibration_rows", "checksum", "load", "train_rows"]

# Offset applied to a seed before it reaches NumPy, as in the submitted paper.
_SEED_BASE = 20260917


@dataclasses.dataclass(frozen=True)
class Split:
    """One training set and the test set it is scored on.

    Attributes:
        regime: ``"representative"`` or ``"biased"``.
        n_train: Training rows.
        seed: The subsample's seed.
        train: The training catalogue.
        test: The test catalogue.
        train_checksum: :func:`checksum` of the training rows.
        n_calibration: Calibration galaxies added to the ``n_train``
            training rows (biased regime only).
    """

    regime: str
    n_train: int
    seed: int
    train: datasets.Catalog
    test: datasets.Catalog
    train_checksum: str
    n_calibration: int = 0


def checksum(rows: npt.ArrayLike) -> str:
    """A short SHA-256 of an index array, as the submitted paper recorded.

    Args:
        rows: Integer row positions.

    Returns:
        The first 16 hex digits of the digest of the rows as int64.
    """
    contiguous = np.ascontiguousarray(np.asarray(rows, dtype=np.int64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def _permutation(n_rows: int, seed: int) -> npt.NDArray[np.int64]:
    """The permutation of ``n_rows`` rows that belongs to ``seed``."""
    return np.random.default_rng(_SEED_BASE + seed).permutation(n_rows)


@functools.cache
def _catalogs() -> tuple[datasets.Catalog, datasets.Catalog, datasets.Catalog]:
    """The DC1 training and test files, and the two merged."""
    train, test = datasets.fetch_dc1(split=True)
    return train, test, datasets.fetch_dc1()


@functools.cache
def _selection() -> dict[str, npt.NDArray[np.int64]]:
    """Rows of the merged catalogue: biased pool, calibration and test."""
    split = datasets.fetch_dc1_biased(**config.SELECTION)
    rows = {
        "pool": np.sort(split.rows["biased"]),
        "calibration": np.sort(split.rows["calibration"]),
        "test": np.sort(split.rows["test"]),
    }
    if rows["calibration"].size != config.CALIBRATION:
        raise RuntimeError(
            f"the calibration sample has {rows['calibration'].size} rows, "
            f"not {config.CALIBRATION}: the selection moved"
        )
    if rows["pool"].size != config.BIASED_TRAIN:
        raise RuntimeError(
            f"the biased pool has {rows['pool'].size} rows, not "
            f"{config.BIASED_TRAIN}: the selection moved"
        )
    if rows["test"].size != config.BIASED_TEST:
        raise RuntimeError(
            f"the biased test set has {rows['test'].size} rows, not "
            f"{config.BIASED_TEST}: the selection moved"
        )
    return rows


def train_rows(regime: str, n_train: int, seed: int) -> npt.NDArray[np.int64]:
    """The training rows of one subsample, sorted.

    Args:
        regime: ``"representative"`` (rows of the training file) or
            ``"biased"`` (rows of the merged catalogue).
        n_train: The subsample's size; the full size gives every row.
        seed: The subsample's seed.

    Returns:
        The rows, sorted, as int64.

    Raises:
        ValueError: If ``n_train`` exceeds the regime's pool.
    """
    if regime == "representative":
        pool = np.arange(config.FULL_TRAIN, dtype=np.int64)
    elif regime == "biased":
        pool = _selection()["pool"]
    else:
        raise ValueError(f"unknown regime: {regime=}")
    if n_train > pool.size:
        raise ValueError(f"{regime} has only {pool.size} rows: {n_train=}")
    if n_train == pool.size:
        return pool.copy()
    return np.sort(pool[_permutation(pool.size, seed)[:n_train]])


def calibration_rows(n_calibration: int, seed: int) -> npt.NDArray[np.int64]:
    """The calibration galaxies added for one seed, sorted.

    Args:
        n_calibration: How many, at most :data:`fm4pz.config.CALIBRATION`;
            the whole sample gives every row.
        seed: The draw's seed; the sizes of one seed are nested.

    Returns:
        Rows of the merged catalogue, sorted, as int64.
    """
    sample = _selection()["calibration"]
    if not 0 <= n_calibration <= sample.size:
        raise ValueError(
            f"the calibration sample has {sample.size} rows: {n_calibration=}"
        )
    if n_calibration == sample.size:
        return sample.copy()
    return np.sort(sample[_permutation(sample.size, seed)[:n_calibration]])


def load(
    regime: str, *, n_train: int, seed: int, n_calibration: int = 0
) -> Split:
    """One training subsample and its test set.

    Args:
        regime: ``"representative"`` or ``"biased"``.
        n_train: Training rows.
        seed: The subsample's seed, also the calibration draw's.
        n_calibration: Calibration galaxies added to the training rows;
            the biased regime only.

    Returns:
        The :class:`Split`.
    """
    if n_calibration and regime != "biased":
        raise ValueError(f"only the biased regime takes calibration: {regime=}")
    train_file, test_file, merged = _catalogs()
    rows = train_rows(regime, n_train, seed)
    if n_calibration:
        rows = np.union1d(rows, calibration_rows(n_calibration, seed))
        if rows.size != n_train + n_calibration:
            raise RuntimeError("calibration rows overlap the biased pool")
    if regime == "representative":
        train, test = train_file.take(rows), test_file
    else:
        train = merged.take(rows)
        test = merged.take(_selection()["test"])
    return Split(
        regime=regime,
        n_train=n_train,
        seed=seed,
        train=train,
        test=test,
        train_checksum=checksum(rows),
        n_calibration=n_calibration,
    )
