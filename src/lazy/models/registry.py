# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The backend registry: which estimators exist and what they are called.

Kept in its own module so that :class:`lazy.models.lazy_model.LazyModel` can
look a backend up without :mod:`lazy.models` having to import it first.

Adding a backend means writing a :class:`lazy.base.BasePhotoZEstimator`
subclass and adding one line to :data:`ESTIMATORS`; nothing else in the library
needs to know it exists.
"""

from __future__ import annotations

from lazy.base import BasePhotoZEstimator
from lazy.models.tabfm import TabFMHistogram
from lazy.models.tabicl import TabICLQuantile
from lazy.models.tabpfn import TabPFNBarDistribution

__all__ = ["ESTIMATORS", "get_estimator", "list_estimators"]

#: Short name -> estimator class. These names are what :class:`LazyModel` and
#: :func:`get_estimator` accept, and what the benchmark tables record.
ESTIMATORS: dict[str, type[BasePhotoZEstimator]] = {
    "tabfm": TabFMHistogram,
    "tabicl": TabICLQuantile,
    "tabpfn": TabPFNBarDistribution,
}


def list_estimators() -> list[str]:
    """The backend names, sorted.

    >>> list_estimators()
    ['tabfm', 'tabicl', 'tabpfn']
    """
    return sorted(ESTIMATORS)


def get_estimator(name: str, **params) -> BasePhotoZEstimator:
    """Construct the *concrete* backend class registered under ``name``.

    Prefer :class:`lazy.models.lazy_model.LazyModel`, which is the same lookup
    with the common API wrapped around it. This is the lower-level door, for
    when you want the backend object itself.

    >>> get_estimator("tabicl", n_estimators=16)
    TabICLQuantile(n_estimators=16)
    """
    try:
        cls = ESTIMATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown estimator {name!r}; known: {list_estimators()}"
        ) from None
    return cls(**params)
