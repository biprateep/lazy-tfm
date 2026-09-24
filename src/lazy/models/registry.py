# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The backend registry: which estimators exist and what they are called.

Kept in its own module so that :class:`lazy.models.lazy_model.LazyModel` can
look a backend up without :mod:`lazy.models` having to import it first.

Adding a backend means writing a :class:`lazy.base.BasePhotoZEstimator`
subclass and adding one line to :data:`ESTIMATORS`; nothing else in the
library needs to know it exists.
"""

from __future__ import annotations

from typing import Any

from lazy import base
from lazy.models import tabfm
from lazy.models import tabicl
from lazy.models import tabpfn

__all__ = ["ESTIMATORS", "get_estimator", "list_estimators"]

#: Short name -> estimator class. These names are what :class:`LazyModel` and
#: :func:`get_estimator` accept, and what the benchmark tables record.
ESTIMATORS: dict[str, type[base.BasePhotoZEstimator]] = {
    "tabfm": tabfm.TabFMHistogram,
    "tabicl": tabicl.TabICLQuantile,
    "tabpfn": tabpfn.TabPFNBarDistribution,
}


def list_estimators() -> list[str]:
    """Returns the backend names, sorted.

    Examples:
        >>> list_estimators()
        ['tabfm', 'tabicl', 'tabpfn']
    """
    return sorted(ESTIMATORS)


def get_estimator(name: str, **params: Any) -> base.BasePhotoZEstimator:
    """Constructs the *concrete* backend class registered under ``name``.

    Prefer :class:`lazy.models.lazy_model.LazyModel`, which is the same lookup
    with the common API wrapped around it. This is the lower-level door, for
    when you want the backend object itself.

    Args:
        name: A backend name; one of :func:`list_estimators`.
        **params: Passed to the backend's constructor.

    Returns:
        The unfitted backend estimator.

    Raises:
        KeyError: If no backend is registered under ``name``.

    Examples:
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
