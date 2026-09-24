# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Progress bars for the prediction loops.

Every backend takes a ``progress`` parameter and hands it to :func:`bar`, but
what the bar counts is the backend's own unit of work, because the three spend
their time in different shapes:

* :class:`~lazy.models.tabpfn.TabPFNBarDistribution` and
  :class:`~lazy.models.tabicl.TabICLQuantile` answer in chunks of query rows,
  so their bars count **galaxies**, and a chunk is one update;
* :class:`~lazy.models.tabfm.TabFMHistogram` is a hierarchy of in-context
  classifications -- per dither, one coarse stage and then one fine stage per
  coarse bin -- each over *every* query row, so its bar counts **stages**, and
  says which dither and level is running and how many context rows it has.

``progress`` is ``"auto"`` by default: shown on a terminal or in a notebook,
silent when output goes to a file, which is what keeps a batch job's log free of
carriage-return spam without anyone having to remember to switch it off.
``True`` forces it on, ``False`` off. The bar writes to stderr, so it never
mixes with what ``verbose`` prints to stdout.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["PROGRESS_MODES", "Progress", "bar", "check_progress"]

#: What ``progress`` accepts on every estimator.
PROGRESS_MODES = ("auto", True, False)

Progress = bool | Literal["auto"]


def check_progress(progress) -> None:
    """Raise unless ``progress`` is ``"auto"``, ``True`` or ``False``.

    Checked by type as well as value, because ``1 in (True, False)`` is true in
    Python and a bar switched on by ``progress=1`` would be an accident.

    >>> check_progress("auto")
    >>> check_progress(1)
    Traceback (most recent call last):
    ...
    ValueError: progress must be one of ('auto', True, False), got 1
    """
    if not (
        isinstance(progress, bool)
        or (isinstance(progress, str) and progress == "auto")
    ):
        raise ValueError(
            f"progress must be one of {PROGRESS_MODES}, got {progress!r}"
        )


def bar(progress: Progress, *, total: int, desc: str, unit: str, **kwargs):
    """A ``tqdm`` bar honouring ``progress``, usable as a context manager.

    Parameters
    ----------
    progress
        ``"auto"``, ``True`` or ``False``; see the module docstring.
    total, desc, unit
        Passed to ``tqdm``.
    **kwargs
        Anything else ``tqdm`` takes.

    Examples
    --------
    >>> with bar(False, total=3, desc="demo", unit="row") as b:
    ...     b.update(3)
    >>> b.disable
    True
    """
    check_progress(progress)
    from tqdm.auto import tqdm

    # tqdm's own vocabulary: disable=None is its "only on a TTY" mode.
    disable = None if progress == "auto" else not progress
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
        disable=disable,
        **kwargs,
    )
