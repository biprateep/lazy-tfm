# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Publication plot style: journal geometry, rcParams and figure helpers.

Every figure in :mod:`lazy.plotting.diagnostics` is drawn through this, and it
is usable on its own for any figure in the same paper::

    from lazy.plotting import style

    style.use_style()                 # AASTeX geometry, house rcParams
    fig, ax = plt.subplots(figsize=style.figsize("column"))

The house sheet is ``paper.mplstyle``, shipped alongside this module: Nimbus
Roman serif with Computer Modern math at 9/10/12 pt, inward ticks on all four
sides, frameless legends, matplotlib's own tab10 / viridis palettes, tight
300 dpi output. Widths come from measured journal geometry rather than guesses
-- see :func:`register_journal` for adding your own.

Colour is deliberately not part of the sheet: figures use exactly what
matplotlib ships unless you call :func:`set_palette`.

Depends only on numpy and matplotlib.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import os
import pathlib
from typing import Any
import warnings

import cycler
import matplotlib as mpl
from matplotlib import axes as mpl_axes
from matplotlib import cm
from matplotlib import colorbar as mpl_colorbar
from matplotlib import figure as mpl_figure
from matplotlib import font_manager
from matplotlib import typing as mpl_typing
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lazy import _typing

__all__ = [
    "PT_PER_INCH",
    "Journal",
    "JOURNALS",
    "DEFAULT_JOURNAL",
    "register_journal",
    "journal",
    "COLUMN_WIDTH",
    "TEXT_WIDTH",
    "SMALL_SIZE",
    "NORMAL_SIZE",
    "BIG_SIZE",
    "FONT_FAMILY",
    "SERIF_FALLBACKS",
    "RC_PARAMS",
    "DEFAULT_PALETTE",
    "set_palette",
    "GOLDEN",
    "ASPECTS",
    "use_style",
    "verify_style",
    "figsize",
    "grid_figsize",
    "better_step",
    "stacked_hist",
    "binned_quantiles",
    "running_median",
    "one_to_one",
    "side_colorbar",
    "save",
]

# --------------------------------------------------------------------------
# Journal geometry
# --------------------------------------------------------------------------
# LaTeX points are 72.27 to the inch (NOT 72 — that is the PostScript point).
PT_PER_INCH = 72.27


@dataclasses.dataclass(frozen=True)
class Journal:
    r"""Column and text width of one journal class, in LaTeX points.

    Measure them in the manuscript itself: put ``\showthe\columnwidth`` and
    ``\showthe\textwidth`` in the body, compile, and read the two numbers
    off the log. Register the result with :func:`register_journal`.

    Attributes:
        name: Short name of the journal class, such as ``"aastex"``.
        column_pt: Width of one column, in LaTeX points.
        text_pt: Full text width of the page, in LaTeX points.
    """

    name: str
    column_pt: float
    text_pt: float

    @property
    def column_width(self) -> float:
        """One column, in inches."""
        return self.column_pt / PT_PER_INCH

    @property
    def text_width(self) -> float:
        """Full page width, in inches."""
        return self.text_pt / PT_PER_INCH

    def width(self, kind: str) -> float:
        """Returns the ``"column"`` or ``"text"`` width, in inches.

        Args:
            kind: ``"column"`` or ``"text"``.

        Returns:
            The requested width, in inches.
        """
        try:
            return {"column": self.column_width, "text": self.text_width}[kind]
        except KeyError:
            raise ValueError(
                f"width must be 'column' or 'text', got {kind!r}"
            ) from None


# Only measured classes go in here. Add a journal with register_journal(),
# never by guessing — a wrong width silently rescales every figure in the
# paper.
JOURNALS: dict[str, Journal] = {
    "aastex": Journal(
        "aastex", column_pt=242.26653, text_pt=513.11743
    ),  # AJ / ApJ
}
DEFAULT_JOURNAL = "aastex"
# Mutable on purpose: use_style() selects the journal that figsize() and
# grid_figsize() follow afterwards, the way rcParams follow plt.style.use().
_active_journal: Journal = JOURNALS[DEFAULT_JOURNAL]


def register_journal(name: str, column_pt: float, text_pt: float) -> Journal:
    r"""Adds (or replaces) a journal's measured widths and returns it.

    Args:
        name: Key to register the journal under.
        column_pt: Measured ``\columnwidth``, in LaTeX points.
        text_pt: Measured ``\textwidth``, in LaTeX points.

    Returns:
        The registered :class:`Journal`.
    """
    JOURNALS[name] = Journal(name, float(column_pt), float(text_pt))
    return JOURNALS[name]


def journal(name: str | Journal | None = None) -> Journal:
    """Returns the named journal, or the one activated by :func:`use_style`.

    Args:
        name: A key of :data:`JOURNALS`, a :class:`Journal` (returned as
            is), or None for the active journal.

    Returns:
        The selected :class:`Journal`.

    Raises:
        KeyError: If ``name`` is a string that has not been registered.
    """
    if name is None:
        return _active_journal
    if isinstance(name, Journal):
        return name
    try:
        return JOURNALS[name]
    except KeyError:
        raise KeyError(
            f"unknown journal {name!r}; known: {sorted(JOURNALS)}. "
            "Measure \\columnwidth / \\textwidth and call register_journal()."
        ) from None


# Module constants for the DEFAULT journal (AASTeX), for the inline
# ``figsize=(COLUMN_WIDTH, 0.62 * COLUMN_WIDTH)`` form. For any other journal
# use figsize() / grid_figsize(), which follow the journal passed to
# use_style().
COLUMN_WIDTH = JOURNALS[DEFAULT_JOURNAL].column_width  # 3.352 in
TEXT_WIDTH = JOURNALS[DEFAULT_JOURNAL].text_width  # 7.100 in

# --------------------------------------------------------------------------
# Type: sizes in points, matched to the 10 pt manuscript body
# --------------------------------------------------------------------------
SMALL_SIZE = 9  # tick labels; crowded legends
NORMAL_SIZE = 10  # body: axis labels, legends, axes titles, free text
BIG_SIZE = 12  # panel titles, suptitles

# The URW Times clone that matches the manuscript body text. "Nimbus Roman
# No9 L" is its classic name; newer urw-base35 packages ship the same face as
# plain "Nimbus Roman". Asking matplotlib for the exact classic name on such a
# machine silently falls back to DejaVu Sans, so the rcParams request the
# generic "serif" family and let this chain resolve it. Tail entries are
# Times-metric stand-ins for machines without any Nimbus at all.
FONT_FAMILY = "Nimbus Roman No9 L"
SERIF_FALLBACKS = [
    "Nimbus Roman No9 L",
    "Nimbus Roman",
    "Times New Roman",
    "Liberation Serif",
    "DejaVu Serif",
]

STYLE_FILE = pathlib.Path(__file__).with_name("paper.mplstyle")

# Every piece of text in a figure — ticks, labels, titles, legends, colorbar
# labels, annotations — takes its face and size from these.
RC_PARAMS: dict[mpl_typing.RcKeyType, Any] = {
    "font.family": "serif",
    "font.serif": SERIF_FALLBACKS,
    "font.size": NORMAL_SIZE,
    "axes.titlesize": NORMAL_SIZE,
    "axes.labelsize": NORMAL_SIZE,
    "xtick.labelsize": SMALL_SIZE,
    "ytick.labelsize": SMALL_SIZE,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": NORMAL_SIZE,
    "legend.title_fontsize": NORMAL_SIZE,
    "legend.frameon": False,
    "figure.titlesize": BIG_SIZE,
    "figure.facecolor": "w",
    "mathtext.fontset": "cm",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "png",
}

# Colour is deliberately NOT in RC_PARAMS. Unless the user names a palette,
# figures use exactly what matplotlib ships: the default property cycle
# (tab10, addressed as C0, C1, ...) for categorical series and the default
# colormap (viridis) for continuous data. These are taken from matplotlib's
# own defaults rather than retyped, so they track the installed version and
# override anything a personal matplotlibrc may have changed.
DEFAULT_PALETTE: dict[mpl_typing.RcKeyType, Any] = {
    "axes.prop_cycle": mpl.rcParamsDefault["axes.prop_cycle"],
    "image.cmap": mpl.rcParamsDefault["image.cmap"],
}


def set_palette(
    colors: str | Sequence[Any] | None = None,
    cmap: str | None = None,
    n_colors: int | None = None,
) -> tuple[list[Any], str]:
    """Installs a user-stated palette once, in the preamble.

    Only call this when the user has explicitly asked for a palette. Either
    argument may be omitted; ``None`` for both restores matplotlib's
    defaults.

    Args:
        colors: The new property cycle, so series keep addressing it by
            ``C0``, ``C1``, ...: a list of colours, or the name of a
            matplotlib colormap — a qualitative one (``"Dark2"``,
            ``"Set2"``, ``"tab20"``) contributes its colours as-is, a
            continuous one is sampled at ``n_colors`` evenly spaced points.
        cmap: Name of the default colormap for ``scatter(c=...)``,
            ``imshow``, ``pcolormesh`` and friends.
        n_colors: Number of colours taken from a colormap named by
            ``colors``: the first ``n_colors`` of a qualitative map (default
            all of them), or the number of samples of a continuous one
            (default 10).

    Returns:
        The ``(colors, cmap)`` now in effect, as a tuple.

    Raises:
        KeyError: If ``colors`` or ``cmap`` names an unknown colormap.
    """
    if colors is None and cmap is None:
        mpl.rcParams.update(DEFAULT_PALETTE)
    if colors is not None:
        if isinstance(colors, str):
            colormap = mpl.colormaps[colors]
            # Qualitative maps are ListedColormaps with a handful of entries
            # (tab20 is the largest at 20); viridis & co. are Listed too but
            # carry 256 samples, so they are sampled rather than copied.
            listed = getattr(colormap, "colors", None)
            if listed is not None and len(listed) <= 20:
                cycle_colors = list(listed)[: n_colors or len(listed)]
            else:
                cycle_colors = [
                    colormap(v) for v in np.linspace(0, 1, n_colors or 10)
                ]
        else:
            cycle_colors = list(colors)
        mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=cycle_colors)
    if cmap is not None:
        mpl.colormaps[cmap]  # Fails early on an unknown name.
        mpl.rcParams["image.cmap"] = cmap
    return (
        mpl.rcParams["axes.prop_cycle"].by_key()["color"],
        mpl.rcParams["image.cmap"],
    )


def use_style(
    style_file: str | os.PathLike[str] | None = None,
    journal: str | Journal = DEFAULT_JOURNAL,
    palette: str | Sequence[Any] | None = None,
    cmap: str | None = None,
) -> Journal:
    """Applies the publication rcParams globally and selects the journal.

    Prefers the ``paper.mplstyle`` sheet shipped alongside this module and
    falls back to the equivalent ``RC_PARAMS`` dict if it is missing. Colour
    is reset to matplotlib's default palettes unless ``palette`` and/or
    ``cmap`` are given.

    Args:
        style_file: A matplotlib style sheet to use instead of the shipped
            ``paper.mplstyle``.
        journal: The journal whose geometry :func:`figsize` and
            :func:`grid_figsize` use from now on: a key of :data:`JOURNALS`
            or a :class:`Journal`.
        palette: Property cycle, as for ``colors`` in :func:`set_palette`.
            Pass it only when the user has explicitly stated a palette.
        cmap: Default colormap, as in :func:`set_palette`. Pass it only when
            the user has explicitly stated a palette.

    Returns:
        The :class:`Journal` now active.
    """
    global _active_journal
    path = pathlib.Path(style_file) if style_file is not None else STYLE_FILE
    if path.exists():
        plt.style.use(str(path))
    else:
        mpl.rcParams.update(RC_PARAMS)
    set_palette(palette, cmap)
    # The `journal` parameter shadows the module-level function of that name.
    _active_journal = globals()["journal"](journal)
    return _active_journal


def verify_style(
    strict: bool = True, *, strict_font: bool = False
) -> dict[str, Any]:
    """Checks that the house rcParams are active and the serif face resolved.

    The palette is reported, not enforced — a user-stated one is legitimate.
    A missing paper font is a property of the machine, not of the code, so
    it warns rather than raises unless ``strict_font`` is set.

    Args:
        strict: Raise if an rcParam is wrong, rather than only reporting it.
        strict_font: Raise, too, if the serif face fell through to DejaVu,
            rather than warning.

    Returns:
        A dict with the keys ``font`` (the path of the font file matplotlib
        will use), ``palette`` (the colours of the active property cycle),
        ``cmap`` (the default colormap's name), ``default_palette`` (whether
        the palette is matplotlib's default) and ``problems`` (a list of
        messages, empty when the style is intact).

    Raises:
        RuntimeError: With ``strict=True`` (default), if any rcParam differs
            from :data:`RC_PARAMS`; with ``strict_font=True`` as well, if
            the font chain fell through to a DejaVu face — which is what
            happens on a machine without Nimbus Roman, and which matplotlib
            otherwise reports only as a debug-level log line.

    Warns:
        UserWarning: If the font chain fell through to DejaVu and
            ``strict_font`` is False.
    """
    expected = mpl.RcParams(RC_PARAMS)  # Runs values through the validators.
    problems = [
        f"{key} is {mpl.rcParams[key]!r}, expected {expected[key]!r}"
        for key in RC_PARAMS
        if mpl.rcParams[key] != expected[key]
    ]
    font_path = pathlib.Path(
        font_manager.findfont(
            font_manager.FontProperties(family=mpl.rcParams["font.serif"])
        )
    )
    fatal = list(problems)
    if "dejavu" in font_path.name.lower():
        font_problem = (
            f"serif font resolved to {font_path.name}; install Nimbus Roman "
            "(package urw-base35 / gsfonts / fonts-urw-base35) and clear "
            "~/.cache/matplotlib"
        )
        problems.append(font_problem)
        if strict_font:
            fatal.append(font_problem)
        else:
            warnings.warn(font_problem, UserWarning, stacklevel=2)
    if strict and fatal:
        bullets = "\n  - ".join(fatal)
        raise RuntimeError(f"plot style not in effect:\n  - {bullets}")
    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", [])
    return {
        "font": str(font_path),
        "palette": colors,
        "cmap": mpl.rcParams["image.cmap"],
        "default_palette": colors
        == DEFAULT_PALETTE["axes.prop_cycle"].by_key()["color"]
        and mpl.rcParams["image.cmap"] == DEFAULT_PALETTE["image.cmap"],
        "problems": problems,
    }


# --------------------------------------------------------------------------
# Figure geometry helpers
# --------------------------------------------------------------------------
GOLDEN = 2 / (1 + 5**0.5)  # 0.618: height / width of a golden rectangle

# Named height:width ratios. The bracket is what the source manuscripts used:
# single-column panels 0.56–0.75 (0.7 most often), full-width two-panel rows
# 0.35–0.4, full-width single panels 0.5–0.65, comparisons square.
ASPECTS = {
    "golden": GOLDEN,  # default single panel
    "wide": 0.5625,  # 16:9 — histograms, trends with little vertical range
    "tall": 0.75,  # 4:3 — scatter with a running median or a legend inside
    "square": 1.0,  # one-to-one comparisons, sky maps, anything aspect="equal"
}


def _resolve_width(
    width: str | float, journal_: str | Journal | None = None
) -> float:
    if isinstance(width, str):
        return journal(journal_).width(width)
    return float(width)


def _resolve_aspect(aspect: str | float) -> float:
    if isinstance(aspect, str):
        try:
            return ASPECTS[aspect]
        except KeyError:
            raise ValueError(
                f"aspect must be a float or one of {sorted(ASPECTS)}"
            ) from None
    return float(aspect)


def figsize(
    width: str | float = "column",
    aspect: str | float = "golden",
    journal: str | Journal | None = None,
) -> tuple[float, float]:
    """Returns ``(w, h)`` in inches for a journal-width figure.

    Args:
        width: ``"column"`` for a single column, ``"text"`` for the full
            page width (of the journal selected by :func:`use_style`, unless
            ``journal`` overrides it), or an explicit width in inches.
        aspect: Height as a fraction of the width, as a float or a key of
            :data:`ASPECTS`. Default is the golden ratio.
        journal: Take the widths from this journal instead of the active
            one.

    Returns:
        The figure size ``(w, h)`` in inches, as a tuple.
    """
    w = _resolve_width(width, journal)
    return (w, _resolve_aspect(aspect) * w)


def grid_figsize(
    nrows: int = 1,
    ncols: int = 1,
    width: str | float = "text",
    panel_aspect: str | float = "golden",
    journal: str | Journal | None = None,
) -> tuple[float, float]:
    """Returns ``(w, h)`` for a grid whose *panels* have ``panel_aspect``.

    The figure spans ``width``; each panel is nominally ``width / ncols``
    wide and ``panel_aspect`` times that tall, so the figure is
    ``nrows * panel_aspect * width / ncols`` tall. Two golden panels across
    the text width give ``0.31 * TEXT_WIDTH``; the manuscripts used 0.35–0.4
    for two-panel rows, i.e. ``panel_aspect`` of roughly 0.7–0.8.

    Args:
        nrows: Number of panel rows.
        ncols: Number of panel columns.
        width: ``"column"``, ``"text"`` or an explicit width in inches, as
            in :func:`figsize`.
        panel_aspect: Height of each panel as a fraction of its width, as a
            float or a key of :data:`ASPECTS`.
        journal: Take the widths from this journal instead of the active
            one.

    Returns:
        The figure size ``(w, h)`` in inches, as a tuple.
    """
    w = _resolve_width(width, journal)
    return (w, nrows * _resolve_aspect(panel_aspect) * w / ncols)


# --------------------------------------------------------------------------
# Figure idioms
# --------------------------------------------------------------------------
def better_step(
    bin_edges: npt.ArrayLike,
    heights: npt.ArrayLike,
    yerr: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    ax: mpl_axes.Axes | None = None,
    **kwargs: Any,
) -> mpl_axes.Axes:
    """A 'better' version of matplotlib's step function.

    Given a set of bin edges and bin heights, this plots the thing that I
    wish matplotlib's ``step`` command plotted. All extra arguments are
    passed directly to matplotlib's ``plot`` command.

    Args:
        bin_edges: The bin edges, shape (n_bins + 1,). This should be one
            element longer than the bin heights array ``heights``.
        heights: The bin heights, shape (n_bins,).
        yerr: Asymmetric error on ``heights``, as a ``(lower, upper)`` pair
            of the lower and upper band limits, each of shape (n_bins,).
        ax: The axis where this should be plotted; the current axes if
            None.
        **kwargs: Passed straight through to matplotlib's ``plot``.

    Returns:
        The axes drawn into.
    """
    edges = np.asarray(bin_edges)
    # Each bin contributes its left and right edge at its own height, so the
    # line runs flat across every bin and jumps vertically between them.
    new_x = np.column_stack([edges[:-1], edges[1:]]).ravel()
    new_y = np.repeat(np.asarray(heights), 2)
    if ax is None:
        ax = plt.gca()
    lines = ax.plot(new_x, new_y, **kwargs)
    if yerr is not None:
        new_yerr_lo = np.repeat(np.asarray(yerr[0]), 2)
        new_yerr_up = np.repeat(np.asarray(yerr[1]), 2)
        ax.fill_between(
            new_x,
            new_yerr_up,
            new_yerr_lo,
            alpha=0.1,
            color=lines[0].get_color(),
        )
    return ax


def stacked_hist(
    ax: mpl_axes.Axes,
    datasets: Sequence[npt.ArrayLike],
    labels: str | Sequence[str] | None = None,
    bins: int | Sequence[float] = 20,
    colors: Sequence[Any] | None = None,
    alpha: float = 0.5,
    rwidth: float = 0.8,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Stacked histogram in house style: white step underlay + gapped fills.

    Two passes over the same bin edges — a ``histtype="step"`` pass in white
    that traces the full-width stacked envelope, then the semi-transparent
    ``rwidth``-narrowed bars on top. The gap plus the underlay keeps the
    individual bars legible where translucent stacks would otherwise merge.

    Args:
        ax: The axes to draw into.
        datasets: One array of values per stacked component.
        labels: Legend label of each component.
        bins: Number of bins, spanning the pooled data, or the bin edges.
        colors: Colour of each component; ``C0``, ``C1``, ... by default.
        alpha: Opacity of the filled bars.
        rwidth: Width of each bar as a fraction of its bin.
        **kwargs: Passed to both ``ax.hist`` calls.

    Returns:
        The ``(counts, edges, patches)`` tuple of the filled pass, as
        returned by ``ax.hist``.
    """
    datasets = list(datasets)
    n_datasets = len(datasets)
    if colors is None:
        colors = [f"C{i}" for i in range(n_datasets)]
    if np.isscalar(bins):
        pooled = np.concatenate([np.asarray(d).ravel() for d in datasets])
        bins = np.histogram_bin_edges(pooled, bins=bins).tolist()

    ax.hist(
        datasets,
        bins=bins,
        stacked=True,
        histtype="step",
        color=["white"] * n_datasets,
        **kwargs,
    )
    return ax.hist(
        datasets,
        bins=bins,
        stacked=True,
        color=colors,
        alpha=alpha,
        rwidth=rwidth,
        label=labels,
        **kwargs,
    )


def binned_quantiles(
    x_values: npt.ArrayLike,
    y_values: npt.ArrayLike,
    nbins: int = 10,
    equal_count: bool = True,
    bins: npt.ArrayLike | None = None,
    percentiles: Sequence[float] = (25, 50, 75),
) -> tuple[_typing.FloatArray, _typing.FloatArray]:
    """Returns percentiles of ``y_values`` in bins of ``x_values``.

    Points where either value is not finite are dropped first.

    Args:
        x_values: The binned coordinate, shape (n,).
        y_values: The values summarized in each bin, shape (n,).
        nbins: Number of bins, when ``bins`` is not given.
        equal_count: Put an equal number of points in each bin (the
            ``pd.qcut`` behaviour) so the scatter band is equally well
            determined everywhere; set it False for equal-width bins.
        bins: Explicit bin edges, overriding ``nbins`` and ``equal_count``.
        percentiles: The percentiles to compute, between 0 and 100.

    Returns:
        A tuple ``(edges, values)``: the unique bin edges, and the
        percentiles of shape ``(len(percentiles), len(edges) - 1)``. Empty
        bins come back as NaN.
    """
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    good = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[good], y_array[good]

    if bins is not None:
        edges = np.asarray(bins, dtype=float)
    elif equal_count:
        edges = np.quantile(x_array, np.linspace(0.0, 1.0, nbins + 1))
    else:
        edges = np.linspace(x_array.min(), x_array.max(), nbins + 1)
    edges = np.unique(edges)

    idx = np.clip(np.digitize(x_array, edges) - 1, 0, len(edges) - 2)
    values = np.full((len(percentiles), len(edges) - 1), np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            values[:, b] = np.percentile(y_array[sel], percentiles)
    return edges, values


def running_median(
    ax: mpl_axes.Axes,
    x_values: npt.ArrayLike,
    y_values: npt.ArrayLike,
    nbins: int = 10,
    equal_count: bool = True,
    bins: npt.ArrayLike | None = None,
    color: str = "C1",
    global_median: bool = True,
    **kwargs: Any,
) -> tuple[
    _typing.FloatArray,
    _typing.FloatArray,
    _typing.FloatArray,
    _typing.FloatArray,
]:
    """Overlays a binned median with a 25–75 percentile band.

    Draws the median as a :func:`better_step` staircase with a shaded IQR,
    and (by default) the global median as a dashed line in the same colour.

    Args:
        ax: The axes to draw into.
        x_values: The binned coordinate, shape (n,).
        y_values: The values whose median is drawn, shape (n,).
        nbins: Number of bins, as in :func:`binned_quantiles`.
        equal_count: Equal-count rather than equal-width bins, as in
            :func:`binned_quantiles`.
        bins: Explicit bin edges, as in :func:`binned_quantiles`.
        color: Colour of the staircase, band and global median.
        global_median: Also draw the median of all ``y_values``.
        **kwargs: Passed to :func:`better_step`.

    Returns:
        A tuple ``(edges, p25, median, p75)``; the last three have shape
        ``(len(edges) - 1,)``.
    """
    edges, (p25, med, p75) = binned_quantiles(
        x_values,
        y_values,
        nbins=nbins,
        equal_count=equal_count,
        bins=bins,
        percentiles=(25, 50, 75),
    )
    better_step(edges, med, (p25, p75), ax=ax, c=color, **kwargs)
    if global_median:
        overall = float(np.nanmedian(np.asarray(y_values, dtype=float)))
        ax.axhline(overall, color=color, ls="--", lw=1)
    return edges, p25, med, p75


def one_to_one(
    ax: mpl_axes.Axes,
    lo: float,
    hi: float,
    color: str = "k",
    ls: str = "--",
    equal: bool = True,
    **kwargs: Any,
) -> mpl_axes.Axes:
    """Draws a dashed identity line on matched, optionally equal-aspect, axes.

    Args:
        ax: The axes to draw into.
        lo: Lower limit of both axes and of the line.
        hi: Upper limit of both axes and of the line.
        color: Line colour.
        ls: Line style.
        equal: Also set an equal aspect ratio.
        **kwargs: Passed to ``ax.plot``.

    Returns:
        The axes drawn into.
    """
    x = np.linspace(lo, hi, 100)
    ax.plot(x, x, color=color, ls=ls, **kwargs)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if equal:
        ax.set_aspect("equal")
    return ax


def side_colorbar(
    fig: mpl_figure.Figure,
    mappable: cm.ScalarMappable,
    label: str | None = None,
    rect: tuple[float, float, float, float] = (0.95, 0.15, 0.02, 0.7),
    labelpad: float = 15,
) -> mpl_colorbar.Colorbar:
    """Adds a slim colorbar on its own axes to the right of the whole figure.

    The label is rotated -90 deg so it reads top-to-bottom.

    Args:
        fig: The figure to add the colorbar to.
        mappable: The image, mesh or scatter the colorbar describes.
        label: Colorbar label.
        rect: ``[left, bottom, width, height]`` in figure coordinates; the
            default sits just outside a standard axes and is shared by
            every panel.
        labelpad: Spacing between the colorbar and its label, in points.

    Returns:
        The new colorbar.
    """
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(mappable, cax=cax)
    if label is not None:
        cbar.set_label(label, rotation=-90, labelpad=labelpad)
    return cbar


def save(
    fig: mpl_figure.Figure,
    path: str | os.PathLike[str],
    dpi: float = 300,
    **kwargs: Any,
) -> pathlib.Path:
    """Saves as a tight 300 dpi PNG — the house output — creating parent dirs.

    A path whose suffix is not a format matplotlib can save gets ``.png``
    appended, so ``"out/model.v2"`` becomes ``out/model.v2.png``. Pass an
    explicit ``.pdf`` only when
    a vector figure has been specifically requested. The figure stays open:
    it belongs to the caller, who closes it.

    Args:
        fig: The figure to save.
        path: Output file.
        dpi: Resolution, in dots per inch.
        **kwargs: Passed to ``fig.savefig``.

    Returns:
        The path written, with ``.png`` added if its suffix was not a
        format.
    """
    path = pathlib.Path(path)
    formats = fig.canvas.get_supported_filetypes()
    if path.suffix[1:].lower() not in formats:
        path = path.with_name(path.name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi, **kwargs)
    return path
