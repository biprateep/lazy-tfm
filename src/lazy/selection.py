# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The spectroscopic selection function: how a complete catalogue gets biased.

A photo-z model trained on a spectroscopic sample is trained on the galaxies
somebody could get a redshift for -- bright, and preferentially at the
redshifts where strong features fall in the observed window. The test sample
is everything else. That mismatch, not the estimator, is usually what limits a
survey's redshifts, so a benchmark that draws its training set at random
measures the easy half of the problem.

This module builds the hard half. :func:`grid_selection` is a port of RAIL's
``rail.creation.degraders.grid_selection.GridSelection``
(``LSSTDESC/rail_astro_tools``, commit 9c176272), which reproduces the HSC
PDR2 ratio of spectroscopic to photometric galaxies on a 100 x 100 grid of
i-band magnitude (13-26) against g-z colour (-2 to 6):

1. every galaxy is assigned the spec/phot ratio of its ``(i, g - z)`` pixel,
   and zero outside the grid;
2. with ``color_redshift_cut`` each pixel also gets the ``percentile_cut``-th
   percentile of the HSC spectroscopic redshifts that landed in it, and
   galaxies above that ceiling are removed -- pixels with no HSC spectra keep
   nothing. This is the step that makes the selection depend on redshift and
   not only on photometry, and it is what a model cannot undo from the
   features alone;
3. an optional hard ``redshift_cut``;
4. within each distinct ratio ``r``, a fraction ``r * scaling_factor`` of the
   survivors is kept at random.

The grid itself is a data file; :func:`lazy.datasets.fetch_hsc_grid` downloads
and caches it, and :func:`lazy.datasets.make_selection_split` is what you
normally call -- this module is the piece underneath, and the piece to reach
for if you want to bias a catalogue of your own.

Fidelity note: RAIL builds its per-pixel redshift ceiling indexed
``[magnitude][colour]`` and reads it back as ``[colour][magnitude]``, a
transposition its square grid hides. Here both are indexed consistently, i.e.
the intended behaviour. The ratio table itself is stored
``[colour][magnitude]`` in the file and RAIL reads it correctly.

Typical usage example:

  keep, diagnostics = selection.grid_selection(catalog.raw, catalog.redshift)
  print(selection.selection_summary(keep, catalog.raw["I"], catalog.redshift))
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib
import typing

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from lazy import _typing

__all__ = [
    "HSC_COLOR_LIMITS",
    "HSC_MAGNITUDE_LIMITS",
    "HSCGrid",
    "grid_selection",
    "selection_summary",
]

#: i-magnitude span of the HSC grid, and the g-z colour span, read once from
#: RAIL's ``HSC_grid_settings.pkl`` and written down here so that nothing in
#: this library ever unpickles a downloaded file.
HSC_MAGNITUDE_LIMITS = (13.0, 26.0)
HSC_COLOR_LIMITS = (-2.0, 6.0)


@dataclasses.dataclass(frozen=True)
class HSCGrid:
    """The HSC PDR2 spectroscopic success rate on a colour-magnitude grid.

    ``ratios`` is indexed ``[colour bin][magnitude bin]``, matching the file;
    ``spec`` holds the HSC spectroscopic galaxies (``mag``, ``color``,
    ``specz``) that the per-pixel redshift ceiling is computed from.

    Attributes:
        ratios: Spectroscopic / photometric ratio per pixel,
            ``[colour][magnitude]``, shape (n_colour_bins, n_magnitude_bins).
        x_edges: Magnitude bin edges in mag, ``ratios.shape[1] + 1`` of them.
        y_edges: Colour bin edges in mag, ``ratios.shape[0] + 1`` of them.
        spec: The HSC spectroscopic sample: ``mag``, ``color``, ``specz``.

    Examples:
        >>> grid = HSCGrid(
        ...     ratios=np.full((2, 2), 0.5),
        ...     x_edges=np.array([20.0, 22.0, 24.0]),
        ...     y_edges=np.array([0.0, 1.0, 2.0]),
        ...     spec=pd.DataFrame(
        ...         {"mag": [21.0], "color": [0.5], "specz": [0.4]}
        ...     ),
        ... )
        >>> grid.pixel([21.0, 25.0], [0.5, 0.5])
        (array([0, 2]), array([0, 0]))
    """

    ratios: _typing.FloatArray
    x_edges: _typing.FloatArray
    y_edges: _typing.FloatArray
    spec: pd.DataFrame
    # Per-percentile ceiling tables, filled on demand; not part of the value.
    _ceilings: dict[float, _typing.FloatArray] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    def __repr__(self) -> str:
        n_y, n_x = self.ratios.shape
        return (
            f"HSCGrid({n_x}x{n_y} pixels, "
            f"{len(self.spec):,} spectroscopic galaxies)"
        )

    @classmethod
    def from_hdf5(
        cls,
        path: str | pathlib.Path,
        *,
        magnitude_limits: tuple[float, float] = HSC_MAGNITUDE_LIMITS,
        color_limits: tuple[float, float] = HSC_COLOR_LIMITS,
    ) -> HSCGrid:
        """Reads RAIL's ``hsc_ratios_and_specz.hdf5``.

        The bin edges are not in the file -- RAIL keeps them in a pickled
        settings dictionary, which this library declines to execute; the
        limits it holds are the defaults above.

        Args:
            path: The HDF5 file.
            magnitude_limits: The i-magnitude span of the grid, in mag.
            color_limits: The g-z colour span of the grid, in mag.

        Returns:
            The grid, with evenly spaced bin edges across the two spans.
        """
        with h5py.File(path, "r") as saved:
            ratios = np.asarray(saved["ratios"][...], dtype=float)
            columns = [
                name.decode() for name in saved["data/block0_items"][...]
            ]
            spec = pd.DataFrame(
                np.asarray(saved["data/block0_values"][...]), columns=columns
            )
        return cls(
            ratios=ratios,
            x_edges=np.linspace(*magnitude_limits, ratios.shape[1] + 1),
            y_edges=np.linspace(*color_limits, ratios.shape[0] + 1),
            spec=spec,
        )

    def pixel(
        self, magnitude: npt.ArrayLike, color: npt.ArrayLike
    ) -> tuple[_typing.IntArray, _typing.IntArray]:
        """Returns the grid pixel each galaxy falls in.

        Args:
            magnitude: Magnitudes in mag, shape (n_galaxies,).
            color: Colours in mag, shape (n_galaxies,).

        Returns:
            A tuple (magnitude_bin, colour_bin) of bin indices, each of shape
            (n_galaxies,); ``-1`` or ``n`` for values off the grid.
        """
        px = (
            np.searchsorted(self.x_edges, np.asarray(magnitude, dtype=float))
            - 1
        )
        py = np.searchsorted(self.y_edges, np.asarray(color, dtype=float)) - 1
        return px, py

    def inside(
        self, px: _typing.IntArray, py: _typing.IntArray
    ) -> _typing.BoolArray:
        """Returns which pixel indices from :meth:`pixel` land on the grid.

        Args:
            px: Magnitude bin indices, shape (n_galaxies,).
            py: Colour bin indices, shape (n_galaxies,).

        Returns:
            A mask of shape (n_galaxies,), true where both indices are on the
            grid.
        """
        n_y, n_x = self.ratios.shape
        return (px >= 0) & (px < n_x) & (py >= 0) & (py < n_y)

    def max_specz(self, percentile: float = 99.0) -> _typing.FloatArray:
        """Returns the per-pixel percentile of the HSC spectroscopic redshifts.

        Zero where HSC has no spectra, which removes those pixels entirely --
        the selection is as blind as the survey it is modelled on. Tables are
        cached per percentile, because sizing a split re-runs the selection.

        Args:
            percentile: Which percentile of each pixel's redshifts to take,
                between 0 and 100.

        Returns:
            The redshift ceiling per pixel, ``[colour][magnitude]``, shape
            (n_colour_bins, n_magnitude_bins).

        Examples:
            >>> grid = HSCGrid(
            ...     ratios=np.full((2, 2), 0.5),
            ...     x_edges=np.array([20.0, 22.0, 24.0]),
            ...     y_edges=np.array([0.0, 1.0, 2.0]),
            ...     spec=pd.DataFrame(
            ...         {
            ...             "mag": [21.0, 21.5],
            ...             "color": [0.5, 0.5],
            ...             "specz": [0.4, 0.8],
            ...         }
            ...     ),
            ... )
            >>> grid.max_specz(50.0)
            array([[0.6, 0. ],
                   [0. , 0. ]])
        """
        key = float(percentile)
        if key not in self._ceilings:
            px, py = self.pixel(
                self.spec["mag"].to_numpy(), self.spec["color"].to_numpy()
            )
            n_y, n_x = self.ratios.shape
            ok = self.inside(px, py)
            table = np.zeros((n_y, n_x))
            populated = pd.DataFrame(
                {
                    "px": px[ok],
                    "py": py[ok],
                    "z": self.spec["specz"].to_numpy()[ok],
                }
            )
            for pixel, galaxies in populated.groupby(["px", "py"]):
                x, y = typing.cast(tuple[int, int], pixel)
                table[y, x] = np.percentile(galaxies.z.to_numpy(), key)
            self._ceilings[key] = table
        return self._ceilings[key]


def grid_selection(
    raw: pd.DataFrame,
    redshift: npt.ArrayLike,
    *,
    grid: HSCGrid | None = None,
    magnitude: str = "I",
    color: tuple[str, str] = ("G", "Z"),
    seed: int = 12345,
    scaling_factor: float = 1.588,
    color_redshift_cut: bool = True,
    percentile_cut: float = 99.0,
    redshift_cut: float = 100.0,
) -> tuple[_typing.BoolArray, pd.DataFrame]:
    """Selects the galaxies a HSC-like spectroscopic campaign would target.

    That is, which galaxies such a campaign would have got a redshift for.

    Args:
        raw: Photometry, with the magnitude and the two colour bands among its
            columns, in mag. Any frame will do -- it need not have come from
            :func:`~lazy.datasets.fetch_dc1`.
        redshift: True redshifts, one per row, shape (n_galaxies,). The
            selection uses them: that is the point, and it is why the result
            cannot be reproduced from photometry alone.
        grid: The selection function. Defaults to the cached HSC grid,
            downloading it on first use (:func:`lazy.datasets.fetch_hsc_grid`).
        magnitude: Column name for the grid's magnitude axis. The default is
            the LSST column matching HSC's ``i``.
        color: The two bands whose difference is the grid's colour axis. The
            default is the LSST pair matching HSC's ``g - z``.
        seed: Seeds the per-pixel subsampling only. RAIL's default is kept so
            that the same catalogue gives the same training set here and
            there.
        scaling_factor: The per-pixel keep fraction is
            ``ratio * scaling_factor``, capped at keeping everything, and
            applied only when ``color_redshift_cut`` is on.
        color_redshift_cut: Whether to apply the per-pixel redshift ceiling.
        percentile_cut: Which percentile of the HSC spectroscopic redshifts in
            the pixel sets that ceiling, between 0 and 100.
        redshift_cut: A hard redshift ceiling on top of the per-pixel one.

    Returns:
        A tuple (keep, diagnostics). ``keep`` is a boolean mask of shape
        (n_galaxies,): the biased training sample. ``diagnostics`` holds, per
        galaxy, why: ``ratio`` (its pixel's spectroscopic success rate, 0 off
        the grid), ``z_ceiling`` (the redshift above which it could not have
        been targeted) and ``survives_z_cut``. Everything the selection knew,
        for a model that tries to estimate it back.

    Raises:
        OSError: If ``grid`` is None and the HSC grid is not cached and its
            download fails or does not match the checksum (see
            :func:`lazy.datasets.fetch_hsc_grid`).

    Examples:
        >>> grid = HSCGrid(
        ...     ratios=np.full((2, 2), 0.5),
        ...     x_edges=np.array([20.0, 22.0, 24.0]),
        ...     y_edges=np.array([0.0, 1.0, 2.0]),
        ...     spec=pd.DataFrame(
        ...         {"mag": [21.0], "color": [0.5], "specz": [0.4]}
        ...     ),
        ... )
        >>> raw = pd.DataFrame(
        ...     {"I": [21.0] * 10, "G": [21.5] * 10, "Z": [21.0] * 10}
        ... )
        >>> keep, diagnostics = grid_selection(
        ...     raw, np.full(10, 0.3), grid=grid, color_redshift_cut=False
        ... )
        >>> int(keep.sum()), float(diagnostics["ratio"][0])
        (5, 0.5)
    """
    if grid is None:
        # Fetching the grid is a datasets-module concern, and datasets imports
        # this module.
        from lazy import datasets  # noqa: PLC0415 - avoids an import cycle.

        grid = datasets.fetch_hsc_grid()
    blue, red = color
    missing = [
        name for name in (magnitude, blue, red) if name not in raw.columns
    ]
    if missing:
        raise KeyError(f"raw photometry is missing columns: {missing}")
    redshift = np.asarray(redshift, dtype=float)
    if len(redshift) != len(raw):
        raise ValueError(
            f"raw has {len(raw)} rows but redshift has {len(redshift)}"
        )

    n = len(redshift)
    px, py = grid.pixel(
        raw[magnitude].to_numpy(dtype=float),
        raw[blue].to_numpy(dtype=float) - raw[red].to_numpy(dtype=float),
    )
    inside = grid.inside(px, py)

    ratio = np.zeros(n)
    ratio[inside] = grid.ratios[py[inside], px[inside]]
    ceiling = np.full(n, 99.0)
    if color_redshift_cut:
        ceiling[inside] = grid.max_specz(percentile_cut)[py[inside], px[inside]]
    survives = (
        (redshift <= ceiling) & (redshift <= redshift_cut) & (redshift > 0)
    )
    keep = _subsample_by_ratio(
        ratio,
        survives,
        factor=scaling_factor if color_redshift_cut else 1.0,
        rng=np.random.default_rng(seed),
    )
    diagnostics = pd.DataFrame(
        {"ratio": ratio, "z_ceiling": ceiling, "survives_z_cut": survives}
    )
    return keep, diagnostics


def _subsample_by_ratio(
    ratio: _typing.FloatArray,
    survives: _typing.BoolArray,
    *,
    factor: float,
    rng: np.random.Generator,
) -> _typing.BoolArray:
    """Keeps a random ``ratio * factor`` share of the survivors of each ratio.

    Args:
        ratio: Each galaxy's pixel success rate, shape (n_galaxies,).
        survives: Which galaxies passed the redshift cuts, shape
            (n_galaxies,).
        factor: Multiplies each ratio; the share is capped at keeping every
            survivor.
        rng: The source of the random draws.

    Returns:
        The kept galaxies, a mask of shape (n_galaxies,).
    """
    # Group the survivors by ratio once. A stable sort leaves each group's row
    # numbers increasing, so the groups -- and therefore every draw made from
    # them below -- are exactly what testing each ratio against every row
    # gives.
    survivors = np.flatnonzero(survives)
    survivors = survivors[np.argsort(ratio[survivors], kind="stable")]
    values = np.unique(ratio)
    starts = np.searchsorted(ratio[survivors], values, side="left")
    stops = np.searchsorted(ratio[survivors], values, side="right")

    keep = np.zeros(len(ratio), dtype=bool)
    for value, start, stop in zip(values, starts, stops, strict=True):
        if value <= 0:
            continue
        candidates = survivors[start:stop]
        number = len(candidates) * value
        if number * factor <= len(candidates):
            number *= factor
        else:
            number = float(len(candidates))
        extra = 0
        # RAIL keeps the fractional galaxy with probability `value`.
        if int(number) != number and rng.uniform() <= value:
            extra = 1
        count = min(int(np.floor(number)) + extra, len(candidates))
        keep[rng.permutation(candidates)[:count]] = True
    return keep


#: The magnitude and redshift bins :func:`selection_summary` reports in, chosen
#: for DC1's ``i < 25.3``, ``0 < z < 2`` footprint.
DEFAULT_MAGNITUDE_BINS = (16.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.3)
DEFAULT_REDSHIFT_BINS = (0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0)


def selection_summary(
    keep: npt.ArrayLike,
    magnitude: npt.ArrayLike,
    redshift: npt.ArrayLike,
    *,
    magnitude_bins: Sequence[float] = DEFAULT_MAGNITUDE_BINS,
    redshift_bins: Sequence[float] = DEFAULT_REDSHIFT_BINS,
) -> pd.DataFrame:
    """Tabulates what fraction of each magnitude and redshift bin was kept.

    The shape of the bias in two columns: the selected fraction falls by more
    than an order of magnitude from the bright bins to the faint ones, and
    again towards high redshift.

    Args:
        keep: The selection mask, shape (n_galaxies,).
        magnitude: Magnitudes in mag, shape (n_galaxies,).
        redshift: Redshifts, shape (n_galaxies,).
        magnitude_bins: Increasing magnitude bin edges, in mag.
        redshift_bins: Increasing redshift bin edges.

    Returns:
        One row per bin, magnitude bins first: ``bin`` (its label),
        ``n_all``, ``n_selected`` and ``fraction``.

    Examples:
        >>> keep = np.array([True, False, False, True])
        >>> summary = selection_summary(
        ...     keep,
        ...     np.array([19.0, 19.0, 24.5, 24.5]),
        ...     np.array([0.1, 0.2, 0.1, 0.2]),
        ... )
        >>> bright = summary.bin == "i in [16.0, 20.0)"
        >>> float(summary.loc[bright, "fraction"].iloc[0])
        0.5
    """
    keep = np.asarray(keep, dtype=bool)
    magnitude = np.asarray(magnitude, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    rows = []
    for label, values, edges in (
        ("i", magnitude, magnitude_bins),
        ("z", redshift, redshift_bins),
    ):
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            in_bin = (values >= low) & (values < high)
            rows.append(
                {
                    "bin": f"{label} in [{low}, {high})",
                    "n_all": int(in_bin.sum()),
                    "n_selected": int((in_bin & keep).sum()),
                }
            )
    table = pd.DataFrame(rows)
    table["fraction"] = table.n_selected / table.n_all.clip(lower=1)
    return table
