# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Color-matched nearest neighbors (Graham et al. 2018, 2020).

For each test galaxy, CMNN takes the training galaxies whose colors match
within the test galaxy's errors: the Mahalanobis distance
``D = sum_k (c_train,k - c_test,k)**2 / dc_test,k**2`` over the colors, with
the test galaxy's color errors, must not exceed the chi-squared percent-point
function at ``ppf`` with one degree of freedom per color. When fewer than
``min_n`` training galaxies match, the ``min_n`` nearest are used instead.
The original code (github.com/dirac-institute/CMNN_Photoz_Estimator) returns
a point estimate drawn from this subset and the subset's standard deviation,
and RAIL's port returns a single Gaussian and leaves out the square root of
the u-g color error. Following Graham's entry in the Data Challenge (Schmidt
et al. 2020), we instead keep the whole subset: the distribution is the
weighted histogram of the subset's redshifts, smoothed by a Gaussian kernel
and integrated exactly over the bins of the common grid.

The search covers the original code's options as well: the magnitude
pseudo-prior (only training galaxies within a window of i-band percentile
around the test galaxy's), the color pre-cut (within 0.3 mag in g-r and r-i),
and how the neighbors are weighted (by a power of 1/D, so that 0 is the plain
histogram and 1 the Data Challenge's inverse-distance weights). It can also
add the i-band magnitude as a sixth distance coordinate, with its error
broadened by a scale we fit, since the photometric error alone would only
match galaxies of the same magnitude. The kernel's bandwidth, fixed or
proportional to the spread of the subset, is a post-processing choice. We do
not inflate the width of a filled subset as the original code does, since
that made the cross-validated loss worse.

Typical usage example:

  model = METHOD.fit(params, train, z_train, seed=0)
  masses = METHOD.masses(METHOD.predict(model, test), post)
"""

from __future__ import annotations

from collections.abc import Mapping
import concurrent.futures
import dataclasses
import os
from typing import Any

from lazy import datasets
import numpy as np
import numpy.typing as npt
import optuna
from scipy import fft
from scipy import stats
import threadpoolctl

from fm4pz import config
from fm4pz import cv
from fm4pz import features
from fm4pz import scoring

__all__ = ["METHOD", "Cmnn", "Neighbors"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Most neighbors kept per test galaxy. A larger subset is thinned evenly in
#: training-row order, which keeps its redshift distribution.
MAX_NEIGHBORS = 2_000

#: Half-width of the color pre-cut on g-r and r-i (Graham et al. 2018), mag.
COLOR_CUT = 0.3

#: Bounds on any kernel bandwidth, in redshift.
MIN_BANDWIDTH = 0.001
MAX_BANDWIDTH = 0.1

# Test rows times training rows held in one distance block (32 MB).
_BLOCK = 1 << 22
# Order of the Taylor series in each neighbor's offset from its bin center.
_TAYLOR = 3
# Rows turned into masses at once.
_MASS_ROWS = 1_024
# Length of the periodic grid the kernel is applied on: the 4,000 bins and
# 1,400 empty ones, seven of the widest kernels, so that no kernel wraps
# around onto the grid.
_FFT_BINS = 5_400
# Predictions of at most this many rows keep their spectra (21.6 kB per row)
# once masses is called on them a second time, as cross-validation does for
# every post-processing choice.
_CACHE_ROWS = 20_000


def _threads() -> int:
    """Threads this process may use: ``$FM4PZ_THREADS`` or 1."""
    return max(1, int(os.environ.get("FM4PZ_THREADS", "1")))


def _inputs(
    catalog: datasets.Catalog, params: Mapping[str, Any]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The distance coordinates, their errors, and the i-band magnitude.

    Args:
        catalog: A DC1 catalog.
        params: The fit parameters.

    Returns:
        A tuple (coordinates, errors, i_mag) of shapes (n, d), (n, d) and
        (n,), in mag. The coordinates are the five adjacent colors and, for
        the feature set ``colors_i``, the i-band magnitude, whose error is
        broadened in quadrature by ``i_scale``.

    Raises:
        ValueError: If the feature set is unknown.
    """
    color, color_err = features.colors(catalog)
    raw = catalog.raw
    i_mag = raw["I"].to_numpy(np.float64)
    if params["feature_set"] == "colors":
        return color, color_err, i_mag
    if params["feature_set"] == "colors_i":
        i_err = np.hypot(raw["IERR"].to_numpy(np.float64), params["i_scale"])
        return (
            np.hstack([color, i_mag[:, None]]),
            np.hstack([color_err, i_err[:, None]]),
            i_mag,
        )
    raise ValueError(f"unknown feature set: {params['feature_set']=}")


@dataclasses.dataclass(frozen=True)
class _Model:
    """A fitted CMNN, which is the training set and the settings.

    Attributes:
        params: The fit parameters.
        coords: Training coordinates, shape (n_train, d), in mag.
        i_mag: Training i-band magnitudes, shape (n_train,).
        i_sorted: The same, sorted.
        redshift: Training redshifts, shape (n_train,).
        center: Mean of the coordinates, shape (d,).
        design: Centered coordinates squared and as they are, shape
            (n_train, 2 d), so that one matrix product gives the distances.
    """

    params: dict[str, Any]
    coords: FloatArray
    i_mag: FloatArray
    i_sorted: FloatArray
    redshift: npt.NDArray[np.float32]
    center: FloatArray
    design: FloatArray


@dataclasses.dataclass(frozen=True)
class Neighbors:
    """The color-matched subsets of n test galaxies, in CSR form.

    Each neighbor costs 8 bytes. At the full training size and a ppf of 0.95
    the test set holds 474 neighbors per galaxy on average, or 1.5 GB for
    all 390,990 galaxies, and 572 (1.8 GB) at a ppf of 0.995.

    Attributes:
        indptr: Galaxy ``k`` owns entries ``indptr[k]:indptr[k + 1]``,
            shape (n + 1,).
        redshift: Each neighbor's redshift, shape (n_entries,).
        weight: Each neighbor's weight, summing to one per galaxy, shape
            (n_entries,).
        n_matched: Training galaxies within the threshold, before thinning
            or filling, shape (n,).
        cache: Spectra kept between calls of :meth:`Cmnn.masses`.
    """

    indptr: IntArray
    redshift: npt.NDArray[np.float32]
    weight: npt.NDArray[np.float32]
    n_matched: npt.NDArray[np.int32]
    cache: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    def __len__(self) -> int:
        """The number of test galaxies."""
        return len(self.indptr) - 1


def _magnitude_window(
    model: _Model, i_mag: FloatArray, width: float
) -> npt.NDArray[np.bool_]:
    """The magnitude pseudo-prior of Graham et al. (2018).

    A test galaxy may only match training galaxies whose i-band magnitude
    lies within ``width`` in percentile of its own, the window being shifted
    inside [0, 1] at either end.

    Args:
        model: The fitted model.
        i_mag: Test i-band magnitudes, shape (n,).
        width: Half-width of the window, as a fraction of the training set.

    Returns:
        Which training galaxies each test galaxy may match, shape
        (n, n_train).
    """
    n_train = len(model.i_sorted)
    rank = np.searchsorted(model.i_sorted, i_mag) / n_train
    low = np.clip(rank - width, 0.0, max(0.0, 1.0 - 2.0 * width))
    high = np.minimum(low + 2.0 * width, 1.0)
    last = n_train - 1
    i_low = model.i_sorted[np.minimum(np.rint(low * n_train), last).astype(int)]
    i_high = model.i_sorted[
        np.minimum(np.rint(high * n_train), last).astype(int)
    ]
    return (model.i_mag[None, :] >= i_low[:, None]) & (
        model.i_mag[None, :] <= i_high[:, None]
    )


def _distances(
    model: _Model, coords: FloatArray, errors: FloatArray, i_mag: FloatArray
) -> FloatArray:
    """Mahalanobis distances of a block of test galaxies, after the pre-cuts.

    Args:
        model: The fitted model.
        coords: Test coordinates, shape (b, d).
        errors: Their errors, shape (b, d).
        i_mag: Test i-band magnitudes, shape (b,).

    Returns:
        Distances to every training galaxy, shape (b, n_train), infinite
        where a pre-cut excludes the pair.
    """
    params = model.params
    # D = sum_k w_k (x_k - y_k)**2 expanded, so that it is a matrix product.
    weight = 1.0 / errors**2
    shifted = coords - model.center
    query = np.hstack([weight, -2.0 * weight * shifted])
    distance = query @ model.design.T
    distance += np.sum(weight * shifted**2, axis=1, keepdims=True)
    np.maximum(distance, 0.0, out=distance)
    allowed = None
    if params["ppmag"]:
        allowed = _magnitude_window(model, i_mag, params["ppmag_width"])
    if params["ppclr"]:
        near = np.ones(distance.shape, dtype=bool)
        for k in (1, 2):  # g-r and r-i.
            near &= (
                np.abs(model.coords[None, :, k] - coords[:, None, k])
                <= COLOR_CUT
            )
        allowed = near if allowed is None else allowed & near
    if allowed is not None:
        # A galaxy whose pre-cuts leave fewer than min_n candidates (the
        # original code returns no estimate) is matched without them.
        allowed[allowed.sum(axis=1) < params["min_n"]] = True
        distance[~allowed] = np.inf
    return distance


def _select(
    distance: FloatArray, threshold: float, min_n: int
) -> tuple[IntArray, IntArray, IntArray]:
    """The color-matched subsets of a block of test galaxies.

    Args:
        distance: Distances, shape (b, n_train).
        threshold: The largest distance of a match.
        min_n: The smallest subset; a galaxy with fewer matches takes its
            ``min_n`` nearest neighbors instead.

    Returns:
        A tuple (rows, cols, n_matched): the test and training row of every
        neighbor, sorted by test row, and the number of matches of every
        test galaxy.
    """
    min_n = min(min_n, distance.shape[1])
    inside = distance <= threshold
    n_matched = inside.sum(axis=1)
    full = n_matched >= min_n
    # Enough matches: all of them, thinned evenly beyond MAX_NEIGHBORS.
    rows, cols = np.nonzero(inside[full])
    rows = np.flatnonzero(full)[rows]
    count = n_matched[rows]
    position = np.arange(len(rows)) - np.searchsorted(rows, rows)
    cap = np.minimum(count, MAX_NEIGHBORS)
    keep = (position + 1) * cap // count > position * cap // count
    rows, cols = rows[keep], cols[keep]
    # Too few matches: the min_n nearest.
    short = np.flatnonzero(~full)
    if short.size:
        nearest = np.argpartition(distance[short], min_n - 1, axis=1)
        rows = np.concatenate([rows, np.repeat(short, min_n)])
        cols = np.concatenate([cols, nearest[:, :min_n].ravel()])
        order = np.argsort(rows, kind="stable")
        rows, cols = rows[order], cols[order]
    return rows, cols, n_matched


def _block(
    model: _Model, coords: FloatArray, errors: FloatArray, i_mag: FloatArray
) -> tuple[IntArray, IntArray, FloatArray, IntArray]:
    """The neighbors of one block of test galaxies.

    Args:
        model: The fitted model.
        coords: Test coordinates, shape (b, d).
        errors: Their errors, shape (b, d).
        i_mag: Test i-band magnitudes, shape (b,).

    Returns:
        A tuple (rows, cols, distance, n_matched): for every neighbor, sorted
        by test row, the test row, the training row and the Mahalanobis
        distance; and for every test row the number of matches.
    """
    distance = _distances(model, coords, errors, i_mag)
    threshold = stats.chi2.ppf(model.params["ppf"], coords.shape[1])
    rows, cols, n_matched = _select(
        distance, threshold, int(model.params["min_n"])
    )
    return rows, cols, distance[rows, cols], n_matched


def _bandwidths(raw: Neighbors, post: Mapping[str, Any]) -> FloatArray:
    """Each galaxy's kernel bandwidth.

    Args:
        raw: The predictions.
        post: ``{"kernel": "fixed", "bandwidth": h}`` for one bandwidth, or
            ``{"kernel": "spread", "scale": a}`` for ``a`` times the
            weighted standard deviation of the subset's redshifts times
            ``n_eff**-0.2`` (Silverman's rule, with Kish's effective number
            of neighbors).

    Returns:
        Bandwidths, shape (n,), in redshift, clipped to
        [MIN_BANDWIDTH, MAX_BANDWIDTH].

    Raises:
        ValueError: If the kernel is unknown.
    """
    n_rows = len(raw)
    if post["kernel"] == "fixed":
        bandwidth = np.full(n_rows, float(post["bandwidth"]))
    elif post["kernel"] == "spread":
        owner = np.repeat(np.arange(n_rows), np.diff(raw.indptr))
        weight = raw.weight.astype(np.float64)
        redshift = raw.redshift.astype(np.float64)
        mean = np.bincount(owner, weight * redshift, minlength=n_rows)
        var = np.bincount(
            owner, weight * (redshift - mean[owner]) ** 2, minlength=n_rows
        )
        n_eff = 1.0 / np.bincount(owner, weight**2, minlength=n_rows)
        bandwidth = post["scale"] * np.sqrt(var) * n_eff**-0.2
    else:
        raise ValueError(f"unknown kernel: {post['kernel']=}")
    return np.clip(bandwidth, MIN_BANDWIDTH, MAX_BANDWIDTH)


def _spectra(raw: Neighbors, start: int, stop: int) -> npt.NDArray[Any]:
    """Fourier transforms of galaxies' neighbor histograms, before the kernel.

    Each neighbor is a delta function at its redshift. We bin it to the
    common grid and carry its offset from the bin center by a Taylor series
    to third order (exact to better than 1e-4 of the peak for bandwidths
    above 0.001), and convolve with one bin's box, so that the smoothed
    density sampled at the bin centers is the mass in each bin.

    Args:
        raw: The predictions.
        start: First galaxy.
        stop: One past the last galaxy.

    Returns:
        The spectra, complex64, shape (stop - start, _FFT_BINS // 2 + 1).
    """
    width = scoring.BIN_WIDTH
    lo, hi = raw.indptr[start], raw.indptr[stop]
    n_rows = stop - start
    owner = np.repeat(np.arange(n_rows), np.diff(raw.indptr[start : stop + 1]))
    position = (raw.redshift[lo:hi].astype(np.float64) - config.Z_MIN) / width
    bins = np.clip(np.floor(position).astype(np.int64), 0, config.FINE_BINS - 1)
    offset = (position - bins - 0.5) * width
    index = owner * _FFT_BINS + bins
    freq = fft.rfftfreq(_FFT_BINS, d=width)
    derivative = (2j * np.pi * freq).astype(np.complex64)
    term = raw.weight[lo:hi].astype(np.float64)
    out = np.zeros((n_rows, len(freq)), dtype=np.complex64)
    for order in range(_TAYLOR + 1):
        if order:
            term = term * -offset / order
        hist = np.bincount(index, term, minlength=n_rows * _FFT_BINS)
        spectrum = fft.rfft(
            hist.reshape(n_rows, _FFT_BINS).astype(np.float32),
            axis=1,
            workers=_threads(),
        )
        out += spectrum * derivative**order if order else spectrum
    out *= np.sinc(freq * width).astype(np.float32)
    return out


def _smooth(raw: Neighbors, bandwidth: FloatArray) -> FloatArray:
    """Exact bin masses of each galaxy's weighted Gaussian mixture.

    Args:
        raw: The predictions.
        bandwidth: Each galaxy's bandwidth, shape (n,), in redshift.

    Returns:
        Masses on the common grid, shape (n, 4000), rows summing to one.
    """
    n_rows = len(raw)
    # Cache from the second call on, so a single call costs no memory.
    calls = raw.cache["calls"] = raw.cache.get("calls", 0) + 1
    cached = n_rows <= _CACHE_ROWS and calls > 1
    if cached and "spectra" not in raw.cache:
        raw.cache["spectra"] = _spectra(raw, 0, n_rows)
    freq = fft.rfftfreq(_FFT_BINS, d=scoring.BIN_WIDTH)
    shared = bool(np.all(bandwidth == bandwidth[0]))
    kernel = np.exp(-2.0 * (np.pi * freq * bandwidth[0]) ** 2)
    out = np.empty((n_rows, config.FINE_BINS))
    for start in range(0, n_rows, _MASS_ROWS):
        stop = min(start + _MASS_ROWS, n_rows)
        if cached:
            spectrum = raw.cache["spectra"][start:stop]
        else:
            spectrum = _spectra(raw, start, stop)
        if not shared:
            h = bandwidth[start:stop, None]
            kernel = np.exp(-2.0 * (np.pi * freq * h) ** 2)
        smooth = fft.irfft(
            spectrum * kernel.astype(np.float32),
            n=_FFT_BINS,
            axis=1,
            workers=_threads(),
        )
        out[start:stop] = smooth[:, : config.FINE_BINS]
    # Clip the round-off of the transforms and normalize, in place.
    np.maximum(out, 0.0, out=out)
    out /= np.maximum(out.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    return out


class Cmnn(cv.Method):
    """CMNN, with the color-matched subset's redshifts as the distribution."""

    name = "cmnn"

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Proposes the selection and weighting of the neighbors.

        Args:
            trial: The Optuna trial.

        Returns:
            The fit parameters.
        """
        params: dict[str, Any] = {
            "feature_set": trial.suggest_categorical(
                "feature_set", ["colors", "colors_i"]
            ),
        }
        if params["feature_set"] == "colors_i":
            params["i_scale"] = trial.suggest_float(
                "i_scale", 0.1, 3.0, log=True
            )
        params["ppf"] = trial.suggest_float("ppf", 0.3, 0.995)
        params["min_n"] = trial.suggest_int("min_n", 2, 100, log=True)
        params["power"] = trial.suggest_float("power", 0.0, 4.0)
        params["floor"] = trial.suggest_float("floor", 0.05, 20.0, log=True)
        params["ppmag"] = trial.suggest_categorical("ppmag", [False, True])
        if params["ppmag"]:
            params["ppmag_width"] = trial.suggest_float(
                "ppmag_width", 0.02, 0.5, log=True
            )
        params["ppclr"] = trial.suggest_categorical("ppclr", [False, True])
        return params

    def post_grid(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The kernel bandwidths, fixed and proportional to the spread.

        Args:
            params: The fit parameters (unused).

        Returns:
            Seventeen fixed bandwidths from 0.002 to 0.08 and nine scales of
            the spread from 0.1 to 1.6, both log-spaced.
        """
        del params  # The same choices suit every fit.
        fixed = [
            {"kernel": "fixed", "bandwidth": float(h)}
            for h in np.geomspace(0.002, 0.08, 17)
        ]
        spread = [
            {"kernel": "spread", "scale": float(a)}
            for a in np.geomspace(0.1, 1.6, 9)
        ]
        return fixed + spread

    def fit(
        self,
        params: Mapping[str, Any],
        train: datasets.Catalog,
        z_train: FloatArray,
        seed: int,
    ) -> _Model:
        """Stores the training set.

        Args:
            params: The fit parameters.
            train: The training catalog.
            z_train: Its redshifts, shape (n_train,).
            seed: Unused; CMNN is deterministic.

        Returns:
            The model.
        """
        del seed  # CMNN is deterministic.
        coords, _, i_mag = _inputs(train, params)
        center = coords.mean(axis=0)
        shifted = coords - center
        return _Model(
            params=dict(params),
            coords=coords,
            i_mag=i_mag,
            i_sorted=np.sort(i_mag),
            redshift=np.asarray(z_train, dtype=np.float32),
            center=center,
            design=np.hstack([shifted**2, shifted]),
        )

    def predict(self, model: _Model, catalog: datasets.Catalog) -> Neighbors:
        """Finds each test galaxy's color-matched subset.

        Blocks of test galaxies run on ``$FM4PZ_THREADS`` threads, each
        against the whole training set.

        Args:
            model: The fitted model.
            catalog: The test galaxies.

        Returns:
            The subsets and their weights.
        """
        params = model.params
        coords, errors, i_mag = _inputs(catalog, params)
        n_rows = len(coords)
        size = max(1, _BLOCK // len(model.redshift))
        starts = range(0, n_rows, size)

        def run(start: int) -> tuple[IntArray, IntArray, FloatArray, IntArray]:
            part = slice(start, start + size)
            return _block(model, coords[part], errors[part], i_mag[part])

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            with concurrent.futures.ThreadPoolExecutor(_threads()) as pool:
                blocks = list(pool.map(run, starts))
        counts = np.concatenate(
            [
                np.bincount(rows, minlength=len(n_matched))
                for rows, _, _, n_matched in blocks
            ]
        )
        cols = np.concatenate([block[1] for block in blocks])
        distance = np.concatenate([block[2] for block in blocks])
        weight = np.maximum(distance, params["floor"]) ** -params["power"]
        owner = np.repeat(np.arange(n_rows), counts)
        weight /= np.bincount(owner, weight, minlength=n_rows)[owner]
        return Neighbors(
            indptr=np.concatenate([[0], np.cumsum(counts)]),
            redshift=model.redshift[cols],
            weight=weight.astype(np.float32),
            n_matched=np.concatenate([block[3] for block in blocks]).astype(
                np.int32
            ),
        )

    def masses(self, raw: Neighbors, post: Mapping[str, Any]) -> FloatArray:
        """Smooths each galaxy's weighted neighbor redshifts.

        Args:
            raw: The predictions, for n galaxies.
            post: One kernel from :meth:`post_grid`.

        Returns:
            Masses on :data:`fm4pz.scoring.FINE_GRID`, shape (n, 4000).
        """
        return _smooth(raw, _bandwidths(raw, post))


METHOD = Cmnn()
