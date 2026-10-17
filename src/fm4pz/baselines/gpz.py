# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""GPz, the sparse Gaussian process of Almosallam et al. (2016), as a baseline.

The fit calls the Python port of GPz that RAIL ships (pz-rail-gpz-v1, which
carries Hatfield's GPz_py3 as ``rail.estimation.algos._gpz_util``) directly,
not through the RAIL stage, so that every option is ours and the split into
fitting and validation rows follows the seed (RAIL draws it from NumPy's
global state). GPz learns the positions and shapes of ``n_basis`` radial basis
functions by maximising the marginal likelihood on the fitting rows, and keeps
the parameters of the iteration with the best likelihood on the validation
rows; the model it predicts with is never refit on all the rows. We stop the
optimiser after ``patience`` iterations without improvement on the validation
rows (the port's own stopping zeroes the gradient instead, which the current
SciPy L-BFGS-B does not take as convergence, so it ran to ``max_iter``).

The prediction for each object is a Gaussian, its mean and its variance (the
model's uncertainty plus the learned noise, which depends on the inputs when
``heteroscedastic`` is set). The only
post-processing is a calibration of its width, a scale ``s`` and a floor
``sigma0`` added in quadrature, chosen by cross-validation on training rows.

Typical usage example:

  model = METHOD.fit(DEFAULT_PARAMS, train, train.redshift, seed=0)
  raw = METHOD.predict(model, test)
  masses = METHOD.masses(raw, {"scale": 1.0, "floor": 0.0})
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import itertools
import os
from typing import Any
import warnings

from lazy import datasets
import numpy as np
import numpy.typing as npt
import optuna
from rail.estimation.algos import _gpz_util
import threadpoolctl

from fm4pz import cv
from fm4pz import features
from fm4pz import scoring

__all__ = [
    "BASIS_METHODS",
    "DEFAULT_PARAMS",
    "FEATURE_SETS",
    "FLOORS",
    "MAX_ITER",
    "METHOD",
    "SCALES",
    "Fitted",
    "GPz",
]

FloatArray = npt.NDArray[np.float64]

#: The covariance of the basis functions: global (G) or variable (V) across
#: them, and a single length scale (L), one per input (D, diagonal) or a full
#: matrix (C).
BASIS_METHODS: tuple[str, ...] = ("VC", "VD", "VL", "GC", "GD", "GL")

#: The feature sets GPz chooses between (see :mod:`fm4pz.features`).
FEATURE_SETS: tuple[str, ...] = ("mags_logerrs", "common12_logerrs")

#: Iterations of L-BFGS-B at most; the validation rows stop it much earlier.
MAX_ITER = 3_000

#: The scales of the predicted widths and the floors added in quadrature, in
#: redshift, that post-processing chooses between.
SCALES: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(21))
FLOORS: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)

#: RAIL's defaults (VC, 50 basis functions, heteroscedastic noise, PCA
#: decorrelation, a quarter of the rows for validation), with its
#: ``max_attempt`` as the patience.
DEFAULT_PARAMS: dict[str, Any] = {
    "features": "mags_logerrs",
    "method": "VC",
    "n_basis": 50,
    "heteroscedastic": True,
    "decorrelate": True,
    "joint": True,
    "train_frac": 0.75,
    "patience": 100,
}

# Rows predicted at once; the basis functions of a chunk of rows take
# 16,384 x (n_basis + d + 1) floats, under 70 MB for 500 basis functions.
_CHUNK = 16_384


def _threads() -> int:
    """The threads a fit may use, from ``$FM4PZ_THREADS`` (default 1)."""
    return int(os.environ.get("FM4PZ_THREADS", "1"))


class _QuietGP(_gpz_util.GP):
    """The port's GP, silent, stopping when the validation rows stop improving.

    Attributes:
        patience: Iterations without improvement on the validation rows
            after which the optimiser stops.
        best_iter: The iteration whose parameters the model keeps.
        best_valid: Their mean log-likelihood on the validation rows.
    """

    patience: int = 100
    best_iter: int = 0
    best_valid: float

    def callbackF(self, theta: FloatArray) -> None:  # noqa: N802 - the port's name.
        """Records the iteration and stops the optimiser when it is time.

        This replaces the port's callback, which prints every iteration. It
        keeps the port's rule for the best parameters: those of the latest
        iteration whose validation log-likelihood is at least the best so
        far.

        Args:
            theta: The parameters after this iteration (a copy).

        Raises:
            StopIteration: After ``patience`` iterations without improvement,
                which SciPy takes as the signal to stop.
        """
        if self.iter == 1 or self.validLL >= self.best_valid:
            self.best_theta = theta
            self.best_valid = self.validLL
            self.best_iter = self.iter
            self.attempts = 0
        else:
            self.attempts += 1
        self.iter += 1
        if self.attempts >= self.patience:
            raise StopIteration


def _fitting_rows(
    n_rows: int, fraction: float, seed: np.random.SeedSequence
) -> npt.NDArray[np.bool_]:
    """A random fraction of the rows, to fit on; the rest validate.

    Args:
        n_rows: Training rows.
        fraction: The fraction to fit on, in (0, 1).
        seed: Seed of the draw.

    Returns:
        A mask of shape (n_rows,), true for the rows to fit on.
    """
    order = np.random.default_rng(seed).permutation(n_rows)
    training = np.zeros(n_rows, dtype=bool)
    training[order[: round(fraction * n_rows)]] = True
    return training


def _recover(
    gp: _QuietGP,
    inputs: FloatArray,
    target: FloatArray,
    training: npt.NDArray[np.bool_],
) -> None:
    """Finishes a fit whose optimiser failed, from its best parameters.

    The port's ``train`` computes the weights of the best parameters only
    after the optimiser returns; when an iterate makes the posterior
    covariance singular, this does that last step instead.
    """
    decorrelated = gp.pca.transform(inputs) if gp.decorrelate else inputs
    residual = target - gp.muY - decorrelated @ gp.wL
    omega = np.ones_like(target)
    gp.best_w, gp.best_SIGMAi = gp.GPz(
        gp.best_theta,
        decorrelated,
        residual,
        omega,
        training,
        [],
        True,
    )


@dataclasses.dataclass
class Fitted:
    """A fitted GPz model and what it needs to predict.

    Attributes:
        gp: The port's model, with the best validation parameters.
        features: The feature set it was fitted on.
        n_iter: Iterations the optimiser ran.
        best_iter: The iteration whose parameters it keeps.
        valid_loglike: Their mean log-likelihood on the validation rows.
    """

    gp: _QuietGP
    features: str
    n_iter: int
    best_iter: int
    valid_loglike: float


class GPz(cv.Method):
    """GPz through RAIL's Python port, with a seeded validation split."""

    name = "gpz"

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Proposes the feature set and the settings of GPz.

        Args:
            trial: The Optuna trial.

        Returns:
            The fit parameters: ``features``, ``method``, ``n_basis``,
            ``heteroscedastic``, ``decorrelate``, ``joint``, ``train_frac``
            and ``patience``.
        """
        return {
            "features": trial.suggest_categorical("features", FEATURE_SETS),
            "method": trial.suggest_categorical("method", BASIS_METHODS),
            "n_basis": trial.suggest_int("n_basis", 10, 400, log=True),
            "heteroscedastic": trial.suggest_categorical(
                "heteroscedastic", (True, False)
            ),
            "decorrelate": trial.suggest_categorical(
                "decorrelate", (True, False)
            ),
            "joint": trial.suggest_categorical("joint", (True, False)),
            "train_frac": trial.suggest_float(
                "train_frac", 0.6, 0.9, step=0.05
            ),
            "patience": trial.suggest_int("patience", 20, 300, log=True),
        }

    def post_grid(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Every pair of a width scale and a width floor.

        Args:
            params: The fit parameters (unused).

        Returns:
            One ``{"scale": s, "floor": sigma0}`` per pair, 126 in all.
        """
        del params  # Unused.
        return [
            {"scale": scale, "floor": floor}
            for floor, scale in itertools.product(FLOORS, SCALES)
        ]

    def fit(
        self,
        params: Mapping[str, Any],
        train: datasets.Catalog,
        z_train: FloatArray,
        seed: int,
    ) -> Fitted:
        """Fits GPz, holding out a seeded fraction of the rows to stop it.

        Args:
            params: The fit parameters, as :meth:`suggest` returns them.
            train: The training catalogue.
            z_train: Its redshifts, shape (n_train,).
            seed: Seed of the validation split and of the initial basis
                functions.

        Returns:
            The fitted model.
        """
        inputs = features.build(train, params["features"])
        target = np.asarray(z_train, dtype=np.float64).reshape(-1, 1)
        split_seed, init_seed = np.random.SeedSequence(seed).spawn(2)
        training = _fitting_rows(len(target), params["train_frac"], split_seed)
        gp = _QuietGP(
            params["n_basis"],
            method=params["method"],
            joint=params["joint"],
            heteroscedastic=params["heteroscedastic"],
            decorrelate=params["decorrelate"],
            seed=init_seed,
        )
        gp.patience = params["patience"]
        with (
            threadpoolctl.threadpool_limits(limits=_threads()),
            warnings.catch_warnings(),
            np.errstate(all="ignore"),
        ):
            # The port passes L-BFGS-B a "disp" option it no longer takes.
            warnings.filterwarnings("ignore", message="Unknown solver options")
            try:
                gp.train(
                    inputs,
                    target,
                    omega=np.ones_like(target),
                    training=training,
                    validation=~training,
                    maxIter=MAX_ITER,
                    maxAttempts=np.inf,
                )
            except np.linalg.LinAlgError:
                if gp.not_initalized:
                    raise
                _recover(gp, inputs, target, training)
        return Fitted(
            gp=gp,
            features=params["features"],
            n_iter=gp.iter - 1,
            best_iter=gp.best_iter,
            valid_loglike=float(gp.best_valid),
        )

    def predict(self, model: Fitted, catalog: datasets.Catalog) -> FloatArray:
        """The mean and standard deviation of each object's Gaussian.

        Args:
            model: What :meth:`fit` returned.
            catalog: The rows to predict.

        Returns:
            Shape (n_rows, 2): the mean and the standard deviation, in
            redshift.
        """
        inputs = features.build(catalog, model.features)
        out = np.empty((len(inputs), 2))
        with (
            threadpoolctl.threadpool_limits(limits=_threads()),
            np.errstate(all="ignore"),
        ):
            for start in range(0, len(inputs), _CHUNK):
                rows = slice(start, start + _CHUNK)
                mean, variance, *_ = model.gp.predict(inputs[rows])
                out[rows, 0] = mean[:, 0]
                out[rows, 1] = np.sqrt(variance[:, 0])
        return out

    def masses(self, raw: FloatArray, post: Mapping[str, Any]) -> FloatArray:
        """Bin masses of the Gaussians, their widths calibrated.

        The width becomes ``hypot(scale * sigma, floor)``, and the masses are
        differences of the Gaussian CDF at the bin edges, truncated to
        [0, 2].

        Args:
            raw: What :meth:`predict` returned, shape (n, 2).
            post: ``{"scale": s, "floor": sigma0}``.

        Returns:
            Masses on :data:`fm4pz.scoring.FINE_GRID`, shape (n, 4000).
        """
        width = np.hypot(post["scale"] * raw[:, 1], post["floor"])
        return scoring.gaussian_masses(raw[:, 0], width)


METHOD = GPz()
