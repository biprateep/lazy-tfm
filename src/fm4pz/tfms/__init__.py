# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""The tabular foundation models, each a :class:`fm4pz.cv.Method`.

Every model runs through lazy-tfm's ``LazyModel`` and nothing else: we choose
its inputs and settings, and it returns densities on the common grid. A
foundation model learns nothing at fit (it reads the training set as its
context), so its hyperparameters are its inputs, its ensemble and its
preprocessing, which we choose by the same cross-validation as the
baselines' hyperparameters. The densities are those of ``predict_proba`` on
:data:`fm4pz.scoring.FINE_GRID`, which puts the exact probability of the
model's own distribution in each bin, drops what falls outside [0, 2] and
renormalizes each row, as :func:`fm4pz.scoring.normalize_masses` expects.
One post-processing choice needs no refit: raising the masses to a power
alpha and renormalizing them, the sharpening FlexZBoost applies to its
series. This lets the cross-validation correct a model whose distributions
are on average too broad or too narrow, while alpha = 1 leaves the model's
answer as it is.

A fitted model also records what it cost: the wall time of the fit and of
the predictions, the GPU it ran on, and the peak GPU memory PyTorch held.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import gc
import importlib.metadata
import json
import time
from typing import Any

import lazy
from lazy import datasets
from lazy import distributions
import numpy as np
import numpy.typing as npt
import optuna
import torch

from fm4pz import cv
from fm4pz import features
from fm4pz import scoring

__all__ = [
    "ALPHAS",
    "FEATURE_SETS",
    "Fitted",
    "Raw",
    "FoundationModel",
    "build_features",
    "common_model_params",
    "study_size",
]

FloatArray = npt.NDArray[np.float64]

#: The feature sets a foundation model chooses between, beyond
#: :data:`fm4pz.features.FEATURE_SETS`.
FEATURE_SETS: dict[str, str] = {
    "common12": features.FEATURE_SETS["common12"],
    "common12_logerrs": features.FEATURE_SETS["common12_logerrs"],
    "i_colors": features.FEATURE_SETS["i_colors"],
    "mags_errs": (
        "the six magnitudes and their errors (lazy's 'mag', DC1 FlexZBoost)"
    ),
    "all42": (
        "the six magnitudes, all fifteen colors and the errors of all "
        "twenty-one (lazy's 'all')"
    ),
}

#: The powers the masses are raised to before scoring; 1 changes nothing.
ALPHAS: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)


def build_features(
    catalog: datasets.Catalog, name: str
) -> npt.NDArray[np.float64]:
    """One feature set of a catalog, as a float64 array.

    Args:
        catalog: A DC1 catalog.
        name: One of :data:`FEATURE_SETS`.

    Returns:
        The features, shape (n_rows, n_features).
    """
    if name == "all42":
        return catalog.features("all").to_numpy(np.float64)
    return features.build(catalog, name)


@dataclasses.dataclass
class Fitted:
    """A fitted foundation model and the inputs it was fitted on.

    Attributes:
        model: The fitted ``lazy.LazyModel``; None once the method has
            fitted another, which frees its GPU memory.
        feature_set: Which features it reads.
    """

    model: lazy.LazyModel | None
    feature_set: str


class Raw:
    """Distributions predicted for some rows, and their masses once computed.

    Attributes:
        distribution: The model's native distributions, a
            ``lazy.distributions.Distribution``.
    """

    def __init__(self, distribution: distributions.Distribution) -> None:
        """Holds the distributions; the masses are computed when asked for.

        Args:
            distribution: What ``predict_distribution`` returned.
        """
        self.distribution: distributions.Distribution | None = distribution
        self._masses: npt.NDArray[np.float32] | None = None

    def masses(self) -> npt.NDArray[np.float32]:
        """The masses on the common grid, computed once.

        They are what ``predict_proba(X, y_grid=FINE_GRID)`` returns times
        the bin width: the exact probability in each bin, with what falls
        outside [0, 2] dropped and each row renormalized.

        Returns:
            Masses, float32 to halve the memory, shape (n, 4000).
        """
        if self._masses is None:
            assert self.distribution is not None  # Set until masses exist.
            density = self.distribution.on_grid(scoring.FINE_GRID)
            self._masses = (density * scoring.BIN_WIDTH).astype(np.float32)
            self.distribution = None
        return self._masses


def _synchronize() -> None:
    """Waits for the GPU, so that a wall time covers its work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def _out_of_memory_as_runtime_error() -> Iterator[None]:
    """Raises a GPU's out-of-memory error as a RuntimeError.

    A search records a trial that raises RuntimeError as failed and goes on
    (:func:`fm4pz.cv.run_study`). PyTorch's out-of-memory error is one, but
    TabPFN raises its own, which would end the search instead.

    Yields:
        Nothing; the body runs inside.

    Raises:
        RuntimeError: If the body ran out of GPU memory.
    """
    try:
        yield
    except Exception as error:
        if isinstance(error, RuntimeError) or (
            "out of memory" not in str(error).lower()
        ):
            raise
        raise RuntimeError(f"out of GPU memory: {error}") from error


def _json_safe(value: Any) -> Any:
    """``value`` as JSON can hold it, unknown objects as their strings."""
    return json.loads(json.dumps(value, default=str))


class FoundationModel(cv.Method, abc.ABC):
    """What the three foundation models share.

    A subclass names its backend and proposes its own settings; this class
    fits ``LazyModel`` with them, predicts masses on the common grid and
    sharpens them.

    Attributes:
        name: The method's name, as it appears in results.
        backend: lazy-tfm's name for the model.
        package: The upstream package whose version is recorded.
    """

    backend: str
    package: str
    on_gpu = True

    def __init__(self) -> None:
        """Starts with no compute used."""
        self._usage = {"fit_seconds": 0.0, "predict_seconds": 0.0}
        self._provenance: dict[str, Any] = {}
        self._last: Fitted | None = None

    @abc.abstractmethod
    def model_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The ``LazyModel`` arguments a setting stands for.

        Args:
            params: The fit parameters, as :meth:`suggest` returned them.

        Returns:
            Keyword arguments of ``lazy.LazyModel``, the backend's name and
            the seed excluded.
        """

    def post_grid(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Every sharpening power in :data:`ALPHAS`.

        Args:
            params: The fit parameters, which do not matter here.

        Returns:
            ``[{"alpha": alpha}, ...]``.
        """
        del params  # Unused.
        return [{"alpha": alpha} for alpha in ALPHAS]

    def fit(
        self,
        params: Mapping[str, Any],
        train: datasets.Catalog,
        z_train: FloatArray,
        seed: int,
    ) -> Fitted:
        """Fits ``LazyModel`` on the training rows, on the GPU.

        Args:
            params: The fit parameters.
            train: The training catalog.
            z_train: Its redshifts, shape (n_train,).
            seed: The model's ``random_state``.

        Returns:
            The fitted model.
        """
        # A search fits one fold after another, and the previous fold's
        # model, still held by the caller, would keep its GPU memory (and
        # PyTorch's cache would keep the rest) from the processes sharing
        # the GPU.
        if self._last is not None:
            self._last.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model = lazy.LazyModel(
            self.backend,
            device="cuda",
            random_state=seed,
            progress=False,
            **self.model_params(params),
        )
        with _out_of_memory_as_runtime_error():
            model.fit(
                build_features(train, params["feature_set"]),
                np.asarray(z_train, dtype=np.float64),
            )
            _synchronize()
        self._usage["fit_seconds"] += time.perf_counter() - start
        self._provenance = _json_safe(getattr(model, "provenance_", {}))
        self._last = Fitted(model=model, feature_set=params["feature_set"])
        return self._last

    def predict(self, model: Fitted, catalog: datasets.Catalog) -> Raw:
        """The model's distributions, not yet on the common grid.

        Putting them on the grid is CPU work, which :class:`Raw` defers, so
        that a production run can hand it to other processes while the GPU
        predicts the next rows.

        Args:
            model: What :meth:`fit` returned.
            catalog: The rows to predict.

        Returns:
            The distributions.
        """
        start = time.perf_counter()
        if model.model is None:
            raise ValueError("this model was released when another was fitted")
        with _out_of_memory_as_runtime_error():
            distribution = model.model.predict_distribution(
                build_features(catalog, model.feature_set)
            )
            _synchronize()
        self._usage["predict_seconds"] += time.perf_counter() - start
        return Raw(distribution)

    def masses(self, raw: Raw, post: Mapping[str, Any]) -> FloatArray:
        """Sharpens the masses by ``post["alpha"]`` and renormalizes them.

        Args:
            raw: What :meth:`predict` returned, for n rows.
            post: ``{"alpha": alpha}``, alpha > 0.

        Returns:
            Masses on :data:`fm4pz.scoring.FINE_GRID`, shape (n, 4000).
        """
        out = np.asarray(raw.masses(), dtype=np.float64)
        alpha = float(post.get("alpha", 1.0))
        if alpha != 1.0:
            out = out**alpha
        return scoring.normalize_masses(out)

    def resources(self) -> dict[str, Any]:
        """The compute a production run used, and what answered.

        Returns:
            Wall times of fit and prediction (s), the GPU's name, the peak
            GPU memory PyTorch allocated (GB), the model's provenance and
            the versions of the packages that ran it.
        """
        cuda = torch.cuda.is_available()
        return {
            "gpu": torch.cuda.get_device_name() if cuda else None,
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / 1e9 if cuda else None
            ),
            "gpu_fit_seconds": self._usage["fit_seconds"],
            "gpu_predict_seconds": self._usage["predict_seconds"],
            "provenance": self._provenance,
            "backend_versions": {
                name: importlib.metadata.version(name)
                for name in ("lazy-tfm", "torch", self.package)
            },
        }

    def suggest_common(
        self, trial: optuna.Trial, transforms: tuple[str, ...]
    ) -> dict[str, Any]:
        """Proposes the settings every foundation model has.

        Args:
            trial: The Optuna trial.
            transforms: The ``transforms`` values to choose between.

        Returns:
            ``feature_set``, ``n_estimators``, ``transforms`` and
            ``outlier`` (``"auto"`` or ``"none"``).
        """
        return {
            "feature_set": trial.suggest_categorical(
                "feature_set", list(FEATURE_SETS)
            ),
            "n_estimators": trial.suggest_categorical("n_estimators", [8, 16]),
            "transforms": trial.suggest_categorical(
                "transforms", list(transforms)
            ),
            "outlier": trial.suggest_categorical("outlier", ["auto", "none"]),
        }


def common_model_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """The ``LazyModel`` arguments of the shared settings.

    Args:
        params: Fit parameters holding ``n_estimators``, ``transforms`` and
            ``outlier``.

    Returns:
        ``n_estimators``, ``transforms`` and ``outlier_threshold``.
    """
    outlier = params["outlier"]
    return {
        "n_estimators": int(params["n_estimators"]),
        "transforms": params["transforms"],
        "outlier_threshold": None if outlier == "none" else outlier,
    }


def study_size(trial: optuna.Trial) -> int:
    """The training size of the study a trial belongs to.

    Studies are named ``<method>-<regime>-<size>``
    (:func:`fm4pz.cv.open_study`).

    Args:
        trial: The Optuna trial.

    Returns:
        The training size.
    """
    return int(trial.study.study_name.rsplit("-", 1)[1])
