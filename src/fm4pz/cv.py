# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Choosing a baseline's hyperparameters by cross-validation on training rows.

A baseline is a :class:`Method`. Its hyperparameters fall in two kinds: those
that need a refit (the regressor's settings, the feature set), which Optuna
proposes one trial at a time, and those that only post-process a fitted
model's raw predictions (FlexZBoost's sharpening, a kernel width, a scale on
the predicted widths), which every trial evaluates on a grid. A trial fits the
method on four folds of the training set and predicts the fifth, five times
over, and its value is the CDE loss on the common grid, averaged over the five
held-out folds, of its best post-processing choice; that choice is stored with
the trial. The test set is never seen.

Typical usage example:

  study = cv.run_study(method, split, n_trials=100, storage=path)
  best = cv.best_params(study)
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import pathlib
from typing import Any

from lazy import datasets
import numpy as np
import numpy.typing as npt
import optuna
from optuna.storages import journal
from sklearn import model_selection

from fm4pz import config
from fm4pz import scoring
from fm4pz import splits

__all__ = [
    "Method",
    "best_params",
    "cde_loss",
    "fold_rows",
    "open_study",
    "run_study",
]

FloatArray = npt.NDArray[np.float64]


class Method(abc.ABC):
    """A classical baseline, as cross-validation and production runs see it.

    Attributes:
        name: The method's name, as it appears in results.
        on_gpu: Whether it fits and predicts on a GPU, so that a production
            run scores in worker processes forked before the fit.
    """

    name: str
    on_gpu: bool = False

    @abc.abstractmethod
    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Proposes the hyperparameters that need a refit.

        Args:
            trial: The Optuna trial.

        Returns:
            The fit parameters, a JSON-serialisable mapping.
        """

    @abc.abstractmethod
    def post_grid(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The post-processing choices to evaluate for one fit.

        Args:
            params: The fit parameters.

        Returns:
            One mapping per choice; ``[{}]`` when there is none.
        """

    @abc.abstractmethod
    def fit(
        self,
        params: Mapping[str, Any],
        train: datasets.Catalog,
        z_train: FloatArray,
        seed: int,
    ) -> Any:
        """Fits the method.

        Args:
            params: The fit parameters.
            train: The training catalogue.
            z_train: Its redshifts, shape (n_train,).
            seed: Seed for the method's own randomness.

        Returns:
            The fitted model, in whatever form :meth:`predict` takes.
        """

    @abc.abstractmethod
    def predict(self, model: Any, catalog: datasets.Catalog) -> Any:
        """The raw predictions that post-processing turns into masses.

        Args:
            model: What :meth:`fit` returned.
            catalog: The rows to predict.

        Returns:
            The raw predictions, in whatever form :meth:`masses` takes.
        """

    def warm_starts(self) -> list[dict[str, Any]]:
        """Settings worth trying first, as Optuna parameters.

        A search starts with these (and with the best setting of the next
        smaller training size), so that its first trials are informed rather
        than drawn at random from the whole space.

        Returns:
            Mappings from each name :meth:`suggest` proposes to a value;
            empty by default.
        """
        return []

    def production_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The chosen setting as a production fit runs it.

        A method may fit its production runs more thoroughly than its
        search trials (GPz keeps the best of several initialisations);
        the default runs the chosen setting as it is.

        Args:
            params: The fit parameters the search chose.

        Returns:
            The fit parameters of a production run.
        """
        return dict(params)

    def resources(self) -> dict[str, Any]:
        """What a production run cost, beyond its wall times.

        A method that runs on a GPU reports the device and its peak memory
        here; the default reports nothing.

        Returns:
            A JSON-serializable mapping, added to the run's record when not
            empty; empty by default.
        """
        return {}

    @abc.abstractmethod
    def masses(self, raw: Any, post: Mapping[str, Any]) -> FloatArray:
        """Turns raw predictions into masses on the common grid.

        Args:
            raw: What :meth:`predict` returned, for n rows.
            post: One post-processing choice from :meth:`post_grid`.

        Returns:
            Masses on :data:`fm4pz.scoring.FINE_GRID`, shape (n, 4000).
        """


def cde_loss(z_true: npt.ArrayLike, masses: npt.ArrayLike) -> float:
    """The mean CDE loss on the common grid, quickly, for tuning.

    It equals the mean of :func:`fm4pz.scoring.score`'s ``cde`` (the density
    is constant in each bin), without the rest of that function's work.

    Args:
        z_true: True redshifts, shape (n,).
        masses: Masses on the common grid, shape (n, 4000).

    Returns:
        The mean CDE loss.
    """
    masses = scoring.normalize_masses(masses)
    z_true = np.asarray(z_true, dtype=np.float64)
    width = scoring.BIN_WIDTH
    index = np.clip(
        np.floor((z_true - config.Z_MIN) / width).astype(np.int64),
        0,
        masses.shape[1] - 1,
    )
    integral = np.einsum("ij,ij->i", masses, masses) / width
    at_truth = masses[np.arange(len(masses)), index] / width
    return float(np.mean(integral - 2.0 * at_truth))


def fold_rows(n_rows: int) -> list[tuple[npt.NDArray[np.int64], ...]]:
    """The five (fit, held-out) row pairs every study uses.

    Args:
        n_rows: Training rows.

    Returns:
        A list of tuples (fit_rows, held_out_rows).
    """
    folds = model_selection.KFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.CV_SEED
    )
    return [
        (np.asarray(fit), np.asarray(held))
        for fit, held in folds.split(np.zeros(n_rows))
    ]


@dataclasses.dataclass
class _Objective:
    """One study's objective: cross-validated CDE loss of a trial."""

    method: Method
    train: datasets.Catalog
    z_train: FloatArray

    def __call__(self, trial: optuna.Trial) -> float:
        params = self.method.suggest(trial)
        trial.set_user_attr("fit_params", params)
        losses: list[list[float]] = []
        posts: list[dict[str, Any]] | None = None
        for k, (fit_rows, held_rows) in enumerate(fold_rows(len(self.z_train))):
            model = self.method.fit(
                params,
                self.train.take(fit_rows),
                self.z_train[fit_rows],
                seed=k,
            )
            raw = self.method.predict(model, self.train.take(held_rows))
            posts = self.method.post_grid(params)
            losses.append(
                [
                    cde_loss(
                        self.z_train[held_rows], self.method.masses(raw, post)
                    )
                    for post in posts
                ]
            )
            running = np.mean(losses, axis=0)
            trial.report(float(running.min()), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean = np.mean(losses, axis=0)
        best = int(np.argmin(mean))
        assert posts is not None  # At least one fold ran.
        trial.set_user_attr("post", posts[best])
        trial.set_user_attr("fold_losses", [row[best] for row in losses])
        return float(mean[best])


def open_study(
    method: Method,
    split: splits.Split,
    *,
    storage: pathlib.Path,
    seed: int = 0,
    enqueue: Sequence[Mapping[str, Any]] = (),
) -> optuna.Study:
    """Creates or loads the search of one training set.

    Args:
        method: The baseline.
        split: The training set.
        storage: The journal file of the study, which several processes may
            share.
        seed: Seed of this process's sampler.
        enqueue: Settings to run first, as Optuna parameters; queued only if
            the study has no trials yet.

    Returns:
        The study.
    """
    storage.parent.mkdir(parents=True, exist_ok=True)
    backend = optuna.storages.JournalStorage(
        journal.JournalFileBackend(str(storage))
    )
    study = optuna.create_study(
        study_name=f"{method.name}-{split.regime}-{split.n_train}",
        storage=backend,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=1
        ),
        load_if_exists=True,
    )
    if not study.trials:
        for params in enqueue:
            study.enqueue_trial(dict(params), skip_if_exists=True)
    return study


def run_study(
    method: Method,
    split: splits.Split,
    *,
    n_trials: int,
    storage: pathlib.Path,
    seed: int = 0,
) -> optuna.Study:
    """Runs (or resumes) the hyperparameter search of one training set.

    Several processes may run the same study at once: the storage is an
    Optuna journal file, which they share. Settings queued by
    :func:`open_study` run first.

    Args:
        method: The baseline.
        split: The training set; its test set is not used.
        n_trials: Trials this call adds to the study.
        storage: The journal file of the study.
        seed: Seed of this process's sampler.

    Returns:
        The study.
    """
    study = open_study(method, split, storage=storage, seed=seed)
    objective = _Objective(method, split.train, split.train.redshift)
    # A trial whose fit fails numerically is recorded as failed, and the
    # search goes on, rather than one bad corner of the space ending it.
    study.optimize(
        objective,
        n_trials=n_trials,
        gc_after_trial=True,
        catch=(
            ArithmeticError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ),
    )
    return study


def best_params(study: optuna.Study) -> dict[str, Any]:
    """The best trial's fit parameters, post-processing and CV loss.

    Args:
        study: A finished study.

    Returns:
        ``{"params": ..., "post": ..., "cv_loss": ..., "fold_losses": ...,
        "n_trials": ...}``.
    """
    best = study.best_trial
    return {
        "params": dict(best.user_attrs["fit_params"]),
        "post": dict(best.user_attrs["post"]),
        "cv_loss": float(study.best_value),
        "fold_losses": list(best.user_attrs["fold_losses"]),
        "n_trials": len(study.trials),
    }
