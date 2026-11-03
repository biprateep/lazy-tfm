# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Times lazy's own prediction code on a CPU, without any network.

The models are the test stand-ins (``tests/lazy/standins.py``), sized like
the real ones: 5,000 buckets as TabPFN predicts, four bagged members whose
buckets differ, as a bagged TabPFN or LimiX pools, and 999 quantiles as
TabICL predicts. Their own work, a nearest-neighbour lookup, is cached after
the first call, so the timed calls are mostly lazy's: densities, point
estimates, quantiles, scores, ``evaluate`` and pooling.

Run it from the repository root::

    uv run python benchmarks/cpu.py               # this checkout
    uv run python benchmarks/cpu.py --against origin/main --repeats 7

``--against`` runs this checkout and the other commit in turn, each case in a
fresh process, and reports the median time of each and their ratio. Both
commits run this file and the current stand-ins; only ``lazy`` differs.
"""

import argparse
from collections.abc import Callable
import itertools
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from typing import Any

import numpy as np
import scipy
from scipy import stats
import sklearn
import threadpoolctl

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "lazy"))

import standins  # noqa: E402 - found through the path set just above.

from lazy import distributions  # noqa: E402 - after the path is set.
from lazy.models import ensemble  # noqa: E402 - after the path is set.

N_CONTEXT = 1_000
"""Context rows of each fit."""

N_QUERIES = 500
"""Rows each case predicts."""

CALLS = 5
"""Timed calls per process; a process reports their median."""

QUANTILES = (0.16, 0.5, 0.84)
"""The levels ``predict_quantiles`` is asked for."""


class _CachedNeighbours:
    """Finds each block of queries' neighbours once, for repeated calls."""

    def _neighbours(self, member: Any, X: np.ndarray) -> np.ndarray:
        """The stand-in's neighbours of ``X``, from the cache when seen."""
        cache = self.__dict__.setdefault("_found", {})
        key = (id(member), X.tobytes())
        if key not in cache:
            cache[key] = super()._neighbours(member, X)
        return cache[key]


class Buckets(_CachedNeighbours, standins.ScaffoldedHistogramStandIn):
    """A bar model with TabPFN's 5,000 buckets, placed by its own targets."""

    n_buckets = 5_000

    def _fit_group(
        self, X: np.ndarray, y: np.ndarray, group: Any
    ) -> dict[str, Any]:
        """Each group's buckets span its own targets, so bagged ones differ."""
        borders = np.linspace(y.min() - 1.0, y.max() + 1.0, self.n_buckets + 1)
        return {"members": self._members(X, y, group), "borders": borders}

    def _predict_group(
        self, handle: dict[str, Any], X: np.ndarray
    ) -> distributions.HistogramDistribution:
        """Counts each query's neighbours in the buckets, all rows at once."""
        borders = handle["borders"]
        masses = np.full((len(X), borders.size - 1), 1e-3)
        rows = np.arange(len(X))[:, None]
        for member in handle["members"]:
            neighbours = self._neighbours(member, X)
            bucket = np.searchsorted(borders, neighbours, side="right") - 1
            np.add.at(masses, (rows, np.clip(bucket, 0, borders.size - 2)), 1)
        return distributions.HistogramDistribution(borders, masses)


class Quantiles(_CachedNeighbours, standins.QuantileStandIn):
    """A quantile model with TabICL's 999 levels."""

    levels = np.linspace(0.001, 0.999, 999)
    normal_levels = stats.norm.ppf(levels)

    def _predict_group(
        self, handle: dict[str, Any], X: np.ndarray
    ) -> distributions.QuantileDistribution:
        """Normal quantiles at the neighbours' mean and spread, cheaply."""
        (member,) = handle["members"]
        neighbours = self._neighbours(member, X)
        spread = neighbours.std(axis=1, keepdims=True) + 1e-3
        locs = (
            neighbours.mean(axis=1, keepdims=True) + spread * self.normal_levels
        )
        return distributions.QuantileDistribution(self.levels, locs)


def _data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Context and query rows with a smooth target, shapes (n, 4) and (n,)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_CONTEXT + N_QUERIES, 4))
    y = X[:, 0] + 0.5 * np.sin(3 * X[:, 1]) + 0.1 * rng.normal(size=len(X))
    return X[:N_CONTEXT], y[:N_CONTEXT], X[N_CONTEXT:], y[N_CONTEXT:]


MODELS = ("buckets", "bagged", "quantiles", "pooled")
"""The models: one bar model, four bagged ones, a quantile model, and the
first and third pooled by ``LazyEnsembleModel``."""

METHODS = (
    "predict_proba",
    "predict",
    "predict_quantiles",
    "score",
    "evaluate",
)
"""The calls timed on each model."""

CASES = tuple(f"{m}.{x}" for m, x in itertools.product(MODELS, METHODS))
"""Every case, as ``"<model>.<method>"``."""


def _model(name: str) -> Any:
    """The named model, fitted on the context rows."""
    X, y, _, _ = _data(0)
    common = {"progress": False, "random_state": 0}
    if name == "pooled":
        model = ensemble.LazyEnsembleModel(
            [
                Buckets(n_estimators=1, **common),
                Quantiles(n_estimators=1, **common),
            ]
        )
    elif name == "bagged":
        model = Buckets(n_estimators=4, bag_size=600, **common)
    elif name == "buckets":
        model = Buckets(n_estimators=1, **common)
    else:
        model = Quantiles(n_estimators=1, **common)
    return model.fit(X, y)


def _call(case: str) -> Callable[[], object]:
    """The case's call, with its model fitted and its arguments bound."""
    name, method = case.split(".")
    model = _model(name)
    _, _, X, y = _data(0)
    if method in ("score", "evaluate"):
        return lambda: getattr(model, method)(X, y)
    if method == "predict_quantiles":
        return lambda: model.predict_quantiles(X, QUANTILES)
    return lambda: getattr(model, method)(X)


def _measure(call: Callable[[], object]) -> tuple[float, float]:
    """Median seconds of ``CALLS`` calls, after warm-ups, and peak memory.

    Returns:
        A tuple (seconds, peak_mib).
    """
    call()  # The first call's one-time costs count in neither figure.
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    seconds = []
    for _ in range(CALLS):
        start = time.perf_counter()
        call()
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds), peak / 2**20


def _setup() -> dict[str, Any]:
    """The machine, threads and versions a run measured on."""
    return {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "blas": [
            {key: pool[key] for key in ("internal_api", "num_threads")}
            for pool in threadpoolctl.threadpool_info()
        ],
        "thread_variables": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _commit(directory: pathlib.Path) -> str:
    """The checkout's commit, ending in ``-dirty`` if it has changes."""
    return subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
        cwd=directory,
    ).stdout.strip()


def _run_once(tree: pathlib.Path, case: str) -> tuple[float, float]:
    """One measurement of one case, in a fresh process importing ``tree``.

    Returns:
        A tuple (seconds, peak_mib).
    """
    environment = {**os.environ, "PYTHONPATH": str(tree / "src")}
    done = subprocess.run(
        [sys.executable, __file__, "--case", case],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if done.returncode:
        raise RuntimeError(f"{case} failed on {tree}:\n{done.stderr}")
    seconds, peak = json.loads(done.stdout)
    return seconds, peak


def _run(trees: dict[str, pathlib.Path], repeats: int) -> dict[str, Any]:
    """Every case on every tree, the trees taking turns within each repeat.

    Prints one row per case: each tree's median seconds and peak MiB, and
    with two trees the ratio of the first's median to the second's.
    """
    two = len(trees) == 2
    header = ["case".ljust(28)]
    header += [f"{name + ' s':>9s}" for name in trees]
    header += [" speed-up"] * two + [f"{name + ' MiB':>9s}" for name in trees]
    print(*header)
    results: dict[str, Any] = {}
    for case in CASES:
        runs: dict[str, list[tuple[float, float]]] = {
            name: [] for name in trees
        }
        for repeat in range(repeats):
            # Alternate which tree goes first, so neither always runs warm.
            order = list(trees.items())
            for name, tree in reversed(order) if repeat % 2 else order:
                runs[name].append(_run_once(tree, case))
        medians = [statistics.median(s for s, _ in v) for v in runs.values()]
        peaks = [max(p for _, p in v) for v in runs.values()]
        results[case] = {"runs": runs, "median_s": medians, "peak_mib": peaks}
        row = [case.ljust(28)] + [f"{m:9.4f}" for m in medians]
        if two:
            row.append(f"{medians[0] / medians[1]:8.2f}x")
        print(*row, *(f"{p:9.1f}" for p in peaks), flush=True)
    return results


def main() -> int:
    """Runs the benchmark; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--against", help="a commit to compare with")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=pathlib.Path, help="write results")
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(_measure(_call(args.case))))
        return 0
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    setup = _setup()
    print(json.dumps(setup, indent=1))
    with tempfile.TemporaryDirectory() as scratch:
        trees = {"head": ROOT}
        if args.against:
            trees = {"base": pathlib.Path(scratch) / "base", **trees}
        try:
            if args.against:
                add = [
                    "worktree",
                    "add",
                    "--detach",
                    trees["base"],
                    args.against,
                ]
                subprocess.run(["git", *add], check=True, cwd=ROOT)
            commits = {name: _commit(tree) for name, tree in trees.items()}
            print(json.dumps(commits, indent=1))
            results = _run(trees, args.repeats)
        finally:
            if args.against:
                # Not checked, so a failed removal cannot hide the real error;
                # git prunes the record once the directory is gone.
                remove = ["worktree", "remove", "--force", trees["base"]]
                subprocess.run(["git", *remove], check=False, cwd=ROOT)
    if args.json:
        record = {"setup": setup, "commits": commits, "results": results}
        args.json.write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
