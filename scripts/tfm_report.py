# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The foundation models' scaling curves, in tables
#
# For every model, regime and training size: the setting the
# cross-validation chose, its CV loss, and the test-set CDE loss on the
# common grid and on the DC1 grid, as mean and standard deviation over the
# five seeds, with what a production run cost on its GPU. Reads
# `$FM4PZ_RESULTS` (default `results/`) and writes `tfm_report.md` there.

# %% Imports
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
import json
import pathlib
from typing import Any

import numpy as np

from fm4pz import config
from fm4pz import tune

# %% [markdown]
# ## Configuration

# %% Configuration
METHODS = ("tabpfn", "limix", "kumo")
SIZES = {
    "representative": config.REPRESENTATIVE_SIZES,
    "biased": config.BIASED_SIZES,
}
RESULTS = config.results_dir()

# %% [markdown]
# ## Reading the records
#
# A search's exported setting, and the records of the five production runs
# at that size; a missing run leaves its seed out.


# %% Readers
def read_best(method: str, regime: str, size: int) -> dict[str, Any] | None:
    """The exported setting of a search, or None if it has none."""
    try:
        return json.loads(
            tune.latest_best_path(method, regime, size).read_text()
        )
    except FileNotFoundError:
        return None


def read_runs(method: str, regime: str, size: int) -> list[dict[str, Any]]:
    """The production records of one size, in seed order."""
    paths = [
        RESULTS / "runs" / method / f"{regime}_n{size}_s{seed}.json"
        for seed in config.SEEDS
    ]
    return [json.loads(path.read_text()) for path in paths if path.exists()]


# %% [markdown]
# ## One row per training size
#
# The setting is written compactly: feature set, members, transforms,
# outlier clip, then the model's own choices and the sharpening power.


# %% Row builder
def setting(best: dict[str, Any]) -> str:
    """The chosen setting as one short string."""
    params = dict(best["params"])
    parts = [
        params.pop("feature_set"),
        f"n={params.pop('n_estimators')}",
        f"tf={params.pop('transforms')}",
        f"clip={params.pop('outlier')}",
    ]
    parts += [f"{key}={value}" for key, value in sorted(params.items())]
    parts.append(f"alpha={best['post'].get('alpha', 1.0)}")
    return ", ".join(parts)


def mean_std(values: list[float]) -> str:
    """Mean and standard deviation over seeds, as text."""
    if not values:
        return "-"
    spread = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{np.mean(values):.3f} ± {spread:.3f}"


def row(method: str, regime: str, size: int) -> str:
    """One markdown table row."""
    best = read_best(method, regime, size)
    runs = read_runs(method, regime, size)
    if best is None:
        return f"| {size} | (no search) |" + " |" * 7
    cde = [run["metrics"]["cde_loss"] for run in runs]
    dc1 = [run["metrics"]["cde_loss_dc1"] for run in runs]
    fit = [run["fit_seconds"] for run in runs]
    pred = [run["predict_seconds"] for run in runs]
    memory = [run["resources"]["peak_gpu_memory_gb"] for run in runs]
    cells = [
        str(size),
        setting(best),
        f"{best['cv_loss']:.3f} ({best['n_trials']})",
        mean_std(cde),
        mean_std(dc1),
        str(len(runs)),
        f"{np.mean(fit):.0f}" if fit else "-",
        f"{np.mean(pred):.0f}" if pred else "-",
        f"{max(memory):.1f}" if memory else "-",
    ]
    return "| " + " | ".join(cells) + " |"


# %% [markdown]
# ## The tables

# %% Write the report
HEADER = (
    "| n | chosen setting | CV loss (trials) | test CDE loss | "
    "test CDE loss (DC1 grid) | seeds | fit s | predict+score s | "
    "peak GPU GB |\n|" + "---|" * 9
)
lines = ["# Foundation models on DC1: chosen settings and test losses", ""]
for method in METHODS:
    for regime, sizes in SIZES.items():
        lines += [f"## {method}, {regime}", "", HEADER]
        lines += [row(method, regime, size) for size in sizes]
        lines.append("")
report = "\n".join(lines)
pathlib.Path(RESULTS / "tfm_report.md").write_text(report)
print(report)
