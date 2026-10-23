# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How many calibration galaxies does a biased training set need?
#
# Each biased training set (1,000 to 35,011 galaxies) takes $k$ galaxies of
# the representative calibration sample, from 500 to all 10,000 in steps of
# 500, and is refitted with the setting tuned for the biased size alone. The
# test set is the biased regime's representative hold-out. The runs are
# those of `fm4pz.produce --calibration` under
# `$FM4PZ_RESULTS/calibration/`; $k = 0$ is the biased production run.

# %%
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
import json
import os
import pathlib

from matplotlib import cm
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotstyle

# %% [markdown]
# ## Configuration

# %%
RESULTS = pathlib.Path(os.environ.get("FM4PZ_RESULTS", "../results"))
METHOD = "tabpfn"
LABEL = "TabPFN-3.5"
NEURIPS_TEXT_PT = 397.48499
# Fractions of the gain from all 10,000 calibration galaxies to report.
FRACTIONS = (0.5, 0.9)

plotstyle.register_journal(
    "neurips", column_pt=NEURIPS_TEXT_PT, text_pt=NEURIPS_TEXT_PT
)
plotstyle.use_style(journal="neurips")
plotstyle.verify_style()

# %% [markdown]
# ## The runs
#
# One row per run, then the test CDE loss as a table of biased training size
# against calibration galaxies added, written beside the runs as a CSV.

# %%
rows = []
paths = sorted(RESULTS.glob(f"calibration/{METHOD}/*.json"))
paths += sorted(RESULTS.glob(f"runs/{METHOD}/biased_n*.json"))
for path in paths:
    record = json.loads(path.read_text())
    if record["regime"] != "biased":
        continue
    rows.append(
        {
            "n_train": record["n_train"],
            "n_calibration": record.get("n_calibration", 0),
            "seed": record["seed"],
            "cde_loss": record["metrics"]["cde_loss"],
            "sigma_iqr": record["metrics"]["sigma_iqr"],
        }
    )
runs = pd.DataFrame(rows)
# The calibration runs use seed 0 alone; so does k = 0 here.
runs = runs[runs["seed"] == 0]
table = runs.pivot_table(
    index="n_train", columns="n_calibration", values="cde_loss"
)
table.to_csv(RESULTS / "calibration" / f"{METHOD}_cde_loss.csv")
print(f"{len(runs)} runs")
print(table.round(2).to_string())

# %% [markdown]
# ## How much of the gain comes early
#
# For each biased size: the gain from all 10,000 calibration galaxies, and
# the fewest galaxies that reach half and nine tenths of it.


# %%
def first_reaching(gains: pd.Series, target: float) -> int | None:
    """The smallest calibration size whose gain reaches a target.

    Args:
        gains: Gain in CDE loss (positive is better) by calibration size.
        target: The gain to reach.

    Returns:
        The calibration size, or None if none reaches it.
    """
    reached = gains[gains >= target]
    return int(reached.index.min()) if not reached.empty else None


summary = []
for n_train, losses in table.iterrows():
    gains = losses.iloc[0] - losses
    total = gains.iloc[-1]
    row = {
        "n_train": n_train,
        "cde_loss_0": losses.iloc[0],
        "cde_loss_10k": losses.iloc[-1],
        "gain_10k": total,
    }
    for fraction in FRACTIONS:
        row[f"k_{int(100 * fraction)}pct"] = first_reaching(
            gains, fraction * total
        )
    summary.append(row)
summary_table = pd.DataFrame(summary)
print(summary_table.round(2).to_string(index=False))

# %% [markdown]
# ## The figure
#
# Test CDE loss against calibration galaxies added, one curve per biased
# training size, coloured along the default colour map by size.

# %%
fig, ax = plt.subplots(
    figsize=plotstyle.figsize("text", 0.55), layout="constrained"
)
sizes = list(table.index)
colours = cm.viridis(np.linspace(0, 0.9, len(sizes)))
for colour, n_train in zip(colours, sizes, strict=True):
    losses = table.loc[n_train]
    label = (
        f"{n_train // 1000}k"
        if n_train % 1000 == 0
        else f"{n_train / 1000:.0f}k (all)"
    )
    ax.plot(
        losses.index,
        losses.values,
        color=colour,
        marker=".",
        markersize=4,
        linewidth=1.2,
        label=label,
    )
ax.set_title(LABEL, fontsize=plotstyle.BIG_SIZE)
ax.set_xlabel("Calibration Galaxies Added")
ax.set_ylabel(r"$\mathcal{L}_{\rm CDE}$")
ax.xaxis.set_major_locator(ticker.MultipleLocator(2000))
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda value, _: f"{value / 1000:g}k" if value else "0"
    )
)
ax.grid(which="major", color="k", alpha=0.15, linestyle="--", linewidth=0.6)
fig.legend(
    frameon=False,
    title="Biased Size",
    loc="outside right center",
    fontsize=plotstyle.SMALL_SIZE,
)
plotstyle.save(fig, f"./figs/calibration_{METHOD}")
plt.close(fig)
