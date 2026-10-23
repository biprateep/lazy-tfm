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
# # Scaling curves
#
# The paper's scaling figures: Figure 1 for the representative training sets
# and the appendix figure for the biased ones. Each shows, against the
# training size, the CDE loss on the common grid (left) and the scatter of the
# mode, $\sigma_{\rm IQR}$ (right, logarithmic), for the three bespoke
# estimators and the three tabular foundation models, as the mean and
# standard deviation over the five training subsamples. The run records are
# those of `fm4pz.produce` under `$FM4PZ_RESULTS/runs/`, one per method,
# regime, training size and seed.

# %%
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
import json
import os
import pathlib

from matplotlib import lines
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotstyle

# %% [markdown]
# ## Configuration
#
# The paper uses the NeurIPS class, whose text is 397.48499 pt wide in a
# single column. Bespoke estimators are drawn dashed and foundation models
# solid, each with its own marker.

# %%
RESULTS = pathlib.Path(os.environ.get("FM4PZ_RESULTS", "../results"))
NEURIPS_TEXT_PT = 397.48499
# Method: (label, colour, marker, family).
METHODS = {
    "flexzboost": ("FlexZBoost", "C0", "v", "Bespoke"),
    "cmnn": ("CMNN", "C1", "D", "Bespoke"),
    "gpz": ("GPz", "C2", "P", "Bespoke"),
    "limix": ("LimiX-2", "C3", "o", "TFM"),
    "tabpfn": ("TabPFN-3.5", "C4", "s", "TFM"),
    "kumo": ("Kumo", "C5", "^", "TFM"),
}
# Regime: the output name the manuscript includes.
REGIMES = {
    "representative": "unbiased_scaling",
    "biased": "biased_scaling",
}
SEEDS = 5
SIGMA_TICKS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)

plotstyle.register_journal(
    "neurips", column_pt=NEURIPS_TEXT_PT, text_pt=NEURIPS_TEXT_PT
)
plotstyle.use_style(journal="neurips")
plotstyle.verify_style()

# %% [markdown]
# ## The run records
#
# One row per run, then the mean and standard deviation over the seeds of
# every method, regime and training size. Every one of the 84 experiments
# must have all five seeds.

# %%
rows = []
for path in sorted(RESULTS.glob("runs/*/*.json")):
    record = json.loads(path.read_text())
    rows.append(
        {
            "method": path.parent.name,
            "regime": record["regime"],
            "n_train": record["n_train"],
            "seed": record["seed"],
            "cde_loss": record["metrics"]["cde_loss"],
            "sigma_iqr": record["metrics"]["sigma_iqr"],
        }
    )
runs = pd.DataFrame(rows)
summary = (
    runs[runs["method"].isin(METHODS)]
    .groupby(["regime", "method", "n_train"])
    .agg(
        cde_mean=("cde_loss", "mean"),
        cde_std=("cde_loss", "std"),
        sigma_mean=("sigma_iqr", "mean"),
        sigma_std=("sigma_iqr", "std"),
        n_seeds=("seed", "count"),
    )
    .reset_index()
)
incomplete = summary[summary["n_seeds"] != SEEDS]
if not incomplete.empty:
    raise ValueError(f"experiments without {SEEDS} seeds:\n{incomplete}")
print(f"{len(summary)} experiments, {len(runs)} runs")
print(summary.round(4).to_string(index=False))

# %% [markdown]
# ## The figures
#
# One figure per regime, two panels side by side and one legend below that
# groups the estimators by family.


# %%
def sigma_ticks(low: float, high: float) -> list[float]:
    """The ticks of the scatter axis that fall within a range.

    Args:
        low: The smallest value drawn, in units of 10^-2.
        high: The largest value drawn, in units of 10^-2.

    Returns:
        The tick values inside [low, high], from :data:`SIGMA_TICKS`.
    """
    return [tick for tick in SIGMA_TICKS if low <= tick <= high]


def legend_entries(
    axis: plt.Axes,
) -> tuple[list[lines.Line2D], list[str]]:
    """The legend's handles and labels, each family led by its name.

    Args:
        axis: An axis holding one line per method, labelled as in
            :data:`METHODS`.

    Returns:
        A tuple (handles, labels) for a legend of two rows, one per family,
        each led by a blank handle carrying the family's name. The legend
        fills its columns first, so the two families are interleaved.
    """
    by_label = dict(zip(*axis.get_legend_handles_labels()[::-1], strict=True))
    rows = []
    for family in ("Bespoke", "TFM"):
        row = [(lines.Line2D([], [], linestyle="none"), family)]
        row += [
            (by_label[label], label)
            for label, _, _, method_family in METHODS.values()
            if method_family == family
        ]
        rows.append(row)
    interleaved: list[tuple[object, str]] = []
    for column in zip(*rows, strict=True):
        interleaved.extend(column)
    handles, labels = zip(*interleaved, strict=True)
    return list(handles), list(labels)


def scaling_figure(table: pd.DataFrame, regime: str) -> plt.Figure:
    """Draws one regime's two-panel scaling figure.

    Args:
        table: The per-experiment summary of the runs, as built above.
        regime: ``"representative"`` or ``"biased"``.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=plotstyle.grid_figsize(1, 2, "text", panel_aspect=0.8),
        layout="constrained",
    )
    rows = table[table["regime"] == regime]
    for method, (label, colour, marker, family) in METHODS.items():
        curve = rows[rows["method"] == method].sort_values("n_train")
        style = {
            "color": colour,
            "marker": marker,
            "markersize": 3.5,
            "linestyle": "--" if family == "Bespoke" else "-",
            "linewidth": 1.2,
            "capsize": 1.5,
            "elinewidth": 0.8,
            "label": label,
        }
        axes[0].errorbar(
            curve["n_train"], curve["cde_mean"], yerr=curve["cde_std"], **style
        )
        axes[1].errorbar(
            curve["n_train"],
            100 * curve["sigma_mean"],
            yerr=100 * curve["sigma_std"],
            **style,
        )
    sizes = np.sort(rows["n_train"].unique())
    shown = [size for size in sizes if size not in (15000, 30000)]
    for axis in axes:
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(ticker.FixedLocator(shown))
        axis.xaxis.set_major_formatter(
            ticker.FixedFormatter([f"{size // 1000}k" for size in shown])
        )
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.set_xlabel("Training Set Size")
        # A grid at the major ticks, as in the submitted figures (PI request).
        axis.grid(
            which="major", color="k", alpha=0.15, linestyle="--", linewidth=0.6
        )
    axes[0].set_ylabel(r"$\mathcal{L}_{\rm CDE}$")
    axes[1].set_yscale("log")
    sigma = 100 * rows["sigma_mean"]
    axes[1].yaxis.set_major_locator(
        ticker.FixedLocator(sigma_ticks(0.95 * sigma.min(), 1.05 * sigma.max()))
    )
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    axes[1].yaxis.set_minor_formatter(ticker.NullFormatter())
    axes[1].set_ylabel(r"$\sigma_{\rm IQR}$ [$10^{-2}$]")
    handles, labels = legend_entries(axes[0])
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="outside lower center",
        ncols=len(labels) // 2,
        handlelength=1.8,
        columnspacing=1.0,
        fontsize=plotstyle.SMALL_SIZE,
    )
    return fig


for regime, name in REGIMES.items():
    figure = scaling_figure(summary, regime)
    plotstyle.save(figure, f"./figs/{name}")
    plt.close(figure)
