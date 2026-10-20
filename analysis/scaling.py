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
# # Scaling curves of the production runs
#
# Reads the run records of `fm4pz.produce` (one per method, regime, training
# size and seed) under `$FM4PZ_RESULTS/runs/`, and draws for each regime the
# CDE loss on the common grid and the scatter of the mode against the
# training size, with the mean and standard deviation over the five
# subsamples.

# %%
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
import json
import os
import pathlib

from matplotlib import ticker
import matplotlib.pyplot as plt
import pandas as pd
import plotstyle

# %% [markdown]
# ## Configuration
#
# The paper uses the NeurIPS class, whose text is 5.5 in wide in a single
# column.

# %%
RESULTS = pathlib.Path(os.environ.get("FM4PZ_RESULTS", "../results"))
METHODS = {
    "tabpfn": "TabPFN-3.5",
    "limix": "LimiX-2",
    "kumo": "Kumo Tabular",
}
REGIMES = {
    "representative": "Representative",
    "biased": "Biased",
}
NEURIPS_TEXT_PT = 397.48499

plotstyle.register_journal(
    "neurips", column_pt=NEURIPS_TEXT_PT, text_pt=NEURIPS_TEXT_PT
)
plotstyle.use_style(journal="neurips")
plotstyle.verify_style()

# %% [markdown]
# ## The run records
#
# One row per run, with the method, regime, size, seed and metrics.

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
print(summary.round(4).to_string(index=False))

# %% [markdown]
# ## The figures
#
# One figure per regime: the CDE loss (left) and the scatter of the mode
# (right, on a logarithmic scale) against the training size.


# %%
def scaling_figure(table: pd.DataFrame, regime: str) -> plt.Figure:
    """Draws one regime's two-panel scaling figure.

    Args:
        table: The per-size summary of the runs, as built above.
        regime: The regime to draw.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=plotstyle.grid_figsize(1, 2, "text", panel_aspect=0.75),
        layout="constrained",
    )
    for k, (method, label) in enumerate(METHODS.items()):
        rows = table[(table["regime"] == regime) & (table["method"] == method)]
        if rows.empty:
            continue
        for ax, column in zip(axes, ("cde", "sigma"), strict=True):
            ax.errorbar(
                rows["n_train"],
                rows[f"{column}_mean"],
                yerr=rows[f"{column}_std"],
                color=f"C{k}",
                marker="o",
                markersize=3,
                capsize=2,
                label=label,
            )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Training Size")
    axes[0].set_ylabel(r"$\mathcal{L}_{\rm CDE}$")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_locator(
        ticker.LogLocator(subs=(1.0, 1.2, 1.4, 1.6, 1.8, 2.0))
    )
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    axes[1].yaxis.set_minor_formatter(ticker.NullFormatter())
    axes[1].set_ylabel(r"$\sigma_{\rm IQR}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="outside lower center",
        ncols=len(labels),
    )
    fig.suptitle(REGIMES[regime])
    return fig


for regime in REGIMES:
    figure = scaling_figure(summary, regime)
    plotstyle.save(figure, f"./figs/scaling_tfm_{regime}")
    plt.close(figure)
