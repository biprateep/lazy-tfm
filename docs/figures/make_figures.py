# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # The documentation's figures
#
# Every figure of the landing page and the user guide is made here, so each
# one can be remade from code. The figures of a model's output use
# TabPFN-3.5-fast at LAZY's defaults, which runs on a CPU. Running the
# script rewrites every SVG in `docs/figures/figs/`:
#
#     python docs/figures/make_figures.py

# %%
import pathlib
import runpy

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import stats

import lazy
from lazy import datasets
from lazy import distributions
from lazy import metrics
from lazy.models import ensemble

FIGS = pathlib.Path(__file__).resolve().parent / "figs"

# Figure defaults for the documentation site, not a journal. The figures fill
# the theme's content column, so their type is scaled with the column's width.
TEXT_WIDTH = 7.25  # in inches
SMALL_SIZE = 10.5  # in pts
NORMAL_SIZE = 12  # in pts; near the body text
BIG_SIZE = 13.5  # in pts
# Dark grey ink on a transparent background, so the figures sit on the page's
# own. docs/_static/gallery.css inverts them in the dark theme, so each one
# is included with `:class: lazy-figure`.
INK = "#404040"
SERIF_FALLBACKS = ["CMU Serif", "Latin Modern Roman", "cmr10"]

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": SERIF_FALLBACKS,
    "font.size": NORMAL_SIZE,
    "axes.titlesize": NORMAL_SIZE,
    "axes.labelsize": NORMAL_SIZE,
    "xtick.labelsize": SMALL_SIZE,
    "ytick.labelsize": SMALL_SIZE,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": NORMAL_SIZE,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.dpi": 300,
    # Vector figures stay sharp at any display size and pixel density, which a
    # raster scaled down by the browser does not. Text is drawn as outlines, so
    # the figures do not depend on the reader having the serif font, and a
    # fixed salt keeps the ids in the files, and so their diffs, stable.
    "svg.fonttype": "path",
    "svg.hashsalt": "lazy",
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
}
plt.rcParams.update(RC_PARAMS)
# No creation date in the files, so a rerun leaves them untouched.
METADATA = {"Date": None}

# %% [markdown]
# ## Filling in a curve
#
# The landing page's demo, as it appears there: a noisy chirp with three
# chunks cut out, and TabPFN-3.5-fast's mean and 68% interval in the gaps.

# %%
# The landing page includes this file, so the model code exists once.
demo = runpy.run_path(
    str(pathlib.Path(__file__).resolve().parents[1] / "snippets/quickstart.py")
)
X_train, X_test, y_train, y_test = (
    demo[name] for name in ("X_train", "X_test", "y_train", "y_test")
)
y_mean, y_low, y_high = (demo[name] for name in ("y_mean", "y_low", "y_high"))


# %%
def _gaps(x: np.ndarray) -> list[slice]:
    """Splits sorted test positions into one slice per contiguous gap."""
    breaks = np.flatnonzero(np.diff(x) > 1.5 * np.min(np.diff(x))) + 1
    edges = [0, *breaks, len(x)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]


x_train, x_test = X_train["x"].to_numpy(), X_test["x"].to_numpy()
fig, ax = plt.subplots(
    figsize=(TEXT_WIDTH, 0.4 * TEXT_WIDTH), layout="constrained"
)
# The training rows, broken where the gaps are.
y_line = np.where(np.diff(x_train, prepend=x_train[0]) > 0.05, np.nan, y_train)
ax.plot(x_train, y_line, c="C0", lw=1, label="Training data")
for i, gap in enumerate(_gaps(x_test)):
    first = i == 0
    ax.plot(
        x_test[gap],
        y_test[gap],
        c="0.5",
        ls="--",
        lw=1,
        label="Truth" if first else None,
    )
    ax.fill_between(
        x_test[gap],
        y_low[gap],
        y_high[gap],
        color="C1",
        alpha=0.3,
        lw=0,
        label="68% interval" if first else None,
    )
    ax.plot(
        x_test[gap],
        y_mean[gap],
        ls="none",
        marker=".",
        ms=3,
        c="C1",
        label="Prediction (mean)" if first else None,
    )
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_xlim(0, 10)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=4)
FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / "chirp_demo.svg", bbox_inches="tight", metadata=METADATA)
plt.close(fig)

# %% [markdown]
# ## What a tabular foundation model sees
#
# A schematic for the explainer page: the labelled rows go in as context, the
# unlabelled rows as queries, and one pass of the pretrained network returns a
# distribution for each query.

# %%
fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 0.36 * TEXT_WIDTH))
ax.set_xlim(0, 10)
ax.set_ylim(0.3, 3.6)
ax.set_axis_off()
n_context, n_query, n_cols = 6, 3, 5
cell_w, cell_h = 0.42, 0.3
top = 3.2
for row in range(n_context + n_query):
    is_query = row >= n_context
    y0 = top - (row + 1) * cell_h - (0.12 if is_query else 0.0)
    for col in range(n_cols):
        is_target = col == n_cols - 1
        x0 = 0.3 + col * cell_w + (0.1 if is_target else 0.0)
        ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                cell_w,
                cell_h,
                facecolor="C1" if is_query else "C0",
                alpha=0.15 if is_query and is_target else 0.45,
                edgecolor="w",
                lw=1,
            )
        )
        if is_query and is_target:
            ax.text(
                x0 + cell_w / 2, y0 + cell_h / 2, "?", ha="center", va="center"
            )
table_mid = 0.3 + (n_cols * cell_w + 0.1) / 2
ax.text(
    table_mid,
    top + 0.1,
    r"features $X$ $\;|\;$ target $y$",
    ha="center",
    va="bottom",
)
ax.text(
    0.2,
    top - n_context * cell_h / 2,
    "context",
    rotation=90,
    ha="right",
    va="center",
)
ax.text(
    0.2,
    top - n_context * cell_h - 0.12 - n_query * cell_h / 2,
    "queries",
    rotation=90,
    ha="right",
    va="center",
)
arrow = dict(arrowstyle="-|>", color=INK, lw=1)
ax.annotate("", xy=(4.0, 1.9), xytext=(2.75, 1.9), arrowprops=arrow)
ax.add_patch(
    mpl.patches.FancyBboxPatch(
        (4.1, 1.2),
        2.2,
        1.4,
        boxstyle="round,pad=0.05",
        facecolor="0.92",
        edgecolor=INK,
        lw=1,
    )
)
ax.text(
    5.2,
    1.9,
    "pretrained\ntransformer\n(weights fixed)",
    ha="center",
    va="center",
)
ax.annotate("", xy=(7.3, 1.9), xytext=(6.45, 1.9), arrowprops=arrow)
grid = np.linspace(-3, 3, 200)
for i, (loc, scale) in enumerate([(-0.8, 0.5), (0.4, 0.9), (1.0, 0.35)]):
    y_base = 2.65 - i * 0.8
    pdf = np.exp(-0.5 * ((grid - loc) / scale) ** 2)
    if i == 1:  # A bimodal answer, which a single number cannot describe.
        pdf = 0.6 * pdf + np.exp(-0.5 * ((grid + 1.6) / 0.3) ** 2)
    xs = 7.5 + (grid + 3) / 6 * 2.2
    ax.fill_between(
        xs,
        y_base,
        y_base + 0.55 * pdf / pdf.max(),
        color="C1",
        alpha=0.45,
        lw=0,
    )
    ax.plot(xs, y_base + 0.55 * pdf / pdf.max(), c="C1", lw=1)
    ax.plot([7.5, 9.7], [y_base, y_base], c=INK, lw=0.5)
ax.text(8.6, 3.35, r"$p(y \mid x, \mathrm{context})$", ha="center", va="bottom")
fig.savefig(FIGS / "tfm_schematic.svg", bbox_inches="tight", metadata=METADATA)
plt.close(fig)

# %% [markdown]
# ## Datasets from a prior
#
# Six draws from a toy prior over one-feature regression problems: a random
# small network with random activations, random noise, random inputs. These
# illustrate the idea only; the models' own priors are far richer (structural
# causal models over many features, mixed types, missing values, ...).

# %%
rng = np.random.default_rng(3)
activations = [np.tanh, np.sin, np.abs, lambda u: np.maximum(u, 0.0), np.cos]


def _draw_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One toy dataset from a random two-layer network with noise."""
    n = rng.integers(40, 200)
    x = rng.uniform(-3, 3, n) if rng.random() < 0.7 else rng.normal(0, 1.2, n)
    width = rng.integers(2, 6)
    hidden = rng.normal(0, 1.5, (width, 1)) @ x[None, :] + rng.normal(
        0, 1, (width, 1)
    )
    act = activations[rng.integers(len(activations))]
    y = rng.normal(0, 1, width) @ act(hidden)
    y = (y - y.mean()) / (y.std() + 1e-9)
    y += rng.normal(0, rng.uniform(0.02, 0.5), n) * (
        1 + rng.uniform(0, 1) * np.abs(x) if rng.random() < 0.5 else 1
    )
    return x, y


fig, axes = plt.subplots(
    2,
    3,
    figsize=(TEXT_WIDTH, 0.5 * TEXT_WIDTH),
    sharex=True,
    layout="constrained",
)
for ax in axes.flat:
    x, y = _draw_dataset(rng)
    ax.scatter(x, y, s=4, marker=".", c="C0")
    ax.set_yticks([])
for ax in axes[-1]:
    ax.set_xlabel(r"$x$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$y$")
fig.savefig(FIGS / "prior_draws.svg", bbox_inches="tight", metadata=METADATA)
plt.close(fig)

# %% [markdown]
# ## Learning from the context
#
# The same model and curve as the landing page, without the gaps, given ever
# more points of the curve, drawn at random, as context: its answer narrows
# as the context grows, with no weights changed.

# %%
full = datasets.load_dataset("chirp", n_gaps=0, random_state=0)
x_full, y_full = full.X.to_numpy(), full.y
# The noise-free curve on a 400-point grid, from the chirp itself.
grid = datasets.load_dataset("chirp", n_samples=400, noise=0.0, n_gaps=0)
x_grid, truth = grid.X.to_numpy(), grid.y
sizes = (20, 100, 400)
fig, axes = plt.subplots(
    1,
    len(sizes),
    figsize=(TEXT_WIDTH, 0.3 * TEXT_WIDTH),
    sharey=True,
    layout="constrained",
)
for ax, size in zip(axes, sizes, strict=True):
    rows = np.sort(rng.choice(len(x_full), size, replace=False))
    model = lazy.LazyModel("tabpfn", version="v3.5-fast")
    model.fit(x_full[rows], y_full[rows])
    low, high = model.predict_interval(x_grid, coverage=0.68).T
    ax.fill_between(
        x_grid[:, 0],
        low,
        high,
        color="C1",
        alpha=0.3,
        lw=0,
        label="68% interval",
    )
    ax.plot(
        x_grid[:, 0],
        model.predict(x_grid, method="mean"),
        c="C1",
        lw=1,
        label="Mean",
    )
    ax.plot(x_grid[:, 0], truth, c="0.5", ls="--", lw=1, label="Truth")
    ax.scatter(
        x_full[rows, 0],
        y_full[rows],
        s=2,
        marker=".",
        c="C0",
        label="Context",
        zorder=3,
    )
    ax.set_title(f"{size:,} context rows")
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel(r"$x$")
    ax.set_xlim(0, 10)
axes[0].set_ylabel(r"$y$")
fig.legend(
    *axes[0].get_legend_handles_labels(),
    frameon=False,
    loc="outside upper center",
    ncols=4,
)
fig.savefig(FIGS / "context_size.svg", bbox_inches="tight", metadata=METADATA)
plt.close(fig)

# %% [markdown]
# ## Point estimates of a bimodal density
#
# The four reductions of `predict` on one skewed, bimodal density, computed
# by `lazy.metrics.grid_point_estimates`: the mode and the peak mean sit on
# the main peak, while the mean falls between the peaks.

# %%
y_grid = lazy.Grid.linear(0.0, 2.0, 400)
pdf = 0.6 * stats.skewnorm.pdf(
    y_grid.centers, 3, loc=0.35, scale=0.25
) + 0.4 * stats.norm.pdf(y_grid.centers, loc=1.45, scale=0.15)
estimates = metrics.grid_point_estimates(y_grid, pdf[None, :])
fig, ax = plt.subplots(
    figsize=(TEXT_WIDTH, 0.32 * TEXT_WIDTH), layout="constrained"
)
ax.fill_between(y_grid.centers, pdf, color="C0", alpha=0.3, lw=0)
ax.plot(y_grid.centers, pdf, c="C0", lw=1)
for i, (method, ls) in enumerate(
    [("mode", "-"), ("peak_mean", "--"), ("mean", "-"), ("median", "--")]
):
    ax.axvline(
        estimates[method][0],
        c=f"C{1 + i // 2}",
        ls=ls,
        lw=1.2,
        label=method.replace("_", " "),  # The font has no underscore.
    )
ax.set_xlabel(r"$y$")
ax.set_ylabel(r"$p(y)$")
ax.set_xlim(0, 2)
ax.set_ylim(0, None)
ax.legend(frameon=False, loc="upper right")
fig.savefig(
    FIGS / "point_estimates.svg", bbox_inches="tight", metadata=METADATA
)
plt.close(fig)

# %% [markdown]
# ## Three ways to pool
#
# The three pools, drawn from their definitions in the table of
# docs/guide/ensembles.md rather than by `LazyEnsembleModel`, applied to two
# members: a bimodal one and a unimodal one. The linear pool keeps
# both modes, the geometric pool keeps only the shared one and is sharpest,
# and the quantile pool averages the quantile functions with
# `QuantileDistribution.average`.

# %%
y = np.linspace(-0.5, 3.5, 801)
# The first member predicts a second mode, which the second rules out.
member_pdfs = np.stack(
    [
        0.7 * stats.norm.pdf(y, 1.0, 0.3) + 0.3 * stats.norm.pdf(y, 2.6, 0.2),
        stats.norm.pdf(y, 1.3, 0.25),
    ]
)
linear = member_pdfs.mean(axis=0)
# LAZY's floor and levels: the floor as a fraction of the uniform density, and
# quantiles at every level, the ends at the grid's ends.
floor = ensemble.LOG_FLOOR / (y[-1] - y[0])
geometric = np.exp(np.log(member_pdfs + floor).mean(axis=0))
geometric /= np.trapezoid(geometric, y)
levels = ensemble.QUANTILE_LEVELS
member_cdfs = integrate.cumulative_trapezoid(member_pdfs, y, initial=0)
member_cdfs /= member_cdfs[:, -1:]
quantile = distributions.QuantileDistribution.average(
    [
        distributions.QuantileDistribution(levels, np.interp(levels, cdf, y))
        for cdf in member_cdfs
    ]
).pdf(y)[0]
fig, axes = plt.subplots(
    1,
    3,
    figsize=(TEXT_WIDTH, 0.3 * TEXT_WIDTH),
    sharey=True,
    layout="constrained",
)
for ax, (name, pooled) in zip(
    axes,
    [("Geometric", geometric), ("Linear", linear), ("Quantile", quantile)],
    strict=True,
):
    for k, ls in enumerate(["--", ":"]):
        ax.plot(
            y,
            member_pdfs[k],
            c="0.5",
            ls=ls,
            lw=1,
            label=f"Member {k + 1}",
        )
    ax.fill_between(y, pooled, color="C1", alpha=0.3, lw=0)
    ax.plot(y, pooled, c="C1", lw=1.2, label="Pool")
    ax.set_title(f"{name} pool")
    ax.set_xlabel(r"$y$")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, None)
axes[0].set_ylabel(r"$p(y)$")
fig.legend(
    *axes[0].get_legend_handles_labels(),
    frameon=False,
    loc="outside upper center",
    ncols=3,
)
fig.savefig(FIGS / "pooling.svg", bbox_inches="tight", metadata=METADATA)
plt.close(fig)

# %% [markdown]
# ## A grid that ends inside the distribution
#
# One distribution put on a grid that covers it and on one that ends inside
# it, by `HistogramDistribution.on_grid`: the probability beyond the short
# grid is dropped, the rest is renormalized, so the density inside rises, and
# a true value beyond the grid has no density to score.

# %%
buckets = np.linspace(-1.0, 4.0, 1001)
skewed = stats.skewnorm(4, loc=0.5, scale=1.0)
dist = distributions.HistogramDistribution(
    buckets, np.diff(skewed.cdf(buckets))[None, :]
)
wide, short = lazy.Grid.linear(0.0, 3.5, 140), lazy.Grid.linear(0.0, 1.5, 60)
y_true = 2.2
fig, ax = plt.subplots(
    figsize=(TEXT_WIDTH, 0.32 * TEXT_WIDTH), layout="constrained"
)
ax.axvspan(short.edges[-1], wide.edges[-1], color="0.92", lw=0)
ax.text(2.5, 1.0, "outside the short grid", ha="center", va="top")
ax.plot(
    wide.centers,
    dist.on_grid(wide)[0],
    c="C0",
    lw=1.2,
    label=f"Grid on [0, {wide.edges[-1]:g}]",
)
ax.plot(
    short.centers,
    dist.on_grid(short)[0],
    c="C1",
    lw=1.2,
    label=f"Grid on [0, {short.edges[-1]:g}]",
)
ax.axvline(y_true, ymax=0.8, c=INK, ls="--", lw=1, label="True value")
ax.set_xlabel(r"$y$")
ax.set_ylabel(r"$p(y)$")
ax.set_xlim(0, wide.edges[-1])
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.85))
fig.savefig(FIGS / "grid_cutoff.svg", bbox_inches="tight", metadata=METADATA)
plt.close(fig)

# %% [markdown]
# ## Chunks and bags
#
# A schematic for the scaling page: every query row attends to every context
# row its member sees. Chunking splits the queries into passes; bagging leaves
# a member only the columns of its bag. The context's own work, once per pass
# without the key/value cache, is not drawn.

# %%
n_ctx, n_qry, chunk, bag = 16, 12, 4, 8
in_bag = np.isin(
    np.arange(n_ctx),
    np.random.default_rng(1).choice(n_ctx, bag, replace=False),
)
fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.3 * TEXT_WIDTH))
for ax, columns, title in [
    (axes[0], np.ones(n_ctx, bool), "Every context row"),
    (axes[1], in_bag, f"A bag of {bag} context rows"),
]:
    for row in range(n_qry):
        # A gap between chunks, which run as separate passes.
        y0 = -(row + 1) - 0.4 * (row // chunk)
        for col in range(n_ctx):
            ax.add_patch(
                mpl.patches.Rectangle(
                    (col, y0),
                    1,
                    1,
                    facecolor="C1" if columns[col] else "0.92",
                    alpha=0.45 if columns[col] else 1.0,
                    edgecolor="w",
                    lw=1,
                )
            )
    for i in range(n_qry // chunk):
        ax.text(
            n_ctx + 0.4,
            -(i + 0.5) * chunk - 0.4 * i,
            f"chunk {i + 1}",
            ha="left",
            va="center",
        )
    ax.text(n_ctx / 2, 0.4, "context rows", ha="center", va="bottom")
    ax.text(
        -0.4,
        -n_qry / 2 - 0.4,
        "query rows",
        rotation=90,
        ha="right",
        va="center",
    )
    ax.set_title(title, pad=24)
    ax.set_xlim(-1.5, n_ctx + 3.5)
    ax.set_ylim(-n_qry - 0.4 * (n_qry // chunk - 1) - 0.2, 0.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.savefig(FIGS / "chunks_bags.svg", bbox_inches="tight", metadata=METADATA)
plt.close(fig)
