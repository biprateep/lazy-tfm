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
# Every figure in the documentation that shows a model's output is made here,
# so each one can be remade from the code that the page shows. The models are
# small enough at these sizes to run on a CPU. Running the script rewrites
# every PNG in `docs/figures/figs/`:
#
#     python docs/figures/make_figures.py

# %%
import pathlib

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import lazy
from lazy import datasets

FIGS = pathlib.Path(__file__).resolve().parent / "figs"

# Figure defaults for AASTeX (AJ).
COLUMN_WIDTH = 242.26653 / 72.27  # in inches
TEXT_WIDTH = 513.11743 / 72.27
SMALL_SIZE = 9  # in pts
NORMAL_SIZE = 10
BIG_SIZE = 12
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
    "figure.facecolor": "w",
    "figure.dpi": 300,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
}
plt.rcParams.update(RC_PARAMS)

# %% [markdown]
# ## Filling in a curve
#
# The landing page's demo, as it appears there: a noisy chirp with three
# chunks cut out, and TabPFN-3.5-fast's mean and 68% interval in the gaps.

# %%
X_train, X_test, y_train, y_test = datasets.make_chirp(random_state=0)

model = lazy.LazyModel(
    "tabpfn",
    version="v3.5-fast",
    n_estimators=8,
    transforms="auto",
    feature_shuffle=True,
    bag_size=None,
    kv_cache=True,
    y_grid=None,
    device="auto",
    random_state=0,
    chunk_size=8192,
    softmax_temperature="auto",
    mixed_precision=True,
    outlier_threshold="auto",
)
model.fit(X_train, y_train)

y_mean = model.predict(X_test, method="mean")
y_low, y_high = model.predict_interval(X_test, coverage=0.68).T


# %%
def _gaps(x: np.ndarray) -> list[slice]:
    """Splits sorted test positions into one slice per contiguous gap."""
    breaks = np.flatnonzero(np.diff(x) > 1.5 * np.min(np.diff(x))) + 1
    edges = [0, *breaks, len(x)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]


x_train, x_test = X_train[:, 0], X_test[:, 0]
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
fig.savefig(FIGS / "chirp_demo.png", bbox_inches="tight", dpi=300)
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
arrow = dict(arrowstyle="-|>", color="k", lw=1)
ax.annotate("", xy=(4.0, 1.9), xytext=(2.75, 1.9), arrowprops=arrow)
ax.add_patch(
    mpl.patches.FancyBboxPatch(
        (4.1, 1.2),
        2.2,
        1.4,
        boxstyle="round,pad=0.05",
        facecolor="0.92",
        edgecolor="k",
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
    ax.plot([7.5, 9.7], [y_base, y_base], c="k", lw=0.5)
ax.text(8.6, 3.35, r"$p(y \mid x, \mathrm{context})$", ha="center", va="bottom")
fig.savefig(FIGS / "tfm_schematic.png", bbox_inches="tight", dpi=300)
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
fig.savefig(FIGS / "prior_draws.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# %% [markdown]
# ## Learning from the context
#
# The same model and curve as the landing page, without the gaps, given ever
# more points of the curve, drawn at random, as context: its answer narrows
# as the context grows, with no weights changed.

# %%
x_full, _, y_full, _ = datasets.make_chirp(n_gaps=0, random_state=0)
x_grid = np.linspace(0.0, 10.0, 400)[:, None]
truth = (1 + 0.2 * x_grid[:, 0]) * np.sin(
    2 * np.pi * (0.2 * x_grid[:, 0] + 0.03 * x_grid[:, 0] ** 2)
)
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
    low, high = model.predict_interval(x_grid, coverage=0.95).T
    ax.fill_between(
        x_grid[:, 0],
        low,
        high,
        color="C1",
        alpha=0.3,
        lw=0,
        label="95% interval",
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
        rasterized=True,
    )
    ax.set_title(f"{size:,} context rows")
    ax.set_xlabel(r"$x$")
    ax.set_xlim(0, 10)
axes[0].set_ylabel(r"$y$")
fig.legend(
    *axes[0].get_legend_handles_labels(),
    frameon=False,
    loc="outside upper center",
    ncols=4,
)
fig.savefig(FIGS / "context_size.png", bbox_inches="tight", dpi=300)
plt.close(fig)
