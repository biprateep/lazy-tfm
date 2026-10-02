# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: T4
#     provenance: []
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: accelerator,colab
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Introduction to `lazy`: photo-$z$ PDFs from a foundation model
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biprateep/lazy-tfm/blob/tutorials/introduction.ipynb)
# [![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?logo=github)](https://github.com/biprateep/lazy-tfm/blob/tutorials/introduction.ipynb)
#
# `lazy` (**L**azy but **A**ccurate ***z*** for **Y**inz) predicts the full
# conditional distribution of a continuous target from tabular features, using
# pretrained tabular foundation models. The models are never trained or
# fine-tuned: you hand them labelled rows as *context*, and they answer new
# rows in a single forward pass. The interface follows scikit-learn
# (`fit` → `predict_proba` → `evaluate`), with `predict_proba` returning a
# probability density over the target instead of class probabilities.
#
# This tutorial uses photometric redshift estimation as the example: from a
# galaxy's magnitudes, predict the PDF of its redshift $z$. Nothing in the
# package is specific to redshifts; the same code works for any continuous
# target. More documentation is on
# [Read the Docs](https://lazy-tfm.readthedocs.io).

# %% [markdown]
# ## Setup
#
# The cell below installs the package, with the TabPFN and TabICL backends,
# when the notebook runs on Google Colab. Elsewhere, install it yourself first
# with `pip install 'lazy-tfm[tabpfn,tabicl]'`.
#
# TabPFN, the default model, wants a GPU. On Colab choose
# *Runtime → Change runtime type → T4 GPU*; `lazy` warns you if it finds no GPU.

# %%
import sys

if "google.colab" in sys.modules:
    # %pip install -q 'lazy-tfm[tabpfn,tabicl]'
    pass

# %% [markdown]
# Let's import the modules we need.

# %%
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Arrays
import pandas as pd  # Tables

import lazy  # The topic of this tutorial
from lazy import datasets
from lazy import metrics
from lazy import plotting

# %% [markdown]
# `lazy.plotting` ships the figure style used by the paper. We fix one seed
# for everything random: the models' ensembles and our choice of galaxies.

# %%
SEED = 299792458  # The one constant to rule them all

plotting.use_style()
plt.rcParams["figure.dpi"] = 110  # Readable in a notebook

# %% [markdown]
# ## The data
#
# We use the catalogue of the LSST DESC Photo-$z$ Data Challenge 1
# ([Schmidt, Malz et al. 2020](https://arxiv.org/abs/2001.03621)), in which
# a dozen photo-$z$ codes were compared on the same simulated galaxies.
# `datasets.fetch_dc1` downloads it (about 1 GB, checksummed) the first time
# and reads it from a local cache after that. With `split=True` it returns the
# challenge's own training and test sets.

# %%
train, test = datasets.fetch_dc1(split=True)
train, test

# %% [markdown]
# Each `Catalog` keeps the raw columns: magnitudes in six bands ($ugrizy$),
# their errors, and the true redshift.

# %%
train.raw.head()

# %% [markdown]
# The model sees a table of features built from the magnitudes.
# `features("mag-color")` gives the $i$-band magnitude and the five adjacent
# colours ($u-g$, $g-r$, ...), with their errors propagated in quadrature.
# Any table of numbers works here: a `pandas.DataFrame` or a NumPy array.
#
# The test set has 391,000 galaxies. We take a random 10,000 of them to keep
# this tutorial quick.

# %%
X_train = train.features("mag-color")
z_train = train.redshift

rng = np.random.default_rng(SEED)
rows = rng.choice(len(test), size=10_000, replace=False)
X_test = test.features("mag-color").iloc[rows]
z_test = test.redshift[rows]

print(f"{len(X_train):,} context rows, {len(X_test):,} test rows")
X_train.head()

# %% [markdown]
# ## The model
#
# `lazy.LazyModel` chooses a backend by name and passes any other arguments
# to it. With no name it uses TabPFN-3.5. `lazy.list_estimators()` shows the
# other models, and the
# [Supported models](https://lazy-tfm.readthedocs.io/en/latest/models/index.html)
# page compares their size, speed and licences.

# %%
lazy.list_estimators()

# %% [markdown]
# `fit` does not train anything. It checks the features and stores the
# labelled rows as the model's context. The first call also downloads the
# model's weights (about 880 MB for TabPFN-3.5), which are cached after that.

# %%
model = lazy.LazyModel(random_state=SEED)
model.fit(X_train, z_train)

# %% [markdown]
# ## Redshift PDFs
#
# `predict_proba` returns one density per galaxy on a grid of the target. We
# ask for the Data Challenge's grid, `lazy.datasets.DC1_GRID` (200 bins over
# $0 \le z \le 2$), so that the numbers below are comparable with the
# challenge's. Without a grid, each model answers on its own native grid
# (5,000 bins for TabPFN).
#
# This is where the model does its work: one forward pass through the
# context for all 10,000 galaxies.

# %%
grid = lazy.datasets.DC1_GRID
pdfs = model.predict_proba(X_test, grid)
pdfs.shape

# %% [markdown]
# Here are the PDFs of a few random galaxies, with their true redshifts.
# Most are narrow; some have two peaks, where different redshifts give
# similar colours.

# %%
axes = plotting.plot_pdfs(
    grid, pdfs, y_true=z_test, n_objects=12, random_state=SEED
)

# %% [markdown]
# ## Point estimates
#
# Many analyses need a single number per galaxy. `predict` reduces each
# density to a point estimate (by default its mode, `"mode"`), and
# `metrics.grid_point_estimates` computes the same from densities you already
# have, without running the model again.

# %%
z_mode = metrics.grid_point_estimates(grid, pdfs)["mode"]

ax = plotting.plot_actual_vs_predicted(z_test, z_mode, outlier_lines=True)

# %% [markdown]
# Each point is a galaxy, coloured by how many share its place (on a log
# scale). The dotted lines, which `outlier_lines=True` adds, mark the Data
# Challenge's outlier cut,
# $|z_\mathrm{pred} - z_\mathrm{true}| > 0.06\,(1+z_\mathrm{true})$: most
# galaxies sit well inside it, and the outliers come from colour degeneracies,
# where a galaxy at one redshift looks like one at another.

# %% [markdown]
# ## How good are the PDFs?
#
# `metrics.summarize` scores densities with the Data Challenge metrics:
#
# - the **CDE loss**, which scores whole PDFs (lower is better);
# - the **PIT** (probability integral transform) statistics, which test
#   calibration, whether the PDFs are as wide as they should be;
# - the **point** metrics of the mode: bias, scatter and outlier fraction.
#
# The point metrics measure plain residuals, $z_\mathrm{pred} -
# z_\mathrm{true}$, by default. Photometric-redshift errors grow with
# $1 + z$, so the Data Challenge divided each residual by $1 + z_\mathrm{true}$;
# `scale="1+y"` asks for that convention.

# %%
table = metrics.summarize(z_test, grid, pdfs, label="TabPFN-3.5", scale="1+y")
table.T

# %% [markdown]
# `plotting.diagnostic_panel` shows the same things in one figure: the point
# estimates against the truth; their residuals, as a running median with its
# 68% band; the PIT quantiles against those of a uniform distribution (on the
# diagonal if the PDFs are calibrated); and the stacked PDFs against the true
# redshift distribution $n(z)$. `scale="1+y"` again uses the photo-$z$
# convention, for the residuals and the outlier lines.

# %%
fig = plotting.diagnostic_panel(
    z_test, grid, pdfs, label="TabPFN-3.5", scale="1+y"
)

# %% [markdown]
# ## Credible intervals
#
# `predict_quantiles` returns quantiles of each galaxy's distribution. They
# are computed from the model's own answer rather than from a grid, so they
# are exact. Here we ask for the median and a 68% interval, and check how
# often the true redshift falls inside it; for calibrated PDFs that is 68% of
# the time.

# %%
q = model.predict_quantiles(X_test, [0.16, 0.5, 0.84])
inside = (q[:, 0] <= z_test) & (z_test <= q[:, 2])
print(
    f"True redshift inside the 68% interval for {inside.mean():.1%} of galaxies"
)

# %% [markdown]
# ## Another model
#
# Every model has the same interface, so trying another is one line. TabICL
# is small, BSD-licensed and fast on a CPU; it is the one to use on a laptop.
# We run it on the same galaxies and compare.
#
# We ask for 4 ensemble members instead of the default 8. Caching the context
# for 8 members briefly needs about 14 GB of GPU memory with this many rows,
# more than Colab's T4 has to spare; 4 need about 7 GB.

# %%
tabicl = lazy.LazyModel("tabicl", n_estimators=4, random_state=SEED)
tabicl.fit(X_train, z_train)
pdfs_tabicl = tabicl.predict_proba(X_test, grid)

pd.concat(
    [
        table,
        metrics.summarize(
            z_test, grid, pdfs_tabicl, label="TabICL", scale="1+y"
        ),
    ]
).set_index("model").T

# %% [markdown]
# On most galaxies the two models agree closely. They differ where the
# answer is uncertain, so let's compare their PDFs for six of the hardest
# galaxies: random picks from the 5% whose TabPFN 68% interval is widest.

# %%
width = q[:, 2] - q[:, 0]
hardest = np.flatnonzero(width > np.quantile(width, 0.95))
idx = rng.choice(hardest, size=6, replace=False)

fig, axes = plt.subplots(
    2, 3, figsize=(9, 4.5), sharex=True, layout="constrained"
)
for ax, i in zip(axes.flat, idx):
    ax.plot(grid.centers, pdfs[i], label="TabPFN-3.5")
    ax.plot(grid.centers, pdfs_tabicl[i], label="TabICL")
    ax.axvline(z_test[i], color="k", ls="--", lw=1, label="true $z$")
for ax in axes[-1]:
    ax.set_xlabel("$z$")
for ax in axes[:, 0]:
    ax.set_ylabel("$p(z)$")
axes[0, 0].legend()

# %% [markdown]
# ## Beyond redshifts
#
# Nothing above depends on the target being a redshift. For any other
# continuous target, pass your own features and values to `fit`, and your own
# grid to `predict_proba`, for example `lazy.Grid.from_edges(np.linspace(0, 10,
# 201))` or no grid at all. The point metrics and plots then measure plain
# residuals, $y_\mathrm{pred} - y_\mathrm{true}$, in the target's units:
# `scale="1+y"` and `outlier_lines=True` were only for the photo-$z$
# convention.
#
# Next steps:
#
# - [Supported models](https://lazy-tfm.readthedocs.io/en/latest/models/index.html):
#   which model to choose, and what hardware it needs.
# - [Backends](https://lazy-tfm.readthedocs.io/en/latest/guide/backends.html):
#   ensembling, bagging, and large contexts.
# - [Metrics and figures](https://lazy-tfm.readthedocs.io/en/latest/guide/metrics.html).
