# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: T4
#     provenance: []
#   gallery:
#     description: Which features and context rows drive a prediction.
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: accelerator,colab,gallery
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
# # Feature importance
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biprateep/lazy-tfm/blob/tutorials/feature_importance.ipynb)
# [![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?logo=github)](https://github.com/biprateep/lazy-tfm/blob/tutorials/feature_importance.ipynb)
#
# A tabular foundation model has no coefficients to read and no tree splits to
# count. To learn which features a prediction depends on, we change the inputs
# and watch the outputs. `lazy.interp` does this for a whole predicted
# distribution, not only for a point estimate. It attributes three statistics
# of each row's distribution to the features: the CRPS and the negative log
# likelihood (NLL), which score the distribution against the true target, and
# the predicted mean.
#
# In this tutorial, we use it on a small engineering dataset in four ways:
# permutation importance, Shapley values by refit, integrated gradients, and
# attributions to the rows of the context. The
# [Interpretability](https://lazy-tfm.readthedocs.io/en/latest/guide/interpretability.html)
# page of the user guide gives the equations of each method.

# %% [markdown]
# ## Setup
#
# On Google Colab, the cell below installs the package with the TabICL
# backend. Elsewhere, install it first with `pip install 'lazy-tfm[tabicl]'`.

# %%
import sys

if "google.colab" in sys.modules:
    # %pip install -q 'lazy-tfm[tabicl]'
    pass

# %%
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Arrays
import pandas as pd  # Tables
from sklearn import model_selection  # Splits

import lazy  # The topic of this tutorial
from lazy import datasets
from lazy import interp
from lazy import plotting

SEED = 299792458  # One seed for the split, the models and the explanations

plotting.use_style()
plt.rcParams["figure.dpi"] = 110  # Readable in a notebook

# %% [markdown]
# ## The data and the model
#
# The concrete dataset (see
# [Demo datasets](https://lazy-tfm.readthedocs.io/en/latest/guide/datasets.html))
# gives the compressive strength, in MPa, of 1,030 concrete samples from the
# amounts of their seven ingredients, in kg per cubic meter of concrete, and
# their age in days. It is a convenient test of feature importance, since the
# physics is well known: concrete gains strength as it cures, with more
# cement, and with less water for the same cement. We hold out 20% of the rows
# for testing, and write the column names with spaces for the figures.

# %%
data = datasets.load_dataset("concrete")
X = data.X.rename(columns=lambda name: name.replace("_", " "))
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, data.y, test_size=0.2, random_state=SEED
)
print(f"{len(X_train):,} context rows, {len(X_test):,} test rows")
X_train.head()

# %% [markdown]
# We use TabICL, which is small and quick on any hardware, since the methods
# below predict the test set many times over. Any other model of
# `lazy.LazyModel` can be swapped in without changing the rest of the code. As
# a reminder of how well the model does before we ask why, we report its CRPS
# (in MPa, lower is better) and the fraction of test rows inside their central
# 68% interval.

# %%
model = lazy.LazyModel("tabicl", random_state=SEED, progress=False)
model.fit(X_train, y_train)

low, high = model.predict_interval(X_test, coverage=0.68).T
coverage = np.mean((y_test >= low) & (y_test <= high))
scores = model.evaluate(X_test, y_test, method="mode")
print(f"CRPS: {scores['crps'].iloc[0]:.2f} MPa")
print(f"68% interval coverage: {coverage:.0%}")
print(f"Median width of the 68% interval: {np.median(high - low):.1f} MPa")

# %% [markdown]
# Every method below explains the test rows, not the context rows. A model
# recognizes its context rows and predicts their targets whatever the
# features, so every feature would look less important than it is.
#
# ## Permutation importance
#
# Permutation importance shuffles one feature across the test rows, which
# breaks its relation to the target while keeping its distribution, and
# measures how much each statistic changes as a result. A feature the model
# does not use loses nothing when shuffled, and the more the model relies on a
# feature, the larger the loss. The shuffle is repeated `n_repeats` times (5
# by default), and the losses are averaged. The method keeps the fitted model
# and its context, so `interp.explain` needs only the rows to explain and
# their targets.

# %%
permutation = interp.explain(
    model,
    X_test,
    y_test,
    method="permutation",
    random_state=SEED,
)
permutation.importance().round(3)

# %% [markdown]
# `importance()` averages each feature's attributions over the test rows. The
# scores are negated, so a positive value means that the feature improves the
# prediction: it lowers the CRPS (in MPa) or the NLL (in nats). For the mean,
# the importance is the absolute change of the predicted mean, averaged over
# the shuffles and then over the rows, in MPa.
# `Explanation.plot` draws one statistic's importance as bars, with the most
# important feature at the top.

# %%
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), layout="constrained")
for ax, statistic in zip(axes, ["crps", "nll"], strict=True):
    permutation.plot(statistic, ax=ax)

# %% [markdown]
# Age and cement should dominate, followed by blast furnace slag and water.
# This matches the physics of concrete: strength grows with curing time and
# with the cement and slag that bind the mix, and falls with excess water.
#
# Permutation importance has a known weakness with correlated features. A
# shuffled row can pair values that never occur together, such as much fly
# ash with much cement, so the model is asked about mixes unlike any in its
# context. The ingredients of concrete are correlated, so these numbers rank
# the features rather than measure their effects.

# %% [markdown]
# ## Shapley values by refit
#
# Shapley values, from cooperative game theory, give each feature its average
# gain over every order in which the features can be added to the model
# ([Lundberg & Lee 2017](https://arxiv.org/abs/1705.07874)). The values of a
# row add up to the difference between its statistic with every feature and
# with none.
#
# Every removal method must decide what the model predicts without a feature.
# `interp.explain` removes the feature's column from the context and from the
# test rows, and fits a clone of the model on the columns that remain. No
# weights are trained, so each refit costs about one pass over the context and
# the test rows.
#
# Setting a feature to a missing value instead would mean something different
# on each model, since each fills a missing value in its own way. Removal by
# refit means the same on every backend, so it is the default: given a
# context, `explain` computes Shapley values by refit. With eight features, it
# evaluates all 256 subsets exactly.

# %%
shapley = interp.explain(
    model,
    X_test,
    y_test,
    X_train,
    y_train,
    method="shapley",
    random_state=SEED,
)

# %% [markdown]
# The figure shows the Shapley importance of each feature for the three
# statistics. Averaged over rows, the Shapley values of a score measure how
# much each feature lowers the average loss.

# %% tags=["thumbnail"]
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), layout="constrained")
for ax, statistic in zip(axes, interp.STATISTICS, strict=True):
    shapley.plot(statistic, ax=ax)

# %% [markdown]
# The Shapley values should agree with permutation importance on the features
# that matter most.
# The mean ranks features only by how they move the predicted value. Both
# scores also reward a feature that narrows the distribution around the
# target, and the NLL, which reads the density at the target, weighs this
# most. So a feature that ranks higher for the NLL than for the mean shapes
# the uncertainty more than the predicted value. The table puts the three
# importances side by side, ranked by the NLL.

# %%
shapley.importance().sort_values("nll", ascending=False).round(3)

# %% [markdown]
# The values of each row also give the direction of an effect.
# `table("mean")` holds each test row's contributions to its predicted mean.
# The figure plots the contribution of age against age.

# %%
fig, ax = plt.subplots(figsize=(5.0, 3.6), layout="constrained")
ax.scatter(X_test["age"], shapley.table("mean")["age"], s=8)
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("Age (days)")
ax.set_ylabel("Contribution to the mean (MPa)")
plt.show()

# %% [markdown]
# The effect of age should be strongly nonlinear: young samples are predicted
# weaker and old samples stronger, while most samples, tested at the common
# curing age of 28 days, sit near zero.

# %% [markdown]
# ## Integrated gradients
#
# The removal methods need only `fit` and `predict`, so they work on any model.
# Their cost grows with the number of features, since each subset or shuffle
# is a new prediction of every row. A gradient method instead takes the
# derivatives of a statistic in the features through the model's
# differentiable forward pass, with no refit. Integrated gradients
# ([Sundararajan et al. 2017](https://arxiv.org/abs/1703.01365)) averages the
# gradient along the straight path from a baseline row to each row, in
# `n_steps` steps (32 by default), and multiplies it by the row's offset from
# the baseline. The baseline stands in for no feature at all; by default it is
# the mean of the context.
#
# The values of a row add up to the statistic at the row minus that at the
# baseline, up to the error of the sum over the steps.
#
# The gradient is taken in the columns as the model receives them, so a
# feature built by a pipeline step, such as an encoder, gets the credit, not
# the column it came from. The forward pass is differentiable only when
# nothing between the columns and the network lacks a gradient. TabICL's own
# recipe applies a power transform to half of its members and clips outliers,
# so we fit a second model without them: `transforms=("none",)` and
# `outlier_threshold=None`. TabICL still standardizes every column. The
# gradient methods also check that their forward pass gives the same means as
# `predict`. On a GPU, TabICL computes in float16 by default, whose rounding
# can fail that check, so the second model computes in float32:
# `mixed_precision=False`. We check what the changes cost in CRPS.

# %%
smooth = lazy.LazyModel(
    "tabicl",
    transforms=("none",),
    outlier_threshold=None,
    mixed_precision=False,
    random_state=SEED,
    progress=False,
)
smooth.fit(X_train, y_train)
smooth_crps = smooth.evaluate(X_test, y_test, method="mode")["crps"].iloc[0]
print(f"CRPS without transforms or clip: {smooth_crps:.2f} MPa")

# %% [markdown]
# The CRPS should be close to that of the default recipe, so the second model
# is a fair stand-in for the first.
#
# The gradient methods need the context the model was fitted on, as
# `X_context` and `y_context`. The predicted mean bends along the path from
# the baseline, so we take 128 steps rather than the default 32, and check the
# error of the sum.

# %%
gradients = interp.explain(
    smooth,
    X_test,
    y_test,
    X_train,
    y_train,
    method="integrated_gradients",
    n_steps=128,
)
gap = gradients.full_values - gradients.base_values - gradients.values.sum(-1)
mean_gap = np.abs(gap[interp.STATISTICS.index("mean")])
print(f"Error of the sum for the mean: median {np.median(mean_gap):.2f} MPa,")
print(f"largest {mean_gap.max():.2f} MPa")

# %% [markdown]
# The error of the sum should be small next to the attributions themselves,
# which reach several MPa; a larger `n_steps` lowers it further.
# To compare the methods on the same model, we compute Shapley values of the
# second model too. The table gives each feature's importance for the
# predicted mean under the two methods.

# %%
smooth_shapley = interp.explain(
    smooth,
    X_test,
    y_test,
    X_train,
    y_train,
    method="shapley",
    random_state=SEED,
)
pd.DataFrame(
    {
        "Shapley (refit)": smooth_shapley.importance()["mean"],
        "Integrated gradients": gradients.importance()["mean"],
    }
).sort_values("Shapley (refit)", ascending=False).round(2)

# %% [markdown]
# Both methods should put age and cement first, though possibly in another
# order, and integrated gradients gives blast furnace slag much more weight.
# The methods need not agree. Shapley values remove a feature, while
# integrated gradients moves it from the context mean. `method="gradient"`
# (gradient times input) is cheaper still: one gradient at the row instead of
# `n_steps`, which is exact only for a statistic linear in the features.

# %%
gradients.plot("mean")
plt.show()

# %% [markdown]
# ## Which context rows drive a prediction
#
# A prediction depends on the context as much as on the features of the row.
# `interp.explain_context` attributes the same three statistics to the rows of
# the context. Its default method, `"gradient"`, gives each context row the
# gradient of the statistic in that row's features, times their offsets from
# the baseline, summed over the features. It takes three backward passes per
# explained row, with no refit, and needs the same differentiable fit. We
# explain three test rows.

# %%
X_few, y_few = X_test.iloc[:3], y_test[:3]
context_explanation = interp.explain_context(
    smooth, X_few, y_few, X_train, y_train
)
X_few.assign(strength=y_few).round(1)

# %% [markdown]
# `top(k, statistic)` lists, for each explained row, the `k` context rows with
# the largest absolute attribution, by their position in `X_train`. We rank by
# the predicted mean: a row's attributions to the NLL can be noisy, since the
# NLL depends on the tails of the distribution. We join the rows to their
# features and targets.

# %%
top = context_explanation.top(3, "mean")
neighbours = X_train.iloc[top["context_row"]].reset_index(drop=True)
neighbours["strength"] = y_train[top["context_row"]]
pd.concat([top, neighbours], axis=1).round(2)

# %% [markdown]
# The most influential context rows of a test row should share its mix, cement
# included: often the same recipe, tested at another age.
# The gradient does not see a context row's target, so a row with a wrong
# target but typical features can get a small value. `method="loo"` refits the
# model without each context row instead: it sees the targets and works on any
# model, at one fit per context row; `rows` limits it to some of them.

# %% [markdown]
# ## Summary
#
# `interp.explain` attributes the CRPS, the NLL and the predicted mean of each
# row to the features, so it shows which features drive the uncertainty of a
# prediction as well as its value. Permutation importance keeps the fitted
# model and needs no context. Shapley values by refit, the default when a
# context is given, remove a feature the same way on every backend.
# Integrated gradients needs no refit and gives a cheap per-row attribution,
# but needs a fit with a differentiable forward pass, and attributes to the
# columns as the model receives them. `interp.explain_context` asks the same
# question of the context rows. Every method explains rows outside the
# context, which the model would otherwise recall.
