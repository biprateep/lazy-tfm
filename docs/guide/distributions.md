# From model output to distribution

The estimators in LAZY follow scikit-learn's API, with one difference. The
natural output of these models is a distribution, so
{meth}`~lazy.base.BaseDensityRegressor.predict_proba` returns a density on a
grid of the target rather than class probabilities, and `predict` is a
documented reduction of it. This page describes how each model's answer
becomes a distribution, how it is put on a grid and reduced to a number, and
how LAZY scores it. The table lists every method of an estimator and what it
returns:

| Method                              | Returns                                                     |
| ----------------------------------- | ----------------------------------------------------------- |
| `fit(X, y)`                         | The model. Stores the labeled context; no weights change.   |
| `predict_proba(X, y_grid)`          | `(n_samples, n_bins)` densities on `y_grid`.                |
| `predict_pdf(X, y_grid)`            | The same; an alias for when "pdf" reads better.             |
| `predict_cdf(X, y_grid)`            | The cumulative distributions.                               |
| `predict_distribution(X)`           | The model's own distributions, before any grid (see below). |
| `predict_quantiles(X, quantiles)`   | `(n_samples, n_quantiles)` values, exact.                   |
| `predict_interval(X, coverage)`     | `(n_samples, 2)` bounds of the central interval, exact.     |
| `predict_pit(X, y)`                 | `(n_samples,)` PIT values (CDF at the true value), exact.   |
| `predict(X, method="mode")`         | One value per row: `mode`, `peak_mean`, `mean` or `median`. |
| `score(X, y)`                       | Negative CDE loss, so higher is better.                     |
| `evaluate(X, y)`                    | A one-row table of every diagnostic metric.                 |

## Each model's native output

TabPFN and LimiX-2 predict the probability of each of several thousand
buckets, TabICL predicts 999 quantiles, and TabFM predicts the probability of
equal-mass bins, averaged over shifted copies of them. LAZY keeps each of these
as one of three kinds of distribution: {class}`~lazy.distributions.HistogramDistribution` (bucket
masses, for TabPFN and LimiX), {class}`~lazy.distributions.QuantileDistribution`
(for TabICL) and {class}`~lazy.distributions.MixtureDistribution` (histograms
on different buckets, as from bagging or TabFM's shifted bins). Every
operation is exact on the piecewise-linear CDF that the model implies.

`predict_distribution` returns this answer as it is, as a
{mod}`lazy.distributions` object whose methods follow scipy.stats and LSST
DESC's [qp](https://github.com/LSSTDESC/qp):

```python
dist = model.predict_distribution(X_test)   # one per row
dist.pdf(z)            # densities at any values of z
dist.cdf(z)            # cumulative probabilities
dist.pit(z_true)       # each row's CDF at its own value
dist.ppf([0.16, 0.5, 0.84])                # quantiles, exact
dist.mean(), dist.median(), dist.mode(), dist.std()
dist.interval(0.68)    # central credible interval
dist.rvs(100, random_state=0)              # samples
dist[:10]              # the first ten rows
ensemble = dist.to_qp()   # a qp.Ensemble, for RAIL (the qp extra)
```

Since the operations are exact, `predict_quantiles(X, q)` is
`predict_distribution(X).ppf(q)` and never goes through a grid, and
`predict_interval(X, coverage)` gives the bounds of the central interval in the
same way. The `pdf` and `cdf` methods evaluate every row at every value, so
`dist.cdf(z_true)` has shape `(n, n)`. The `pit` method instead evaluates
each row at its own value, $F_i(y_i)$, which is the diagonal of that array,
and `predict_pit(X, y)` is `predict_distribution(X).pit(y)`. For RAIL and
other qp users, `to_qp` converts any of the three kinds to a `qp.Ensemble`, and
{func}`~lazy.distributions.from_qp` converts a histogram or quantile ensemble
back (both need the `qp` extra, see {doc}`../installation`).

## The grid belongs to the prediction

Fitting does not depend on the output binning. The models place their
internal bins according to the distribution of the *context* targets, and a
grid only enters at the final, exact integration of the distribution over its
bins. Therefore, one fitted model can answer on any number of grids without
being refitted:

```python
import numpy as np
from lazy import Grid
from lazy.datasets import DC1_GRID

model.predict_proba(X_test)                     # the model's native grid
model.predict_proba(X_test, Grid.linear(0.0, 3.0, 300))
model.predict_proba(X_test, DC1_GRID)           # the Data Challenge's grid
model.predict_proba(X_test, np.linspace(0.005, 2.995, 300))   # bin centers
```

A grid given at call time takes precedence, and the constructor's `y_grid`
is the default for when every call would pass the same grid. When both
are `None`, a foundation model answers on its **native grid**,
`model.native_grid_`, which loses no information. The table lists the native
grid of each backend:

| Backend  | Native grid                                                         |
| -------- | ------------------------------------------------------------------- |
| `tabpfn` | The bar distribution's buckets, in full (5,000 on v3).               |
| `limix`  | LimiX-2's 5,000 buckets, mapped with the context's mean and spread. |
| `tabicl` | 1,500 equal bins over the context targets, padded by 25% each side. |
| `tabfm`  | The union of every shifted copy's bin edges.                        |

The native bucket grids reach far into both tails, including below zero.
Passing a grid restricts the range, and `"native"` asks for the native grid
explicitly. A plain
{class}`~lazy.base.BaseDensityRegressor` subclass, which has no native grid,
defaults to 200 equal bins over the range of the training targets, padded by
5% on each side. Probability outside the grid is dropped and each density is
renormalized, so a grid should cover the targets (see {doc}`limits`).

On any grid, `predict_proba(X, grid)` is
`predict_distribution(X).on_grid(grid)`. Each bin receives the exact
probability that the distribution places inside it, divided by the bin width,
and each row is then renormalized by the grid's normalization (see below).
Therefore, the densities differ from the bin averages of `dist.pdf` when part
of the distribution lies outside the grid (e.g., the native buckets below zero
on {data}`~lazy.datasets.DC1_GRID`), and on a trapezoid grid, whose rule gives
the first and last bins only half their width. The probability in each bin
before the renormalization is `dist.histogramize(grid.edges).masses`.

## Two normalizations

A density on a grid can be normalized in two ways, and a
{class}`~lazy.grid.Grid` records which one it uses. The default,
**`"trapezoid"`**, is the DC1 convention, with unit trapezoid mass over the bin
centers, `np.trapezoid(pdf, grid.centers) == 1`.
Every grid built with `Grid.linear`, `from_edges` or `from_centers` uses it,
and so does {data}`~lazy.datasets.DC1_GRID`, so the numbers computed on those
grids are directly comparable with the Data Challenge. Under
**`"histogram"`**, the density is constant across each bin,
`(pdf * grid.widths).sum() == 1`. The native grids use this normalization,
because their bins are far from uniform and the trapezoid rule would misplace
mass. In either case, multiplying by `grid.widths` gives the probability mass
in each bin.

## Point estimates

`predict` reduces each density to one value in one of four ways. The `mode`
is the center of the highest bin (DC1's `z_PEAK`), and `peak_mean` is the
probability-weighted mean over the main peak (DC1's `z_WEIGHT`), where the
main peak is the contiguous run of bins around the mode in which the density
stays above 5% of its peak value. The `mean` and the `median` are those of the
whole density. The four definitions disagree exactly when a density is
multimodal: `mean` lands between two peaks, where there is no probability,
while `mode` and `peak_mean` pick one of them. To obtain several of them, we
recommend calling `predict_proba` once and reducing its output with
{meth}`~lazy.base.BaseDensityRegressor.point_estimates` or
{func}`lazy.metrics.grid_point_estimates`, rather than rerunning the model for
each definition. The `median` is also `predict_quantiles(X, [0.5])`, which
is computed without a grid.

## Metrics

{mod}`lazy.metrics` implements the metrics of the LSST DESC PZ Data Challenge,
read off the challenge's own scripts rather than paraphrased, so that the
numbers are directly comparable with those of Schmidt et al. (2020). Each
metric takes the {class}`~lazy.grid.Grid` that the densities are on and scores
them according to the grid's own normalization. On a model's
histogram-normalized native grid it scores them exactly, bin by bin, and on
grids such as {data}`~lazy.datasets.DC1_GRID` it uses the Data Challenge's
trapezoid rule, so that the numbers are the challenge's, byte for byte. Bin
centers alone also work, and are read as a trapezoid grid. The metrics
take `bin_edges=` to score a histogram grid exactly, and `score`, `evaluate`
and `point_estimates` pass it automatically.

The PDF metrics apply to any target. The conditional density estimate (CDE)
loss, {func}`~lazy.metrics.cde_loss`, is the mean over objects of the integral
of $p^2$ minus twice the density at the true value, which is the L2 distance
between the estimated and true conditional densities up to a constant that
does not depend on the estimator, so it ranks methods without knowing the true
density (lower is better, and `score` returns its negative). The probability
integral transform (PIT) of an object is its predicted CDF at the true value,
and the PIT values of a well-calibrated set of densities are uniform on [0,
1]. The metrics compute it from the densities on a grid, while
`predict_pit` computes it exactly from the native distribution.
{func}`~lazy.metrics.pit_statistics` measures their departure from
uniformity in the ways DC1 quotes, each sensitive to a different departure:
the Kolmogorov-Smirnov statistic (and its p-value), the Cramér-von Mises
statistic, DC1's Anderson-Darling statistic on two cuts, (0.05, 0.95) and
(0.01, 0.99), the RMS difference from the uniform quantiles, the KL divergence
of the binned PIT histogram from uniform, and the fraction of PIT values
within 1e-4 of 0 or 1, which marks catastrophic failures. Three further
scores also apply to any target. The continuous ranked probability score,
{func}`~lazy.metrics.crps`, is the integral of the squared difference between
the predicted CDF and the step function at the true value. The negative log
likelihood, {func}`~lazy.metrics.nll`, is the mean of minus the logarithm of
the density at the true value. The pinball loss,
{func}`~lazy.metrics.pinball_loss`, scores a predicted quantile at level
$\tau$ by $\tau$ times the residual when the truth lies above it and $1 -
\tau$ times it when the truth lies below. Lower is better for all three.

The point metrics measure plain residuals, `y_pred - y_true`, in the units of
the target: the bias (their median), $\sigma_{\rm IQR}$ (the interquartile
range divided by 1.349), $\sigma_{\rm MAD}$, the outlier rate beyond
`max(0.06, 3 * sigma_IQR)`, the rate beyond a fixed 0.15 and the median
absolute error. The outlier thresholds are therefore in the units of the
target too. For photometric redshifts, whose errors grow with `1 + z`, passing
`scale="1+y"` to `summarize` or `evaluate` divides each residual by `1 +
y_true`, as the Data Challenge did, which is needed to reproduce its published
numbers.

{func}`~lazy.metrics.summarize` scores densities that the user already has,
from any method, and returns every metric as a one-row table, with `crps` and
`nll` columns beside the CDE loss and the PIT statistics. For one fitted model,
`model.evaluate(X, y)` returns the same table:

```python
import pandas as pd
from lazy.metrics import summarize

table = pd.concat([
    summarize(y_true, grid, pdfs_a, label="TabPFN"),
    summarize(y_true, grid, pdfs_b, label="TabICL"),
])
```

{mod}`lazy.plotting` draws the standard diagnostics (accuracy, calibration and
the sample distribution) identically for every method, so that
comparisons between methods hold up by eye:

```python
from lazy.plotting import diagnostic_panel

fig = diagnostic_panel(y_true, grid, pdfs, label="TabPFN")   # a Grid, or bin centers
fig = diagnostic_panel(y_true, grid, pdfs, scale="1+y")      # photo-z residuals
```
