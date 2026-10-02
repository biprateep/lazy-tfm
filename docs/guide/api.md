# The estimator API

The estimators in LAZY follow scikit-learn's API, with one difference. The
natural output of these models is a distribution rather than a number, so
{meth}`~lazy.base.BaseDensityRegressor.predict_proba` returns a density on a
grid of the target rather than class probabilities, and `predict` is a
documented reduction of it. The table lists every method of an estimator and
what it returns:

| Method                              | Returns                                                     |
| ----------------------------------- | ----------------------------------------------------------- |
| `fit(X, y)`                         | The model. Stores the labeled context; no weights change.   |
| `predict_proba(X, y_grid)`          | `(n_samples, n_bins)` densities on `y_grid`.                |
| `predict_pdf(X, y_grid)`            | The same; an alias for when "pdf" reads better.             |
| `predict_cdf(X, y_grid)`            | The cumulative distributions.                               |
| `predict_distribution(X)`           | The model's own distributions, before any grid (see below). |
| `predict_quantiles(X, quantiles)`   | `(n_samples, n_quantiles)` values, exact.                |
| `predict_interval(X, coverage)`     | `(n_samples, 2)` bounds of the central interval, exact.     |
| `predict(X, method="mode")`       | One value per row: `mode`, `peak_mean`, `mean` or `median`. |
| `score(X, y)`                       | Negative CDE loss, so higher is better.                     |
| `evaluate(X, y)`                    | A one-row table of every diagnostic metric.                 |

Since `get_params`, `set_params` and {func}`sklearn.base.clone` all work, the
models can be used in scikit-learn pipelines and search objects without any
modification. `LazyModel` flattens the parameters of its backend into its own,
so `GridSearchCV(model, {"n_estimators": [4, 8]})` needs no prefix. As in
scikit-learn, constructing a model only stores its parameters, which are
validated (and the checkpoint loaded) at `fit`.

## Input data

The features can be given in any of the forms that scikit-learn users expect:
a NumPy array, a structured or record array, a pandas DataFrame, an astropy
`Table` or `QTable` (whose quantities give their values), or anything with a
`to_pandas()` method. When the input has column names, they are recorded as
`feature_names_in_` and checked at prediction time. The same columns in a
different order are reordered, while an unnamed table meeting a named fit, or
the reverse, is used by position with a warning.

Missing values are marked with `NaN`, and the masked entries of an astropy
table or masked array also become `NaN`. Each model handles them in its own
way: TabPFN and LimiX add missing-value indicators, while TabICL and TabFM
impute them inside their preprocessing. Infinities and non-numeric columns are
rejected, and the target values must be finite.

## Distributions, quantiles and densities

Each model has a native form for its answer. TabPFN and LimiX-2 predict the
probability of each of several thousand buckets, TabICL predicts 999
quantiles, and TabFM predicts the probability of equal-mass bins, averaged
over shifted copies of them. `predict_distribution` returns this answer as it
is, in a {mod}`lazy.distributions` object whose methods follow scipy.stats and
LSST DESC's [qp](https://github.com/LSSTDESC/qp):

```python
dist = model.predict_distribution(X_test)   # one per row
dist.pdf(z)            # densities at any values of z
dist.cdf(z)            # cumulative probabilities
dist.ppf([0.16, 0.5, 0.84])                # quantiles, exact
dist.mean(), dist.median(), dist.mode(), dist.std()
dist.interval(0.68)    # central credible interval
dist.rvs(100, random_state=0)              # samples
dist[:10]              # the first ten rows
ensemble = dist.to_qp()   # a qp.Ensemble, for RAIL (the qp extra)
```

There are three kinds of distribution:
{class}`~lazy.distributions.HistogramDistribution` (bucket masses, for TabPFN
and LimiX), {class}`~lazy.distributions.QuantileDistribution` (for TabICL) and
{class}`~lazy.distributions.MixtureDistribution` (histograms on different
buckets, as from bagging or TabFM's shifted bins). Every operation is exact on
the piecewise-linear CDF that the model implies. Therefore,
`predict_quantiles(X, q)` is `predict_distribution(X).ppf(q)` and never goes
through a grid.

## The grid belongs to the prediction

Nothing about fitting depends on the output binning. The models place their
internal bins according to the distribution of the *context* targets, and a
grid only enters at the final, exact integration of the distribution over its
bins. Therefore, one fitted model can answer on as many grids as needed,
without being refitted:

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
serves as the default for when every call would pass the same grid. When both
are `None`, a foundation model answers on its **native grid**,
`model.native_grid_`, which loses no information. The table lists the native
grid of each backend:

| Backend  | Native grid                                                         |
| -------- | ------------------------------------------------------------------- |
| `tabpfn` | The bar distribution's buckets, in full (5,000 on v3).               |
| `limix`  | LimiX-2's 5,000 buckets, mapped with the context's mean and spread. |
| `tabicl` | 1,500 equal bins over the context targets, padded by 25% each side. |
| `tabfm`  | The union of every shifted copy's bin edges.                        |

The native bucket grids reach far into both tails, including below zero,
because that is where the buckets are. Passing a grid restricts the range, and
`"native"` asks for the native grid explicitly. A plain
{class}`~lazy.base.BaseDensityRegressor` subclass, which has no native grid,
defaults to 200 equal bins over the range of the training targets, padded by
5% on each side.

## Two normalizations

A density on a grid can be normalized in two ways, and a
{class}`~lazy.grid.Grid` records which one it uses. The default,
**`"trapezoid"`**, is the DC1 convention, in which the density has unit
trapezoid mass over the bin centers, `np.trapezoid(pdf, grid.centers) == 1`.
Every grid built with `Grid.linear`, `from_edges` or `from_centers` uses it,
and so does {data}`~lazy.datasets.DC1_GRID`, so the numbers computed on those
grids are directly comparable with the Data Challenge. Under
**`"histogram"`**, the density is constant across each bin,
`(pdf * grid.widths).sum() == 1`. The native grids use this normalization,
because their bins are far from uniform and the trapezoid rule would misplace
mass.

In either case, multiplying by `grid.widths` gives the probability mass in
each bin. The metrics in {mod}`lazy.metrics` take `bin_edges=` to score a
histogram grid exactly, and `score`, `evaluate` and `point_estimates` pass it
automatically.

```{note}
Probability outside the grid is dropped and each density is renormalized. If
the targets reach beyond a grid, pass one that covers them.
```

## Point estimates

The four definitions of a point estimate disagree exactly when a density is
multimodal: `mean` lands between two peaks, where there is no probability,
while `mode` and `peak_mean` pick one of them. To obtain several of them, call
`predict_proba` once and reduce its output with
{meth}`~lazy.base.BaseDensityRegressor.point_estimates` or
{func}`lazy.metrics.grid_point_estimates`, rather than running the model again
for each definition. The `median` is also `predict_quantiles(X, [0.5])`, which
is computed without a grid.
