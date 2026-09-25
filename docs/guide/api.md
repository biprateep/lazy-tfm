# The estimator API

It is scikit-learn's, with one difference. A photo-z model's natural output is a
distribution rather than a number, so
{meth}`~lazy.base.BasePhotoZEstimator.predict_proba` returns a density on a
redshift grid rather than class probabilities, and `predict` is a documented
reduction of it:

| Method                              | Returns                                                     |
| ----------------------------------- | ----------------------------------------------------------- |
| `fit(X, y)`                         | The model. Stores the labelled context; no weights change.  |
| `predict_proba(X, z_grid)`          | `(n_samples, n_bins)` densities on `z_grid`.                |
| `predict_pdf(X, z_grid)`            | The same; an alias for when "pdf" reads better.             |
| `predict_cdf(X, z_grid)`            | The cumulative distributions.                               |
| `predict_distribution(X)`           | The model's own distributions, before any grid (see below). |
| `predict_quantiles(X, quantiles)`   | `(n_samples, n_quantiles)` redshifts, exact.                |
| `predict(X, method="z_peak")`       | One redshift per row: `z_peak`, `z_weight`, `z_mean` or `z_median`. |
| `score(X, y)`                       | Negative CDE loss, so higher is better.                     |
| `evaluate(X, y)`                    | A one-row table of every diagnostic metric.                 |

`get_params`, `set_params` and {func}`sklearn.base.clone` all work, so the
models drop into scikit-learn pipelines and search objects unmodified.
`LazyModel` flattens its backend's parameters into its own, so
`GridSearchCV(model, {"n_estimators": [4, 8]})` needs no prefix. As in
scikit-learn, constructing a model only stores its parameters: they are
validated, and the checkpoint is loaded, at `fit`.

## Input data

Features can come in any of the forms scikit-learn users expect: a NumPy
array, a structured or record array, a pandas DataFrame, an astropy `Table` or
`QTable` (quantities give their values), or anything with a `to_pandas()`
method. Column names, when the input has them, are recorded as
`feature_names_in_` and checked at prediction time (the same columns in another
order are reordered; an unnamed table meeting a named fit, or the reverse, is
used by position with a warning).

Mark missing values with `NaN`; masked entries of an astropy table or masked
array become `NaN` too. Each model handles them its own way: TabPFN and LimiX
add missing-value indicators, TabICL and TabFM impute inside their
preprocessing. Infinities and non-numeric columns are rejected. Redshifts must
be finite.

## Distributions, quantiles and densities

Each model has a native form for its answer. TabPFN and LimiX-2 predict the
probability of each of several thousand buckets; TabICL predicts 999
quantiles; TabFM predicts the probability of equal-mass bins, averaged over
shifted copies of them. `predict_distribution` returns that answer as it is,
in a {mod}`lazy.distributions` object whose methods follow scipy.stats and
LSST DESC's [qp](https://github.com/LSSTDESC/qp):

```python
dist = model.predict_distribution(X_test)   # one per galaxy
dist.pdf(z)            # densities at any redshifts
dist.cdf(z)            # cumulative probabilities
dist.ppf([0.16, 0.5, 0.84])                # quantiles, exact
dist.mean(), dist.median(), dist.mode(), dist.std()
dist.interval(0.68)    # central credible interval
dist.rvs(100, random_state=0)              # samples
dist[:10]              # the first ten galaxies
ensemble = dist.to_qp()   # a qp.Ensemble, for RAIL (the qp extra)
```

The three kinds are {class}`~lazy.distributions.HistogramDistribution`
(bucket masses; TabPFN, LimiX), {class}`~lazy.distributions.QuantileDistribution`
(TabICL) and {class}`~lazy.distributions.MixtureDistribution` (histograms on
different buckets, as from bagging or TabFM's shifted bins). Every operation
is exact on the piecewise-linear CDF the model implies, so
`predict_quantiles(X, q)` is `predict_distribution(X).ppf(q)` and never goes
through a grid.

## The grid belongs to the prediction

Nothing about fitting depends on the output binning: the models place their
internal bins by the distribution of the *context* redshifts, and a grid only
enters at the final, exact integration of the distribution over its bins. So
one fitted model answers on as many grids as you like, without refitting:

```python
import numpy as np
from lazy import DC1_GRID, RedshiftGrid

model.predict_proba(X_test)                     # the model's native grid
model.predict_proba(X_test, DC1_GRID)           # the Data Challenge grid
model.predict_proba(X_test, RedshiftGrid.linear(0.0, 3.0, 300))
model.predict_proba(X_test, np.linspace(0.005, 2.995, 300))   # bin centres
```

A grid given at call time wins; the constructor's `z_grid` is the default for
when every call would pass the same thing; and when both are `None` a
foundation model answers on its **native grid**, `model.native_grid_`, which
loses nothing:

| Backend  | Native grid                                                         |
| -------- | ------------------------------------------------------------------- |
| `tabpfn` | The bar distribution's buckets, in full (5,000 on v3).               |
| `limix`  | LimiX-2's 5,000 buckets, mapped with the context's mean and spread. |
| `tabicl` | 1,000 equal bins over the context redshifts, padded by 2%.          |
| `tabfm`  | The union of every shifted copy's bin edges.                        |

Native bucket grids reach far into both tails, below zero included, because
that is where the buckets are. Pass a grid to restrict the range; `"native"`
asks for the native grid explicitly. A plain
{class}`~lazy.base.BasePhotoZEstimator` subclass defaults to
{data}`~lazy.grid.DC1_GRID`, 200 bins over 0 < z < 2.

## Two normalisations

A density on a grid can be normalised two ways, and a
{class}`~lazy.grid.RedshiftGrid` records which it uses:

- **`"trapezoid"`**, the default and the DC1 convention: unit trapezoid mass
  over the bin centres, `np.trapezoid(pdf, grid.centers) == 1`. Every grid you
  build with `RedshiftGrid.linear`, `from_edges` or `from_centers` uses it,
  and so does {data}`~lazy.grid.DC1_GRID`, so numbers on those grids are
  directly comparable with the Data Challenge.
- **`"histogram"`**: the density is constant across each bin,
  `(pdf * grid.widths).sum() == 1`. Native grids use it, because their bins are
  far from uniform and the trapezoid rule would misplace mass.

Multiply by `grid.widths` for per-bin probability masses in either case. The
metrics in {mod}`lazy.metrics` take `bin_edges=` to score a histogram grid
exactly, and `score`, `evaluate` and `point_estimates` pass it for you.

```{note}
Probability outside the grid is dropped and each density renormalised. If your
galaxies reach beyond a grid, pass one that covers them.
```

## Point estimates

The four definitions disagree exactly when a density is multimodal: `z_mean`
lands between two peaks, where there is no probability, while `z_peak` and
`z_weight` pick one. To get several, call `predict_proba` once and reduce it
with {meth}`~lazy.base.BasePhotoZEstimator.point_estimates` or
{func}`lazy.metrics.grid_point_estimates`, rather than re-running the model for
each definition. `z_median` is also `predict_quantiles(X, [0.5])`, computed
without a grid.
