# The estimator API

It is scikit-learn's, with one difference. A photo-z model's natural output is a
density rather than a number, so
{meth}`~lazy.base.BasePhotoZEstimator.predict_proba` returns a density on a
redshift grid rather than class probabilities, and `predict` is a documented
reduction of it:

| Method                         | Returns                                                     |
| ------------------------------ | ----------------------------------------------------------- |
| `fit(X, y)`                    | The model. Stores the labelled context; no weights change.  |
| `predict_proba(X, z_grid)`     | `(n_samples, n_bins)` densities on `z_grid`.                |
| `predict_pdf(X, z_grid)`       | The same; an alias for when "pdf" reads better.             |
| `predict_cdf(X, z_grid)`       | The cumulative distributions.                               |
| `predict(X, method="z_peak")`  | One redshift per row: `z_peak`, `z_weight`, `z_mean` or `z_median`. |
| `score(X, y)`                  | Negative CDE loss, so higher is better.                     |
| `evaluate(X, y)`               | A one-row table of every diagnostic metric.                 |

The returned arrays are **densities**, normalised to unit trapezoid mass over
the bin centres, the convention every metric in {mod}`lazy.metrics` uses.
Multiply by `grid.widths` for per-bin probability masses.

`get_params`, `set_params` and {func}`sklearn.base.clone` all work, so the
models drop into scikit-learn pipelines and search objects unmodified.
`LazyModel` flattens its backend's parameters into its own, so
`GridSearchCV(model, {"n_estimators": [4, 8]})` needs no prefix. As in
scikit-learn, constructing a model only stores its parameters: they are
validated, and the checkpoint is loaded, at `fit`.

## The grid belongs to the prediction

Nothing about fitting depends on the output binning: the models place their
internal bins by the distribution of the *context* redshifts, and the grid only
enters at the final, exact rebinning step. So one fitted model answers on as
many grids as you like, without refitting:

```python
import numpy as np
from lazy import RedshiftGrid

model.predict_proba(X_test, RedshiftGrid.linear(0.0, 3.0, 300))
model.predict_proba(X_test, np.linspace(0.005, 2.995, 300))   # or bin centres
```

The constructor also takes `z_grid`, as a per-model default for when every call
would pass the same thing. A grid given at call time wins, and `None` at both
levels means {data}`~lazy.grid.DC1_GRID`, 200 bins over 0 < z < 2.

```{note}
Probability outside the grid is dropped and each density renormalised. If your
galaxies reach beyond z = 2, pass a grid that covers them.
```

## Point estimates

The four definitions disagree exactly when a density is multimodal: `z_mean`
lands between two peaks, where there is no probability, while `z_peak` and
`z_weight` pick one. To get several, call `predict_proba` once and reduce it
with {meth}`~lazy.base.BasePhotoZEstimator.point_estimates` or
{func}`lazy.metrics.grid_point_estimates`, rather than re-running the model for
each definition.
