# Metrics and figures

{mod}`lazy.metrics` implements the metrics of the LSST DESC PZ Data Challenge,
read off the challenge's own scripts rather than paraphrased, so that the
numbers are directly comparable with those of Schmidt et al. (2020).
{func}`~lazy.metrics.summarize` scores densities that the user already has,
from any method:

```python
import pandas as pd
from lazy.metrics import summarize

table = pd.concat([
    summarize(y_true, grid, pdfs_a, label="TabPFN"),
    summarize(y_true, grid, pdfs_b, label="TabICL"),
])
```

For one fitted model, `model.evaluate(X, y)` returns the same table.

Each metric takes the {class}`~lazy.grid.Grid` that the densities are on and
scores them according to the grid's own normalization. On a model's
histogram-normalized native grid it scores them exactly, bin by bin, and on
grids such as {data}`~lazy.datasets.DC1_GRID` it uses the Data Challenge's
trapezoid rule, so that the numbers are then the challenge's, byte for byte.
Bin centers alone also work, and are read as a trapezoid grid.

The PDF metrics (CDE loss, PIT and its goodness-of-fit statistics) apply to
any target, while the point metrics (bias, scatter and outlier rates) measure
plain residuals, `y_pred - y_true`, in the units of the target. For
photometric redshifts, whose errors grow with `1 + z`, passing `scale="1+y"`
to `summarize` or `evaluate` divides each residual by `1 + y_true`, as the
Data Challenge did, which is needed to reproduce its published numbers.

{mod}`lazy.plotting` draws the standard diagnostics (accuracy, calibration and
the sample distribution) identically for every method, so that the
comparisons between methods hold up by eye:

```python
from lazy.plotting import diagnostic_panel

fig = diagnostic_panel(y_true, grid, pdfs, label="TabPFN")   # a Grid, or bin centers
fig = diagnostic_panel(y_true, grid, pdfs, scale="1+y")      # photo-z residuals
```
