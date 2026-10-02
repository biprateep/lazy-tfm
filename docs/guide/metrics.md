# Metrics and figures

{mod}`lazy.metrics` implements the LSST DESC PZ Data Challenge metrics, read off
the challenge's own scripts rather than paraphrased, so the numbers are directly
comparable with Schmidt et al. (2020). {func}`~lazy.metrics.summarize` scores
densities you already have, from any method:

```python
import pandas as pd
from lazy.metrics import summarize

table = pd.concat([
    summarize(y_true, grid, pdfs_a, label="TabPFN"),
    summarize(y_true, grid, pdfs_b, label="TabICL"),
])
```

`model.evaluate(X, y)` is the same table for one fitted model.

Pass the {class}`~lazy.grid.Grid` the densities are on, and each metric scores
them by the grid's own normalisation: exactly, bin by bin, on a model's
histogram-normalised native grid, and by the Data Challenge's trapezoid rule
on grids such as {data}`~lazy.datasets.DC1_GRID`, where the numbers are then the
challenge's, byte for byte. (Bin centres alone also work, and are read as a
trapezoid grid.)

The PDF metrics -- CDE loss, PIT and its goodness-of-fit statistics -- apply
to any target. The point metrics (bias, scatter, outlier rates) measure plain
residuals, `y_pred - y_true`, in the target's units. For photometric redshifts,
whose errors grow with `1 + z`, pass `scale="1+y"` to `summarize` or `evaluate`
to divide each residual by `1 + y_true`, as the Data Challenge did; its
published numbers need that.

{mod}`lazy.plotting` draws the standard diagnostics (accuracy, calibration and
the sample distribution) identically for every method, so comparisons hold up
by eye:

```python
from lazy.plotting import diagnostic_panel

fig = diagnostic_panel(y_true, grid, pdfs, label="TabPFN")   # a Grid, or bin centres
fig = diagnostic_panel(y_true, grid, pdfs, scale="1+y")      # photo-z residuals
```
