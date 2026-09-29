# Metrics and figures

{mod}`lazy.metrics` implements the LSST DESC PZ Data Challenge metrics, read off
the challenge's own scripts rather than paraphrased, so the numbers are directly
comparable with Schmidt et al. (2020). {func}`~lazy.metrics.summarize` scores
densities you already have, from any method:

```python
import pandas as pd
from lazy.metrics import summarize

table = pd.concat([
    summarize(z_true, grid, pdfs_a, label="TabPFN"),
    summarize(z_true, grid, pdfs_b, label="TabICL"),
])
```

`model.evaluate(X, y)` is the same table for one fitted model.

Pass the {class}`~lazy.grid.Grid` the densities are on, and each metric scores
them by the grid's own normalisation: exactly, bin by bin, on a model's
histogram-normalised native grid, and by the Data Challenge's trapezoid rule
on grids such as {data}`~lazy.grid.DC1_GRID`, where the numbers are then the
challenge's, byte for byte. (Bin centres alone also work, and are read as a
trapezoid grid.)

The PDF metrics -- CDE loss, PIT and its goodness-of-fit statistics -- apply
to any target. The point metrics (bias, scatter, outlier rates) follow the
photo-z convention of dividing each residual by `1 + z_true`; for any other
target pass `scale="none"` to `summarize` or `evaluate`, which uses plain
residuals `z_pred - z_true`.

{mod}`lazy.plotting` draws the standard diagnostics (accuracy, calibration and
N(z)) identically for every method, so comparisons hold up by eye:

```python
from lazy.plotting import diagnostic_panel

fig = diagnostic_panel(z_true, grid.centers, pdfs, label="TabPFN")
```
