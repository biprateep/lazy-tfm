# Metrics and figures

{mod}`lazy.metrics` implements the LSST DESC PZ Data Challenge metrics, read off
the challenge's own scripts rather than paraphrased, so the numbers are directly
comparable with Schmidt et al. (2020). {func}`~lazy.metrics.summarize` scores
densities you already have, from any method:

```python
import pandas as pd
from lazy.metrics import summarize

table = pd.concat([
    summarize(z_true, grid.centers, pdfs_a, label="TabPFN"),
    summarize(z_true, grid.centers, pdfs_b, label="TabICL"),
])
```

`model.evaluate(X, y)` is the same table for one fitted model.

Densities on a histogram-normalised grid -- a model's native grid, whose bins
are far from uniform -- are scored exactly by passing the grid's edges:
`summarize(z_true, grid.centers, pdfs, bin_edges=grid.histogram_edges)`, and
likewise for {func}`~lazy.metrics.cde_loss` and the other metrics. The
estimators' `score` and `evaluate` do this themselves. On the usual
trapezoid grids, such as {data}`~lazy.grid.DC1_GRID`, leave it out: the
numbers are then the Data Challenge's, byte for byte.

{mod}`lazy.plotting` draws the standard diagnostics (accuracy, calibration and
N(z)) identically for every method, so comparisons hold up by eye:

```python
from lazy.plotting import diagnostic_panel

fig = diagnostic_panel(z_true, grid.centers, pdfs, label="TabPFN")
```
