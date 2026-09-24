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

{mod}`lazy.plotting` draws the standard diagnostics (accuracy, calibration and
N(z)) identically for every method, so comparisons hold up by eye:

```python
from lazy.plotting import diagnostic_panel

fig = diagnostic_panel(z_true, grid.centers, pdfs, label="TabPFN")
```
