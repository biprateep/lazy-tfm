# Quickstart

Install the package with the default model's backend, TabPFN:

```console
pip install 'lazy-tfm[tabpfn]'
```

The first `fit` downloads TabPFN-3.5's weights (~880 MB), which Prior Labs
releases under a **non-commercial** licence; {doc}`guide/models` lists every
model's size, licence and hardware needs.

```python
import numpy as np

import lazy
from lazy.datasets import fetch_dc1

train, test = fetch_dc1(split=True)            # ~1 GB, cached after the first call
# 391k test rows in all, sorted by redshift, so start with a random 20k.
test = test.take(np.random.default_rng(0).choice(len(test), 20_000, replace=False))
X_train, z_train = train.features("mag-color"), train.redshift
X_test, z_test = test.features("mag-color"), test.redshift

model = lazy.LazyModel()                        # TabPFN-3.5, the default
model.fit(X_train, z_train)

pdfs = model.predict_proba(X_test, lazy.datasets.DC1_GRID)    # (20000, 200) densities
mode = lazy.metrics.grid_point_estimates(lazy.datasets.DC1_GRID.centers, pdfs)["mode"]
# scale="1+y": the photo-z convention, residuals divided by 1 + z, as in DC1.
print(lazy.metrics.summarize(z_test, lazy.datasets.DC1_GRID, pdfs, label="TabPFN-3.5", scale="1+y"))
```

````{admonition} No GPU?
:class: tip
TabPFN-3.5 is built for a GPU: on a CPU it is slow, and it refuses contexts
larger than 5,000 rows, so `fit` above fails without one (`lazy` warns when
it finds no GPU). On a laptop, use TabICL, which is small, BSD-licensed and
quick on a CPU:

```python
# pip install 'lazy-tfm[tabicl]'
model = lazy.LazyModel("tabicl")
```

or TabPFN's distilled fast checkpoint, which runs on a CPU at more than twice
TabPFN-3.5's speed, less accurately, with a context of at most 5,000 rows:

```python
model = lazy.LazyModel("tabpfn", version="v3.5-fast")
```

or give TabPFN-3.5 a context of at most 5,000 rows, drawn at random (the
catalogue's files are not in random order), or pass
`ignore_pretraining_limits=True` and be patient.
````

Scoring the densities you already have, as above, costs nothing; calling
`model.evaluate(X_test, z_test, y_grid=lazy.datasets.DC1_GRID, scale="1+y")` instead runs the
model again.
Pass `lazy.datasets.DC1_GRID` to reproduce the Data Challenge's numbers: without a
grid, a model answers on its own native grid (5,000 buckets for TabPFN), which
is finer but not the challenge's convention.

## The data

{func}`~lazy.datasets.fetch_dc1` downloads the LSST DESC PZ Data Challenge 1
catalogue (about 1 GB, checksummed, cached) in one call. `split=True` returns
the challenge's own train/test split; the default returns both files
concatenated, with `source` marking where each row came from, for making your
own splits.

## The features

{meth}`~lazy.datasets.Catalog.features` builds the table the model sees:

- `"mag"`: the magnitudes and their errors;
- `"mag-color"` (the default): the reference magnitude and the adjacent
  colours, with errors propagated in quadrature;
- `"all"`: every magnitude, every colour and all their errors.

It works on a plain {class}`pandas.DataFrame`, so your own catalogue goes
through the same code via {meth}`~lazy.datasets.Catalog.from_frame` or
{func}`~lazy.datasets.build_features`. Any tabular features work; the
DC1 helpers are a convenience, not a requirement.

## The model

`LazyModel(name, ...)` picks the backend by name (TabPFN-3.5 when no name is
given) and passes everything else to it; see {doc}`guide/models` for choosing
one and {doc}`guide/backends` for the parameters they share. `fit` stores the
labelled rows as context, and each prediction is one forward pass over them.
The first `fit` downloads the backend's checkpoint.

## The outputs

`predict_proba` returns densities, one row per galaxy, on the grid you pass,
or on the model's native grid if you pass none; `predict_quantiles` gives
exact credible bounds, and `predict_distribution` the model's whole answer
before any grid; `predict` reduces each density to a point estimate;
`evaluate` scores both against the true redshifts with the Data Challenge
metrics. The next steps are in {doc}`guide/api`, and the parameters every
backend shares -- the key/value cache, ensembling, transforms, bagging -- in
{doc}`guide/backends`.
