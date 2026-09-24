# Quickstart

```python
from lazy import LazyModel, RedshiftGrid
from lazy.datasets import fetch_dc1

train, test = fetch_dc1(split=True)
X_train, X_test = train.features("mag-color"), test.features("mag-color")

model = LazyModel("tabpfn", version="v3.5")
model.fit(X_train, train.redshift)

pdfs = model.predict_proba(X_test, RedshiftGrid.linear(0, 2, 200))
z = model.predict(X_test, method="z_peak")
print(model.evaluate(X_test, test.redshift))
```

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
{meth}`~lazy.datasets.Catalog.build_features`. Any tabular features work; the
DC1 helpers are a convenience, not a requirement.

## The model

`LazyModel(name, ...)` picks the backend by name and passes everything else to
it; see {doc}`guide/backends`. `fit` stores the labelled galaxies as context,
and each prediction is one forward pass over them. The first `fit` downloads
the backend's checkpoint.

## The outputs

`predict_proba` returns densities, one row per galaxy, on the grid you pass;
`predict` reduces each density to a point estimate; `evaluate` scores both
against the true redshifts with the Data Challenge metrics. The next steps are
in {doc}`guide/api`.
