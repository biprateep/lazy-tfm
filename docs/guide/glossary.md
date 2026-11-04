# Glossary

This page defines the terms that the guide uses. Each definition links to the
page that explains the term.

```{glossary}
:sorted:

backend
  One of the five models in LAZY, reached by its registered name
  (`"tabpfn"`, `"limix"`, `"tabicl"`, `"tabfm"` or `"kumo"`) through
  {class}`~lazy.models.lazy_model.LazyModel`, or by its class. See
  {doc}`interface`.

bag
  The random subset of the context rows that one member sees, with
  `bag_size` rows. Bagging draws the rows without replacement within a member
  and independently across members. See {doc}`scaling`.

CDE loss
  The conditional density estimate loss: the mean over rows of the integral
  of $p^2$ minus twice the density at the true value. Lower is better, and
  `score` returns its negative. See {doc}`distributions`.

checkpoint
  The pretrained weights, one file or a directory, at a pinned revision of its repository on
  the Hugging Face Hub. Each version of a backend is one checkpoint. See
  {doc}`reproducibility`.

chunk
  The query rows that one forward pass predicts: at most `chunk_size` of
  them, or all of them when `chunk_size=0`. A row's answer does not depend on the other rows in its chunk,
  except under TabFM's `prior_shift="em"`. See {doc}`scaling`.

context
  The labelled rows passed to `fit`. They play the role of a training set:
  the model reads them at every prediction, and no weights change. See
  {doc}`tfm`.

coverage
  The probability that a central interval holds, as the `coverage` argument
  of `predict_interval` gives it. On a test set, the empirical coverage is
  the fraction of true values inside their intervals. See
  {doc}`distributions`.

CRPS
  The continuous ranked probability score: the integral of the squared
  difference between the predicted CDF and the step function at the true
  value. Lower is better. See {doc}`distributions`.

density
  A row's predicted probability density, in float64 at the bin centres of a
  grid and normalized as the grid states. `predict_proba` returns one per
  row. See {doc}`distributions`.

ensemble
  The `n_estimators` members of one model, whose answers are combined into
  one distribution per row. A combination of several models is a pool. See
  {doc}`interface`.

grid
  A {class}`~lazy.grid.Grid`: the bin edges and centres on which densities
  are given, and their normalization. `predict_proba`, `predict`, `score`
  and `evaluate` take one as `y_grid`, and a fitted model needs no refit
  for a new one. See {doc}`distributions`.

key/value cache
  The network's processing of the context, which `fit` stores when
  `kv_cache=True` (TabFM: each `predict` call), so that each chunk of
  queries reuses it. Without it, each chunk processes the context again.
  See {doc}`scaling`.

member
  One run of the model in an ensemble, with its own transform, column order,
  seed and, under bagging, bag. See {doc}`interface`.

model cache
  The networks that LAZY keeps loaded for the rest of the Python process, so
  that a later fit with the same checkpoint, device and precision reuses
  them. {func}`lazy.clear_model_cache <lazy.models.clear_model_cache>`
  releases them. See {doc}`scaling`.

native distribution
  A model's own answer before any grid: a
  {class}`~lazy.distributions.HistogramDistribution`,
  {class}`~lazy.distributions.QuantileDistribution` or
  {class}`~lazy.distributions.MixtureDistribution` per row, which
  `predict_distribution` returns. Quantiles, intervals and PIT values come
  from it. See {doc}`distributions`.

native grid
  The grid on which a model answers when no grid is given,
  `native_grid_`. {doc}`distributions` lists each backend's.

NLL
  The negative log likelihood: the mean of minus the logarithm of the density
  at the true value. See {doc}`distributions`.

normalization
  How a density on a grid integrates to one: `"trapezoid"`, with unit
  trapezoid mass over the bin centres, or `"histogram"`, constant across each
  bin. See {doc}`distributions`.

PIT
  The probability integral transform: a row's predicted CDF at its true
  value. The PIT values of well-calibrated densities are uniform on [0, 1].
  See {doc}`distributions`.

point estimate
  One value per row that `predict` reduces a density to: the `mode` (the
  default), `peak_mean`, `mean` or `median`. See {doc}`distributions`.

pool
  A {class}`~lazy.models.ensemble.LazyEnsembleModel`: several models fitted
  on the same context, whose distributions are combined. Also called an
  See {doc}`ensembles`.

provenance
  The record in a fitted model's `provenance_` of what answered: the
  checkpoint, the package versions, the device and the recipe. See
  {doc}`reproducibility`.

query
  A row to predict, given by its features alone. See {doc}`tfm`.

recipe
  Every setting of a backend that changes a prediction, pinned by LAZY for
  each version rather than read from the installed upstream package.
  `transforms="auto"` selects the model's own recipe. See {doc}`interface`.

scaffolded
  Built by LAZY around a model that lacks a feature, such as a transform or
  bagging. See {doc}`interface`.

transform
  What a member does to the features before the model sees them, such as
  `power` or `quantile`, named in `transforms`. See {doc}`interface`.

version
  The name of one pinned checkpoint of a backend, such as `"v3.5"`.
  {func}`lazy.list_versions <lazy.models.list_versions>` lists them. See
  {doc}`reproducibility`.
```
