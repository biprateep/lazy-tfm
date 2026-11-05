# Limits and pitfalls

A tabular foundation model answers from its context and from the prior it was
pretrained on, and nothing else (see {doc}`tfm`). This page collects the cases
in which that answer, or the way LAZY scores it, should not be taken at face
value.

## Extrapolation

None of the models extrapolated a trend beyond the range of the context. We
trained every model (TabPFN-3.5, TabPFN-3.5-fast, LimiX-2, TabICL and TabFM)
on the noisy chirp without gaps,
{func}`load_dataset("chirp", n_gaps=0) <lazy.datasets.load_dataset>`,
restricted to x < 8 or x < 7, and predicted beyond that range. Every model
reverted toward a flat mean, with an interval that did not cover the curve.
Against the noise-free curve, the mean of each density had an rms error of 1.8
to 2.5, and the 68% interval a coverage of 21% to 55% (see [Measurement
setup](scaling.md#measurement-setup)). Therefore, the width of the predicted
distribution is not a reliable warning of extrapolation, and we recommend
identifying queries outside the range of the context before they are
predicted.

## Context and feature limits

Each model was pretrained on contexts and tables up to a maximum size, beyond
which its performance is not guaranteed. TabPFN enforces the limits stored in
each checkpoint (see [Versions](../models/tabpfn.md#versions)) unless
`ignore_pretraining_limits=True`, on the context each member sees, which under
bagging is its bag (see {doc}`scaling`).

LimiX-2 states no maximum context size, but on DC1 photometry its accuracy
degrades above about 20,000 context rows (see {doc}`../models/limix`). On any
data, LAZY warns when a member's context is larger, and bagging with a smaller
`bag_size` keeps every member within it (see {doc}`scaling`).

Under `transforms="auto"`, TabFM gives each member a random subset of 500
columns when a table has more, and an explicit `transforms` turns that cap off. At the other end, TabFM sizes its
bins to the context, with about five context rows per bin below 500 rows, so a
small context gives it a coarser histogram rather than an overconfident one.
The model pages under {doc}`../models/index` give each model's limits in
full.

On a CPU, TabPFN also refuses contexts above 1,000 rows (v2 to v2.6) or 5,000
rows (v3 and later). In that case we recommend a GPU, TabICLv2, or bags of at
most 5,000 rows (`bag_size=5_000`, with enough members to cover the context;
see {doc}`choosing`).

## Probability outside the grid

Probability outside the grid is dropped and each density is renormalized. A
grid that ends inside the support of a distribution therefore moves its mass
inward and makes it look narrower than it is, and a true value outside the
grid has no density to score. If the targets reach beyond a grid, pass one
that covers them, or use the native grid, which reaches far into both tails
(see {doc}`distributions`). When TabFM takes its bins from a constructor
`y_grid` (`n_coarse_bins="grid"`, see {doc}`../models/tabfm`), a bin of that
grid with no context rows gets zero probability, and a true value in it has
zero density. Such a grid should therefore have context rows in every bin.

```{figure} ../figures/figs/grid_cutoff.svg
:alt: One skewed density on a grid that covers it and on a grid that ends at 1.5; the short grid's density is higher, and a true value at 2.2 lies beyond it.
:width: 100%

**Figure 1.** One distribution on a grid that covers it (blue) and on a grid
that ends inside it (orange), from
{meth}`~lazy.distributions.HistogramDistribution.on_grid`. The short grid
drops the probability beyond 1.5 and renormalizes the rest, so its density is
higher, and it gives the true value at 2.2 no density.
```

## Discrete targets and the PIT

The PIT statistics assume a continuous target. When the target takes only a
few distinct values (counts, classes stored as numbers, or values rounded to a
coarse precision), the predicted CDF at the true value is not expected to be
uniform even for a well-calibrated model. The PIT histogram and the statistics
computed from it will then flag a miscalibration that may not be there.

## A constant target

When every training target has the same value, every model answers each row
with a point mass at that value, as scikit-learn's regressors do, and runs no
network. Its quantiles, intervals and point estimates are the value, and
{meth}`~lazy.base.BaseDensityRegressor.predict_proba` puts all of the
probability in the grid bin that holds it. The native grid then has three
bins, the middle one centred on the value, spanning 1% of the value on each
side and at least 0.01. The PIT of a target equal to the value is 1. In
`provenance_`, `constant_target` records the value. A context of one row is
a constant target. A {class}`~lazy.models.ensemble.LazyEnsembleModel` gives
the same point mass, under every pooling, and its pooling grid is that grid.

## Ties at flat-topped peaks

The `mode` is the center of the highest bin, and when several adjacent bins
share the highest density, the first of them is taken. This happens most often
on TabFM, whose histograms are built from equal-mass bins and can have flat
tops several bins wide. On such a row the mode sits at the low edge of the
plateau rather than at a location the model prefers, so `peak_mean` or the
`median` is the better point estimate there (see {doc}`distributions`).

## Very fine output grids

The densities of TabPFN and LimiX-2 come from buckets and those of TabFM from a
histogram, so each model has a resolution of its own, the width of its buckets
or bins. On a much finer grid, the densities of neighboring bins can jitter
around the shape the model predicts, and counting the peaks of a density or
taking its mode can find peaks that are not there. On the
insurance dataset, for example, a grid of 25 USD bins gave 24 rows with two
peaks, while a grid of 100 USD bins gave 3 to 4. Therefore, wherever the peaks
matter, we recommend a grid no finer than the native resolution of the model,
which its native grid (`native_grid_`) shows.

## Categorical columns

Every model treats every column as numeric and rejects a column that is not.
A categorical feature must therefore be encoded as numbers before it is passed
(e.g., ordinal codes or one-hot columns), and the model then treats an ordinal
code as an ordered quantity. None of the models guesses that a numeric column
with few distinct values is a category, as some of them do upstream.

## Biased training sets

In-context learning assumes that the context and the queries come from the
same population. When the training set is biased, for example because only
the brightest or easiest objects were labeled, the predicted distributions
inherit the distribution of the context targets, and this mismatch, rather
than the estimator, is usually what limits the accuracy. On DC1,
LimiX-2 performs best in this regime (see {doc}`choosing`). TabFM
also takes `prior_shift="em"`, a label-shift correction estimated by
expectation maximization, for when the context's targets are distributed
unlike the queries' while the features given the target are not.
{func}`~lazy.datasets.fetch_dc1_biased` provides such a training set, with a
representative calibration sample and a representative test set, and
{func}`~lazy.selection.grid_selection` applies the same selection to any
photometry (see {doc}`datasets`).
