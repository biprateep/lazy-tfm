# Limits and pitfalls

A tabular foundation model answers from its context and from the prior it was
pretrained on, and nothing else (see {doc}`tfm`). This page collects the cases
in which that answer, or the way LAZY scores it, should not be taken at face
value.

## Extrapolation

In our tests, none of the models extrapolated a trend beyond the range of the
context. We trained every model
(TabPFN-3.5, TabPFN-3.5-fast, LimiX-2, TabICL and TabFM) on the noisy chirp of
{func}`~lazy.datasets.make_chirp` restricted to x < 8 or x < 7, and predicted
beyond that range. Every model reverted toward a flat mean beyond the training
range, with an interval that did not cover the curve, giving an rms error of
1.8 to 2.4 and a 68%-interval coverage of 21% to 55%. Therefore, the width of
the predicted distribution is not a reliable warning of extrapolation, and we
recommend identifying queries outside the range of the context before they are
predicted.

## Context and feature limits

Each model was pretrained on contexts and tables up to a maximum size, beyond
which its performance is not guaranteed. TabPFN enforces the limits stored in
each checkpoint unless `ignore_pretraining_limits=True`, on the context each
member sees, which under bagging is its bag (see {doc}`scaling`):

| TabPFN `version` | Max. context rows | Max. features |
| ---------------- | ----------------: | ------------: |
| `"v2"`           | 10,000            | 500           |
| `"v2.5"`         | 50,000            | 2,000         |
| `"v2.6"`         | 100,000           | 2,000         |
| `"v3"`           | 1,000,000         | 2,000         |
| `"v3.5"`         | 1,000,000         | 20,000        |
| `"v3.5-fast"`    | 1,000,000         | 20,000        |

LimiX states no maximum context size, but in our tests on DC1 photometry its
accuracy degrades above about 20,000 context rows, so LAZY warns when a
member's context is larger, whether the whole context without bagging or a bag
of more than 20,000 rows, and bagging with a smaller `bag_size` keeps every
member within it (see {doc}`scaling`). Under
`transforms="auto"`, TabFM gives each member a random subset of 500 columns
when a table has more, and an explicit `transforms` turns that cap off. The model pages under {doc}`../models/index`
give each model's limits in full.

On a CPU, TabPFN also refuses contexts above 1,000 rows (v2 to v2.6) or 5,000
rows (v3 and later). We recommend a GPU, TabICLv2, or bags of at most 5,000
rows (`bag_size=5_000`, with enough members to cover the context) in that case
(see {doc}`choosing`).

## Probability outside the grid

Probability outside the grid is dropped and each density is renormalized. A
grid that ends inside the support of a distribution therefore moves its mass
inward and makes it look narrower than it is, and a true value outside the
grid has no density to score. If the targets reach beyond a grid, pass one
that covers them, or use the native grid, which reaches far into both tails
(see {doc}`distributions`).

## Discrete targets and the PIT

The PIT statistics assume a continuous target. When the target takes only a
few distinct values (counts, classes stored as numbers, or values rounded to a
coarse precision), the predicted CDF at the true value is not expected to be
uniform even for a well-calibrated model, so the PIT histogram and the
statistics computed from it will flag a miscalibration that may not be there.

## Ties at flat-topped peaks

The `mode` is the center of the highest bin, and when several adjacent bins
share the highest density, the first of them is taken. This happens most often
on TabFM, whose histograms are built from equal-mass bins and so can have flat
tops several bins wide. On such a row the mode sits at the low edge of the
plateau rather than at a location the model prefers, so `peak_mean` or the
`median` is the better point estimate there (see {doc}`distributions`).

## Very fine output grids

The densities of TabPFN and LimiX-2 come from buckets and those of TabFM from a
histogram, so each model has a resolution of its own, the width of its buckets
or bins. On a grid much finer than that, the densities of neighboring bins can
jitter around the shape the model predicts, and counting the peaks of a density
or taking its mode on such a grid can find peaks that are not there. On the
insurance dataset, for example, a grid of 25 USD bins gave 24 rows with two
peaks, while a grid of 100 USD bins gave 3 to 4. Therefore, wherever the peaks
matter, we recommend a grid no finer than the native resolution of the model,
which its native grid (`native_grid_`) shows.

## Categorical columns

Every column is treated as numeric on every model, and a column that is not
numeric is rejected. A categorical feature must therefore be encoded as numbers
before it is passed (e.g., ordinal codes or one-hot columns), and the model
then treats an ordinal code as an ordered quantity. None of the models guesses
that a numeric column with few distinct values is a category, as some of them
do upstream.

## Biased training sets

In-context learning assumes that the context and the queries come from the
same population. When the training set is biased, for example because only
the brightest or easiest objects were labeled, the predicted distributions
inherit the distribution of the context targets, and this mismatch, rather
than the estimator, is usually what limits the accuracy. Our own testing
shows that LimiX-2 performs best in this regime (see {doc}`choosing`). TabFM
also takes `prior_shift="em"`, a label-shift correction estimated by
expectation maximization, for when the context's targets are distributed
unlike the queries' while the features given the target are not.
{func}`~lazy.datasets.fetch_dc1_biased` provides such a training set, with a
representative calibration sample and a representative test set, and
{func}`~lazy.selection.grid_selection` applies the same selection to any
photometry (see {doc}`datasets`).
