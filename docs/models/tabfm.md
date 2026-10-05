# TabFM

TabFM, from Google Research, is a tabular foundation model pretrained on
synthetic tables generated from structural causal models. It alternates
attention across the columns and the rows of the table, compresses each row
into a single vector, and then passes the rows to a large transformer that
does in-context learning. TabFM's own regressor predicts a single value per
row, so LAZY instead builds the distribution from TabFM's classifier, which
predicts at most ten classes: a classifier first predicts which of
`n_coarse_bins` equal-mass bins of the target a row falls in, and one
classifier per coarse bin then predicts which of `n_fine_bins` sub-bins.
This gives a histogram over at most 100 bins, which is 10 by 10 for a context
of 500 rows or more. TabFM is the largest model in LAZY and needs a GPU to be
practical.

```python
model = lazy.LazyModel("tabfm")
```

**Paper:** [Kong et al. 2026](https://arxiv.org/abs/2609.37959). **Code:**
[google-research/tabfm](https://github.com/google-research/tabfm). **Blog
post:** [Introducing
TabFM](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/).
**Weights:** [google/tabfm-1.0.0-pytorch on Hugging
Face](https://huggingface.co/google/tabfm-1.0.0-pytorch), under a
non-commercial license. For more than a few thousand query rows, run
`lazy setup` to install TabFM's repository build (see {doc}`../installation`).

## Defaults

The table lists the creators' defaults in their `TabFMClassifier`, the part of
TabFM that LAZY uses, next to LAZY's.

| Setting | Creators' default | LAZY default |
| ------- | ----------------- | ------------ |
| Output for regression | one value per row, from the regression checkpoint | a histogram from the classification checkpoint (`n_coarse_bins="auto"`, `n_fine_bins="auto"`) |
| Ensemble members | 32 | `n_estimators=8` |
| Feature preprocessing | none and a power transform, alternating | the same (`transforms="auto"`) |
| Feature shuffling | random permutations, with the class labels shifted | the same (`feature_shuffle=True`) |
| Softmax temperature | 0.9, after averaging the members' logits | the same (`softmax_temperature="auto"`) |
| Outlier clipping | soft clip at 4 standard deviations | the same (`outlier_threshold="auto"`) |
| Maximum features | 500 | 500 |
| Categorical columns | detected by data type and ordinal-encoded | off: every column is numeric |
| Key/value cache | off; the context is re-encoded for every batch | on, with LAZY's own cache (`kv_cache=True`) |
| Precision | bfloat16, on a GPU and on a CPU | bfloat16 on a GPU, float32 on a CPU |
| Random seed | 42 | `random_state=0` |
| Row subsampling | none | none (`bag_size=None`) |

## Built-in standardizations

Before the network, TabFM imputes missing values with the column mean, drops
constant columns, z-scores every column on the context and clips it at 100
standard deviations, then applies each member's transform (none, or a power
transform) and soft-clips outliers at 4 standard deviations. Inside the
network, any remaining missing value is replaced by $-100$; there are no
missing-value indicators. Since LAZY uses the classifier, the target is never
standardized: it is only binned, into equal-mass bins of the context targets
or, on request, the bins of `y_grid` (below), and a target outside the bins is clipped to
them with a warning.

## Where LAZY differs

The largest difference is that LAZY does not use TabFM's regression output,
since a single value per row has no distribution to keep. The two-level
hierarchy of classifiers is LAZY's own construction, and its resolution is set
by `n_coarse_bins` and `n_fine_bins`; `n_dither` adds copies of the hierarchy
on shifted bin edges, which smooths the histogram at a proportional cost, and
`prior_shift="em"` applies a label-shift correction, estimated by expectation
maximization, for when the context's targets are distributed unlike the
queries' while the features given the target are not. Within each classifier,
the settings that affect a prediction are the creators', except that we run 8
members rather than 32, as for every other model, use the seed 0 rather than
42, and treat every column as numeric. We compute in float32 on a CPU, where
TabFM uses bfloat16 everywhere, and LAZY caches the context's keys and values
in its own implementation, which gives the same answers as re-encoding the
context up to rounding and, in our tests, about 26 times less compute per
query row. Under an explicit `transforms`, LAZY turns off both the soft clip
and the 500-feature cap.

## Choosing the bins

By default (`"auto"`), TabFM picks its bins from the size of the context. Each
level gets $\lfloor\sqrt{n/5}\rfloor$ bins, between 2 and 10, for a context of
$n$ rows, so that a final bin holds about five context rows, and every context
of 500 rows or more gets 10 by 10. With fewer rows per bin the histogram is
overconfident. On the 277 context rows of `yacht` with 4 members, for
example, the 7 by 7 bins that `"auto"` picks lower the CRPS from 0.631 to 0.517 and the NLL from
1.28 to 0.50 relative to 10 by 10. An integer sets a level's bin count
explicitly, from 2 to 10, and always gives equal-mass bins.

The equal-mass bins are used whether or not a `y_grid` is passed to the
constructor. With `n_coarse_bins="grid"` and `n_fine_bins="grid"` (always the
two together), the bins of the constructor's `y_grid`, which must have at most
100 bins, become the classes instead: the fine classes are its bins, and the
coarse classes are the fewest consecutive runs of at most ten of them. The
density is then built on that grid and never rebinned onto it. However, a bin
with no context rows gets zero probability, since the classifier cannot
predict a class it has not seen, so a true value that falls in such a bin has
zero density. On `yacht`, whose targets crowd at the low end, a grid of 50
equal-width bins left 2 of the 31 test values in empty bins and gave a CRPS of
0.566 and an NLL of 2.66, against 0.517 and 0.50 for the default equal-mass
bins. We therefore recommend `"grid"` only for a grid whose every bin is well
populated by the context. It does not take `n_dither > 1`, since shifted
copies would not keep the edges of the grid. A grid passed to `predict_proba`
never changes the bins and only rebins the answer, and `provenance_["bins"]`
records which bins a fit used.

`n_dither=1` is the default. When accuracy matters, we recommend `n_dither=3`,
which in our tests on `yacht`, `energy`, `concrete` and DC1 lowered the NLL
in all but one of 13 settings, at two to three times the cost of a prediction.
