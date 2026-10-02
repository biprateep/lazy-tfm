# TabFM

TabFM, from Google Research, is a tabular foundation model pretrained on
synthetic tables generated from structural causal models. It alternates
attention across the columns and the rows of the table, compresses each row
into a single vector, and then passes the rows to a large transformer that
does in-context learning. TabFM's own regressor predicts a single value per
row, so LAZY instead builds the distribution from TabFM's classifier, which
predicts at most ten classes: a classifier first predicts which of
`n_coarse_bins` equal-mass bins of the target a row falls in, and one
classifier per coarse bin then predicts which of `n_fine_bins` sub-bins,
giving a histogram over 100 bins by default. TabFM is the largest model in
LAZY and needs a GPU to be practical.

```python
model = lazy.LazyModel("tabfm")
```

**Paper:** [Kong et al. 2026](https://arxiv.org/abs/2609.37959). **Code:**
[google-research/tabfm](https://github.com/google-research/tabfm). **Blog
post:** [Introducing
TabFM](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/).
**Weights:** [google/tabfm-1.0.0-pytorch on Hugging
Face](https://huggingface.co/google/tabfm-1.0.0-pytorch), under a
non-commercial license. For more than a few thousand query rows, install
TabFM's repository build as well (see {doc}`../installation`).

## Defaults

The table lists the creators' defaults in their `TabFMClassifier`, the part of
TabFM that LAZY uses, next to LAZY's.

| Setting | Creators' default | LAZY default |
| ------- | ----------------- | ------------ |
| Output for regression | one value per row, from the regression checkpoint | a histogram from the classification checkpoint (`n_coarse_bins=10`, `n_fine_bins=10`) |
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
standardized: it is only binned, into equal-mass bins of the context targets,
and a target outside the bins is clipped to them with a warning.

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
