# One interface for every model

Every model in LAZY is reached through the same estimator, takes the same
parameters with the same defaults, and accepts the same kinds of input. This
page describes that shared interface and how each model implements it.

| Name     | Method                                                                          | Weights        |
| -------- | ------------------------------------------------------------------------------- | -------------- |
| `tabpfn` | Bucket masses of TabPFN's bar distribution.                                     | 41 MB – 880 MB |
| `limix`  | Bucket masses of LimiX-2's 5,000-bucket head.                                   | ~1.6 GB        |
| `tabicl` | Quantiles of TabICLv2's regression head.                                        | ~100 MB        |
| `tabfm`  | A hierarchy of TabFM in-context classifiers over equal-mass bins of the target. | ~6.6 GB        |

`LazyModel("tabpfn", ...)` and the concrete classes
({class}`~lazy.models.tabpfn.TabPFNBarDistribution`,
{class}`~lazy.models.limix.LimiXBarDistribution`,
{class}`~lazy.models.tabicl.TabICLQuantile`,
{class}`~lazy.models.tabfm.TabFMHistogram`) are the same models, and the
class pages document every parameter. We recommend `LazyModel` when the
backend is a configuration value, as in a benchmark loop. All of the backends
write onto whatever {class}`~lazy.grid.Grid` the user asks for, and each has
a native grid on which it answers by default (see {doc}`distributions`).

## Shared parameters and defaults

Every backend takes the same parameters, with the same defaults, and means the
same thing by them. Where a model has a setting of its own, the parameter is
translated to it, and where it does not, LAZY builds the feature around the
model. Every upstream setting that changes a prediction is either driven by
one of these parameters or pinned by LAZY itself, so neither a model's own
default nor an upgrade of its package can decide an answer without the user
knowing. Only `version` differs between backends, because the checkpoints do.

| Parameter             | Meaning                                                                     | Default   |
| --------------------- | --------------------------------------------------------------------------- | --------- |
| `n_estimators`        | Ensemble members; exactly this many run.                                    | `8`       |
| `transforms`          | Per-member feature transforms (below).                                      | `"auto"`  |
| `feature_shuffle`     | Each member sees the feature columns in a different order.                  | `True`    |
| `bag_size`            | Context rows per member: a count, a fraction, or `None` for all.            | `None`    |
| `kv_cache`            | Process the context once and reuse it for every query chunk.                | `True`    |
| `y_grid`              | Default output grid; `None` is the native grid.                             | `None`    |
| `random_state`        | The ensemble's seed; `None` draws one and records it.                       | `0`       |
| `chunk_size`          | Query rows per forward pass, to bound memory; `0` for one pass.             | `8192`    |
| `softmax_temperature` | Divides the output logits; `"auto"` is the checkpoint's calibrated value.   | `"auto"`  |
| `mixed_precision`     | The model's reduced-precision path on a GPU; float32 otherwise.             | `True`    |
| `outlier_threshold`   | Soft-clip each feature at this many standard deviations (below).            | `"auto"`  |
| `device`              | `"auto"`, `"cuda"`, `"cuda:1"`, `"cpu"`.                                    | `"auto"`  |
| `progress`            | A progress bar: `"auto"`, `True` or `False`.                                | `"auto"`  |
| `verbose`             | Log messages to stdout.                                                     | `False`   |
| `version`             | Which pinned checkpoint.                                                    | per model |

The `softmax_temperature`, `mixed_precision` and `outlier_threshold` actually
used are recorded in `provenance_`, together with the rest of the recipe (see
{doc}`reproducibility`). Every column is treated as numeric on every model, so
none of them guesses that a column with few distinct values is a category.
{doc}`scaling` describes `kv_cache`, `chunk_size` and `bag_size`, which
affect the cost of a prediction.

## How LAZY scaffolds what a model lacks

The table below shows how each model provides these parameters, where
"scaffolded" means that LAZY builds the feature itself.

| Parameter             | TabPFN                                   | LimiX-2                                   | TabICL                                | TabFM                                     |
| --------------------- | ---------------------------------------- | ----------------------------------------- | ------------------------------------- | ----------------------------------------- |
| `kv_cache`            | its fit-time cache, at full precision    | ported                                    | its key/value cache                   | its streaming prefill/decode              |
| `feature_shuffle`     | `FEATURE_SHIFT_METHOD="shuffle"`         | its column shuffler                       | Latin-square shuffles                 | `feat_shuffle_method="random"`            |
| `transforms`          | `none` native, the rest scaffolded       | `none`, `quantile_uniform` native         | `none` native, the rest scaffolded    | `none`, `power` native                    |
| `bag_size`            | scaffolded: one regressor per bag        | scaffolded: one member per bag            | scaffolded: one regressor per bag     | its members, given `lazy`'s bags          |
| `softmax_temperature` | its own; `"auto"` 0.9, v3.5 and fast 1.0 | its own; `"auto"` 0.9                     | no softmax: only `"auto"`             | its own; `"auto"` 0.9                     |
| `mixed_precision`     | float16 autocast                         | float16 autocast                          | `use_amp=True`                        | bfloat16 weights                          |
| `outlier_threshold`   | `OUTLIER_REMOVAL_STD`                    | scaffolded (`lazy`'s clip)                | its `outlier_threshold`               | its `outlier_threshold`                   |
| members combined      | averaged buckets / mixture               | averaged buckets / mixture                | weighted average of quantiles         | logits averaged; dithers and groups mixed |

This contract is enforced: {func}`lazy.models.registry.register` refuses a
backend that lacks any of these parameters or their shared defaults, and the
conformance tests run every registered backend through every feature. A new
model joins by subclassing {class}`~lazy.models.ContextEnsembleEstimator`.

## Ensembling and feature shuffling

Each model is run as an ensemble of `n_estimators` members, and exactly that
many run on every model, including where a model's own default asks for a
different count. The members differ in their feature transform, taken from
`transforms` in round-robin order, and with `feature_shuffle=True` each member
also sees the feature columns in a different order. Their answers are combined
as the last row of the table above shows. With `bag_size` set, each member
also sees its own random subset of the context (below, and {doc}`scaling`).

## Bagging

With `bag_size` set, each member sees its own random subset of the context
rows, and LAZY draws these subsets for every model. The rows are drawn without
replacement within a member and independently across members, from one
generator seeded with `random_state`, exactly as the paper's bagged LimiX-2
runs drew them. Therefore, member `i` sees the same rows whichever model runs
it, TabFM included, which would otherwise draw rows of its own. On TabFM the
member's coarse classifier sees its bag, and its classifier for coarse bin `j`
sees the rows of its bag that fall in that bin.

Everything a member fits on the context is fitted on its own bag: a scaffolded
transform, the outlier clip and the standardization of the target, along with
each model's own preprocessing. TabICL and LimiX-2 run each bagged member as a
model of its own, and so does TabPFN, because its own row subsampling would
standardize the target and place the buckets using the whole context. TabFM
keeps the members of its own transforms (`none` and `power`) in one classifier,
where they keep their column orders, class shifts and averaged logits, and
refits each member's standardization, norm method and clip on that member's
rows. A member with a scaffolded transform is a model of its own on every
backend. Since each member of TabPFN and LimiX-2 places its buckets from its
own targets, their native grid under bagging is the union of every member's
buckets, with up to `n_estimators` times as many bins as an unbagged one, so we
recommend passing `y_grid` for a large query set. Two things are taken from the
whole context rather than from a bag: TabFM's equal-mass bins, so that its
members answer over the same classes, and the range of TabICL's native grid,
which spans every context target. Each member is also seeded on its own (see
{doc}`reproducibility`).

## Transforms

`transforms` names what each member does to the features before the model
sees them, and the members take the names in round-robin order.

| Name               | Transform                                                     |
| ------------------ | ------------------------------------------------------------- |
| `none`             | no transform                                                  |
| `power`            | Yeo-Johnson power transform, standardized                     |
| `quantile`         | quantile transform to a normal distribution                   |
| `quantile_uniform` | quantile transform to a uniform distribution                  |
| `quantile_rtdl`    | RTDL's quantile transform (normal, with a little noise)       |
| `robust`           | median and interquartile-range scaling                        |
| `…+original`       | the transformed columns *and* the originals                   |

A name means the same transform on every model. A model runs it natively only
where its own implementation is this one, and otherwise LAZY applies it,
fitted on the context rows, before the model sees the features.

`"auto"` (the default) is each model's own tuned recipe, which we write out in
LAZY for every version rather than read from the installed package, so
upgrading TabPFN or TabICL cannot change what it runs. It includes each
model's extras: TabPFN's fingerprint feature, SVD components, target
transforms, polynomial features (v2.6) and 12-sigma clip (v3.5); TabICL's and
TabFM's 4-sigma clip; and LimiX-2's eight pipelines, half of them
quantile-uniform with the originals and SVD components and half power.

Any explicit value is all the model sees. Those extras are turned off, and
outlier clipping runs only if `outlier_threshold` asks for it, which leaves
each model's unavoidable input handling, listed in the table below. Since
TabICL and TabFM z-score whatever they are given, an affine transform such as
`robust` changes nothing on them.

| Model   | Runs whatever the recipe                                                                                  |
| ------- | --------------------------------------------------------------------------------------------------------- |
| TabPFN  | drops constant columns; standardizes inside the network                                                   |
| LimiX-2 | drops constant and all-missing columns; fills missing values and standardizes inside the network; standardizes the target for its buckets |
| TabICL  | fills missing values with the context mean; drops constant columns; z-scores, clipped at ±100; standardizes the target |
| TabFM   | fills missing values with the context mean; drops constant columns; z-scores, clipped at ±100              |

`"limix"` is LimiX-2's two pipelines written in the shared vocabulary,
`("quantile_uniform+original", "power")`, and it can be used on any model; it
is what the paper's "with LimiX transforms" runs use. It lacks the SVD
components of LimiX-2's own quantile pipeline, so on LimiX-2 it is not the
same as `"auto"`.

## Outlier clipping

`outlier_threshold` controls one clip, the two-pass soft clip that TabPFN,
TabICL and TabFM share. The bounds are placed at the mean plus and minus that
many standard deviations of the context, and then recomputed without the
values beyond the first bounds. A value past a bound is pulled back to the
bound plus the logarithm of its magnitude, so extreme values keep their order
but lose their leverage. Under `transforms="auto"`, `"auto"` is the model's
own clip (4 on TabICL and TabFM, 12 on TabPFN-3.5, none on the others), and
under an explicit recipe it means no clip. A number clips at that many
standard deviations on every model (LimiX-2 has no clip of its own, so LAZY
applies it there), and `None` turns clipping off everywhere.

## Input data

The features can be given in any of the forms that scikit-learn users expect:
a NumPy array, a structured or record array, a pandas DataFrame, an astropy
`Table` or `QTable` (whose quantities give their values), or anything with a
`to_pandas()` method. When the input has column names, they are recorded as
`feature_names_in_` and checked at prediction time. The same columns in a
different order are reordered, while an unnamed table meeting a named fit, or
the reverse, is used by position with a warning.

Missing values are marked with `NaN`, and the masked entries of an astropy
table or masked array also become `NaN`. Each model handles them in its own
way: TabPFN and LimiX add missing-value indicators, while TabICL and TabFM
impute them inside their preprocessing. Infinities and non-numeric columns are
rejected, and the target values must be finite. Therefore, a categorical
column has to be encoded as numbers before it is passed (see {doc}`limits`).

## scikit-learn compatibility

Since `get_params`, `set_params` and {func}`sklearn.base.clone` all work, the
models can be used in scikit-learn pipelines and search objects without any
modification. `LazyModel` flattens the parameters of its backend into its own,
so `GridSearchCV(model, {"n_estimators": [4, 8]})` needs no prefix. As in
scikit-learn, constructing a model only stores its parameters, which are
validated (and the checkpoint loaded) at `fit`. `score` returns the negative
CDE loss, so a search object that maximizes it prefers the better
distributions.
