# Choosing a backend

| Name     | Method                                                                     | Weights        |
| -------- | -------------------------------------------------------------------------- | -------------- |
| `tabpfn` | Bucket masses of TabPFN's bar distribution.                                | 41 MB – 880 MB |
| `limix`  | Bucket masses of LimiX-2's 5,000-bucket head.                              | ~1.6 GB        |
| `tabicl` | Quantiles of TabICLv2's regression head.                                   | ~100 MB        |
| `tabfm`  | A hierarchy of TabFM in-context classifiers over equal-mass bins of the target. | ~6.6 GB        |

`LazyModel("tabpfn", ...)` and the concrete classes
({class}`~lazy.models.tabpfn.TabPFNBarDistribution`,
{class}`~lazy.models.limix.LimiXBarDistribution`,
{class}`~lazy.models.tabicl.TabICLQuantile`,
{class}`~lazy.models.tabfm.TabFMHistogram`) are the same models, and the
class pages document every parameter. We recommend `LazyModel` when the
backend is a configuration value, as in a benchmark loop. All of the backends
write onto whatever {class}`~lazy.grid.Grid` the user asks for, and each has
a native grid on which it answers by default (see {doc}`api`).

## One set of parameters

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
| `feature_shuffle`     | Each member sees the feature columns in a different order.                 | `True`    |
| `bag_size`            | Context rows per member: a count, a fraction, or `None` for all.            | `None`    |
| `kv_cache`            | Process the context once and reuse it for every query chunk (below).        | `True`    |
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

The table below shows how each model provides these parameters, where
"scaffolded" means that LAZY builds the feature itself.

| Parameter             | TabPFN                                   | LimiX-2                                   | TabICL                                | TabFM                                     |
| --------------------- | ---------------------------------------- | ----------------------------------------- | ------------------------------------- | ----------------------------------------- |
| `kv_cache`            | its fit-time cache, at full precision    | ported (below)                            | its key/value cache                   | its streaming prefill/decode              |
| `feature_shuffle`     | `FEATURE_SHIFT_METHOD="shuffle"`         | its column shuffler                       | Latin-square shuffles                 | `feat_shuffle_method="random"`            |
| `transforms`          | `none` native, the rest scaffolded       | `none`, `quantile_uniform` native         | `none` native, the rest scaffolded    | `none`, `power` native                    |
| `bag_size`            | `SUBSAMPLE_SAMPLES`, given `lazy`'s bags | scaffolded: one member per bag            | scaffolded: one regressor per bag     | `max_num_rows` (TabFM draws the rows)     |
| `softmax_temperature` | its own; `"auto"` 0.9, v3.5 and fast 1.0 | its own; `"auto"` 0.9                     | no softmax: only `"auto"`             | its own; `"auto"` 0.9                     |
| `mixed_precision`     | float16 autocast                         | float16 autocast                          | `use_amp=True`                        | bfloat16 weights                          |
| `outlier_threshold`   | `OUTLIER_REMOVAL_STD`                    | scaffolded (`lazy`'s clip)                | its `outlier_threshold`               | its `outlier_threshold`                   |
| members combined      | averaged buckets                         | averaged buckets / mixture                | weighted average of quantiles         | logits averaged; dithers and groups mixed |

The `softmax_temperature`, `mixed_precision` and `outlier_threshold` actually
used are recorded in `provenance_`, together with the rest of the recipe.
Every column is treated as numeric on every model, so none of them guesses
that a column with few distinct values is a category. This contract is
enforced: {func}`lazy.models.registry.register` refuses a backend that lacks
any of these parameters or their shared defaults, and the conformance tests
run every registered backend through every feature. A new model joins by
subclassing {class}`~lazy.models.ContextEnsembleEstimator`.

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

## The key/value cache

An in-context model reads the whole context for every query. With
`kv_cache=True` it reads the context once and each chunk of queries attends
to what was stored, so a large query set costs little more than its own rows.
The cache is exact on a CPU. Since queries only ever attend to the context and
never to each other, the answer is the uncached one up to float rounding.
Under mixed precision on a GPU the two agree to the rounding of that precision
(for TabICL, about 1e-3 of the quantile spread).

TabPFN caches at full precision. Its own default, which is also what earlier
`fit_mode="fit_with_cache"` runs used, is an int8 cache that moves densities
by up to about half a percent, and `kv_cache="int8"` (or `"fp8"`) asks for it.
TabICL caches keys and values, while `kv_cache="repr"` stores the smaller row
representations instead and re-runs its in-context layers. TabFM uses its
streaming prefill/decode path, which needs the repository build (see
{doc}`../installation`) and otherwise falls back with a
{class}`~lazy.models.PerformanceWarning`. Since TabFM fits and prefills inside
every `predict` call, the cache serves the chunks of that call. The paper's
TabFM numbers come from the streaming path in bfloat16. LimiX-2 has no cache
upstream, so LAZY ports one. It costs about 2 GB of GPU memory per member at
20,000 context rows, and falls back, with a warning, when the device lacks the
room.

## Bagging, and LimiX-2's context limit

`bag_size` gives each member a random subset of the context, with rows drawn
without replacement within a member and independently across members. This
matters when the context outgrows what a model was pretrained on. LimiX-2
degrades above about 20,000 context rows, and a
{class}`~lazy.models.ContextSizeWarning` says so when a larger context arrives
without bagging. The remedy it suggests is the one used in the paper:

```python
model = lazy.LazyModel("limix", bag_size=20_000, n_estimators=32)
```

## LimiX-2

LimiX is not on PyPI, so its code is installed from the repository, at the
commit LAZY was validated on, and its dependencies with the extra:

```bash
pip install "LimiX @ git+https://github.com/limix-ldm-ai/LimiX@516bf396333feb3198cf7aff8a6c10421f218e24"
pip install "lazy-tfm[limix]"
```

Alternatively, `$LAZY_LIMIX_SRC` can point at a checkout. LAZY loads two parts
of it under private module names and runs LimiX-2's network directly, with
three changes that make a row's answer independent of which other rows share
its chunk. The first is that each member's preprocessing is fitted on the
context alone (upstream's column filter, categorical detection and category
encoding also see the queries, and LAZY treats every column as numeric). The
second is that the feature positional embedding has its own random generator
(upstream draws it from the global one after advancing it by an amount that
depends on the chunk's size). The third is the ported cache. The answers are
statistically, not bitwise, those of upstream's `LimiXPredictor`.

Built with StableAI LimiX. LimiX-2's weights are released under the StableAI
LimiX Non-Commercial License 1.0, which also requires distributions,
derivatives and publications to display "Built with StableAI LimiX"; LimiX's
code is under the Apache-2.0-based Stable AI Technology Co., Ltd. License
1.0.

## Which model, exactly

A backbone is a family rather than a model, so every backend takes a
`version`, and each version is a separately pinned checkpoint:

```python
import lazy

lazy.list_versions("tabpfn")
# ['v2', 'v2.5', 'v2.6', 'v3', 'v3.5', 'v3.5-fast']

model = lazy.LazyModel("tabpfn", version="v2.5")
model.name_          # 'tabpfn:v2.5', the label evaluate() puts on its row
```

What actually answered is recorded on the fitted model in `provenance_`, ready
to be written out beside the numbers. It holds the weights, the code that read
them and the ensemble that answered, and no local paths, so it means the same
thing on another machine and survives a trip through JSON:

```python
model.fit(X_train, z_train).provenance_
# {'backend': 'tabpfn', 'version': 'v2.5',
#  'repo_id': 'Prior-Labs/tabpfn_2_5',
#  'filename': 'tabpfn-v2.5-regressor-v2.5_default.ckpt',
#  'revision': '6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b',
#  'package': 'tabpfn 9.0.0', 'lazy': 'lazy-tfm 0.1.0', 'device': 'cuda',
#  'n_estimators': 8, 'transforms': ['auto', ...], 'kv_cache': True, ...}
```

Versions are not interchangeable. TabPFN `v2` is pretrained for at most 10,000
context rows and `v2.5` for 50,000, compared with a million for `v3`, and `v2`
is also the only one whose weights allow commercial use.
{func}`lazy.get_checkpoint <lazy.models.get_checkpoint>` returns each
version's license and size notes.

## Memory and speed

Peak memory is bounded by `chunk_size` on every backend (8,192 query rows by
default), and on TabFM's streaming path `query_block_rows` also bounds the
host memory of a block of queries. A row's answer never depends on which other
rows share its chunk: on a CPU it is the same bit for bit on TabPFN and TabICL
and to float rounding on LimiX-2 and TabFM, and under mixed precision on a GPU
it is the same to the rounding of that precision. We recommend lowering
`chunk_size` on a small GPU, and setting it to `0` gives a single pass. Every
backend shows a progress bar while it predicts (with `progress="auto"`, on a
terminal or in a notebook but not when output goes to a file).
