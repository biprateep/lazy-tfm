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
{class}`~lazy.models.tabfm.TabFMHistogram`) are the same models; the class
pages document every parameter. Use `LazyModel` when the backend is a
configuration value, as in a benchmark loop.

All of them write onto whatever {class}`~lazy.grid.Grid` you ask for,
and each has a native grid it answers on by default; see {doc}`api`.

## One set of parameters

Every backend takes the same parameters for the same features, and means the
same thing by them. Where a model has a feature of its own, the parameter is
translated to it, so each model's tuned defaults are used unchanged; where it
does not, `lazy` builds the feature around the model.

| Parameter         | Meaning                                                                | Default    |
| ----------------- | ---------------------------------------------------------------------- | ---------- |
| `kv_cache`        | Process the context once, at fit, and reuse it for every query chunk. Exact. | `True` |
| `n_estimators`    | Ensemble members.                                                      | 8 (TabFM 4) |
| `feature_shuffle` | Each member sees the feature columns in a different order.            | `True`     |
| `transforms`      | Per-member feature transforms (below).                                 | `"auto"`   |
| `bag_size`        | Context rows per member: a count, a fraction, or `None` for all.       | `None`     |
| `z_grid`          | Default output grid; `None` is the native grid.                        | `None`     |
| `version`         | Which pinned checkpoint.                                               | per model  |
| `random_state`    | The ensemble's seed.                                                   | per model  |
| `device`          | `"auto"`, `"cuda"`, `"cuda:1"`, `"cpu"`.                               | `"auto"`   |
| `chunk_size`      | Query rows per forward pass, to bound memory; `0` for one pass.       | per model  |
| `progress`        | A progress bar: `"auto"`, `True` or `False`.                           | `"auto"`   |
| `verbose`         | Log messages to stdout.                                                | `False`    |

How each model provides them ("scaffolded" means `lazy` builds it):

| Feature           | TabPFN                  | LimiX-2                  | TabICL                   | TabFM                   |
| ----------------- | ----------------------- | ------------------------ | ------------------------ | ----------------------- |
| `kv_cache`        | its fit-time cache, at full precision | ported (below) | its K/V cache       | its streaming decoder   |
| `feature_shuffle` | `FEATURE_SHIFT_METHOD`  | its column shuffler      | Latin-square shuffles    | `feat_shuffle_method`   |
| `transforms`      | `PREPROCESS_TRANSFORMS` | its rebalancing step     | `norm_methods`           | `norm_methods`          |
| `bag_size`        | `SUBSAMPLE_SAMPLES`     | scaffolded: one member per bag | scaffolded: one regressor per bag | `max_num_rows` |
| members combined  | averaged buckets        | averaged buckets / mixture | averaged quantiles     | mixture of shifted bins |

This contract is enforced: {func}`lazy.models.registry.register` refuses a
backend that lacks any of these parameters or their shared defaults, and the
conformance tests run every registered backend through every feature. A new
model joins by subclassing {class}`~lazy.models.ContextEnsembleEstimator`.

## Transforms

`transforms` names what each member does to the features before the model
sees them. Members take the names round robin.

| Name               | Transform                                                     |
| ------------------ | ------------------------------------------------------------- |
| `none`             | the features as given                                          |
| `power`            | Yeo-Johnson power transform, standardised                      |
| `quantile`         | quantile transform to a normal distribution                   |
| `quantile_uniform` | quantile transform to a uniform distribution                  |
| `quantile_rtdl`    | RTDL's quantile transform (normal, with a little noise)        |
| `robust`           | median and interquartile-range scaling                        |
| `…+original`       | the transformed columns *and* the originals                   |

`"auto"` (the default) is each model's own recipe, untouched. `"limix"` is
LimiX-2's recipe, `("quantile_uniform+original", "power")`, on any model: the
paper's "with LimiX transforms" runs. Transforms a model lacks are fitted on
the context rows and applied by `lazy`, so every name works everywhere;
{mod}`lazy.models._transforms` has the details.

## The key/value cache

An in-context model reads the whole context for every query. With
`kv_cache=True` it reads it once, at `fit`, and each chunk of queries attends
to what was stored, so a large query set costs little more than its own rows.
The cache is exact: queries only ever attend to the context, never to each
other, so the answer is the one the uncached pass gives, up to float rounding.

- **TabPFN** caches at full precision. Its own default, and what earlier
  `fit_mode="fit_with_cache"` runs used, is an int8 cache that moves densities
  by up to about half a per cent; `kv_cache="int8"` (or `"fp8"`) asks for it.
- **TabICL** caches keys and values (`kv_cache="repr"` stores the smaller row
  representations and re-runs its in-context layers).
- **TabFM** uses its streaming prefill/decode path, which needs the
  repository build (see {doc}`../installation`); it falls back with a
  {class}`~lazy.models.PerformanceWarning` otherwise. On CPU the two paths
  agree to 5e-8; on a GPU, where both run in mixed precision, they differ by
  about 0.02 in the mean CDE loss on DC1, and the paper's TabFM numbers come
  from the streaming path.
- **LimiX-2** has no cache upstream; `lazy` ports one. It costs about 2 GB of
  GPU memory per member at 20,000 context rows, and falls back, with a
  warning, when the device lacks the room.

## Bagging, and LimiX-2's context limit

`bag_size` gives each member a random subset of the context: rows drawn
without replacement within a member and independently across members. It
matters when the context outgrows what a model was pretrained on. LimiX-2
degrades above about 20,000 context rows, and a
{class}`~lazy.models.ContextSizeWarning` says so when a larger context arrives
without bagging. The remedy it suggests is the paper's:

```python
model = lazy.LazyModel("limix", bag_size=20_000, n_estimators=32)
```

## LimiX-2

LimiX is not on PyPI. Install its code from the repository, at the commit
`lazy` was validated on, and its dependencies with the extra:

```bash
pip install "LimiX @ git+https://github.com/limix-ldm-ai/LimiX@516bf396333feb3198cf7aff8a6c10421f218e24"
pip install "lazy-tfm[limix]"
```

or point `$LAZY_LIMIX_SRC` at a checkout. `lazy` loads two parts of it under
private module names and runs LimiX-2's network directly, with three changes
that make a row's answer independent of which other rows share its
chunk: each member's preprocessing is fitted on the context alone (upstream
fits one step on context and queries together), the feature positional
embedding has its own random generator (upstream draws it from the global one
after advancing it by an amount that depends on the chunk's size), and the
ported cache. The answers are statistically, not bitwise, those of upstream's
`LimiXPredictor`.

Built with StableAI LimiX. LimiX-2's code and weights are released under the
Stable AI Technology Co., Ltd. License 1.0, Apache-2.0 with attribution terms:
anything built with it and made available must display "Built with StableAI
LimiX".

## Which model, exactly

A backbone is a family, not a model, so every backend takes a `version`, and
each version is a separately pinned checkpoint:

```python
import lazy

lazy.list_versions("tabpfn")
# ['v2', 'v2.5', 'v2.6', 'v3', 'v3.5', 'v3.5-fast']

model = lazy.LazyModel("tabpfn", version="v2.5")
model.name_          # 'tabpfn:v2.5', the label evaluate() puts on its row
```

What actually answered is recorded on the fitted model, ready to be written out
beside the numbers:

```python
model.fit(X_train, z_train).provenance_
# {'backend': 'tabpfn', 'version': 'v2.5',
#  'repo_id': 'Prior-Labs/tabpfn_2_5',
#  'filename': 'tabpfn-v2.5-regressor-v2.5_default.ckpt',
#  'revision': '6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b',
#  'package': 'tabpfn 9.0.0', 'lazy': 'lazy-tfm 0.1.0', 'device': 'cuda',
#  'n_estimators': 8, 'transforms': ['auto', ...], 'kv_cache': True, ...}
```

It holds the weights, the code that read them and the ensemble that answered,
and no local paths, so it means the same thing on another machine and survives
a trip through JSON.

Versions are not interchangeable. TabPFN `v2` is pretrained for at most 10,000
context rows and `v2.5` for 50,000, against a million for `v3`; `v2` is also
the only one whose weights allow commercial use.
{func}`lazy.get_checkpoint` returns each version's licence and size notes.

## Memory and speed

Peak memory is bounded by `chunk_size` on every backend, except on TabFM's
streaming path, where `query_block_rows` and `decode_chunk_rows` do the same
job. Chunking is exact: a row's answer never depends on which other
rows share its chunk, bit for bit on TabPFN and TabICL and to float
rounding on LimiX-2. Lower `chunk_size` on a small GPU; set it to `0` for a
single pass.

Every backend shows a progress bar while it predicts (`progress="auto"`: on a
terminal or in a notebook, not when output goes to a file).
