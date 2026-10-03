# Reproducibility and provenance

A prediction from LAZY is fixed by the checkpoint, the context, the seed and
the shared parameters, and by nothing on the machine it runs on. This page
describes how each of these is pinned and recorded, and which settings change
a prediction.

## Seeds

Every model takes `random_state`, the ensemble's seed. The default is `0` on
every model, including TabICL and TabFM, whose creators use 42. With
`random_state=None`, a fresh seed is drawn at `fit` and recorded as
`random_state_` and in `provenance_`, so that the run can be repeated.

Each member has a seed of its own, derived from `random_state` and its
position `i` in the ensemble by NumPy's `SeedSequence([random_state, i])`, and
kept below 2^31 so that every model accepts it. A group of members that one
call of a model serves (e.g., every member of TabPFN without bagging) is run
with the seed of its first member. Hashing the pair keeps the seeds of
neighboring ensembles apart, whereas an offset such as `random_state + i`
would give member `i` at `random_state=r` the seed of member `i - 1` at
`r + 1`, so that two runs differing only in their seed would share most of
their members. The bags are drawn from a generator seeded with `random_state`
itself, and the same bags are drawn on every model (see {doc}`interface`).

## Pinned checkpoints

A backbone is a family rather than a model, so every backend takes a
`version`, and each version is a separately pinned checkpoint:

```python
import lazy

lazy.list_versions("tabpfn")
# ['v2', 'v2.5', 'v2.6', 'v3', 'v3.5', 'v3.5-fast']

model = lazy.LazyModel("tabpfn", version="v2.5")
model.name_          # 'tabpfn:v2.5', the label evaluate() puts on its row
```

The checkpoint that a model loads is exactly the one
{func}`~lazy.models.download_checkpoint` fetches, because the pinned revision
is passed to the backend rather than left to its own default. Each model's
`"auto"` recipe is also written out in LAZY for every version rather than
read from the installed package, so upgrading TabPFN or TabICL cannot change
what it runs (see {doc}`interface`).

## Provenance

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

The `softmax_temperature`, `mixed_precision` and `outlier_threshold` actually
used are recorded there too.

## Comparing versions

Versions are not interchangeable. TabPFN `v2` is pretrained for at most 10,000
context rows and `v2.5` for 50,000, compared with a million for `v3`, and `v2`
is also the only one whose weights allow commercial use.
{func}`lazy.get_checkpoint <lazy.models.get_checkpoint>` returns each
version's license and size notes. We therefore recommend comparing numbers
only between runs whose `provenance_` agree on the version and the recipe. For
example, the
paper's TabFM numbers come from the streaming path in bfloat16, and the cost
measurements in {doc}`scaling` predate the change to float32 on every CPU.

## What changes a prediction

The version, the context, `random_state`, `n_estimators`, `transforms`,
`feature_shuffle`, `bag_size`, `softmax_temperature` and `outlier_threshold`
all change a prediction, which is why they are recorded. Chunking does not. A
row's answer never depends on which other rows share its chunk: on a CPU it is
the same bit for bit on TabPFN and TabICL and to float rounding on LimiX-2 and
TabFM, and under mixed precision on a GPU it is the same to the rounding of
that precision.

The key/value cache changes a prediction only by rounding. The cache is exact
on a CPU. Since queries only ever attend to the context and never to each
other, the answer is the uncached one up to float rounding. Under mixed
precision on a GPU the two agree to the rounding of that precision (for
TabICL, about 1e-3 of the quantile spread). For TabFM in bfloat16, that
rounding moves a typical row's density by about 3% of its peak and single bins
by up to a quarter of the peak, on either path and at any number of bins,
while the CRPS and NLL moved by at most 0.004 on `yacht` and 0.001 on DC1,
about as much as with a change of seed. TabPFN's own int8 cache, which
LAZY uses only when asked with `kv_cache="int8"`, moves densities by up to
about half a percent. Precision changes a prediction by the same kind of
rounding: `mixed_precision=True` uses each model's reduced-precision path on a
GPU, and every model computes in float32 on a CPU.

LimiX-2 needed three changes to make a row's answer independent of which
other rows share its chunk. The first is that each member's preprocessing is
fitted on the context alone (upstream's column filter, categorical detection
and category encoding also see the queries, and LAZY treats every column as
numeric). The second is that the feature positional embedding has its own
random generator (upstream draws it from the global one after advancing it by
an amount that depends on the chunk's size). The third is the ported cache.
The answers are statistically, not bitwise, those of upstream's
`LimiXPredictor`.
