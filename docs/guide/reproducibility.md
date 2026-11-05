# Reproducibility and provenance

A prediction from LAZY is fixed by the checkpoint, the context, the seed and
the shared parameters, up to the rounding of the machine it runs on, and for
LimiX-2 by the kind of device (see [Across devices](#across-devices)).
This page describes how each is pinned and recorded, and which settings change a
prediction.

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
`r + 1`, so two runs differing only in their seed would share most of their
members. The bags are drawn from a generator seeded with `random_state`
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
`"auto"` recipe is also pinned for every version (see
[Transforms](interface.md#transforms)).

## Provenance

What answered is recorded on the fitted model in `provenance_`, ready to be
written out beside the numbers. It holds the weights, the code that read them
and the ensemble that answered, and no local paths, so it means the same thing
on another machine and survives a trip through JSON:

```python
model.fit(X_train, y_train).provenance_
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
example, the paper's TabFM numbers come from the streaming path in bfloat16,
and the cost measurements in {doc}`scaling` state the precision they ran at.

## What changes a prediction

The version, the context, `random_state`, `n_estimators`, `transforms`,
`feature_shuffle`, `bag_size`, `softmax_temperature` and `outlier_threshold`
all change a prediction, which is why they are recorded. Chunking does not. A
row's answer never depends on which other rows share its chunk: on a CPU it is
the same to float rounding on every backend (a chunk of a single row takes a
different matrix-multiply path in TabICL and Kumo, about $10^{-6}$ apart), and
under mixed precision on a GPU it is the same to the rounding of that
precision. On some CPUs, chunks of more than one row give the same bits; on
others they do not.

The key/value cache changes a prediction only by rounding. Since queries only
ever attend to the context and never to each other, the answer is the uncached
one up to float rounding, on a CPU as on a GPU. Under mixed
precision on a GPU the two agree to the rounding of that precision (for
TabICL, about 1e-3 of the quantile spread). For TabFM in bfloat16 against
float32, with 8 members on `yacht` and on 1,000 context and 1,000 test rows of
DC1, that rounding moved the median row's density by about 2% of its peak and
single bins by up to 15%. The CRPS moved by 0.0002 on `yacht` and 0.00003 on
DC1, and the NLL by 0.0018 and 0.0016 (see [Measurement
setup](scaling.md#measurement-setup)), about as much as with a change of
seed. TabPFN's own int8 cache, which
LAZY uses only when asked with `kv_cache="int8"`, moves densities by about
half a percent of the peak for a typical row, and up to 3% (see
{doc}`scaling`). Precision changes a prediction by the same kind of
rounding: `mixed_precision=True` uses each model's reduced-precision path on a
GPU, and every model computes in float32 on a CPU.

## Across machines

On one machine, the same inputs, settings and seed give the same densities to
the bit, run after run. On another machine they can differ, because the
processor and the libraries' code paths round differently. For most backends
this is float rounding; for TabPFN on a CPU, and for any backend in bfloat16,
it is a few percent of a density's peak:

| Backend | Another thread count, kernel set or BLAS path, on a CPU | Cause |
|---|---:|---|
| Kumo | up to $7 \times 10^{-5}$ of the peak | float32 rounding |
| TabICL | up to $5 \times 10^{-5}$ of the peak | float32 rounding |
| LimiX-2 | within $10^{-4}$ relatively | float32 rounding |
| TabFM, float32 (the CPU default) | up to $6 \times 10^{-6}$ of the peak | float32 rounding |
| TabFM, bfloat16 | 2% to 4% of the peak | bfloat16 keeps 8 bits of mantissa |
| TabPFN | 4% to 8% of the peak | its fingerprint feature (below) |

The figures are the largest difference, over 50 rows and 200 bins of
`DC1_GRID`, between two runs on one 4-core x86 CPU that differed in one thing:
the torch thread count (one or four), torch's vector kernels
(`ATEN_CPU_CAPABILITY`), or MKL's code path (`MKL_CBWR`). Two processors
differ in the same ways: that CPU differed by the same amounts from a GitHub
runner (TabPFN, TabFM) and from a maintainer's machine (Kumo, TabICL). For
LimiX-2, the figure is not a measured difference but the tolerance that
densities recorded on a maintainer's machine meet on GitHub's runners.

TabPFN adds a *fingerprint* feature: a hash of each row of its preprocessed
data, which helps the model tell identical rows apart. The data it hashes is
float32, so a difference in its last bit, which another processor's
arithmetic produces, gives that row a different hash. The model then sees a
different value of that feature, and its density moves by several percent of
the peak. Without the fingerprint, TabPFN differs across machines by about
$10^{-5}$ of the peak. This is upstream TabPFN's behavior, which LAZY's pinned
recipe keeps.

So compare results from different machines with scores over many rows (the
CDE loss, CRPS or NLL), not density by density.

Three things make LimiX-2's answer for a row independent of which other
rows share its chunk. The first is that each member's preprocessing is
fitted on the context alone (upstream's column filter, categorical detection
and category encoding also see the queries, and LAZY treats every column as
numeric). The second is that the feature positional embedding has its own
random generator (upstream draws it from the global one after advancing it by
an amount that depends on the chunk's size). The third is the ported cache.
The answers are statistically, not bitwise, those of upstream's
`LimiXPredictor`.

## Across devices

Kumo Tabular draws each member's sign flips and column shuffles with a torch
random generator. A CPU generator and a CUDA generator give different streams
for the same seed, and NVIDIA's package draws on the device that runs the
model (NVIDIA/structured-data-models#1339). So LAZY draws them on a CPU
generator on every device and moves them to the device, and the same
`random_state` gives the same ensemble on a CPU and on a GPU. On a CPU these
are NVIDIA's own draws. A Kumo model saved by LAZY before 0.6 still draws on
the device it runs on, so it repeats its answers on the same kind of device.

Each backend was fitted afresh on each device, in float32 with
`random_state=0`, on 300 `california_housing` rows, and predicted 50 more:

| Backend | CPU against GPU | Seed 0 against seed 1, on the GPU |
|---|---:|---:|
| TabPFN | $2.5 \times 10^{-5}$ | 0.23 |
| TabICL | $6.8 \times 10^{-6}$ | 0.19 |
| TabFM | $3.9 \times 10^{-6}$ | 0.11 |
| LimiX-2 | 0.023 | 0.28 |
| Kumo Tabular | $6.6 \times 10^{-6}$ | 0.12 |

The figures are the largest change of the 5%, 50% and 95% quantiles over the
same 50 rows, as a fraction of a row's 5% to 95% range, on an NVIDIA GB10 and
its CPU. Each device repeats its own answer to the bit. TabPFN, TabICL, TabFM
and Kumo Tabular agree across devices to float32 rounding. LimiX-2 differs
by more than rounding, but by about a tenth of a change of seed; its cause is
not yet known.

So, to repeat a result exactly, run it on the same kind of device, which
`provenance_` records. Compare results from a CPU and a GPU with scores over
many rows, as across machines.
