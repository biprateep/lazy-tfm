# Scaling and performance

The cost of a prediction is set by the size of the context, since `fit` only
stores the context. This page gives the measured cost of each model, how it
grows with the data, and the parameters that bound it.

## What it costs

We measured the cost of each model on one NVIDIA GB10 (a Grace-Blackwell
desktop with 20 Arm CPU cores and 128 GB of memory shared by CPU and GPU),
with every model at its default settings, on DC1 photometry (12 features). The
*GPU* run fits on 10,000 context rows and predicts 10,000 more, while the
*CPU* run fits on 1,000 and predicts 1,000 with the GPU unused. The CPU run
therefore has a hundredth of the pairs of context and query rows (1,000 ×
1,000 against 10,000 × 10,000), so the two columns do not compare the
devices. Peak GPU memory is what PyTorch allocated at the peak. The last
column is the DC1 CDE loss on the GPU run (lower is better), a rough guide to
accuracy at this context size and not a ranking.

| Model | GPU, 10k + 10k rows | Peak GPU memory | CPU, 1k + 1k rows | CDE loss |
| ----- | -----------------: | --------------: | -----------------: | -------: |
| TabPFN-3.5 | 14 s | 3.2 GB | 36 s | −13.47 |
| TabPFN-3.5-fast | 9 s | 2.3 GB | 11 s\* | −13.00 |
| TabPFN-3 | 9 s | 2.5 GB | — | −13.00 |
| TabPFN-2.5 | 10 s | 5.0 GB | 18 s | −12.91 |
| TabPFN-2 | 11 s | 3.5 GB | — | −12.88 |
| TabICLv2 | 6 s | 4.1 GB | 7 s | −12.33 |
| LimiX-2 | 46 s | 17.5 GB | 102 s | −13.71 |
| TabFM v1.0 | 132 s | 7.8 GB | 8 min 16 s | −13.30 |

\*Measured in a separate run under the same conditions, in which TabPFN-3.5
took 26 s. With 4 members (the fast checkpoint's own count), it takes 6 s.

TabFM was measured with 4 members and the other rows with the default of 8;
8 members roughly double TabFM's time. These runs predate two settings of
the current code: every model computes in float32 on a CPU (TabFM computed in
bfloat16 there), and TabICL uses mixed precision on a GPU at every context
size.

## How time and memory grow

The prediction time grows with the number of context rows times the number of
query rows, times `n_estimators`, since every query attends to every context
row in every member. Therefore, doubling the context roughly doubles the
prediction time. The GPU memory grows with the context and with `chunk_size`
(the number of query rows predicted at once), so lowering `chunk_size` lets a
model fit on a smaller GPU. TabFM runs a hierarchy of `n_dither × (1 +
n_coarse_bins)` in-context classifications (11 by default), which is why it is
the slowest, and it needs its repository build to be even this fast (see
{doc}`../installation`).

```{figure} ../figures/figs/chunks_bags.svg
:class: lazy-figure
:alt: Two grids of query rows by context rows, split into three chunks of query rows; in the second grid only half the context columns are filled.
:width: 100%

**Figure 1.** The queries' work in one member, in which each filled cell is a
query row attending to a context row. The queries run in chunks of
`chunk_size` rows, which bound the memory of each pass, and a bagged member
(right) attends only to the context rows in its bag. Not drawn: the context's
own work, which without the key/value cache each chunk repeats (see
[Chunking](#chunking)).
```

## Chunking

Peak memory is bounded by `chunk_size` on every backend (8,192 query rows by
default), and on TabFM's streaming path `query_block_rows` also bounds the
host memory of a block of queries. We recommend lowering `chunk_size` on a
small GPU, and setting it to `0` gives a single pass. With `kv_cache=False`,
each forward pass processes the context again, so the context's cost is paid
once per chunk, and a smaller `chunk_size` then takes longer. On the host, `predict`,
`score` and `evaluate` work through the query rows a block at a time, and
`predict_quantiles`, `predict_interval` and `predict_pit` a chunk at a time
(TabFM, which predicts all its queries together, excepted), so their memory
does not grow with the number of rows; `predict_proba` and
`predict_distribution` return every row at once. A row's answer does not
depend on which other rows share its chunk (see {doc}`reproducibility`). Every
backend shows a progress bar while it predicts (with `progress="auto"`, on a
terminal or in a notebook but not when output goes to a file).

## The key/value cache

An in-context model reads the whole context for every query. With
`kv_cache=True` it reads the context once and each chunk of queries attends to
what was stored, so a large query set costs little more than its own rows. The
cache keeps the processed context in memory, which for LimiX-2 takes about 3
GB per member at 20,000 rows of DC1, so its eight members want 24 GB or more
at that size (cache on minus cache off, after a fit and a prediction; see
[Measurement setup](#measurement-setup)). LimiX-2 has no cache upstream, so LAZY ports
one, which falls back, with a warning, when the device lacks the room.

TabPFN caches at full precision. Its own default is an int8 cache that moves
densities by about half a percent of the peak for a typical row, and up to 3%
(TabPFN-3.5, 10,000 DC1 context rows and 2,000 test rows; see [Measurement
setup](#measurement-setup)), and `kv_cache="int8"` (or `"fp8"`) asks for it.
TabICL caches keys and values, while `kv_cache="repr"` stores the smaller row
representations instead and re-runs its in-context layers. TabFM uses its
streaming prefill/decode path, which needs the repository build (see
{doc}`../installation`) and otherwise falls back with a
{class}`~lazy.models.PerformanceWarning`. Since TabFM fits and prefills inside
every `predict` call, the cache serves the chunks of that call.

## Bagging

`bag_size` gives each member a random subset of the context (see
[Bagging](interface.md#bagging)). This matters when the context outgrows what
a model was pretrained on. LimiX-2
degrades above about 20,000 context rows, and a
{class}`~lazy.models.ContextSizeWarning` says so whenever a member's context
is larger (the whole context without bagging, or a bag of more than 20,000
rows). The remedy it suggests is the one that the DC1 runs of
{doc}`../models/limix` used:

```python
model = lazy.LazyModel("limix", bag_size=20_000, n_estimators=32)
```

TabPFN checks its limits against what each member sees. A bagged member is a
regressor of its own, fitted on its bag alone, so only the bag has to lie
within the context size the checkpoint declares and within the CPU cap, and a
`bag_size` below them runs a context of any size without
`ignore_pretraining_limits`. A bag (or an unbagged context) above them raises
TabPFN's own error, to which LAZY adds the `bag_size` and `n_estimators` that
would keep each member within the limit while the members together cover the
context.

Bagging also bounds the cost, since each member then attends to `bag_size`
context rows rather than to all of them. The members' contexts differ in
length, so TabFM runs its bagged members one at a time whatever
`member_batch_size` says, and the native grid of TabPFN and LimiX-2 holds
every member's buckets (see {doc}`interface`).

## Reusing loaded weights

Every model reads its checkpoint from disk and builds the network on its
device when it is first fitted (TabFM when it first predicts), and for a small
context this load is most of the cost of a fit. The network learns nothing from
a fit, since the context and its key/value cache belong to the fitted model.
LAZY therefore keeps every network it loads for the rest of the Python process
and reuses it in any later fit with the same version, checkpoint, device and
precision, so a grid search, a cross-validation or a sweep over `n_estimators`
loads each network once. With one member and 500 DC1 context rows on the
GPU, this cuts the median of 7 refits after a warm-up of TabPFN-3.5 from 0.7 s to 0.03 s, and that of LimiX-2 from 1.3 s to 0.1 s (see
[Measurement setup](#measurement-setup)). A shared network gives
predictions bit for bit identical to those of a freshly loaded one.

Each network is held once, and `lazy.clear_model_cache()` releases all of them
(TabFM's backbone alone takes 6.6 GB). A fitted model keeps the network it ran
on, so the memory is returned only once those models are deleted too. Setting
`LAZY_MODEL_CACHE=0` in the environment, or calling
`lazy.set_model_cache(False)`, turns the sharing off, and every fit then loads
its own network. Fits and predictions in threads can share a network: a
network that keeps the context of its last call is locked while it answers,
so threads on it take turns.

## Parallel jobs

With joblib's default backend, the `n_jobs` of scikit-learn's search and
cross-validation tools runs each job in a worker process of its own. Each
worker has its own model cache, so it loads each network that it uses once,
and the workers share none. With `device="auto"` on a machine with a GPU,
every worker runs on the same GPU, so that GPU holds one copy of the network
per worker, beside each job's context and key/value cache. The GPU memory
that a search needs therefore grows with `n_jobs`, and on one GPU `n_jobs=1`
needs the least.

## GPU and CPU

Every model chooses its device when it is fitted, and `device="auto"` selects
CUDA if it is available and the CPU otherwise (see {doc}`../installation`).
Most of the models need a GPU to be practical. TabPFN refuses large CPU
contexts (see [Context and feature limits](limits.md#context-and-feature-limits))
unless `ignore_pretraining_limits=True` or a `bag_size` keeps every bag within
that limit, and `lazy` warns at `fit` when it finds no GPU for TabPFN-3.5.
TabICLv2 fits on 1,000 rows and predicts 1,000 in 7 s on a CPU (in the table above),
and Kumo Tabular Small (`version="v1-small"`, 28 M parameters) is small
enough for a CPU, while LimiX-2, TabFM and Kumo Tabular's Medium and Large
need a GPU.
{doc}`choosing` describes which model to use on each.

## Measurement setup

The figures in these docs that link to this section were measured with this
setup. Kumo Tabular's comparison with NVIDIA's code ran on the CPU, and every
other figure on the GPU.

| Item | Value |
| --- | --- |
| Commit | `main` at ccc88a6 |
| Machine | one NVIDIA GB10 (Grace-Blackwell), 20 Arm cores, 128 GB of memory shared by CPU and GPU |
| Driver, CUDA | 580.178.04, CUDA 13.0 |
| Threads | `torch.set_num_threads(4)`; OMP not set |
| Libraries | Python 3.12.14, torch 2.14.0, numpy 2.5.3, scikit-learn 1.9.1, tabpfn 9.0.0, tabicl 2.2.0, tabfm 1.0.1 (the repository build of `lazy setup`), structured-data-models 0.2.1, LimiX source 516bf39 |
| Data | DC1 rows drawn with `numpy.random.default_rng(0)` from its challenge split; OpenML splits by `train_test_split(test_size=0.1, random_state=0)` |
| Settings | each backend's defaults and `random_state=0` (42 on the golden problem), unless the figure states otherwise |
