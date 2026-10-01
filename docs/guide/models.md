# Supported models

Every model is a pretrained tabular foundation model used in context: `fit`
stores your labelled rows, and each prediction is a forward pass over them.
They differ a great deal in size, speed and what hardware they need.

| Model | `LazyModel(...)` | Weights | Parameters | Licence of the weights | GPU | CPU-friendly |
| ----- | ---------------- | ------: | ---------: | ---------------------- | --- | ------------ |
| TabPFN-3.5 (**default**) | `"tabpfn"` | 0.88 GB | 219 M | Prior Labs, non-commercial | recommended | no |
| TabPFN-3.5-fast | `"tabpfn", version="v3.5-fast"` | 0.33 GB | 84 M | Prior Labs, non-commercial | recommended | no |
| TabPFN-3 | `"tabpfn", version="v3"` | 0.23 GB | 58 M | Prior Labs, non-commercial | recommended | no |
| TabPFN-2.6 | `"tabpfn", version="v2.6"` | 0.05 GB | 13 M | Prior Labs, non-commercial | optional | ≤1,000 rows |
| TabPFN-2.5 | `"tabpfn", version="v2.5"` | 0.04 GB | 10 M | Prior Labs, non-commercial | optional | ≤1,000 rows |
| TabPFN-2 | `"tabpfn", version="v2"` | 0.04 GB | 11 M | Apache-2.0 + attribution | optional | ≤1,000 rows |
| TabICLv2 | `"tabicl"` | 0.11 GB | 29 M | BSD-3-Clause | optional | **yes** |
| LimiX-2 | `"limix"` | 1.6 GB | 406 M | Stable AI 1.0 (Apache-2.0 + attribution) | yes | no (and Linux only) |
| TabFM v1.0 | `"tabfm"` | 6.6 GB | 1.64 B | Google, non-commercial | yes | no |

The weights are downloaded once, at the first `fit`, and cached; see
{doc}`../installation` for the licences in full and {doc}`clusters` for caches
and offline use.

## What it costs

Measured on one NVIDIA GB10 (a Grace-Blackwell desktop: 20 Arm CPU cores and
128 GB of memory shared by CPU and GPU), with every model at its default
settings, on DC1 photometry (12 features). *GPU* is fitting on 10,000 context
rows and predicting 10,000 more; *CPU* is 1,000 and 1,000 with the GPU unused.
Peak GPU memory is what PyTorch allocated at the peak. The last column is the
DC1 CDE loss on the GPU run (lower is better), as a rough guide to accuracy
at this context size, not a ranking.

| Model | GPU: fit + predict | Peak GPU memory | CPU: fit + predict | CDE loss |
| ----- | -----------------: | --------------: | -----------------: | -------: |
| TabPFN-3.5 | 14 s | 3.2 GB | 36 s | −13.47 |
| TabPFN-3.5-fast | 9 s | 2.3 GB | — | −13.00 |
| TabPFN-3 | 9 s | 2.5 GB | — | −13.00 |
| TabPFN-2.5 | 10 s | 5.0 GB | 18 s | −12.91 |
| TabPFN-2 | 11 s | 3.5 GB | — | −12.88 |
| TabICLv2 | 6 s | 4.1 GB | 7 s | −12.33 |
| LimiX-2 | 46 s | 17.5 GB | 102 s | −13.71 |
| TabFM v1.0 | 132 s | 7.8 GB | 8 min 16 s | −13.30 |

TabFM was measured with 4 members, its default at the time; the shared default
is now 8, which roughly doubles its time. The other rows used 8 members. Since
then every model runs in float32 on a CPU (TabFM computed in bfloat16 there),
and TabICL uses mixed precision on a GPU at every context size.

How the cost grows:

- **Time** grows with the number of context rows times the number of query
  rows, times `n_estimators`: every query attends to every context row in
  every member. Doubling the context roughly doubles the prediction time.
- **GPU memory** grows with the context and with `chunk_size`, the number of
  query rows predicted at once; lower `chunk_size` to fit a smaller GPU. The
  key/value cache (`kv_cache=True`, the default) keeps the processed context in
  memory: for LimiX-2 about 2 GB per member at 20,000 context rows, so its
  eight members want a GPU with 16–24 GB or more at that size.
- **TabFM** runs a hierarchy of `n_dither × (1 + n_coarse_bins)` in-context
  classifications (11 by default), which is why it is the slowest; it needs
  its repository build to be even this fast (see {doc}`../installation`).

## Which one to use

- **With a GPU:** TabPFN-3.5, the default: close to the most accurate at a
  fraction of the cost (LimiX-2 edges it at 10,000 context rows for about
  three times the time and five times the memory). On a CPU it is slow,
  and TabPFN itself refuses CPU contexts larger than 5,000 rows (1,000 for the
  v2 family) unless you pass `ignore_pretraining_limits=True`; `lazy` warns
  when it finds no GPU.
- **A laptop, a CPU, or permissive licensing:** TabICLv2. It is small,
  BSD-licensed and nearly as fast on a CPU as on a GPU, at some cost in
  accuracy.
- **Large contexts:** TabPFN-3 and later were pretrained on contexts up to a
  million rows. LimiX-2 degrades above about 20,000 rows unless bagged
  (`bag_size=20_000` with enough members), and TabPFN-2 and 2.5 were
  pretrained on at most 10,000 and 50,000 rows.
- **Commercial use:** TabICLv2, TabPFN-2 and LimiX-2 have weights that allow
  it, under their licences' terms; the others are non-commercial.

All of them take the same parameters and give the same outputs; see
{doc}`backends` for how each implements them.
