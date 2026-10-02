# Supported models

Every model in LAZY is a pretrained tabular foundation model used in context,
in which `fit` stores the labeled rows and each prediction is a forward pass
over them. The models differ a great deal in size, speed and the hardware they
need.

| Model | `LazyModel(...)` | Weights | Parameters | License of the weights | GPU | CPU-friendly |
| ----- | ---------------- | ------: | ---------: | ---------------------- | --- | ------------ |
| TabPFN-3.5 (**default**) | `"tabpfn"` | 0.88 GB | 219 M | Prior Labs, non-commercial | recommended | no |
| TabPFN-3.5-fast | `"tabpfn", version="v3.5-fast"` | 0.33 GB | 84 M | Prior Labs, non-commercial | optional | ≤5,000 rows |
| TabPFN-3 | `"tabpfn", version="v3"` | 0.23 GB | 58 M | Prior Labs, non-commercial | recommended | no |
| TabPFN-2.6 | `"tabpfn", version="v2.6"` | 0.05 GB | 13 M | Prior Labs, non-commercial | optional | ≤1,000 rows |
| TabPFN-2.5 | `"tabpfn", version="v2.5"` | 0.04 GB | 10 M | Prior Labs, non-commercial | optional | ≤1,000 rows |
| TabPFN-2 | `"tabpfn", version="v2"` | 0.04 GB | 11 M | Apache-2.0 + attribution | optional | ≤1,000 rows |
| TabICLv2 | `"tabicl"` | 0.11 GB | 29 M | BSD-3-Clause | optional | **yes** |
| LimiX-2 | `"limix"` | 1.6 GB | 406 M | StableAI LimiX, non-commercial | yes | no (and Linux only) |
| TabFM v1.0 | `"tabfm"` | 6.6 GB | 1.64 B | Google, non-commercial | yes | no |

The weights are downloaded once, at the first `fit`, and cached from then on.
{doc}`../installation` gives the licenses in full, and
{doc}`../guide/clusters` describes caches and offline use.

## What it costs

We measured the cost of each model on one NVIDIA GB10 (a Grace-Blackwell
desktop with 20 Arm CPU cores and 128 GB of memory shared by CPU and GPU),
with every model at its default settings, on DC1 photometry (12 features). The
*GPU* run fits on 10,000 context rows and predicts 10,000 more, while the
*CPU* run fits on 1,000 and predicts 1,000 with the GPU unused. Peak GPU
memory is what PyTorch allocated at the peak. The last column is the DC1 CDE
loss on the GPU run (lower is better), which serves as a rough guide to
accuracy at this context size and not as a ranking.

| Model | GPU: fit + predict | Peak GPU memory | CPU: fit + predict | CDE loss |
| ----- | -----------------: | --------------: | -----------------: | -------: |
| TabPFN-3.5 | 14 s | 3.2 GB | 36 s | −13.47 |
| TabPFN-3.5-fast | 9 s | 2.3 GB | 11 s\* | −13.00 |
| TabPFN-3 | 9 s | 2.5 GB | — | −13.00 |
| TabPFN-2.5 | 10 s | 5.0 GB | 18 s | −12.91 |
| TabPFN-2 | 11 s | 3.5 GB | — | −12.88 |
| TabICLv2 | 6 s | 4.1 GB | 7 s | −12.33 |
| LimiX-2 | 46 s | 17.5 GB | 102 s | −13.71 |
| TabFM v1.0 | 132 s | 7.8 GB | 8 min 16 s | −13.30 |

\*Measured later under the same conditions, alongside TabPFN-3.5 at 26 s. With
4 members (the fast checkpoint's own count), it takes 6 s.

TabFM was measured with 4 members, which was its default at the time. The
shared default is now 8, which roughly doubles its time, and the other rows
used 8 members. Since these measurements, every model runs in float32 on a CPU
(TabFM computed in bfloat16 there), and TabICL uses mixed precision on a GPU
at every context size.

The prediction time grows with the number of context rows times the number of
query rows, times `n_estimators`, since every query attends to every context
row in every member. Therefore, doubling the context roughly doubles the
prediction time. The GPU memory grows with the context and with `chunk_size`,
the number of query rows predicted at once, so lowering `chunk_size` lets a
model fit on a smaller GPU. The key/value cache (`kv_cache=True`, the default)
keeps the processed context in memory, which for LimiX-2 takes about 2 GB per
member at 20,000 context rows, so its eight members want a GPU with 16–24 GB
or more at that size. TabFM runs a hierarchy of `n_dither × (1 +
n_coarse_bins)` in-context classifications (11 by default), which is why it is
the slowest, and it needs its repository build to be even this fast (see
{doc}`../installation`).

## Which one to use

With a GPU, the default TabPFN-3.5 is close to the most accurate model at a
fraction of the cost (LimiX-2 edges it at 10,000 context rows for about three
times the time and five times the memory). On a CPU it is slow, and TabPFN
itself refuses CPU contexts larger than 5,000 rows (1,000 for the v2 family)
unless `ignore_pretraining_limits=True` is passed, so `lazy` warns when it
finds no GPU. On a laptop or a CPU, or when permissive licensing is needed,
TabICLv2 is the model to use, since it is small, BSD-licensed and nearly as
fast on a CPU as on a GPU, at some cost in accuracy. For a TabPFN on a CPU,
TabPFN-3.5-fast is a smaller TabPFN-3.5 at under half its size. It is more
than twice as fast on a CPU (within the 5,000-row CPU limit of TabPFN),
somewhat faster on a GPU, and less accurate than TabPFN-3.5 (−13.05 vs. −13.47
in CDE loss on 10,000 DC1 context rows), and `lazy` does not warn when it runs
on a CPU.

For large contexts, TabPFN-3 and later were pretrained on contexts up to a
million rows. LimiX-2 degrades above about 20,000 rows unless bagged
(`bag_size=20_000` with enough members), and TabPFN-2 and 2.5 were pretrained
on at most 10,000 and 50,000 rows, respectively. For commercial use, only
TabICLv2 and TabPFN-2 have weights that allow it, under the terms of their
licenses, while the others are non-commercial. All of the models take the same
parameters and give the same outputs, and {doc}`../guide/backends` describes
how each one implements them.

```{toctree}
:maxdepth: 1

tabpfn
tabicl
tabfm
limix
```
