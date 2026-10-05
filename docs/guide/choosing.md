# Choosing a model

All models take the same parameters and give the same outputs (see
{doc}`interface`), so the choice comes down to accuracy, cost, hardware, the
size and nature of the training set, and the license of the weights.
{doc}`../models/index` lists each model's size, license and hardware needs, and
{doc}`scaling` gives the measured cost of each.

## With a GPU

With a GPU, the default TabPFN-3.5 is close to the most accurate model at a
fraction of the cost (LimiX-2 edges it at 10,000 context rows for about three
times the time and five times the memory). For large contexts, TabPFN-3 and
later were pretrained on contexts up to a million rows, while TabPFN-2 and 2.5
were pretrained on at most 10,000 and 50,000 rows, respectively.

Our own testing shows that LimiX-2 performs best when the training set is
biased (i.e., unrepresentative of the test set). However, LimiX-2
degrades above about 20,000 context rows, so a larger training set needs
bagging, with `bag_size=20_000` (so that each member sees at most 20,000 rows)
and enough members, as in our own tests:

```python
model = lazy.LazyModel("limix", bag_size=20_000, n_estimators=32)
```

## On a CPU

On a CPU, TabPFN-3.5 is slow, and TabPFN itself refuses CPU contexts larger
than 5,000 rows (1,000 for the v2 family) unless
`ignore_pretraining_limits=True` is passed, so `lazy` warns when it finds no
GPU. On a laptop or a CPU, or when permissive licensing is needed, TabICLv2 is
the model to use, since it is small, BSD-licensed and nearly as fast on a CPU
as on a GPU, at some cost in accuracy. For a TabPFN on a CPU, TabPFN-3.5-fast
is a smaller TabPFN-3.5 at under half its size. It is more than twice as fast
on a CPU (within the 5,000-row CPU limit of TabPFN), somewhat faster on a GPU,
and less accurate than TabPFN-3.5 (−13.05 vs. −13.47 in CDE loss on 10,000 DC1
context rows), and `lazy` does not warn when it runs on a CPU.

## Licenses

For commercial use, only TabICLv2 and TabPFN-2 have weights that allow it,
under the terms of their licenses, while the others are non-commercial.
Please check the license of each backend at its original source before use.
