# FAQ

## Do I need a GPU?

Most of the models need a GPU to be practical, and LimiX-2 and TabFM need one.
Without a GPU, we recommend TabICLv2, which is small, BSD-licensed and nearly
as fast on a CPU as on a GPU, or TabPFN-3.5-fast with a context of at most
5,000 rows. {doc}`choosing` and {doc}`scaling` give the details.

## What does fit do?

`fit` trains no weights. On TabPFN, TabICL and LimiX-2 it fits each member's
preprocessing on the context and, with the key/value cache on (`kv_cache=True`,
the default), also runs the context through the network and stores the result,
so it costs one forward pass over the context per member. TabPFN builds this
cache with `fit_mode="fit_with_cache"`, TabICL with its own key/value cache and
LimiX-2 with the cache that LAZY ports. TabFM only stores the context at `fit`,
and fits its classifiers and prefills their cache inside every `predict` call.
With `kv_cache=False`, `fit` runs no forward pass and every chunk of queries
reads the context again in `predict` (see {doc}`tfm` and {doc}`scaling`). The
first `fit` of a model also downloads and caches its weights.

## Can I fine-tune?

LAZY does not fine-tune the weights. The models are pretrained and used in
context, so everything a model knows about a dataset comes from the context it
is given. LAZY lets the user change the context itself and the shared
parameters (e.g., the ensemble, the transforms and the softmax temperature;
see {doc}`interface`).

## Can I use it commercially?

LAZY itself is released under the MIT license, but the license of the weights
decides. Only TabICLv2 (BSD-3-Clause) and TabPFN-2 (Apache-2.0 with attribution) have
weights that allow commercial use, under the terms of their licenses; the
others are non-commercial. LimiX-2's weights are released under the
StableAI LimiX Non-Commercial License 1.0, which also requires distributions,
derivatives and publications to display "Built with StableAI LimiX", while
LimiX's code is under the Apache-2.0-based Stable AI Technology Co., Ltd.
License 1.0. Please check the license of each backend at its original source
before use.

Built with StableAI LimiX.

## How does this differ from calling TabPFN directly?

LAZY runs the same network with the same weights, but keeps its full
distribution, on any grid, where TabPFN's own `predict` returns a single value
by default (see {doc}`distributions`). Every model in LAZY also takes the same
parameters with the same defaults, so a few settings differ from TabPFN's own:
we run exactly `n_estimators` members, turn off its automatic detection of
categorical columns, cache the context at full precision and compute in
float32 on a CPU (see {doc}`interface` and {doc}`../models/tabpfn`). The
checkpoint and the preprocessing recipe are pinned and recorded in
`provenance_` (see {doc}`reproducibility`).

## Why does my interval look too wide or too narrow?

A predicted distribution is broad when the context holds little information
about a query and narrow when it holds a lot (see {doc}`tfm`), so a small
context gives wide intervals. An interval that is too narrow to cover the
truth can mean that the query lies outside the range of the context or that
the context is unrepresentative of the queries, and a grid that ends inside a
distribution also narrows it (see {doc}`limits`). Over a test set, the PIT
histogram shows whether the intervals are too wide (a hump in the middle) or
too narrow (peaks at 0 and 1), and {doc}`distributions` describes the
statistics that measure it. On the models with a softmax, `softmax_temperature`
sets how sharp the densities are, with lower values sharpening them and higher
values broadening them.

## Which model should I use?

With a GPU, the default TabPFN-3.5 is close to the most accurate model at a
fraction of the cost. Without one, or when permissive licensing is needed,
TabICLv2 is the model to use, and in our own testing LimiX-2 performs best
when the training set is biased. {doc}`choosing` gives the reasons.
