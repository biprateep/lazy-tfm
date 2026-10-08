# Ensembles of models

Different foundation models make different errors, so the densities they
predict for the same row differ, and a combination of them can be better than
any one of them. {class}`~lazy.models.ensemble.LazyEnsembleModel` fits several
models on the same context and combines (pools) their predicted distributions
into one. The pooled model has the interface of a single model (see
{doc}`interface`), so `predict_proba`, `predict_quantiles`, `predict_pit`,
`score` and `evaluate` work unchanged:

```python
import lazy

model = lazy.LazyEnsembleModel(
    [
        "tabpfn",
        lazy.LazyModel("limix", bag_size=20_000, n_estimators=32),
        "tabfm",
    ]
)
model.fit(X_train, y_train)
pdfs = model.predict_proba(X_test)
```

Each member is given as a backend name, which builds `LazyModel(name)` at its
defaults, or as a configured {class}`~lazy.models.lazy_model.LazyModel`. A
member is named after its backend, with `_2`, `_3`, etc. appended to repeats,
and its parameters are reachable as `<name>__<parameter>` (e.g.,
`tabfm__n_estimators`), as in scikit-learn's `VotingRegressor`, so `clone` and
the search objects work. The ensemble's `random_state` and `device` are passed
to every member whose own value is still the default (0 and `"auto"`), while a
member given another value keeps it. The fitted members are in `estimators_`
and `named_estimators_`, and `provenance_` holds each member's own provenance
together with the pooling, the weights and the pooling grid.

## Three ways to pool

All three pools weight the members equally, and the weights are recorded in
`weights_`. For $K$ members with densities $p_k$ and quantile functions $Q_k$,
the table defines each pool and when we would use it:

| `pooling`     | Pooled distribution                                                | When to use it                                                                   |
| ------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `"geometric"` | $\prod_k p_k^{1/K}$, renormalized (also called a log pool).        | The default. Sharper where the members agree, and drops a mode only one predicts. |
| `"linear"`    | $\frac{1}{K} \sum_k p_k$, the mixture of the members.              | Members that each miss part of the distribution; keeps every mode, but is wider.  |
| `"quantile"`  | $\frac{1}{K} \sum_k Q_k$, the average quantile function (Vincentization). | Members that agree on the shape and differ in location or width.                 |

The members are first evaluated exactly, each on its own native distribution.
For the geometric and the linear pool, we then compute the probability that
each member places in every bin of one fine grid of equal bins (the
`native_grid_` of the pooled model), pool these bin by bin and renormalize, so
that the pooled model answers with a histogram on that grid. The grid spans the
range of the training targets padded by 25% on each side (cut to the union of
the members' own grids where that is narrower), so the pool does not cut off
the mass that the members place beyond the targets. Its bins are as narrow as
the median bin of the finest member within that range, with between 200 and
10,000 bins, and `pool_bins` sets their number by hand. Before taking
logarithms, the geometric pool adds a floor of $10^{-10}$ times the uniform
density over the grid to every member's density, so that a bin which one member
rules out completely is suppressed by a factor of about $10^{-10/K}$ rather
than excluded. The quantile pool averages the members' quantiles at the 1,001
levels 0, 0.001, ..., 1, each clipped to the pooling grid (since the ends of a
bar distribution reach far into the tails), and answers with a
{class}`~lazy.distributions.QuantileDistribution` that lies within the grid.

## Our own testing

Our own testing on the DC1 photometric redshift benchmark shows that pooling
different models helps, while pooling the members of one model does not. On a
random subset of 50,000 test galaxies with a context of 43,486 galaxies, scored
on a grid of 0.001 in redshift, the equal geometric pool of TabPFN-3.5, bagged
LimiX-2 (32 members on random bags of 20,000 rows) and TabFM lowered the CDE
loss by 0.277 ± 0.024 relative to TabPFN-3.5, and it was the best calibrated of
the estimators we tested (a Kolmogorov-Smirnov distance of its PIT values from
uniform of 0.0092 vs. 0.0100 for TabPFN-3.5). On the biased training set, the
same pool lowered the CDE loss by 0.301 ± 0.033 relative to LimiX-2, the best
single model there. The cheaper pair of TabPFN-3.5 and TabFM lowered it by
0.308 ± 0.022 on DC1, but only by 0.214 on the biased training set, so we
recommend the three models together as the configuration that holds up in both
regimes. In contrast, pooling the eight members of TabPFN-3.5 geometrically,
instead of averaging them, lowered the CDE loss on its native grid by only
0.0075.

The pools need a large enough context. The equal geometric pool of the three
models beat the best single model at every context size from 5,000 rows in both
regimes. However, on 1,000 rows of DC1, where TabFM is far behind the other two
models (by 1.38 in CDE loss), the pool gained only 0.06 over the best single
model, and the pair of TabPFN-3.5 and TabFM lost 0.07, since an equal pool is
dragged down by its weakest member. Therefore, `fit` warns with an
{class}`~lazy.models.ensemble.EnsembleSizeWarning` when a pool of two or more
members gets fewer than 3,000 context rows, and we suggest fewer members, or
the best single model, at those sizes.

## Cost

Every member is a full model, so a pool costs about the sum of its members in
time and memory, and the members run one after another. LimiX-2 (bagged) and
TabFM each took about 50 times the wall-clock time of TabPFN-3.5 on our 50,000
test galaxies (4,000 s vs. 80 s), so the pool of the three models costs about
50 to 100 times as much as TabPFN-3.5 alone. The pool predicts `chunk_size`
rows at a time (8,192 by default), calling each member once per chunk, and the
members' distributions of one chunk set the peak memory.

The weights are equal in this version. Weights fitted to labeled data, and the
beta-transformed linear pool (BLP), are planned for a future version. While
fitted exponents lowered the CDE loss of the geometric pool further in our
tests, they also made its densities too sharp.
