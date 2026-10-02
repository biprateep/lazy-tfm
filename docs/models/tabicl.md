# TabICL

TabICLv2, from Inria's SODA team, is a tabular foundation model pretrained on
synthetic datasets. It works in three stages: it first embeds each column with
a set transformer, then compresses each row into a fixed-size vector with a
row-wise transformer, and finally passes the rows to a transformer that does
in-context learning, in which the query rows attend to the context rows. For
regression it predicts 999 quantiles of the target, at levels 0.001 to 0.999,
from which LAZY builds the distribution. It is small (29 million parameters),
quick on a CPU and released under a permissive license, which makes it the
model we recommend without a GPU.

```python
model = lazy.LazyModel("tabicl")
```

**Papers:** TabICL ([Qu et al. 2025](https://arxiv.org/abs/2502.05564)) and
TabICLv2 ([Qu et al. 2026](https://arxiv.org/abs/2602.11139)). **Code:**
[soda-inria/tabicl](https://github.com/soda-inria/tabicl). **Documentation:**
[tabicl.readthedocs.io](https://tabicl.readthedocs.io/en/latest/).
**Weights:** [jingang/TabICL on Hugging
Face](https://huggingface.co/jingang/TabICL), under the BSD-3-Clause license.

## Defaults

The table lists the creators' defaults in their `TabICLRegressor`, next to
LAZY's.

| Setting | Creators' default | LAZY default |
| ------- | ----------------- | ------------ |
| Ensemble members | 8, silently fewer for tables with fewer than 4 features | `n_estimators=8`, exactly |
| Feature preprocessing | none and a power transform, alternating | the same (`transforms="auto"`) |
| Feature shuffling | Latin-square permutations | the same (`feature_shuffle=True`) |
| Softmax temperature | none (the output is quantiles) | none (`softmax_temperature="auto"`) |
| Outlier clipping | soft clip at 4 standard deviations | the same (`outlier_threshold="auto"`) |
| Categorical columns | ordinal-encoded when the input marks them | off: every column is numeric |
| Key/value cache | off | on (`kv_cache=True`) |
| Mixed precision | on a GPU above 1,024 rows or 60 features | on for every GPU run (`mixed_precision=True`) |
| Random seed | 42 | `random_state=0` |
| Row subsampling | none | none (`bag_size=None`) |

## Built-in standardizations

TabICL z-scores the target with the mean and standard deviation of the context
and maps the predicted quantiles back to the original units. The features go
through a fixed pipeline before the network: missing values are imputed with
the column mean, constant columns are dropped, every column is z-scored on the
context and clipped at 100 standard deviations, then each member applies its
own transform (none, or a power transform) and soft-clips outliers at 4
standard deviations. LAZY keeps all of these.

## Where LAZY differs

Most of the settings that affect a prediction are the creators'. We run
exactly `n_estimators` members: when a table has so few features that TabICL
cannot form eight distinct combinations of column order and transform, it
quietly runs fewer, and LAZY adds members to make up the count. We also fix
the order of the members, which upstream takes from a Python set and which can
therefore change between sessions. We use the same seed (0) as every other
model rather than TabICL's 42, and we treat every column as numeric. LAZY also
caches the context's keys and values, and uses mixed precision on every GPU
run rather than only for large tables, which change the predictions only by
rounding. The distribution itself is the piecewise-linear cumulative
distribution through the 999 quantiles, so the 0.1% of probability beyond each
outermost quantile is placed at it, rather than in the exponential tails
TabICL uses for its own sampling. Its native grid is 1,500 equal bins over the
range of the context targets, padded by 25% on each side. Under an explicit
`transforms`, the soft clip is off unless `outlier_threshold` is given.
