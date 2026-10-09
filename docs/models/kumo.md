# Kumo Tabular

Kumo Tabular, from NVIDIA, is a tabular foundation model which, according to
NVIDIA's announcement, is pretrained only on synthetic tables drawn from
structural causal models. It works in two stages.
It first turns each row into a fixed-size vector with an encoder that
alternates attention across the rows of each column (an induced set attention,
in which the query rows see only the context rows) with attention across the
columns of each row, and then passes the rows to a transformer that does
in-context learning, in which the query rows attend to the context rows. Each
cell enters the encoder through learned Fourier features of its value, with a
learned embedding for a missing value, and the attention logits are scaled by
the logarithm of the table size, which NVIDIA introduced so that the attention
stays as concentrated on large tables as on small ones. For regression it predicts 999 quantiles of the target, at levels 0.001
to 0.999, from which LAZY builds the distribution. It comes in three sizes,
from 28 million to 216 million parameters, which LAZY loads as versions
`v1-small`, `v1-medium` and `v1-large` (the default), and its weights are
released under a license that permits commercial use.

```python
model = lazy.LazyModel("kumo")                       # Kumo Tabular Large
model = lazy.LazyModel("kumo", version="v1-small")   # on a CPU
```

**Announcement:** [NVIDIA Kumo Tabular Sets a New Accuracy-Efficiency Frontier
for Tabular Prediction](https://huggingface.co/blog/nvidia/kumo-tabular)
(there is no paper yet). **Code:**
[NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models).
**Documentation:**
[nvidia.github.io/structured-data-models](https://nvidia.github.io/structured-data-models).
**Weights:** [nvidia/Kumo-Tabular on Hugging
Face](https://huggingface.co/nvidia/Kumo-Tabular), under the OpenMDW License
1.1.

## Installation

Kumo Tabular is part of NVIDIA's `structured-data-models` package (imported as
`sdm`), which is not on PyPI yet. The extra installs its dependencies, and
`lazy setup kumo` installs the package itself from GitHub, at the commit LAZY
was validated on (the backend is named because its extra lists only common
packages, which do not show that it was asked for):

```console
pip install 'lazy-tfm[kumo]'
lazy setup kumo
```

Without a name, `lazy setup` checks that an installed `structured-data-models`
is that commit.

## Defaults

The table lists the creators' defaults in their `KumoTabular` model and its
default recipe, next to LAZY's.

| Setting | Creators' default | LAZY default |
| ------- | ----------------- | ------------ |
| Model size | large | `version="v1-large"` |
| Ensemble members | 8 in the model card; 8 (small, medium) and 16 (large) in NVIDIA's benchmarks | `n_estimators=8` |
| Feature preprocessing | none, a power transform, a robust scaling with a soft clip at 3, and a Gaussian rank transform, in turn | the same (`transforms="auto"`) |
| Feature shuffling | Latin-square permutations | the same (`feature_shuffle=True`) |
| Softmax temperature | none (the output is quantiles) | none (`softmax_temperature="auto"`) |
| Outlier clipping | two-pass soft clip at 4 standard deviations | the same (`outlier_threshold="auto"`) |
| Combining the members | mean of the quantiles, dropping the highest and lowest 20% of members at each level | the same, over every member |
| Categorical columns | encoded by the recipe when the input marks them | off: every column is numeric |
| Key/value cache | off in a single call, on with `fit` and `predict` | on (`kv_cache=True`) |
| Mixed precision | float16 autocast on a GPU | the same (`mixed_precision=True`) |
| Random seed | none (42 in NVIDIA's benchmarks) | `random_state=0` |
| Row subsampling | none | none (`bag_size=None`) |

## Built-in standardizations

Kumo Tabular z-scores the target on the context, flips its sign at random for
each member, and maps the predicted quantiles back to the original units. The
features go through a fixed pipeline before the network: constant columns are
dropped, every column is z-scored on the context in double precision and
clipped at 100 standard deviations, then each member applies its own
transform and soft-clips outliers at 4 standard deviations, flips the sign of
each column with probability one half, and shuffles the columns, after which
at most 500 columns are kept. There is no imputation step, since the network
embeds a missing value itself. LAZY keeps all of these.

## Where LAZY differs

LAZY runs the creators' recipe, written out in `lazy.models.kumo` rather than
read from the installed package, so that an upgrade of `sdm` cannot change
what `transforms="auto"` runs. We use 8 members for every size, the same seed
(0) as every other model, and we treat every column as numeric. We standardize
the target in double precision before Kumo standardizes it again, which keeps
a narrow spread about a large offset that single precision would round away.
The members' quantiles are combined by the creators' trimmed mean, but over
every member of the ensemble at once, so that a bagged ensemble, whose members
run one at a time, is combined the same way as one that runs in a single call.
On a CPU, LAZY's answer with the small size matches the creators' own call to
about $3\times 10^{-7}$ in the units of the target on our test problem, and
on a GPU the cached and uncached answers of the large size agree to about
$1.3\times 10^{-3}$, which is of the size we expect from float16 rounding. The checkpoint is read
from a pinned revision of the Hugging Face repository rather than through
upstream's download. The distribution itself is the piecewise-linear
cumulative distribution through the 999 quantiles, so the 0.1% of probability
beyond each outermost quantile is placed at it. Its native grid is 1,500 equal
bins over the range of the context targets, padded by 25% on each side. Under
an explicit `transforms`, the soft clip is off unless `outlier_threshold` is
given.
