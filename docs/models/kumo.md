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
stays as concentrated on large tables as on small ones. For regression it
predicts 999 quantiles of the target, at levels 0.001 to 0.999, from which LAZY
builds the distribution. It comes in three sizes,
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
`sdm`), which the extra installs from PyPI:

```console
pip install 'lazy-tfm[kumo]'
```

LAZY was validated on the package's commit
[2b76468](https://github.com/NVIDIA/structured-data-models/commit/2b76468cb5b4ac1f4bc59f7f07f0506a4a2e8298).
Its Kumo Tabular code is identical to that of release 0.2.1, the extra's
lowest version.

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
| Combining the members | mean of the quantiles, dropping the highest and lowest 20% of members at each level | the same, over every member (a plain mean under an explicit `transforms`) |
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

We write NVIDIA's recipe out in `lazy.models.kumo` instead of reading it from
the installed package, so an upgrade of `sdm` cannot change what
`transforms="auto"` runs. We use 8 members for every size, the same seed (0)
as every other model, and we treat every column as numeric. We also
standardize the target in double precision before Kumo standardizes it again
(a bag whose targets are all equal is left unscaled), since a narrow spread
about a large offset can be rounded away in single precision. The members' quantiles are
combined by NVIDIA's trimmed mean, which we take over every member of the
ensemble at once, so that a bagged ensemble, whose members run one at a time,
is combined in the same way as one that runs in a single call. Under an
explicit `transforms`, the recipe's extras are dropped as on every other model:
the soft clip is off unless `outlier_threshold` is given, and the members are
combined by a plain mean, which `trim_fraction_` records. The random sign
flips of the columns and the target stay, because they are part of how the
network is run and not a choice of the recipe.

On a CPU, our prediction with the small size matches NVIDIA's own
implementation to about $2\times 10^{-7}$ in the units of the target, with 5
members, on the 200 context and 60 query rows of the synthetic problem in the
first Kumo test (commit bd0bb9e; see [Measurement
setup](../guide/scaling.md#measurement-setup)). The cached and uncached predictions agree to float rounding on
a CPU. A query predicted on its own takes a
different matrix-multiply path and differs by up to about $10^{-6}$ of the
quantiles' spread, and a bagged member with missing values whose columns LAZY
permutes differs by up to about $10^{-5}$. On a GPU, the cached and
uncached quantiles of the large size differ by at most $5\times 10^{-4}$ in the
units of the target (0.6% of a row's quantile spread) on the 300 context and 50 test rows of
the golden problem, which we attribute to float16 rounding (see [Measurement
setup](../guide/scaling.md#measurement-setup)).
The checkpoint is read from a pinned revision of the Hugging Face repository
instead of through upstream's download. The distribution itself is the
piecewise-linear cumulative distribution through the 999 quantiles, with the
0.1% of probability beyond each outermost quantile spread evenly over one
quantile spacing past it, so that a fine grid shows no spike there. Its native
grid is 1,500 equal bins over the range of the context targets, padded by 25%
on each side. Kumo's preprocessing runs in double precision on the device,
which Apple's MPS devices do not support, so LAZY refuses `device="mps"`
before it loads the weights. LAZY draws the members' random sign flips and
column shuffles on a CPU, so a CPU and a GPU give the same ensemble for one
seed (see {doc}`../guide/reproducibility`). NVIDIA pretrained Kumo Tabular on tables of up to
60,000 rows, and LAZY warns above that size, since its accuracy on larger
contexts has not been measured.
