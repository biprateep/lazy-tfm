# LimiX-2

LimiX-2, from Stable AI, is a large structured-data model pretrained on
synthetic data from structural causal models with a context-conditional
masked-modeling objective, which it uses to approximate the joint distribution
of the features and the target given the context. A single
pretrained model performs classification, regression and missing-value
imputation. For regression it predicts the probability of each of 5,000
buckets of the standardized target, which LAZY keeps as the model's native
distribution. With 406 million parameters it is the second-largest model in
LAZY, and it needs a GPU (and, for now, Linux).

```python
model = lazy.LazyModel("limix")
```

**Papers:** LimiX ([LimiX team 2025](https://arxiv.org/abs/2509.03505)) and
LimiX-2 ([Zhang et al. 2026](https://arxiv.org/abs/2609.17488)). **Code:**
[limix-ldm-ai/LimiX](https://github.com/limix-ldm-ai/LimiX), which is not on
PyPI (see {doc}`../installation`). **Project page:**
[limix.ai](https://www.limix.ai/). **Weights:** [stable-ai/LimiX-2 on Hugging
Face](https://huggingface.co/stable-ai/LimiX-2), under the StableAI LimiX
Non-Commercial License, which also requires the attribution "Built with
StableAI LimiX".

## Defaults

The table lists the creators' defaults in their `LimiXPredictor` and the
regression configuration shipped with LimiX, next to LAZY's.

| Setting | Creators' default | LAZY default |
| ------- | ----------------- | ------------ |
| Output for regression | the mean of the predicted distribution | the full 5,000-bucket distribution |
| Ensemble members | 8, fixed by the configuration | `n_estimators=8`; any number in blocks of the same recipe |
| Feature preprocessing | 4 members with a quantile transform plus the original features and an SVD, 4 with a power transform | the same (`transforms="auto"`) |
| Feature shuffling | on | on (`feature_shuffle=True`) |
| Softmax temperature | 0.9, then the members' probabilities averaged | the same (`softmax_temperature="auto"`) |
| Outlier clipping | none | none (`outlier_threshold="auto"`) |
| Categorical columns | detected (fewer than 4 distinct values in at least 100 rows) and encoded | off: every column is numeric |
| Preprocessing fitted on | the context and the queries together | the context only |
| Key/value cache | none | on, ported from LimiX (`kv_cache=True`) |
| Precision | float16 autocast on a GPU, float32 on a CPU | the same (`mixed_precision=True`) |
| Random seed | 0 | `random_state=0` |
| Row subsampling | none | none (`bag_size=None`) |

## Built-in standardizations

LimiX-2 z-scores the target with the mean and the sample standard deviation of
the context (a constant target is left unscaled), and its buckets are fixed in
that standardized space. Before the network, each member drops constant and
empty columns and applies its transform: a quantile transform with one
quantile per five rows, kept beside the original features and followed by a
singular value decomposition, or a power transform followed by
standardization. Inside the network, missing and infinite values are replaced
by the context mean and flagged with indicators, every feature is z-scored on
the context and clipped at 100 standard deviations, and the features are
rescaled by the fraction that are valid. LAZY keeps all of these.

## Where LAZY differs

LAZY keeps the distribution that LimiX-2 predicts, while LimiX's own `predict`
reduces it to its mean. We also fit every preprocessing step on the context
alone (LimiX fits its column filter and transforms on the context and the
queries together), so that the prediction for a query does not depend on the
other queries, and we treat every column as numeric. The member recipe,
the transforms, the column permutations and their seeds are LimiX's. However,
the random feature embeddings inside the network are drawn from a separate
generator per member rather than from PyTorch's global one, so LAZY's answers
agree with LimiX's only within the variation between random seeds, not bit
for bit. LAZY also adds a key/value cache, which upstream LimiX lacks, and
lets `n_estimators` take any value by repeating the eight-member recipe. LimiX
states no maximum context size, but in our tests on DC1 photometry its
accuracy degrades above about 20,000 context rows, so LAZY warns above that
size; bagging (e.g., `bag_size=20_000`, so that each member sees at most
20,000 rows) keeps every member within that size.
