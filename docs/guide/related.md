# LAZY and related tools

LAZY does one task: it predicts the distribution of a continuous target with
pretrained tabular foundation models used in context, with no change to their
weights. Other tools cover other tasks, or the same models in other ways. This
page states what LAZY adds to each model's own package, and when another tool
suits a task better. Each statement about another tool links to its source,
as read in November 2026: the tool's own documentation, or, for what a model's
package returns, the LAZY page that describes that model.

| Tool | What it is | Use it instead of LAZY for | Source |
| ---- | ---------- | -------------------------- | ------ |
| `tabpfn` | Prior Labs' package for TabPFN | classification, fine-tuning | [API](https://docs.priorlabs.ai/api-reference/python/tabpfn/regressor/fitting-and-prediction), [fine-tuning](https://docs.priorlabs.ai/capabilities/fine-tuning) |
| `tabicl` | Inria's package for TabICL | classification, fine-tuning | [API](https://tabicl.readthedocs.io/en/latest/api.html) |
| `tabfm` | Google Research's package for TabFM | classification, TabFM's own regression checkpoint | [repository](https://github.com/google-research/tabfm) |
| LimiX | Stable AI's code for LimiX and LimiX-2 | classification, missing-value imputation | [repository](https://github.com/limix-ldm-ai/LimiX) |
| `structured-data-models` | NVIDIA's library of foundation models for structured data, imported as `sdm`, which holds Kumo Tabular | relational data, time series, anomaly detection, fine-tuning | [repository](https://github.com/NVIDIA/structured-data-models), [fine-tuning](https://nvidia.github.io/structured-data-models/finetuning.html) |
| TabTune | Lexsi Labs' library for inference and fine-tuning of 24 tabular foundation models | fine-tuning, LoRA adapters, classification, time series, models that LAZY does not include | [repository](https://github.com/Lexsi-Labs/TabTune) |
| AutoGluon-Tabular | An automated machine-learning library that fits and combines many models | a search over many kinds of models, stacked ensembles, classification | [repository](https://github.com/autogluon/autogluon), [foundation models](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html) |

## The models' own packages

LAZY runs each model through its own package and weights, so the network is
the same. What each package returns for a regression differs:

- TabPFN's `TabPFNRegressor.predict` returns the mean by default.
  `output_type` selects the median, the mode, quantiles at chosen levels, or
  the full output, which holds the logits and the bar distribution
  ([API](https://docs.priorlabs.ai/api-reference/python/tabpfn/regressor/fitting-and-prediction),
  [predictive
  distribution](https://docs.priorlabs.ai/capabilities/predictive-distribution)).
  TabPFN therefore already gives the full distribution.
- TabICL's `TabICLRegressor.predict` returns the mean by default, and
  `output_type="quantiles"` returns quantiles at the levels in `alphas`
  ([API](https://tabicl.readthedocs.io/en/latest/api.html)).
- TabFM's regressor predicts one value per row, LimiX's `predict` returns
  the mean of its distribution, and Kumo Tabular predicts quantiles (see
  {doc}`../models/tabfm`, {doc}`../models/limix` and {doc}`../models/kumo`).

LAZY adds the following to every model:

- One estimator with the same parameters and defaults for every model (see
  {doc}`interface`).
- Densities on any grid, integrated exactly over its bins, from one fitted
  model (see {doc}`distributions`).
- A distribution object of the same kind for every model, from which
  quantiles, intervals, CDF values and PIT values are exact.
- A pinned checkpoint and recipe for each version, independent of the
  installed package, recorded with the seed in `provenance_` (see
  {doc}`reproducibility`).
- Pools of different models' densities (see {doc}`ensembles`).
- Metrics for densities: the CDE loss, the CRPS, the negative log-likelihood
  and PIT statistics (see [Metrics](distributions.md#metrics)).

The packages do things that LAZY does not. TabPFN
(`FinetunedTabPFNRegressor`), TabICL (`FinetunedTabICLRegressor`) and
`structured-data-models` fine-tune the weights on a dataset, and LAZY does not
(see [Can I fine-tune?](faq.md#can-i-fine-tune)). TabPFN, TabICL and TabFM
also classify, and several packages encode categorical columns, which LAZY treats as numbers (see
[Categorical columns](limits.md#categorical-columns)). The FAQ lists the
settings in which LAZY differs from TabPFN's own (see [How does this differ
from calling TabPFN
directly?](faq.md#how-does-this-differ-from-calling-tabpfn-directly)).

## TabTune

TabTune gives one scikit-learn-style interface to 24 tabular foundation
models, including TabPFN, TabICLv2, TabFM, LimiX-2 and Kumo Tabular, and to
eleven time-series models
([README](https://github.com/Lexsi-Labs/TabTune)). It does classification and
regression, zero-shot or after fine-tuning: episodic meta-learning, supervised
fine-tuning or LoRA adapters. Its regression metrics are the RMSE, the MAE and
R², and it gives prediction intervals by conformal prediction. LAZY does no
fine-tuning, but it returns each row's full density and scores it with the
density metrics above.

## AutoGluon-Tabular

AutoGluon fits many kinds of models on a table and finds the combination that
works best, for tabular data, time series and multimodal data
([README](https://github.com/autogluon/autogluon)). Version 1.6.1 includes
TabPFN v2 to v3, TabICL, TabICLv2, Mitra, TabDPT and Nori, and fitting several
of them adds a weighted ensemble of them; Mitra can also be fine-tuned
([foundation
models](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html)).
Its `TabularPredictor` has a `"quantile"` problem type, scored by the pinball
loss
([API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html)).
LAZY does no model search, but it returns each row's full density rather than
chosen quantiles.
