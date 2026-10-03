# What are Tabular Foundation Models?

A tabular dataset is a table in which each row is an object (a galaxy, a
patient, a transaction) and each column is a measured property of it. The
usual way to learn from such a table is supervised learning: we choose a model
(e.g., a gradient-boosted tree ensemble, a neural network, a random forest,
etc.), fit its parameters to the labeled rows by minimizing a loss, tune its
hyperparameters on held-out data, and repeat the whole process for every new
dataset. A tabular foundation model (TFM) replaces this per-dataset training
with a single, very expensive pretraining step that happens once, before the
model ever sees our data. During pretraining, the network is trained to
predict held-out labels from the labeled rows of many tables, so when we give
it our labeled rows it can predict the target for new rows directly, without
updating any of its weights. This page gives a brief overview of how this
works and what it means for the user, and links to the papers that describe
each model in detail.

## Learning in context

A TFM receives the labeled rows and the unlabeled rows as one input. The
labeled rows, which we call the *context* (they play the role of the training
set), and the rows we want predictions for (the *queries*) are passed to the
network together, and it returns a prediction for every query in a forward
pass (see Figure 1). This is called in-context learning, since everything the
model knows about our dataset comes from the context it is given at prediction
time rather than from gradient descent on it. As the TabPFN v2 paper puts it,
the model "performs training and prediction on this dataset in a single neural
network forward pass" ([Hollmann et al.
2025](https://doi.org/10.1038/s41586-024-08328-6)). Therefore, `fit` in LAZY
only stores the context (and, with the key/value cache on, the network's
encoding of it), and the predictions are computed in `predict`.

```{figure} ../figures/figs/tfm_schematic.png
:alt: Context rows with known targets and query rows with unknown targets go into a pretrained transformer, which returns a distribution for each query.
:width: 100%

**Figure 1.** In-context learning with a tabular foundation model. The context
rows (blue) carry both their features and their target, while the query rows
(orange) carry only their features. The pretrained transformer reads both and
returns the conditional distribution of the target for each query, which may
be narrow, broad or have more than one peak. No weights are updated at any
point.
```

Inside the network, TabPFN v2 uses attention in two directions: each cell
attends to the other features in its row, and to the same feature in the other
rows ([Hollmann et al. 2025](https://doi.org/10.1038/s41586-024-08328-6)). The
other models in LAZY use variants of this design, such as TabICL's, which
first summarizes each row in a single vector and then attends across rows ([Qu
et al. 2025](https://arxiv.org/abs/2502.05564)). In TabPFN, the query rows
attend only to the context rows and not to each other, so the prediction for a
query does not depend on which other queries are predicted with it. The cost
of a prediction grows with the number of context rows (quadratically for the
attention between them), which is why the size of the context, rather than
training time, is the main computational constraint of a TFM. For more details
on each architecture, we refer the reader to the papers linked on each model's
page under {doc}`../models/index`.

## Pretraining on synthetic data

The idea behind TFMs was laid out by [Müller et al.
(2022)](https://arxiv.org/abs/2112.10510), who called these networks
prior-data fitted networks (PFNs). They proposed to draw a very large number
of datasets from a *prior*, a random process that generates plausible
datasets, hide some of the labels in each, and train a transformer to predict
the hidden labels from the rest. They showed that a network trained this way
learns to approximate Bayesian inference: for a new dataset, its output
approximates the posterior predictive distribution (PPD) of the target given
the context, under the prior it was trained on. Since the prior generates the
datasets, no real data is needed for pretraining. For example, TabPFN v2 was
pretrained on about 130 million synthetic datasets generated from structural
causal models, random directed graphs of causes and effects between the
features and the target ([Hollmann et al.
2025](https://doi.org/10.1038/s41586-024-08328-6)).

```{figure} ../figures/figs/prior_draws.png
:alt: Six small synthetic one-feature regression datasets with different shapes and noise levels.
:width: 100%

**Figure 2.** Six datasets drawn from a toy prior over one-feature regression
problems, in which each dataset comes from a small random network with a
random activation function and random noise. The priors the models in LAZY
were pretrained on are far richer (many features, causal structure between
them, mixed types, missing values, etc.); this toy prior only illustrates the
idea. A TFM is pretrained to predict held-out points in millions of datasets
like these.
```

This view also tells us what to expect from a TFM in practice. Since the
output approximates a posterior predictive distribution, it should be broad
when the context holds little information about a query and narrow when it
holds a lot. Figure 3 shows this for the noisy curve from the landing page:
with 20 context rows TabPFN-3.5-fast cannot follow the oscillations and its
68% interval is broad, while with 100 and 400 context rows it follows the true
curve closely and its interval narrows to roughly the spread of the noise. The
model and its weights are the same in all three panels.

```{figure} ../figures/figs/context_size.png
:alt: Three panels showing TabPFN-3.5-fast's mean and 68% interval for a noisy curve given 20, 100 and 400 context points.
:width: 100%

**Figure 3.** TabPFN-3.5-fast's predictions for the noisy chirp of
{func}`load_dataset("chirp") <lazy.datasets.load_dataset>` given 20, 100 and 400 context rows drawn at
random. The blue points show the context, the gray dashed line the true curve,
the orange line the mean of each predicted distribution and the orange band
its central 68% interval. We observe that the predicted distributions narrow
and follow the true curve as the context grows, with no weights changed.
```

## Predicting distributions

Most TFMs can predict a full probability distribution for the target rather
than a single value, although they represent it in different ways. TabPFN and
LimiX-2 predict the probability of each of several thousand bins of the target
(also called a bar or Riemann distribution, from Müller et al. 2022); TabICLv2
predicts 999 quantiles; and Google's TabFM predicts a single value for
regression, so LAZY builds a distribution from its classifier instead. LAZY
converts all of these to a common form, so that every model answers the same
calls (`predict_proba`, `predict_quantiles`, `predict_interval`, etc.) with
the same kind of output. {doc}`distributions` describes how.

## Strengths and limits

TFMs have been shown to match or outperform tuned classical methods on many
small and medium-sized tabular benchmarks ([Hollmann et al.
2025](https://doi.org/10.1038/s41586-024-08328-6)), without any training or
hyperparameter search on the user's side, and most of them can provide
distributions rather than only point estimates. However, the cost of every
prediction grows with the size of the context, and each model supports
contexts only up to a maximum number of rows and features (from 10,000 rows
for TabPFN v2 to a million for TabPFN-3), beyond which its performance is not
guaranteed. Most of the models need a GPU to be practical, and several have
weights released under non-commercial licenses. The models also inherit the
assumptions of the prior they were pretrained on, so a dataset very unlike
anything the prior generates may be predicted poorly. {doc}`../models/index`
lists each model's limits, hardware needs and license.

## Further reading

The papers introducing each model are linked from its page under
{doc}`../models/index`. For the idea of training on a prior and its connection to
Bayesian inference, see [Müller et al.
(2022)](https://arxiv.org/abs/2112.10510); for a broader argument for tabular
foundation models as a research direction, see [van Breugel & van der Schaar
(2024)](https://arxiv.org/abs/2405.01147).
