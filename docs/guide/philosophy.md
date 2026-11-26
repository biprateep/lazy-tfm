# Design principles

LAZY asks the user to do as little as possible. A user gives the labeled rows
and the rows to predict, and gets a full distribution for each row, with no
training, no tuning and no code beyond scikit-learn's. Three principles
follow from this, and each shapes what a user can expect.

## Pretrained models, used as they are

Every model in LAZY is pretrained, and LAZY keeps its weights frozen. `fit`
stores the labeled rows as the context, and a prediction is a forward pass
over the context and the queries (see {doc}`tfm`). For a user, this means:

- `fit` trains no weights, so it needs no epochs, learning rate or early
  stopping (see [the FAQ](faq.md#what-does-fit-do)).
- A new dataset needs only a new context. The same weights answer for every
  dataset.
- The cost moves from `fit` to the prediction, and it grows with the size
  of the context (see {doc}`scaling`).

## scikit-learn's API and tools first

Every model is a scikit-learn regressor, so `clone`, pipelines,
cross-validation and the hyperparameter searches work with it (see
{doc}`scikit_learn`). Every backend takes the same parameters with the same
defaults (see {doc}`interface`), so a user who knows one model knows them
all.

Some of scikit-learn's tools combine point predictions and cannot carry a
density. Where that happens, LAZY provides its own version that works in the
same way. For example, scikit-learn's `VotingRegressor` averages the
predictions of its members, while
{class}`~lazy.models.ensemble.LazyEnsembleModel` pools their densities, and
both name their members' parameters `<name>__<parameter>` (see
{doc}`ensembles`).

## Methods that need no training

Classical methods that train nothing can improve a prediction from frozen
weights. LAZY offers four of them:

- Each model runs as an ensemble of members, 8 by default (see
  [Ensembling](interface.md#ensembling-and-feature-shuffling)).
- The members differ in their feature transform and in the order of the
  feature columns (see [Transforms](interface.md#transforms)).
- With `bag_size` set, each member sees its own random subset of the
  context, which is called bagging (see [Bagging](interface.md#bagging)).
- {class}`~lazy.models.ensemble.LazyEnsembleModel` pools several models
  into one distribution (see {doc}`ensembles`).

Each of these methods is a parameter with a default that works without
tuning. Since each is a parameter, scikit-learn's searches, such as
{class}`~sklearn.model_selection.GridSearchCV`, can tune it like any other.

## What follows

LAZY trains no weights by default. A method that trains weights, such as
fine-tuning or an adapter, can only be an option that the user turns on, and
never the default; the one so far is the experimental
[context adapter](adapter.md), which trains a small network before a frozen
model. A user who wants to change a prediction changes the context or the
parameters instead (see [the FAQ](faq.md#can-i-fine-tune)).
