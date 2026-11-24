# Interpretability

{func}`lazy.interp.explain` measures which features a model's predicted
distributions rely on. It needs only `fit` and `predict_distribution`, so it
works the same on every backend, and on a pool of them.

```python
from lazy import interp

explanation = interp.explain(model, X_test, y_test, X_train, y_train)
explanation.importance()   # one row per feature, one column per statistic
explanation.table("nll")   # one row per explained row
```

The rows to explain should not be rows of the context. A model recognizes its
context rows and predicts their targets whatever the features, so every
feature would look less important than it is.

## What is attributed

A point prediction hides the features that change only the spread of a
distribution. `explain` therefore attributes three statistics of each row's
predicted distribution $\hat p_i$, given the true target $y_i$:

| Statistic | Value | Units |
| --------- | ----- | ----- |
| `"crps"` | $-\mathrm{CRPS}(\hat p_i, y_i)$ | the target's |
| `"nll"` | $\log \hat p_i(y_i)$ | nats |
| `"mean"` | $\mathrm{E}_{\hat p_i}[y]$ | the target's |

The two scores are negated, so that a positive attribution means that the
feature improves the prediction. Both are computed on one grid for every
feature set, which `y_grid` sets ({doc}`distributions` defines the scores).

`explanation.importance()` averages the attributions over the explained rows.
For the mean it averages their absolute values, since contributions to a
prediction cancel over rows.

## Removing a feature

Every method compares predictions with and without a feature, so it must
decide what the model predicts without it. The models are fitted by storing
their context, which makes the direct answer cheap: remove the feature's
column from the context and from the queries, and fit a clone of the model on
the columns that remain. One fit serves all the explained rows. We write
$v_i(S)$ for a statistic of row $i$ when the model has only the features in
the set $S$.

This removal means the same on every backend. Setting a feature to `NaN` does
not: each model handles a missing value in its own way (see
{doc}`interface`).

For the empty set there is no column to fit on, so the model is fitted on one
column of standard normal noise. Its prediction is then the model's estimate
of the distribution of the target with no feature.

## The methods

### Shapley values

`method="shapley"` gives feature $j$ its average gain over every order in
which the $p$ features can be added:

$$
\phi_{ij} = \sum_{S \subseteq F \setminus \{j\}}
\frac{|S|!\,(p - |S| - 1)!}{p!}\,\bigl(v_i(S \cup \{j\}) - v_i(S)\bigr),
$$

where $F$ is the set of all features. The values of a row sum to what the
features add together, $\sum_j \phi_{ij} = v_i(F) - v_i(\varnothing)$, and
`explanation.full_values` and `explanation.base_values` hold the two terms.
Averaged over rows, the Shapley values of a loss are the SAGE values of
[Covert et al. (2020)](https://arxiv.org/abs/2004.00668).

With up to 10 features, `explain` evaluates all $2^p$ feature sets, which is
exact and takes $2^p$ fits. Above that, it samples orders of the features
(32 by default, set by `n_permutations`) and averages each feature's gain over
them. Each order is paired with its reverse, which makes the values exact
when no more than two features interact.

### Leave one covariate out

`method="loco"` gives feature $j$ what the statistic loses when it alone is
removed, $v_i(F) - v_i(F \setminus \{j\})$, in $p + 1$ fits. A feature that
the others can stand in for gets little credit, even if the model relies on
it, so these values rank features as the Shapley values do only when the
features are independent.

### Permutation

`method="permutation"` shuffles the values of one feature among the explained
rows and gives the feature what the statistic loses, averaged over
`n_repeats` shuffles. It keeps the fitted model and its context, needs no
refit and no context from the caller, and is what `explain` runs when no
context is given. A shuffled row can pair values that never occur together,
so the model is asked about rows unlike any in its context when features are
correlated.

## Groups of columns

`groups` treats several columns as one feature, which is removed, added or
shuffled as a whole:

```python
interp.explain(
    model, X_test, y_test, X_train, y_train,
    groups={"colors": ["u-g", "g-r", "r-i"]},
)
```

This also lowers the cost, which depends on the number of features and not of
columns.
