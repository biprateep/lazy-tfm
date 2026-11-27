# Interpretability

{func}`lazy.interp.explain` measures which features a model's predicted
distributions rely on. Its removal methods need only `fit` and
`predict_distribution`, so they work the same on every backend, and on a pool
of them; its gradient methods need a backend's differentiable forward pass.
{func}`lazy.interp.explain_context` measures which context rows they rely on
(see [Context rows](#context-rows)).

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

`explanation.plot()` draws one statistic's importance as horizontal bars, with
the most important feature at the top, and returns the matplotlib axes:

```python
ax = explanation.plot("nll")   # a new figure, or plot("nll", ax=ax)
```

## Removing a feature

Every removal method compares predictions with and without a feature, so it
must decide what the model predicts without it. The models are fitted by storing
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

### Gradients

The two gradient methods keep the fitted model, and take the derivatives of a
statistic $s$ of row $x$ in its features through the model's forward pass.
Neither refits; both need the context the model was fitted on, as `X_context`
and `y_context`. They compare the row with a baseline row $b$, which stands in
for no feature at all: the mean of `X_context` by default, or the row that
`baseline` gives.

`method="gradient"` multiplies the gradient at the row by the row's offset
from the baseline (gradient times input):

$$
\phi_{ij} = (x_{ij} - b_j)\,\frac{\partial s}{\partial x_j}(x_i).
$$

It costs one forward pass and one backward pass per statistic. It is exact
for a statistic that is linear in the features, and only a local view
otherwise.

`method="integrated_gradients"` averages the gradient along the straight path
from the baseline to the row, in $m$ steps (`n_steps`, 32 by default):

$$
\phi_{ij} = (x_{ij} - b_j)\,\frac{1}{m}\sum_{k=1}^{m}
\frac{\partial s}{\partial x_j}\Bigl(b + \tfrac{k}{m}\,(x_i - b)\Bigr).
$$

The sum approximates the integral of the gradient along the path, so the
values of a row sum to what the path adds,
$\sum_j \phi_{ij} \approx s(x_i) - s(b)$, with an error that falls as $1/m$.
`explanation.full_values` and `explanation.base_values` hold the two terms.
The cost is $m$ times that of `"gradient"`.

The forward pass is differentiable only when nothing between the features and
the model's network is a step that has no gradient. The columns must be
numeric. A transform or a clip that lazy applies itself for the backend has a
gradient, such as `"quantile"` on TabPFN or TabICL; its fitted parameters are
constants, so the gradient does not see how they depend on the context. A
transform or a clip that the model's own package applies has none:
`transforms=("none",)` and `outlier_threshold=None` always avoid them.
`explain` refuses a fit without a differentiable pass, and any model with no
such pass, such as a pipeline.

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
columns. The gradient methods take the gradient of every column at once, so a
group's value is the sum of its columns' values, and the cost does not change.

## Context rows

A model's prediction depends on its context as much as on the features of the
row. {func}`lazy.interp.explain_context` attributes the same three statistics
to the rows of the context, with the same signs: a positive CRPS or NLL value
means that the context row improves the prediction.

```python
explanation = interp.explain_context(model, X_test, y_test, X_train, y_train)
explanation.top(5, "nll")  # each row's 5 most influential context rows
```

`explanation.values` has one value per statistic, explained row and context
row, and `explanation.context_rows` gives the positions of those context rows
in `X_context`. `explanation.top(k)` lists, for each explained row, the $k$
context rows with the largest absolute values.

Explain rows that are not in the context here too. A model recognizes its
context rows, so the prediction of a row of the context rests on its own copy
there.

### Gradient times input

`method="gradient"`, the default, gives context row $r$ the gradient of the
statistic $s_i$ of row $i$ in the row's features, times their offsets from the
baseline $b$, summed over the features:

$$
\psi_{ir} = \sum_j \frac{\partial s_i}{\partial x^{\mathrm{c}}_{rj}}\,
\bigl(x^{\mathrm{c}}_{rj} - b_j\bigr),
$$

where $x^{\mathrm{c}}_{rj}$ is feature $j$ of context row $r$, and $b$ is the
mean of `X_context` by default or the row that `baseline` gives. It measures
what moving the context row from the baseline to where it is does to the
prediction, to first order. It does not see the row's target, so a row whose
target is wrong but whose features are typical can get a small value.

Every statistic of a row depends on the whole context, so each takes a
backward pass of its own: the cost is one forward pass per 64 rows and three
backward passes per row, each about as costly as a forward pass. The model must
be fitted on `X_context` and `y_context` with a differentiable forward pass,
as for the gradient methods of `explain`.

### Leave one out

`method="loo"` refits the model without each context row:

$$
\psi_{ir} = s_i(C) - s_i(C \setminus \{r\}),
$$

where $C$ is the whole context. It works on any model, sees the targets, and
costs one fit per context row, plus one. `rows` limits it to some context rows:
a number of them, drawn with `random_state`, or their positions. A model that
bags its context rows (`bag_size`) draws other bags without $r$, so its values
hold that redraw as well as $r$'s own effect; fit it without bagging to
measure $r$ alone.
