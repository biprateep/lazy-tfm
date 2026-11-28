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
`explanation.full_values` and `explanation.base_values` hold the two terms,
and `explanation.sum_error` their difference from the sum, which shows whether
`n_steps` is enough. The cost is $m$ times that of `"gradient"`.

### Models that rank their inputs

TabPFN v3.5 (`version="v3.5"` and `"v3.5-fast"`) feeds its network, beside
each standardized value, the value's midrank among the context's values of its
column:

$$
r(v) = \operatorname{clip}\Bigl(\frac{1}{n}\sum_{k} m_k\,H(v - e_k),\;
r_{\min},\; r_{\max}\Bigr), \qquad
H(z) = \begin{cases} 0 & z < 0 \\ 1/2 & z = 0 \\ 1 & z > 0, \end{cases}
$$

where $e_k$ are the distinct context values of the column, $m_k$ the number of
context rows at each, and $n$ the number of context rows. From tabpfn 9.2, a
value outside the context's range is held at the midrank of the nearest
extreme, $r_{\min} = m_1/(2n)$ and $r_{\max} = 1 - m_K/(2n)$ for the lowest
and the highest distinct values; before, $r_{\min} = 0$ and $r_{\max} = 1$.
The rank is a step function of the query: its gradient is zero between the
context values, and it jumps at each of them. A gradient misses those jumps,
and so does every step of integrated gradients. On TabPFN v3.5, integrated
gradients missed 9% to 36% of the gain on continuous data, and more on data
with many ties, at any `n_steps`.

For such a model, the gradient methods attribute a smoothed model instead.
Its rank of each query value replaces every step by a normal CDF $\Phi$:

$$
\tilde r(v) = \operatorname{clip}\Bigl(\frac{1}{n}\sum_{k} m_k\,
\Phi\Bigl(\frac{v - e_k}{h}\Bigr),\; r_{\min},\; r_{\max}\Bigr),
\qquad h = \min\Bigl(\frac{1}{2}, \frac{d}{2m}\Bigr)\,g,
$$

where $g$ is the median gap between consecutive distinct context values of the
column, $d$ their number, and $m$ is `n_steps` (1 for `"gradient"`). The ranks
of the context rows stay exact. A Riemann sum with spacing $s$ integrates a
normal bump of width $h$ to within $2e^{-2\pi^2 h^2/s^2}$ of its mass, 1.4% at
$h = s/2$. One step of the path crosses about $d/m$ gaps or fewer, so
$h = dg/(2m)$ keeps $h \geq s/2$; at $h = g/2$ neighbouring steps overlap, and
$\tilde r$ is linear between context values to 1.4%. As $m$ grows, $h$
shrinks, and the smoothed model tends to the model itself.

`explanation.full_values`, `base_values` and `sum_error` are then the smoothed
model's, and `explanation.smoothing_gap` holds each statistic of the smoothed
model minus the model's own at each row; `full_values - smoothing_gap` are the
model's own statistics, those of `predict_distribution`. A value that many
context rows share is a large step, around which the smoothed model departs
most from the model itself, so `explain` warns when one value fills at least a
tenth of a column's context rows.

`explain_context` takes the gradient in the context rows, whose ranks set the
steps themselves. Smoothing the context's ranks changed TabPFN's predictions
by up to 1.29 standard deviations of the target on kin8nm, so it is not done:
its gradients go through the exact ranks and miss what they carry, and it
warns so.

### How far a gradient reaches

A gradient passes from the predicted density back through the model, and then
through each step before it. Each step has one gradient kind:

| Kind | Meaning | Examples |
|---|---|---|
| `differentiable` | The gradient passes. | a logarithm, a scaling, PCA, a power or quantile transform |
| `index_map` | The gradient passes to the input each output comes from. | a subset or a reordering of columns |
| `mask` | The gradient passes, and a masked input gets none. | an imputed value |
| `barrier` | The gradient stops at the step's output. | text hashing, a transformer that lazy does not know |

The model itself is differentiable when its fit applied no transform or clip
of the model's own package: `transforms=("none",)` and
`outlier_threshold=None` always qualify, and so does any transform of the
vocabulary that lazy applies for the backend, such as `"quantile"` on TabPFN
or TabICL. The columns must be numeric. `explain` refuses any other fit, and
names what stands in the way.

In a {class}`~lazy.LazyPipeline`, lazy knows the forward of these fitted
scikit-learn transformers, read from their fitted parameters:
`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`,
`PowerTransformer`, `QuantileTransformer`, `PCA`, `TruncatedSVD`,
`SimpleImputer` (a mask), `FunctionTransformer` with a numpy function such as
`np.log1p` or `np.arcsinh`, and `Pipeline` and `ColumnTransformer` of these.
The fitted parameters are constants: the gradient does not see how they
depend on the context.

A {class}`~lazy.Tabularizer` is entered column by column. Its reduction is
differentiable, and so are the classical encoders of images and time series
(`image_pixels`, `image_statistics`, `time_statistics`, `time_spectrum`), so
the gradient reaches the pixels of an image or the values of a series when
the items of a column have one shape. The value of such a column is the sum
over its elements, and `explanation.item_values` keeps the value of each
element, such as a map over the pixels:

```python
model = lazy.make_pipeline(
    lazy.Tabularizer({"cutout": "image_pixels"}), lazy.LazyModel(...)
)
explanation = interp.explain(
    model.fit(X_train, y_train), X_test, y_test, X_train, y_train,
    method="integrated_gradients",
)
explanation.item_values["cutout"]  # (n_statistics, n_rows, height, width, channels)
```

The networks of `lazy.time.Pretrained`, but Chronos-T5, also take series of
different lengths, and series with missing values. The series of a column
are padded on the left with `NaN` to the longest of the explained rows,
and the network reads a `NaN` as a missing value, whose attribution is 0.
`item_values` then has shape (n_statistics, n_rows, longest), and the
attributions of a series of `k` values, from its first finite value to its
last, are the last `k`:

```python
values = explanation.item_values["curve"]
own = values[:, row, -len(series):]  # one per value of the row's series
```

Integrated gradients keep a missing value missing along the whole path. The
baseline of a column of items is the mean of the context's items, value by
value, over the values that the context has. `explain` takes no gradient in
the context: it reads the context through the encoders' `transform`, with
their cache, and holds the items of the explained rows alone.

A token of a text has no gradient, but the vector that a network looks it up
as, its input embedding $e_j$, does. Every network of `lazy.text.Pretrained`
takes the input embeddings of the tokens it reads of each text, padded on the
right with `NaN` to the longest of the explained rows. The baseline of a
text keeps its special tokens and the prefix, such as E5's `"query: "`, and
puts the padding token's embedding $p$ in place of each of the text's own
tokens. Token $j$ then gets one value, summed over its vector:

$$
\phi_j = \sum_{h} (e_{jh} - b_{jh}) \, \frac{1}{m} \sum_{k=1}^{m}
\frac{\partial s}{\partial e_{jh}}\Big|_{b + \frac{k}{m}(e - b)},
\qquad b_j = \begin{cases} p & \text{a token of the text} \\
e_j & \text{a special or prefix token,} \end{cases}
$$

so a special or prefix token, which the baseline keeps, gets 0, and the
values of the text's own tokens sum to the statistic's gain from the text of
padding tokens, to the error of the Riemann sum. `item_values` has shape
(n_statistics, n_rows, longest), in the order of the encoder's `token_spans`,
which locates each token in the text. `word_values` sums a row's values over
each word:

```python
encoder = model[0].encoders_["note"]
values = explanation.item_values["note"]  # (n_statistics, n_rows, longest)
words, sums = encoder.word_values(X_test["note"].iloc[row], values[0, row])
```

A step that is a barrier ends the attribution: its output columns become the
features. `explain` warns with the step's name and the reason, and lists them
in `explanation.barriers`. The classical encoders of text are barriers, since
characters have no gradient, and so is Chronos-T5, which reads each value of a
series as one of a set of tokens. The other networks of `lazy.time.Pretrained`
pass the gradient to each value of a series, those of `lazy.text.Pretrained`
to each token of a text, and every network of `lazy.image.Pretrained` to each
pixel of an image of one size. For an image that is not 8-bit, the gradient
reaches its stretched values ({doc}`encoders`); behind augmentations, whose
forwards would run before the stretch, such a column is a barrier. A
scikit-learn transformer that lazy does not know is a barrier too, although it
still predicts as before.

### Stages

A *stage* is the output of one step of a pipeline. `at` names the step, and
the features are then that step's output columns, named as the table it
returns names them (`x0` onwards for a step that returns an array):

```python
interp.explain(model, X_test, y_test, X_train, y_train, at="tabularizer")
```

Every method takes `at`. A removal method runs the pipeline to the stage,
removes or permutes the stage's columns, and runs the rest: Shapley values and
`"loco"` refit the later steps and the model on the stage's columns, and
`"permutation"` keeps them fitted. A gradient method takes the gradient in the
stage's columns, so every later step needs a forward. Without `at`, a removal
method works on the raw columns, and a gradient method goes as far back as the
gradient reaches. `explanation.stage` records the stage.

### Ensembles

A {class}`~lazy.LazyEnsembleModel` that pools densities (`"geometric"` or
`"linear"`) is one function of the input through every member. The gradient
methods take one backward pass from the pooled density, which gives each
member's path its share: a feature's value is the sum over the members, and
each member's barriers are reported under `<member>__<step>`. A member that
stops at a barrier contributes that stage's columns as features of their own,
named `<member>__<column>`. So does a column of items whose members' encoders
read the items differently, such as two networks of text with other
tokenizers, whose `item_values` would not line up; seeded copies of one
pipeline share the column. `pooling="quantile"` has no differentiable pass.

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
context rows with the largest absolute values, and
`explanation.top(k, X_context=X_train)` adds each context row's features.

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

On a pipeline, the gradient is taken in the columns that the gradient reaches,
as above, except that it stops at the outputs of a Tabularizer's encoders:
going through an encoder would encode the whole context on the autograd graph
for every batch. `through_encoders=True` asks for it, and `at` names a stage.

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
