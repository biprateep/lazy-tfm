# Context adapter (experimental)

{class}`lazy.experimental.ContextAdapter` learns a small network that maps
some of a table's columns to a few new ones, for a frozen model to read. It
can stand in for the principal components that {class}`lazy.Tabularizer`
keeps of a wide embedding: the network starts at those components and is
trained for the model's own likelihood.

It trains weights, the network's and never the model's, so it is opt-in (see
{doc}`philosophy`). It lives in
`lazy.experimental`, which a release may change or remove without a
deprecation. Whether it beats PCA on the benchmarks is an open research
question, [#331](https://github.com/biprateep/lazy-tfm/issues/331).

```python
import lazy
from lazy import experimental

adapter = experimental.ContextAdapter(
    lazy.LazyModel("tabpfn", transforms=("none",), outlier_threshold=None),
    n_components=8,
)
model = lazy.make_pipeline(
    lazy.Tabularizer({"notes": "text_pretrained"}, n_components=None),
    adapter,
)
model.fit(X_train, y_train)
model.predict_proba(X_test)
model[-1].history_   # the training and validation NLL at each step
```

The adapter is a density regressor: it has every prediction method of the
model ({doc}`interface`), and answers them with the model fitted on the mapped
context, `fitted_model_`.

## The network

Let $x$ be the columns of a row that the network maps (`columns`, all of them
by default), and $s(x) = (x - \mu) / \sigma$ their standardization, with the
mean $\mu$ and the standard deviation $\sigma$ of each column over the
context. The network gives $k$ new columns (`n_components`):

$$
f(x) = h + B \tanh(A h + a) + b, \qquad h = W s(x),
$$

where $W$ is a $k \times p$ matrix for $p$ mapped columns, and $A$, $a$, $B$
and $b$ are the weights of a multilayer perceptron (MLP) with one hidden layer
of `hidden` units, added to $h$ as a residual. The model reads $f(x)$, then
the columns that the network does not map, unchanged.

At step 0, $W$ holds the loadings of the first $k$ principal components of
$s(x)$ over the context, and $B$ and $b$ are zero, so that $f(x) = W s(x)$ is
principal component analysis (PCA). $A$ and $a$ are drawn uniformly within
$1 / \sqrt{k}$ of zero. $\mu$ and $\sigma$ stay as fitted.

## Training

`fit` splits the context once, with `random_state`, into three parts: the
validation queries $V$ (`validation_fraction` of the rows), the training
queries $Q$ (`query_fraction`) and the support rows $S$ (the rest). The loss is
the mean negative log-likelihood (NLL) of the training queries under the model
given the support rows, both mapped by the network:

$$
L_Q(\theta) = -\frac{1}{|Q|} \sum_{q \in Q}
\log p\bigl(y_q \mid f_\theta(x_q), \{(f_\theta(x_s), y_s)\}_{s \in S}\bigr),
$$

where $\theta = (W, A, a, B, b)$ and $p$ is the model's predicted density.
The gradient of $L_Q$ passes through the model's differentiable forward pass,
which {doc}`interpretability` also uses, to $\theta$ alone. That pass needs a
model fitted with `transforms=("none",)` and `outlier_threshold=None`, on
numeric columns. Each of the `n_steps` steps of Adam, at `learning_rate`,
updates $\theta$.

The adapter keeps the step with the lowest validation NLL $L_V(\theta_t)$,
over $t = 0, \ldots,$ `n_steps`. Step 0 is a candidate, so the adapter falls
back to PCA when training does not help. `history_` holds $L_Q$ and $L_V$ at
each step, and `best_step_` the step kept. Finally, a clone of the model is
fitted on the whole context mapped by the kept network, and each prediction
maps its rows the same way.

## Cost and limits

- Each step runs the model's forward pass twice: on the training queries, with
  the autograd graph of the support rows and the queries held in memory, and
  on the validation queries, without one. `fit` also fits the model twice:
  on the support rows, for the forward pass, and on the whole context.
- The network trains on the model's device in float64, or in float32 on an
  Apple GPU, which has no float64; the columns are standardized in float64
  first. It maps rows for `fit` and prediction in float64.
- The context must have no missing value and no categorical column. At
  prediction, a missing value in a mapped column makes every new column of
  its row missing.
- A query whose target falls outside the model's buckets has an infinite NLL.
  A step whose validation NLL is infinite is never kept.
- The model's own limits hold for the new columns too. TabPFN, for example,
  refuses a differentiable pass over more columns than it reads per member.
