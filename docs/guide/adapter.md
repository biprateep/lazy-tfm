# Adaptive reducer

{class}`lazy.Tabularizer` reduces its embeddings to a few columns by principal
component analysis (PCA), which is fitted without the target. An *adaptive
reducer* is a small network in its place, trained for the frozen model's own
likelihood: it starts at the principal components and learns which directions
of the embeddings the model needs.

It trains weights, the network's and never the model's, so it is opt-in (see
{doc}`philosophy`). Whether it beats PCA on the benchmarks is an open research
question, [#331](https://github.com/biprateep/lazy-tfm/issues/331).

```python
import lazy

model = lazy.make_pipeline(
    lazy.Tabularizer({"notes": "text"}, n_columns=8, reducer="mlp"),
    lazy.LazyModel("tabpfn", transforms=("none",), outlier_threshold=None),
)
model.fit(X_train, y_train)
model.predict_proba(X_test)
model[0].history_   # the training and validation NLL at each step
```

`reducer="mlp"` uses the default settings.
{class}`lazy.encoders.AdaptiveReducer` holds them, and is given as `reducer`
to change one: `reducer=AdaptiveReducer(n_steps=500)`. A search reaches them
as `tabularizer__reducer__n_steps`.

A scikit-learn step is fitted before the model and never sees it. So the
training is part of {meth}`LazyPipeline.fit <lazy.pipeline.LazyPipeline.fit>`: it fits every step, trains
the reducer through the model, then fits the later steps and the model again
on the trained reducer's columns. A Tabularizer fitted on its own, outside a
`LazyPipeline`, keeps the network as it starts, which is PCA.

## The network

Let $z$ be the standardized, weighted embeddings of a row, side by side
({doc}`encoders` defines them). The network gives $k$ new columns
(`n_columns`):

$$
f(z) = h + B \tanh(A h + a) + b, \qquad h = W z - c,
$$

where $W$ is a $k \times p$ matrix for $p$ embedding columns, $c$ centres the
projection, and $A$, $a$, $B$ and $b$ are the weights of a multilayer
perceptron (MLP) with one hidden layer of `hidden` units, added to $h$ as a
residual. The model reads $f(z)$, then the columns that pass through the
Tabularizer, unchanged.

At step 0, $W$ holds the loadings of the first $k$ principal components of
$z$ over the context, and $B$ and $b$ are zero, so that $f(z) = W z - c$ is
PCA. $A$ and $a$ are drawn uniformly within $1 / \sqrt{k}$ of zero. The
standardization and the weights of the embeddings stay as fitted.

`AdaptiveReducer(network=module)`, or a `torch.nn.Module` given as `reducer`,
trains that module in place of the network above. It maps $z$, of shape
(rows, $p$), to the columns the model reads, and it starts as it is given. A
copy of the module is trained, and the one given is left as it is.

## Training

`fit` splits the context once, with `random_state`, into three parts: the
validation queries $V$ (`validation_fraction` of the rows), the training
queries $Q$ (`query_fraction`) and the support rows $S$ (the rest). The loss is
the mean negative log-likelihood (NLL) of the training queries under the model
given the support rows, both mapped by the network:

$$
L_Q(\theta) = -\frac{1}{|Q|} \sum_{q \in Q}
\log p\bigl(y_q \mid f_\theta(z_q), \{(f_\theta(z_s), y_s)\}_{s \in S}\bigr),
$$

where $\theta = (W, A, a, B, b)$ and $p$ is the model's predicted density.
The gradient of $L_Q$ passes through the forward of each step between the
Tabularizer and the model, and through the model's differentiable forward
pass, which {doc}`interpretability` also uses, to $\theta$ alone. Each of the
`n_steps` steps of Adam, at `learning_rate`, updates $\theta$.

The reducer keeps the step with the lowest validation NLL $L_V(\theta_t)$,
over $t = 0, \ldots,$ `n_steps`. Step 0 is a candidate, so the reducer falls
back to PCA when training does not help. `history_` holds $L_Q$ and $L_V$ at
each step, and `best_step_` the step kept.

## Cost and limits

- Each step runs the model's forward pass twice: on the training queries, with
  the autograd graph of the support rows and the queries held in memory, and
  on the validation queries, without one. `fit` also fits the model three
  times: on the whole context, as every pipeline does; on the support rows,
  for the forward pass; and on the whole context again, mapped by the kept
  network.
- The items are encoded once, and the network trains on their embeddings, in
  float64 on the CPU. The model's pass runs on the model's device.
- The model needs a differentiable forward pass, and every step between the
  Tabularizer and the model a forward ({doc}`interpretability` lists them).
- The columns that pass through the Tabularizer must be numeric, and no value
  may be missing. A missing item takes the context's mean embedding, as in
  every reduction.
- A query whose target falls outside the model's buckets has an infinite NLL.
  A step whose validation NLL is infinite is never kept.
- The model's own limits hold for the new columns too. TabPFN, for example,
  refuses a differentiable pass over more columns than it reads per member.
- Parameters given to `fit` for a step reach the first fit of each step only.
