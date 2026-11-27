# Augmentation

An augmentation shows the model another view of the same data: a rotated
image, a window of a time series, a table without some of its columns.
{mod}`lazy.augment` holds them as scikit-learn steps. An augmentation trains
nothing: `fit` only checks its parameters.

An augmentation has no averaging mode. It applies one transform, the same way
to the context and to the queries, and a random one draws from its
`random_state`. So one pipeline sees one view of the data. To average over
views, pool pipelines that differ only in their seed:

```python
import lazy
from lazy import augment

member = lazy.make_pipeline(
    lazy.Tabularizer({"cutout": [augment.Dihedral(), "image"]}),
    lazy.LazyModel(),
)
model = lazy.LazyEnsembleModel([member] * 8, pooling="linear")
model.fit(X_train, y_train)
```

{class}`~lazy.LazyEnsembleModel` gives step $j$ of member $i$ the seed
`SeedSequence([random_state, i, j])` if the step's `random_state` is still
its default, so the eight copies above draw eight views. The pool of their
predictions with `pooling="linear"` is the mixture

$$
p(y \mid x) = \frac{1}{M} \sum_{i=1}^{M} p_i\bigl(y \mid g_i(x)\bigr),
$$

where $g_i$ is member $i$'s augmentation and $p_i$ its predicted density.
`pooling="geometric"`, the default, multiplies the densities instead
({doc}`ensembles`). Averaging predictions, and not embeddings, keeps what each
view says about the uncertainty.

## Where an augmentation goes

An augmentation of items goes in the {class}`~lazy.Tabularizer`, before the
encoder of its column: `{"cutout": [augment.Crop(0.8), "image"]}`. The
Tabularizer's `random_state` seeds the augmentations of every column, each
with a seed of its own.

An augmentation of a table is a step of the pipeline, before or after the
Tabularizer. After it, the step sees only table columns:

```python
lazy.make_pipeline(
    lazy.Tabularizer(columns, n_columns=None),
    augment.ColumnDropout(0.3, by="prefix"),
    lazy.LazyModel(),
)
```

## The augmentations

| Data | Augmentation | What it does | Gradient kind |
|---|---|---|---|
| Numbers | {class}`~lazy.augment.Rescale` | $f(x / s)$, with $f$ the identity, `asinh`, `log` or `log1p` | differentiable |
| Numbers | {class}`~lazy.augment.Noise` | adds Gaussian noise of a set scale | differentiable |
| Images | {class}`~lazy.augment.Dihedral` | one of the eight rotations and flips of a square | index map |
| Images, time series | {class}`~lazy.augment.Crop` | keeps a window of each axis | index map |
| Images, time series | {class}`~lazy.augment.Mask` | sets patches to zero | mask |
| Text | {class}`~lazy.augment.TokenDropout` | drops words at random | barrier |
| Text | {class}`~lazy.augment.Truncate` | keeps the first words | barrier |
| Tables | {class}`~lazy.augment.ColumnDropout` | drops columns, or every column of one prefix | index map |

`Rescale` and `Noise` work on any array: an image, a time series or the
numeric columns of a table. With `by="prefix"`, `ColumnDropout` drops every
column that shares the text before the first `__` of its name. After a
Tabularizer that does not reduce (`n_columns=None`), that is one column of
items, so a member loses a whole data type.

An augmentation that only moves or zeroes values keeps their type, so an
8-bit image stays 8-bit for a pretrained network. An image can be an array, a
Pillow image or the path of a file.

## A row stands alone

A row's answer does not depend on the rows that come with it. A draw
therefore depends on the seed and on what it is applied to, never on the
batch:

- `Dihedral`, `Crop` and `ColumnDropout` draw once from the seed, for every
  row.
- `Mask` draws from the seed and the item's shape, so items of one shape lose
  the same patches.
- `Noise` and `TokenDropout` draw from the seed and the row's own content
  (for `Noise`, its values rounded to float32), so
  the same row always gets the same draw.

## Gradients

Each augmentation declares its gradient kind, and all but those of text have
a forward on torch tensors. `Noise` is reparameterized, $x + s\,\epsilon$ with
the draw $\epsilon$ held fixed, so a gradient passes it. Each point on a path
of integrated gradients is another input with a draw of its own, so behind
`Noise` those attributions sum to the gain only roughly; gradient times input
is exact. A gradient method of {mod}`lazy.interp` therefore
passes them ({doc}`interpretability`): through a rotation or a crop, the
attribution of a pixel goes back to where the pixel was, and a masked value
gets none.
