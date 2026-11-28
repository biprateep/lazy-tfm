# Adding an encoder

An encoder turns one item that is not a table cell (an image, a text, a time
series) into a vector of fixed length, its embedding. A column of $n$ items
becomes $n$ rows of numeric columns, which the models take as features.
{doc}`guide/encoders` lists the encoders that ship. An
encoder is a subclass of {class}`~lazy.encoders.BaseEncoder`, which already
does everything encoders have in common: it reads the column, encodes it in
batches, gives a missing item a row of `NaN`, checks each answer, files
embeddings in the cache and names the output columns. This page describes the
contract of a subclass and gives a skeleton.

## The contract

`BaseEncoder` is a scikit-learn transformer, so an encoder is a step of a
{class}`~sklearn.compose.ColumnTransformer` or of a
{class}`~lazy.pipeline.LazyPipeline`, and `clone`, `get_params` and
`set_params` work on it. `fit` and `transform` take one column of items: a
sequence, a pandas Series, a DataFrame, or an array whose first axis indexes
the items. A DataFrame is one column that is not numeric, or numeric columns
that hold one item per row, such as time series of equal length.

An encoder is frozen: the same item gives the same embedding whatever the
encoder was fitted on, so `fit` only validates.

A subclass does the following.

- Its constructor takes `batch_size` (default 64) and `cache` as keywords,
  with its own parameters, stores each one verbatim, gives every parameter
  a default and takes no `**kwargs`, so that `cls()` builds a working encoder.
- It sets `modality` to `"image"`, `"text"` or `"time"`.
- It implements `_width`, the length of an embedding, and `_encode`, which
  returns the finite embeddings of a batch of at most `batch_size` items,
  shape `(len(items), width)`. No item of a batch is missing. `fit` calls
  `_width`, so `_width` is also where an encoder checks its own parameters.
- It may implement `_names`, a name for each column of an embedding.
  `get_feature_names_out` puts the modality and the lower-case name of the
  class before each, as in `time_statistics_mean`. Without `_names` the
  columns are numbered, as in `image_pixels_0`.
- Each row of an answer depends on its own item alone, whatever else the
  batch holds, and an encoding changes neither the encoder's parameters nor
  its later answers.

{func}`lazy.encoders.register` refuses a class that falls short of the first
three points, and `transform` checks the shape and the values of every answer.

An encoder that ships with `lazy` lives in the module of its modality
(`lazy.image`, `lazy.text` or `lazy.time`), and `lazy/encoders/registry.py`
registers it under `<modality>_<name>`. `import lazy` imports these modules,
so an encoder imports a heavy package, such as PyTorch, inside its methods.

## A skeleton

```python
import numpy as np

from lazy import encoders


class Length(encoders.BaseEncoder):
    """The number of characters and of words of each text."""

    modality = "text"

    def __init__(self, *, batch_size=64, cache=False):
        self.batch_size = batch_size
        self.cache = cache

    def _width(self):
        return 2

    def _encode(self, items):
        return np.array(
            [[len(item), len(item.split())] for item in items], dtype=float
        )


encoders.register("length", Length)
encoders.get_encoder("length").fit_transform(["a lazy model", None])
# array([[12.,  3.],
#        [nan, nan]])
```

## A pretrained encoder

An encoder that runs a frozen network subclasses
`lazy.encoders._pretrained.PretrainedEncoder`, which handles the download,
the device and the network cache. It runs one of several networks, which its
`model` parameter names, as a backend's `version` does. In place of
`_encode`, it gives:

- For each network, its pinned weights in `ENCODER_CHECKPOINTS` in
  `src/lazy/models/_hub.py`, keyed `"<modality>:<model>"`: the repository on
  the Hugging Face Hub, its commit, the files to fetch, the licence, the size
  and the number of parameters. The first download warns with this licence,
  as the models' checkpoints do, and `lazy.download_checkpoint` fetches it
  ahead of time. A network whose licence restricts commercial use passes
  `commercial_use=False`, and its first download asks the user to read the
  licence. The default network is in `DEFAULT_ENCODER_MODELS`, and allows
  commercial use.
- `recipes`, which maps each network to its recipe: the width of the
  embedding (`width`) and what else differs between its networks, such as the
  pooling or a prompt. `register` refuses a class whose recipes are not exactly the
  networks pinned for its modality.
- `package`, the package that loads the weights, from the `encoders` extra.
- `model` and `device` keywords; `device` defaults to `"auto"`.
- `_load(path, device)`, which builds the network from the downloaded files,
  and `_embed(network, items)`, which runs it on a batch, without gradients.
  The network loads once per process and device.
- `_length(item)`, if it pads a batch to its longest item, so that items of
  similar length share a batch.

`PretrainedEncoder`'s docstring states what the base class does with these:
the download, the cache of networks and the key of the embedding cache. Its
tests run its own code on fakes of the upstream loaders
(`tests/lazy/fakes/encoders.py`), and compare the real weights with golden
embeddings on request; `tests/lazy/record_golden.py encoders` records them.

## The cache

With `cache=True`, an encoder files the embedding of each item on disk, in
`embeddings/v1` under {func}`lazy.datasets.data_home`, and an item that is filed
is not encoded again: not in a later call, in another process, or in another
subset of the rows, such as a fold of a cross-validation. `cache` can also
name a directory to use in place of the default.
{func}`lazy.encoders.clear_embedding_cache` empties the default cache.

An embedding is filed under two keys.

- The content of the item, from `_item_bytes`. The base class covers text,
  bytes and numeric arrays. An encoder of other items returns the bytes that
  set its embedding. An encoder that reads files must override it for a path
  given as text too: the base class would key the path, not the content of
  the file.
- The identity of the encoder: `_identity_head` (by default the version of
  `lazy`), the class and every parameter but those in `_UNKEYED` (by default
  `batch_size` and `cache`). An encoder whose output depends on more, such as
  the revision of its weights, adds it to `_identity_head`. One with a
  parameter that is not a number, text, `None`, or a list or dict of those
  names it in `_UNKEYED` and keys what it sets in `_identity_head`. Each encoder's folder holds its identity
  as text, in `identity.json`.

A change to an encoder's code that changes its output, with no change of
version or parameters, leaves stale embeddings in the cache; clear the cache
after such a change.

An encoder that is cheap to run should default to `cache=False`: reading a
file costs more than a few arithmetic operations.

## Gradients

A step declares how a gradient passes it ({mod}`lazy._steps`). An encoder is a
barrier to gradients by default: a gradient taken through a pipeline ends at
its embedding. To let a gradient reach the items, an encoder does three
things.

1. It declares `gradient_kind = "differentiable"` on the class, or as a
   property where the kind depends on a parameter, as `lazy.time.Pretrained`'s
   does on `model`.
2. It implements `_array(item)`: one item as a float64 array, or `None` for an
   item that has none, such as a series with a gap. The base class stacks the
   arrays of a column (`_stacked`) when no item is missing and all have one
   shape. The attribution is then to the elements of that array. If a column
   does not stack, attribution ends at the encoder's embedding, as at a
   barrier. An encoder whose items differ in length, or lack some elements,
   sets `pads_items` to the side it aligns them on, `"left"` or `"right"`:
   its `_array` gives `NaN` for a missing element, and `_stacked` pads each
   array with `NaN` on that side to the longest along the first axis, as
   `lazy.time.Pretrained` does on the left.
3. It implements `forward(items)`: the embeddings of the stacked array as a
   torch tensor, on the autograd graph. It must give what `transform` gives for
   the same items, to float rounding, with the type and on the device of
   `items`, and a test should check it against `transform`. It refuses a
   tensor that `_stacked` would not give, with `lazy._steps.BarrierError`. The
   caller chooses how many items one call holds. With `pads_items`, it reads
   a `NaN` as a missing element: it gives what `transform` gives for the item
   without its padding, and the gradient in a `NaN` is 0. No `NaN` may enter
   the arithmetic on the graph, whose backward pass would then multiply it by
   0 and give `NaN`: select with `torch.where` first.

An encoder that cannot have a forward, such as one of text, may say why in
`barrier_reason`. A subclass that changes the embedding of an encoder with a
forward must give its own `forward`, or set `gradient_kind = "barrier"`.
