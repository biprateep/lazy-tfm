# Adding an encoder

An encoder turns one item that is not a table cell (an image, a text, a time
series) into a vector of fixed length, its embedding. A column of $n$ items
becomes $n$ rows of numeric columns, which the models take as features. An
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
  shape `(len(items), width)`. No item of a batch is missing.
- Each row of an answer depends on its own item alone, whatever else the
  batch holds, and an encoding changes neither the encoder's parameters nor
  its later answers.

{func}`lazy.encoders.register` refuses a class that falls short of the first
three points, and `transform` checks the shape and the values of every answer.

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
- The identity of the encoder, from `_cache_identity`: by default the version
  of `lazy`, the class and every parameter but `batch_size` and `cache`. An
  encoder whose output depends on more, such as the revision of its weights,
  adds it there, and so does one with a parameter that is not a number, text,
  `None`, or a list or dict of those. Each encoder's folder holds its identity
  as text, in `identity.json`.

A change to an encoder's code that changes its output, with no change of
version or parameters, leaves stale embeddings in the cache; clear the cache
after such a change.

An encoder that is cheap to run should default to `cache=False`: reading a
file costs more than a few arithmetic operations.
