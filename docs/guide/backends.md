# Choosing a backend

| Name     | Method                                                                  | Weights        |
| -------- | ----------------------------------------------------------------------- | -------------- |
| `tabpfn` | Bucket masses of TabPFN's bar distribution, rebinned onto the grid.     | 41 MB – 880 MB |
| `tabicl` | Quantiles of TabICLv2's regression head, differenced onto the grid.     | ~100 MB        |
| `tabfm`  | A hierarchy of TabFM in-context classifiers over equal-mass redshift bins. | ~6.6 GB     |

`LazyModel("tabpfn", ...)` and the concrete classes
({class}`~lazy.models.tabpfn.TabPFNBarDistribution`,
{class}`~lazy.models.tabicl.TabICLQuantile`,
{class}`~lazy.models.tabfm.TabFMHistogram`) are the same models; the class
pages document every parameter. Use `LazyModel` when the backend is a
configuration value, as in a benchmark loop.

All of them write onto whatever {class}`~lazy.grid.RedshiftGrid` you ask for:
any number of bins, any spacing, any range. TabFM's classifier is limited to
ten classes, but that limit applies to each level of its internal hierarchy,
never to the output grid.

## Which model, exactly

A backbone is a family, not a model, so every backend takes a `version`, and
each version is a separately pinned checkpoint:

```python
import lazy

lazy.list_versions("tabpfn")
# ['v2', 'v2.5', 'v2.6', 'v3', 'v3.5', 'v3.5-fast']

model = lazy.LazyModel("tabpfn", version="v2.5")
model.name_          # 'tabpfn:v2.5', the label evaluate() puts on its row
```

What actually answered is recorded on the fitted model, ready to be written out
beside the numbers:

```python
model.fit(X_train, z_train).provenance_
# {'backend': 'tabpfn', 'version': 'v2.5',
#  'repo_id': 'Prior-Labs/tabpfn_2_5',
#  'filename': 'tabpfn-v2.5-regressor-v2.5_default.ckpt',
#  'revision': '6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b',
#  'package': 'tabpfn 9.0.0', 'lazy': 'lazy-photoz 0.1.0', 'device': 'cuda'}
```

It holds the weights *and* the code that read them, and no local paths, so it
means the same thing on another machine and survives a trip through JSON.
Settings are not in it; `get_params()` has those.

Versions are not interchangeable. TabPFN `v2` is pretrained for at most 10,000
context rows and `v2.5` for 50,000, against a million for `v3`; `v2` is also
the only one whose weights allow commercial use.
{func}`lazy.get_checkpoint` returns each version's licence and size notes.

## Memory and speed

Peak memory is bounded by `chunk_size` on every backend (16,384 query rows by
default). Chunking is exact: the in-context stage builds its keys and values
from the context rows alone, so a galaxy's answer never depends on which other
galaxies share its chunk. The test suite asserts this bit for bit. Lower
`chunk_size` on a small GPU; set it to `0` for a single pass.

Two settings avoid recomputing the context for every chunk, and are worth their
memory whenever the query set is much larger than the context: TabPFN's
`fit_mode="fit_with_cache"`, and TabFM's repository build (see
{doc}`../installation`).

Every backend shows a progress bar while it predicts (`progress="auto"`: on a
terminal or in a notebook, not when output goes to a file).
