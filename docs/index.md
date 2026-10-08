# LAZY: Lazy but Accurate Z\* for Yinz

<small>\*Where **Z** is any quantity you may want to predict, such as galaxy
redshift, metallicity, atomic number, electrical impedance, or whatever you
want (Z stands for [many
things](https://en.wikipedia.org/wiki/Z#Other_uses)).</small>

**LAZY** is a Python package that provides a unified wrapper around [several
pretrained](models/index) [tabular foundation models](guide/tfm). **LAZY** standardizes their interface, defaults and outputs behind one API,
which is similar to and compatible with scikit-learn's. **LAZY** also provides a consistent way to access the full predicted probability distribution of the target for every model, rather than just a point estimate.

## Getting Started

### Installation

**LAZY** is on PyPI as `lazy-tfm` and is imported as `lazy`. To install it with every supported model:

```console
pip install 'lazy-tfm[all]'
```

With [uv](https://docs.astral.sh/uv/), use `uv add 'lazy-tfm[all]'` in a
project or `uv pip install 'lazy-tfm[all]'` in an environment. For detailed
installation instructions, including support for specific models, refer to the {doc}`installation` page.

### Usage

As a quick demonstration, we predict the mean
and 68% interval for the missing segments of a noisy curve. The `"chirp"` dataset of {func}`~lazy.datasets.load_dataset` is a noisy sinusoid, and
`split=True` returns the three gaps cut out of it as the test set.

```python
import lazy
from lazy import datasets

X_train, X_test, y_train, y_test = datasets.load_dataset(
    "chirp", split=True, return_X_y=True, random_state=0
)



model = lazy.LazyModel( # Equivalent to lazy.LazyModel("tabpfn", version="v3.5-fast").
    "tabpfn", # The defaults are spelled out for clarity.
    version="v3.5-fast", # Can be run on CPU
    n_estimators=8,
    transforms="auto",
    feature_shuffle=True,
    bag_size=None,
    kv_cache=True,
    y_grid=None,
    device="auto",
    random_state=0,
    chunk_size=8192,
    softmax_temperature="auto",
    mixed_precision=True,
    outlier_threshold="auto",
)
model.fit(X_train, y_train)

y_mean = model.predict(X_test, method="mean")
y_low, y_high = model.predict_interval(X_test, coverage=0.68).T
```

```{figure} figures/figs/chirp_demo.svg
:alt: A noisy chirp with three gaps, filled by TabPFN-3.5-fast's mean and 68% interval.
:width: 100%
```

<!-- The sidebar. :titlesonly: keeps a page's own sections out of it, so
Installation, Contributing and Citing show no sub-list. -->

```{toctree}
:hidden:
:titlesonly:

installation
```

```{toctree}
:hidden:
:maxdepth: 2

guide/index
tutorials/index
models/index
API reference <autoapi/lazy/index>
```

```{toctree}
:hidden:
:titlesonly:

contributing
adding_a_backend
citing
```

```{toctree}
:hidden:
:maxdepth: 2

changelog
```
