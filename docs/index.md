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

**LAZY** is on PyPI as `lazy-tfm` and is imported as `lazy`. The demo below
uses TabPFN, so install that model:

```console
pip install 'lazy-tfm[tabpfn]'
```

With [uv](https://docs.astral.sh/uv/), use `uv add 'lazy-tfm[tabpfn]'` in a
project or `uv pip install 'lazy-tfm[tabpfn]'` in an environment. For the other
models, and for `[all]`, which installs every supported model, see the
{doc}`installation` page.

### Usage

As a quick demonstration, we predict the mean
and 68% interval for the missing segments of a noisy curve. The `"chirp"` dataset of {func}`~lazy.datasets.load_dataset` is a noisy sinusoid, and
`split=True` returns the three gaps cut out of it as the test set.

```{literalinclude} snippets/quickstart.py
:language: python
:start-at: import lazy
```

Every other argument of `LazyModel` has a default, listed in the
[shared parameters table](guide/interface.md#shared-parameters-and-defaults).
The first run downloads the model's weights, about 334 MB. The snippet prints
the first three rows of each prediction; their last digits differ from one
processor to another:

```text
[0.64850495 0.62743615 0.62331416]
[0.45311083 0.44126895 0.43087525]
[0.84825626 0.82071483 0.82279941]
```

```{figure} figures/figs/chirp_demo.svg
:class: lazy-figure
:alt: A noisy chirp with three gaps, filled by TabPFN-3.5-fast's mean and 68% interval.
:width: 100%
```

### Next

- {doc}`guide/index`: how the models work and how to choose one.
- {doc}`tutorials/index`: worked examples.
- {doc}`models/index`: each model's checkpoints, limits and parameters.

<!-- The sidebar. :titlesonly: keeps a page's own sections out of it, so
Installation and Citing show no sub-list, and Contributing lists only the
pages of its own toctree (Adding a backend and its worked example). -->

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
citing
brand
```

```{toctree}
:hidden:
:maxdepth: 2

changelog
```
