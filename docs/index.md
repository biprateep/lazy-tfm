# LAZY

**Lazy but Accurate *z*\* for Yinz**: full conditional distributions of a
continuous target from pretrained tabular foundation models.

\*where *z* is whatever you want to predict from tabular features: a redshift,
a metallicity, a mass, a yield.

The models are never fine-tuned. You hand them labelled rows as *context* and
they answer queries in one forward pass, so there is no training loop, no
hyper-parameter search and no per-dataset retraining. Hence lazy. The examples
use photometric redshifts, where the package began.

```{warning}
Pre-release. The API is still moving and the numbers in the paper are not
final. Pin a version if you depend on it.
```

```python
import numpy as np

import lazy
from lazy.datasets import fetch_dc1

train, test = fetch_dc1(split=True)
test = test.take(np.random.default_rng(0).choice(len(test), 20_000, replace=False))
X_train, X_test = train.features("mag-color"), test.features("mag-color")

model = lazy.LazyModel().fit(X_train, train.redshift)   # TabPFN-3.5; wants a GPU
pdfs = model.predict_proba(X_test)                      # densities on its native grid
print(model.evaluate(X_test, test.redshift, y_grid=lazy.datasets.DC1_GRID, scale="1+y"))
```

The API is scikit-learn's, with `predict_proba` returning a density on a grid
of the target rather than class probabilities.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc
pip or uv, the optional backends, checkpoints and their licences.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
From a catalogue to densities, point estimates and metrics.
:::

:::{grid-item-card} Tutorial
:link: tutorials/introduction
:link-type: doc
A notebook to run on Colab: photo-z PDFs, their quality, and two models compared.
:::

:::{grid-item-card} User guide
:link: guide/index
:link-type: doc
The API, choosing a backend, clusters and offline use, biased training sets.
:::

:::{grid-item-card} API reference
:link: autoapi/lazy/index
:link-type: doc
Every public class and function, from the docstrings.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
Tutorial <tutorials/introduction>
guide/index
API reference <autoapi/lazy/index>
changelog
contributing
citing
```
