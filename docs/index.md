# LAZY

**Lazy but Accurate photo-Z for Yinz**: photometric redshift probability
densities from pretrained tabular foundation models.

The models are never fine-tuned. You hand them labelled galaxies as *context*
and they answer queries in one forward pass, so there is no training loop, no
hyper-parameter search and no per-survey retraining. Hence lazy.

```{warning}
Pre-release. The API is still moving and the numbers in the paper are not
final. Pin a version if you depend on it.
```

```python
from lazy import LazyModel
from lazy.datasets import fetch_dc1

train, test = fetch_dc1(split=True)
X_train, X_test = train.features("mag-color"), test.features("mag-color")

model = LazyModel("tabpfn", version="v3.5").fit(X_train, train.redshift)
pdfs = model.predict_proba(X_test)            # densities on a redshift grid
print(model.evaluate(X_test, test.redshift))  # the full metric table
```

The API is scikit-learn's, with `predict_proba` returning a density on a
redshift grid rather than class probabilities, because that is the natural
output of a photo-z model.

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
guide/index
API reference <autoapi/lazy/index>
changelog
contributing
citing
```
