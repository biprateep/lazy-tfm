# Warnings

LAZY's own warnings are subclasses of {class}`lazy.LazyWarning`, which is a
`UserWarning`. Each points at the first line outside LAZY's code that led to
it: the caller's line, or a line of scikit-learn when a scikit-learn tool
calls the model. `ImportNameWarning`, issued during the import, is the
exception. Deprecations are `FutureWarning`s, and each names
the version in which the deprecated behaviour ends.

## LAZY's warning classes

Every class below can be imported from `lazy`.

| Class | Base | Issued when |
| ----- | ---- | ----------- |
| {class}`~lazy.LazyWarning` | `UserWarning` | A `bag_size` gives each member fewer than 10 context rows. Prediction features have column names and the fit's did not, or the reverse, so the columns are used by position. TabFM's context targets lie outside the range of its `y_grid` and are clipped to its ends. A cached benchmark file has the wrong size and is fetched again. {func}`~lazy.plotting.style.verify_style` finds that the serif font fell back to DejaVu. |
| {class}`ContextSizeWarning <lazy.models.ContextSizeWarning>` | `LazyWarning` | At `fit`, a member's context is larger than the model handles well: 20,000 rows for LimiX-2 and 60,000 for Kumo Tabular (see {doc}`scaling`). |
| {class}`PerformanceWarning <lazy.models.PerformanceWarning>` | `LazyWarning` | At `fit`, `device="auto"` finds no GPU for a model that is slow on a CPU. LimiX-2 predicts some members without their key/value cache, because the device lacks the memory or the checkpoint does not support it. |
| {class}`~lazy.models.tabfm.TabFMPerformanceWarning` | `PerformanceWarning` | At `fit`, the installed `tabfm` lacks the API of TabFM's streaming path, so every chunk of queries processes the context again (see {doc}`../installation`). `kv_cache=False` asks for that path and silences it. |
| {class}`GridCoverageWarning <lazy.models.GridCoverageWarning>` | `LazyWarning` | On the native grid of an unbagged TabICL or Kumo Tabular model, rows put more than 1% of their probability outside the grid, and their densities are renormalized. A wider `y_grid` avoids it (see {doc}`limits`). |
| {class}`~lazy.models.ensemble.EnsembleSizeWarning` | `LazyWarning` | At `fit`, a pool of two or more members gets fewer than 3,000 context rows (see {doc}`ensembles`). |
| {class}`LicenseWarning <lazy.models.LicenseWarning>` | `LazyWarning` | Before LAZY downloads weights whose upstream package asks the user to accept a licence: TabPFN v2.5 and later. It names the licence. |
| {class}`LimiXSourceWarning <lazy.models.LimiXSourceWarning>` | `LazyWarning` | LimiX's source is not at the commit that LAZY was validated on, or `LAZY_LIMIX_SRC` names another source than the one the process loaded. |
| {class}`~lazy.ImportNameWarning` | `LazyWarning` | At `import lazy`, the unrelated PyPI package `lazy` is installed in the same environment, and the two overwrite each other's files. |

LAZY also issues warnings of other classes:

| Class | Issued when |
| ----- | ----------- |
| `FutureWarning` | A deprecated name, argument or input is used. |
| `RuntimeWarning` | A cached checkpoint, loaded with downloads disabled, cannot be confirmed complete. {func}`~lazy.models.tabpfn.bucket_masses` gets no regressor and some float32 bucket borders coincide. |
| {class}`sklearn.exceptions.DataConversionWarning` | The target has shape `(n_samples, 1)` and is flattened, as in scikit-learn. |

The upstream packages issue warnings of their own, which LAZY does not
filter.

## Filtering warnings

{func}`warnings.filterwarnings` selects warnings by class, and a filter on a
class also selects its subclasses:

```python
import warnings

import lazy

# One class.
warnings.filterwarnings("ignore", category=lazy.LicenseWarning)

# PerformanceWarning, and TabFMPerformanceWarning with it.
warnings.filterwarnings("ignore", category=lazy.PerformanceWarning)

# Every warning class of LAZY's own, but not FutureWarning or RuntimeWarning.
warnings.filterwarnings("ignore", category=lazy.LazyWarning)

# Raise an error instead, for example in a test suite.
warnings.filterwarnings("error", category=lazy.ContextSizeWarning)
```

To filter warnings in one block of code only, put the filter inside
{class}`warnings.catch_warnings`. `ImportNameWarning` is issued during
`import lazy`, so its filter must come before the import, and select the
warning by its message:

```python
warnings.filterwarnings("ignore", message="The PyPI package 'lazy'")
import lazy
```
