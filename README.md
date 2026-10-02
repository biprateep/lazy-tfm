<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/biprateep/lazy-tfm/main/docs/_static/lazy-logo-white.svg">
    <img src="https://raw.githubusercontent.com/biprateep/lazy-tfm/main/docs/_static/lazy-logo.svg" alt="LAZY" width="440">
  </picture>
</h1>

**L**azy but **A**ccurate ***z***\* for **Y**inz — full conditional distributions
of a continuous target from pretrained tabular foundation models.

\*where *z* is whatever you want to predict from tabular features: a
redshift, a metallicity, a mass, a yield.

[![PyPI](https://img.shields.io/pypi/v/lazy-tfm)](https://pypi.org/project/lazy-tfm/)
[![Python](https://img.shields.io/pypi/pyversions/lazy-tfm)](https://pypi.org/project/lazy-tfm/)
[![Unit test and code coverage](https://github.com/biprateep/lazy-tfm/actions/workflows/testing-and-coverage.yml/badge.svg)](https://github.com/biprateep/lazy-tfm/actions/workflows/testing-and-coverage.yml)
[![Documentation](https://readthedocs.org/projects/lazy-tfm/badge/?version=latest)](https://lazy-tfm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/biprateep/lazy-tfm/blob/main/LICENSE)

The models are pretrained and never fine-tuned. You hand them labelled rows as
*context* and they answer queries in one forward pass — no training loop, no
hyper-parameter search, no per-dataset retraining. Hence lazy.

The package grew out of photometric redshifts, so its worked examples and
benchmark tools (the LSST DESC DC1 catalogue, the HSC spectroscopic selection,
the Data Challenge metrics) are photo-z's; the estimators work on any tabular
regression problem.

> **Pre-release.** The API is still moving and the paper numbers are not final.
> Pin a version if you depend on this.

**Documentation:** [lazy-tfm.readthedocs.io](https://lazy-tfm.readthedocs.io) ·
**Tutorial:** [open in Colab](https://colab.research.google.com/github/biprateep/lazy-tfm/blob/tutorials/introduction.ipynb) ·
**Changelog:** [CHANGELOG.md](https://github.com/biprateep/lazy-tfm/blob/main/CHANGELOG.md) · **Contributing:** [CONTRIBUTING.md](https://github.com/biprateep/lazy-tfm/blob/main/CONTRIBUTING.md)

## Install

```console
pip install lazy-tfm              # grid, metrics, plots, datasets
pip install 'lazy-tfm[tabfm]'     # + the TabFM backend
pip install 'lazy-tfm[tabicl]'    # + the TabICLv2 backend
pip install 'lazy-tfm[tabpfn]'    # + the TabPFN backend (v2 to v3.5)
pip install 'lazy-tfm[limix]'     # + the LimiX-2 backend (its code installs separately)
pip install 'lazy-tfm[qp]'        # + qp interoperability, for RAIL
pip install 'lazy-tfm[all]'       # + everything above
```

Pretrained weights are not bundled: they are fetched from the Hugging Face Hub
at the first `fit` and cached thereafter. TabFM's classification checkpoint is
~6.6 GB and carries a **non-commercial** licence from Google; TabICLv2's is
~100 MB (BSD-3-Clause); the TabPFN checkpoints run from ~41 MB to ~880 MB and
are **non-commercial** from Prior Labs, except `v2`, which is Apache-2.0 with an
attribution clause; LimiX-2's is ~1.6 GB under Stable AI's Apache-2.0-based
licence, which requires the attribution "Built with StableAI LimiX". Every licence is quoted in `lazy.CHECKPOINTS`. Warm the
cache ahead of time with `lazy.download_checkpoint("tabfm")`.

## Use

```python
import numpy as np

import lazy
from lazy import LazyModel, Grid
from lazy.datasets import fetch_dc1

train, test = fetch_dc1(split=True)        # or fetch_dc1() for both, concatenated
# 391k test rows in all, sorted by redshift, so take a random 20k.
test = test.take(np.random.default_rng(0).choice(len(test), 20_000, replace=False))
X_train, X_test = train.features("mag-color"), test.features("mag-color")
z_test = test.redshift

model = LazyModel()                          # TabPFN-3.5, the default; wants a GPU
model.fit(X_train, train.redshift)

pdfs = model.predict_proba(X_test, Grid.linear(0, 2, 200))
z = model.predict(X_test, method="z_peak")  # or z_mean, z_weight, z_median
lo, med, hi = model.predict_quantiles(X_test, [0.16, 0.5, 0.84]).T
print(model.evaluate(X_test, z_test, z_grid=lazy.DC1_GRID))
```

No GPU? TabPFN is slow on a CPU and refuses contexts above 5,000 rows there;
`LazyModel("tabicl")` is small, BSD-licensed and quick on a laptop, and
`LazyModel("tabpfn", version="v3.5-fast")` is TabPFN's distilled checkpoint,
usable on a CPU for contexts up to 5,000 rows. The
[supported models](https://lazy-tfm.readthedocs.io/en/latest/guide/models.html)
page compares them all.

`LazyModel(name, ...)` picks the backend by name, so switching is a string
change; the concrete classes (`TabPFNBarDistribution`, `LimiXBarDistribution`,
`TabICLQuantile`, `TabFMHistogram`) are importable and identical. The API is scikit-learn's, except `predict_proba` returns a
density on a grid of the target rather than class probabilities — the natural
output of a model that knows its own uncertainty. `get_params`/`set_params`/`clone` work, so models drop
into scikit-learn pipelines and search objects unmodified.

**`z_grid` belongs to the prediction, not the fit.** Nothing about fitting
depends on the output binning — these models place their internal bins by the
distribution of the context targets, and the grid only enters at the final exact
rebinning step — so one fitted model answers on as many grids as you like
without refitting:

```python
import numpy as np

model.predict_proba(X_test)                                  # the native grid
model.predict_proba(X_test, Grid.linear(0, 3, 300))
model.predict_proba(X_test, np.linspace(0.005, 2.995, 300))  # or bin centres
```

With no grid a model answers on its *native* grid, the one it thinks in (for
TabPFN and LimiX-2, their 5,000 buckets), so nothing is lost to rebinning.
`predict_distribution(X)` returns that native answer itself, with exact
`pdf`/`cdf`/`ppf`/`mean`/`interval`/`rvs` in the vocabulary of scipy.stats and
LSST DESC's qp, and `.to_qp()` for RAIL.

| Backend  | Method                                                           | Versions | Weights  |
| -------- | ---------------------------------------------------------------- | -------- | -------- |
| `tabpfn` | Bucket masses of the bar distribution | `v2` … `v3.5` | 41 MB – 880 MB |
| `limix`  | Bucket masses of LimiX-2's 5,000-bucket head | `v2` | ~1.6 GB |
| `tabicl` | Quantiles of an in-context regression head | `v2` | ~100 MB |
| `tabfm`  | Hierarchy of in-context classifiers over equal-mass bins of the target | `v1.0` | ~6.6 GB  |

**One set of parameters, with one set of defaults, on every model.**
`n_estimators`, `transforms` (a shared vocabulary — `power`, `quantile`,
`robust`, … — plus recipes such as `"limix"`; `"auto"` is each model's own
recipe, pinned in `lazy`), `feature_shuffle`, `bag_size` (per-member subsets of
the context), `kv_cache` (process the context once; on by default),
`random_state`, `chunk_size`, `softmax_temperature`, `mixed_precision`,
`outlier_threshold` and `z_grid` mean the same thing on every backend. Each is
translated to the model's own machinery where it has it and built around the
model where it does not, every other upstream setting that changes an answer is
pinned, and registering a new backend without them fails. LimiX-2 degrades above ~20,000 context rows and warns when
a larger context arrives without `bag_size`.

Memory is bounded by `chunk_size` on every backend, and chunking is exact: a
query row's answer never depends on which other query rows share its chunk
(bit for bit on TabPFN and TabICL, to float rounding on LimiX-2).

### For TabFM, install its repository build

`pip install 'lazy-tfm[tabfm]'` installs TabFM's PyPI release, which works
but has no KV-cache API: every chunk of query rows re-encodes the whole training
context, about **13.7 ms per member-row against 0.53 ms** on the cached path,
roughly **26× the compute** for identical answers. For anything beyond a few
thousand query rows, install the repository build as well:

```bash
pip install 'lazy-tfm[tabfm]'
pip install --force-reinstall --no-deps 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm@fbb665569425fd2f490c6576b3af967876fe11ff'
```

`--force-reinstall` matters: the repository build calls itself 1.0.1, like the
release, so without it pip keeps the release and changes nothing.
A checkout of this repository gets the repository build automatically (`uv sync
--extra all`), pinned through `[tool.uv.sources]`. On the release build the
backend warns at `fit` with a `TabFMPerformanceWarning` rather than silently
taking 26× longer. To check which you have:

```python
from lazy.models._icl_stream import streaming_available
streaming_available()   # True means the fast path is in use
```

Pass `kv_cache=False` to choose the slow path deliberately and silence the
warning.

### For LimiX-2, install its code

LimiX is not on PyPI. The `limix` extra brings its dependencies; the code
comes from the repository, at the commit this package was validated on (or
from a checkout named by `LAZY_LIMIX_SRC`):

```bash
pip install 'lazy-tfm[limix]'
pip install 'LimiX @ git+https://github.com/limix-ldm-ai/LimiX@516bf396333feb3198cf7aff8a6c10421f218e24'
```

Upstream LimiX-2 has no key/value cache and its answers depend on how the
queries are chunked; the backend runs its network with a ported cache,
context-only preprocessing and a dedicated random generator, which removes
both. LimiX's network needs `triton`, so the backend is Linux-only for now.
Built with StableAI LimiX.

They all write onto any `Grid` you ask for — any number of bins, any
spacing, any range. (TabFM's ten-class ceiling constrains its internal
hierarchy, never your output grid.)

### Progress bars

Every backend draws a `tqdm` bar while it predicts, counting whatever that
model's loop actually iterates over. For 20,000 DC1 test galaxies on a
1,000-galaxy context (one GB10 GPU):

```
TabPFN v3.5: 100%|██████████| 20000/20000 [00:15<00:00, 1279.46row/s, buckets=5000, context=1000]
TabICL v2: 100%|██████████| 20000/20000 [00:10<00:00, 1968.82row/s, context=1000, quantiles=999]
TabFM v1.0 (20,000 rows): 100%|██████████| 33/33 [08:47<00:00, 16.00s/stage, context=34, dither=3/3, level=fine 10/10]
```

TabPFN, LimiX and TabICL answer in chunks of query rows, so their bars count
rows. TabFM is a hierarchy of in-context classifications, each over every
query row, so its bar counts *stages* — `n_dither × (1 + n_coarse_bins)` of
them — and shows which dither and level is running on how many context rows.

`progress="auto"` (the default) shows the bar on a terminal or in a notebook and
stays silent when output goes to a file, so batch logs stay clean. Pass
`progress=True` to force it on or `progress=False` to turn it off. The bar goes
to stderr; `verbose=True` log messages go to stdout.

## The harder benchmark: a biased training set

DC1 hands every training galaxy a redshift. Real spectroscopic samples do not
look like that — they are bright, incomplete, and cut in redshift by which
features fall in the observed window — and that mismatch, not the estimator, is
usually what limits a survey's redshifts. `fetch_dc1_biased` merges both DC1
files and cuts the realistic case out of them:

```python
from lazy.datasets import fetch_dc1_biased

split = fetch_dc1_biased()          # 35,011 biased / 10,000 calibration / 19,383 test
split.biased                        # what a HSC-like campaign would have got
split.calibration                   # representative, yours to repair the model with
split.test                          # representative, held out from all of it
print(split.summary())              # each subset against the parent catalogue
```

Both files are merged and reshuffled, then cut in two: the first part is run
through the selection and what it keeps is the training set, the second is
divided at random into the calibration sample and the test set. Where the cut
falls is solved for, not chosen, because the selection keeps a near-fixed
fraction of whatever it sees — `n_train` is the knob. The calibration sample
comes *out* of the hold-out rather than on top of it, so no galaxy is both given
to a model and scored on.

`n_train` cannot reach the DC1 training file's 43,486: the selection keeps 8.66
per cent of a representative sample, so all 434,476 galaxies yield at most
37,626, and that with nothing left to test on. 35,000 is the largest round
number that still leaves a usable hold-out.

`control=True` adds `split.unbiased`, exactly as many galaxies drawn at random
from the same pool — the control that separates the selection from the sample
size. It is drawn last, so asking for it changes nothing else in the split.

The selection is a port of RAIL's HSC `GridSelection`: each galaxy is kept with
the HSC spectroscopic success rate of its (i, g−z) pixel, below a
colour-dependent redshift ceiling. It is reproduced galaxy-for-galaxy against
the runs this work reports. It also acts on any photometry, so a catalogue of
your own can be biased the same way:

```python
from lazy.selection import grid_selection, selection_summary

keep, diagnostics = grid_selection(catalog.raw, catalog.redshift)
print(selection_summary(keep, catalog.raw["I"], catalog.redshift))
#  i in [16.0, 20.0)  ...  fraction 0.661
#  i in [24.0, 25.3)  ...  fraction 0.0005
```

`diagnostics` carries each galaxy's pixel success rate and redshift ceiling —
everything the selection knew, for a method that tries to estimate it back.

## Which model, exactly

A backbone is a family, not a model, so every backend takes a `version` and each
version is a separately pinned checkpoint:

```python
lazy.list_versions("tabpfn")
# ['v2', 'v2.5', 'v2.6', 'v3', 'v3.5', 'v3.5-fast']

model = LazyModel("tabpfn", version="v2.5")
model.name_          # 'tabpfn:v2.5' -- also the label evaluate() puts on its row
```

Sweeping `version` compares model versions on equal terms, and a results table
says which one produced each row rather than only which family. What actually
answered is recorded on the fitted model, ready to be written out beside the
numbers:

```python
model.fit(X_train, z_train).provenance_
# {'backend': 'tabpfn', 'version': 'v2.5',
#  'repo_id': 'Prior-Labs/tabpfn_2_5',
#  'filename': 'tabpfn-v2.5-regressor-v2.5_default.ckpt',
#  'revision': '6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b',
#  'package': 'tabpfn 9.0.0', 'lazy': 'lazy-tfm 0.1.0', 'device': 'cuda',
#  'n_estimators': 8, 'transforms': ['auto', ...], 'kv_cache': True, ...}
```

It holds the weights, the code that read them and the ensemble that answered,
and no local paths, so it means the same thing on another machine and
survives a trip through JSON. The rest of the settings are in `get_params()`. `CHECKPOINTS` is keyed
`"backend:version"` and is the authority on what each name loads;
`download_checkpoint("tabpfn", "v2.5")` warms exactly that one.

Versions are not interchangeable. TabPFN `v2` is pretrained for at most 10,000
context rows and `v2.5` for 50,000, against a million for `v3`; `v2` is also the
only one under a commercial-use licence. `lazy.get_checkpoint(name, version)`
carries both notes.

## Caches and offline use

Nothing platform-specific is baked in: the wheel is pure Python
(`py3-none-any`), the lockfile resolves for macOS/Linux/Windows on Python
3.12–3.14, and the device is chosen at run time (`device="auto"`).

| Cache | Default | Override |
| ----- | ------- | -------- |
| Pretrained weights | `~/.cache/huggingface/hub` | `HF_HOME` |
| Benchmark catalogues | `~/.cache/lazy-tfm` | `LAZY_DATA_HOME`, else `XDG_CACHE_HOME` |

For a compute node with no network, warm both on a login node first, then run
with `download_checkpoint`/`is_cached` and `fetch_dc1(..., download_if_missing=False)`
so a missing file fails immediately instead of hanging. The ~14 MB HSC selection
grid caches beside the catalogues and `fetch_hsc_grid` takes the same flag. The
checkpoint a model loads is the one `download_checkpoint` fetches — the pinned revision is handed
to the backend rather than left to its default.

Memory is bounded by default on every backend via `chunk_size` (8,192 query
rows), and a row's answer does not depend on which rows share its chunk, so the
default costs nothing.

## What's in the box

| Module           | Contents                                                                     |
| ---------------- | ---------------------------------------------------------------------------- |
| `lazy.models`    | `LazyModel`, the concrete backends, and the name registry                    |
| `lazy.grid`      | `Grid`: binning, normalisation, mass-conserving rebinning            |
| `lazy.metrics`   | LSST DESC PZ Data Challenge point and PDF metrics, and `summarize`           |
| `lazy.selection` | The HSC spectroscopic selection function, for biasing a catalogue of your own |
| `lazy.plotting`  | Publication figure style, and the standard diagnostic figures                |
| `lazy.datasets`  | The DC1 catalogue (download, checksum, cache), `Catalog.features`, `fetch_dc1_biased` |

## Develop

```console
git clone https://github.com/biprateep/lazy-tfm
cd lazy-tfm
uv sync                  # locked environment, including dev tooling
uv run pre-commit install
uv run pytest
```

The test suite needs neither a GPU nor a checkpoint. See
[CONTRIBUTING.md](https://github.com/biprateep/lazy-tfm/blob/main/CONTRIBUTING.md) for the checks CI runs, building the docs and
making a release.

## Repository layout

This repository is the library alone: the public, installable package, its
tests and its documentation. The production runs and figures behind the paper
live in a separate repository that uses this package through its public API.

## Citing

A paper is in preparation. Until it appears, cite the repository and the
checkpoint revisions recorded in `lazy.CHECKPOINTS` — a foundation model's
weights are part of the method. The `provenance_` of a fitted model is that
citation for the run you actually did.

Built from the
[LINCC Frameworks Python Project Template](https://github.com/lincc-frameworks/python-project-template).

## License

MIT, for this code. The pretrained checkpoints carry their own licences; TabFM's
is non-commercial, and LimiX-2's requires the attribution "Built with StableAI
LimiX". The HSC selection grid is redistributed by DESC's
[rail_astro_tools](https://github.com/LSSTDESC/rail_astro_tools) (MIT) and
derives from HSC PDR2 (Aihara et al. 2019); `lazy.selection` downloads it from
there, at a pinned commit, rather than bundling it.
