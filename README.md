# LAZY

**L**azy but **A**ccurate photo-**Z** for **Y**inz — photometric redshift PDFs
from pretrained tabular foundation models.

[![Unit test and code coverage](https://github.com/biprateep/lazy-photoz/actions/workflows/testing-and-coverage.yml/badge.svg)](https://github.com/biprateep/lazy-photoz/actions/workflows/testing-and-coverage.yml)
[![Documentation](https://readthedocs.org/projects/lazy-photoz/badge/?version=latest)](https://lazy-photoz.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

The models are pretrained and never fine-tuned. You hand them labelled galaxies
as *context* and they answer queries in one forward pass — no training loop, no
hyper-parameter search, no per-survey retraining. Hence lazy.

> **Pre-release.** The API is still moving and the paper numbers are not final.
> Pin a commit if you depend on this.

## Install

```console
pip install lazy-photoz              # grid, metrics, plots, datasets
pip install 'lazy-photoz[tabfm]'     # + the TabFM backend
pip install 'lazy-photoz[tabicl]'    # + the TabICLv2 backend
pip install 'lazy-photoz[tabpfn]'    # + the TabPFN-3 backend
pip install 'lazy-photoz[all]'       # + all three
```

Pretrained weights are not bundled: they are fetched from the Hugging Face Hub
on first prediction and cached thereafter. TabFM's classification checkpoint is
~6.6 GB and carries a **non-commercial** licence from Google; TabICLv2's is
~100 MB (BSD-3-Clause); the TabPFN checkpoints run from ~41 MB to ~880 MB and
are **non-commercial** from Prior Labs, except `v2`, which is Apache-2.0 with an
attribution clause. Every licence is quoted in `lazy.CHECKPOINTS`. Warm the
cache ahead of time with `lazy.download_checkpoint("tabfm")`.

## Use

```python
from lazy import LazyModel, RedshiftGrid
from lazy.datasets import fetch_dc1

train, test = fetch_dc1(split=True)        # or fetch_dc1() for both, concatenated
X_train, X_test = train.features("mag-color"), test.features("mag-color")

model = LazyModel("tabfm", n_estimators=4, n_dither=3)
model.fit(X_train, train.redshift)

pdfs = model.predict_proba(X_test, RedshiftGrid.linear(0, 2, 200))
z = model.predict(X_test, method="z_peak")  # or z_mean, z_weight, z_median
print(model.evaluate(X_test, test.redshift))
```

`LazyModel(name, ...)` picks the backend by name, so switching is a string
change; the concrete classes (`TabFMHistogram`, `TabICLQuantile`,
`TabPFNBarDistribution`) are importable and identical. The API is scikit-learn's, except `predict_proba` returns a
density on a redshift grid rather than class probabilities — that is the natural
output of a photo-z model. `get_params`/`set_params`/`clone` work, so models drop
into scikit-learn pipelines and search objects unmodified.

**`z_grid` belongs to the prediction, not the fit.** Nothing about fitting
depends on the output binning — these models place their internal bins by the
quantiles of the context redshifts, and the grid only enters at the final exact
rebinning step — so one fitted model answers on as many grids as you like
without refitting:

```python
model.predict_proba(X_test, RedshiftGrid.linear(0, 3, 300))
model.predict_proba(X_test, np.linspace(0.005, 2.995, 300))  # or bin centres
```

| Backend  | Method                                                           | Versions | Weights  |
| -------- | ---------------------------------------------------------------- | -------- | -------- |
| `tabfm`  | Hierarchy of in-context classifiers over equal-mass redshift bins | `v1.0` | ~6.6 GB  |
| `tabicl` | Quantiles of an in-context regression head, differenced onto the grid | `v2` | ~100 MB |
| `tabpfn` | Bucket masses of the bar distribution, rebinned onto the grid | `v2` … `v3.5` | 41 MB – 880 MB |

Every backend runs correctly on its released dependency. Memory is bounded by
`chunk_size` on all of them, which is exact — the in-context stage builds its keys and
values from the context rows alone, so a query row's answer never depends on
which other query rows share its chunk (asserted bit-identical in the test
suite).

### TabFM's KV cache is pinned for you

`lazy-photoz[tabfm]` installs TabFM from its repository at a pinned commit, not
from PyPI, and that is deliberate rather than fussiness. The PyPI release has no
KV-cache API, so every chunk of query rows re-encodes the entire training
context: about **13.7 ms per member-row against 0.53 ms** on the cached path,
roughly **26× the compute**.

Both paths give the same answers, which is exactly what makes it dangerous — the
only symptom is a prediction that takes a day instead of an hour, and there is
nothing wrong with the result to tip you off. This project lost 17 GPU-hours to
it on a run we had every reason to believe was simply large.

A version specifier cannot express the requirement, because the repository build
also calls itself `1.0.1`; only the URL distinguishes them. Nothing is needed
from you:

```bash
uv sync --extra all                  # or: pip install 'lazy-photoz[tabfm]'
```

If you end up on the release build anyway — a stale lock, a deliberate override,
or an index that strips direct URLs — the backend says so at `fit` time with a
`TabFMPerformanceWarning` rather than silently taking 26× longer. To check:

```python
from lazy.models._icl_stream import streaming_available
streaming_available()   # True means the fast path is in use
```

Pass `inference="predict_proba"` to choose the slow path deliberately and
silence the warning.

The same concern has a different name on `tabpfn`: `fit_mode="fit_with_cache"`
keeps the context's key/value tensors so each chunk of queries skips the
context forward pass. It costs memory and is worth it whenever the query set is
much larger than the context.

They all write onto any `RedshiftGrid` you ask for — any number of bins, any
spacing, any range. (TabFM's ten-class ceiling constrains its internal
hierarchy, never your output grid.)

### Progress bars

Every backend draws a `tqdm` bar while it predicts, counting whatever that
model's loop actually iterates over. For 20,000 DC1 test galaxies on a
1,000-galaxy context (one GB10 GPU):

```
TabPFN v3.5: 100%|██████████| 20000/20000 [00:15<00:00, 1279.46gal/s, buckets=5000, context=1000]
TabICL v2: 100%|██████████| 20000/20000 [00:10<00:00, 1968.82gal/s, context=1000, quantiles=999]
TabFM v1.0 (20,000 gal): 100%|██████████| 33/33 [08:47<00:00, 16.00s/stage, context=34, dither=3/3, level=fine 10/10]
```

TabPFN and TabICL answer in chunks of query rows, so their bars count
galaxies. TabFM is a hierarchy of in-context classifications, each over every
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
# {'backend': 'tabpfn', 'version': 'v3',
#  'repo_id': 'Prior-Labs/tabpfn_3',
#  'filename': 'tabpfn-v3-regressor-v3_default.ckpt',
#  'revision': '24a16a89d245878b846555110985634aa2e656d7',
#  'package': 'tabpfn 9.0.0', 'lazy': 'lazy-photoz 0.1.0.dev0', 'device': 'cuda'}
```

It holds the weights *and* the code that read them, and no local paths, so it
means the same thing on another machine and survives a trip through JSON.
Settings are not in it — `get_params()` has those. `CHECKPOINTS` is keyed
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
| Benchmark catalogues | `~/.cache/lazy-photoz` | `LAZY_DATA_HOME`, else `XDG_CACHE_HOME` |

For a compute node with no network, warm both on a login node first, then run
with `download_checkpoint`/`is_cached` and `fetch_dc1(..., download_if_missing=False)`
so a missing file fails immediately instead of hanging. The ~14 MB HSC selection
grid caches beside the catalogues and `fetch_hsc_grid` takes the same flag. The
checkpoint a model loads is the one `download_checkpoint` fetches — the pinned revision is handed
to the backend rather than left to its default.

Memory is bounded by default on every backend via `chunk_size` (16384 query
rows), and chunking is numerically exact, so the default costs nothing but a
little repeated context work.

## What's in the box

| Module           | Contents                                                                     |
| ---------------- | ---------------------------------------------------------------------------- |
| `lazy.models`    | `LazyModel`, the concrete backends, and the name registry                    |
| `lazy.grid`      | `RedshiftGrid`: binning, normalisation, mass-conserving rebinning            |
| `lazy.metrics`   | LSST DESC PZ Data Challenge point and PDF metrics, and `summarize`           |
| `lazy.selection` | The HSC spectroscopic selection function, for biasing a catalogue of your own |
| `lazy.plotting`  | Publication figure style, and the standard diagnostic figures                |
| `lazy.datasets`  | The DC1 catalogue (download, checksum, cache), `Catalog.features`, `fetch_dc1_biased` |

## Develop

```console
git clone https://github.com/biprateep/lazy-photoz
cd lazy-photoz
uv sync                  # locked environment, including dev tooling
uv run pre-commit install
uv run pytest
```

`uv sync --locked` is what CI runs; if it fails, `uv lock` and commit the
result. The test suite needs neither a GPU nor a checkpoint.

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
is non-commercial. The HSC selection grid is redistributed by DESC's
[rail_astro_tools](https://github.com/LSSTDESC/rail_astro_tools) (MIT) and
derives from HSC PDR2 (Aihara et al. 2019); `lazy.selection` downloads it from
there, at a pinned commit, rather than bundling it.
