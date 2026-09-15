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
pip install 'lazy-photoz[all]'       # + both
```

Pretrained weights are not bundled: they are fetched from the Hugging Face Hub
on first prediction and cached thereafter. TabFM's classification checkpoint is
~6.6 GB and carries a **non-commercial** licence from Google; TabICLv2's is
~100 MB (BSD-3-Clause). Warm the cache ahead of time with
`lazy.download_checkpoint("tabfm")`.

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
change; the concrete classes (`TabFMHistogram`, `TabICLQuantile`) are importable
and identical. The API is scikit-learn's, except `predict_proba` returns a
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

| Backend  | Method                                                           | Weights  |
| -------- | ---------------------------------------------------------------- | -------- |
| `tabfm`  | Hierarchy of in-context classifiers over equal-mass redshift bins | ~6.6 GB  |
| `tabicl` | Quantiles of an in-context regression head, differenced onto the grid | ~100 MB |

Neither backend needs a pinned or patched dependency. Memory is bounded by
`chunk_size` on both, which is exact — the in-context stage builds its keys and
values from the context rows alone, so a query row's answer never depends on
which other query rows share its chunk (asserted bit-identical in the test
suite). A `tabfm` build with the KV-cache API, if you have one, is picked up
automatically and avoids re-running the context forward pass per chunk; it is a
speed optimisation, not a correctness requirement.

Both write onto any `RedshiftGrid` you ask for — any number of bins, any
spacing, any range. (TabFM's ten-class ceiling constrains its internal
hierarchy, never your output grid.)

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
so a missing file fails immediately instead of hanging. The checkpoint a model
loads is the one `download_checkpoint` fetches — the pinned revision is handed
to the backend rather than left to its default.

Memory is bounded by default on both backends via `chunk_size` (16384 query
rows), and chunking is numerically exact, so the default costs nothing but a
little repeated context work.

## What's in the box

| Module           | Contents                                                                     |
| ---------------- | ---------------------------------------------------------------------------- |
| `lazy.models`    | `LazyModel`, the concrete backends, and the name registry                    |
| `lazy.grid`      | `RedshiftGrid`: binning, normalisation, mass-conserving rebinning            |
| `lazy.metrics`   | LSST DESC PZ Data Challenge point and PDF metrics, and `summarize`           |
| `lazy.plotting`  | Publication figure style, and the standard diagnostic figures                |
| `lazy.datasets`  | The DC1 catalogue (download, checksum, cache) and `Catalog.features`          |

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

- `main` — this library. The public, installable package.
- `paper` — the production code behind the paper: the exact runs, in the exact
  order, that produce every number and figure in it. See [paper/](paper/).

## Citing

A paper is in preparation. Until it appears, cite the repository and the
checkpoint revisions recorded in `lazy.models.CHECKPOINTS`, which pin the exact
pretrained weights — a foundation model's weights are part of the method.

Built from the
[LINCC Frameworks Python Project Template](https://github.com/lincc-frameworks/python-project-template).

## License

MIT, for this code. The pretrained checkpoints carry their own licences; TabFM's
is non-commercial.
