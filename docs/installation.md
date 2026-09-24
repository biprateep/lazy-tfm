# Installation

`lazy-photoz` needs Python 3.12 or newer. The core package has no
deep-learning dependencies; each foundation-model backend is an optional extra:

```console
pip install lazy-photoz              # the grid, metrics, plots and datasets
pip install 'lazy-photoz[tabpfn]'    # + the TabPFN backend (v2 to v3.5)
pip install 'lazy-photoz[tabicl]'    # + the TabICLv2 backend
pip install 'lazy-photoz[tabfm]'     # + the TabFM backend
pip install 'lazy-photoz[all]'       # + all three
```

With [uv](https://docs.astral.sh/uv/), use `uv add 'lazy-photoz[tabpfn]'` in a
project or `uv pip install` in an environment. The wheel is pure Python, so it
installs the same way on Linux, macOS and Windows.

## PyTorch and GPUs

The backends run on PyTorch, which the extras pull in. The default PyTorch
wheel from PyPI suits most Linux machines with an NVIDIA GPU. For a specific
CUDA version, or ROCm, install PyTorch first by following
[pytorch.org](https://pytorch.org/get-started/locally/), then install
`lazy-photoz`.

Every model chooses its device when it is fitted (`device="auto"`: CUDA if
available, otherwise CPU). The models run on CPU, but a foundation model on a
large query set is much faster on a GPU. Passing `device="cuda"` explicitly
raises an error on a machine without one, rather than quietly running on CPU.

## TabFM: install the repository build too

TabFM's PyPI release works, but it has no KV-cache API, so every chunk of query
rows re-encodes the whole training context: about 26 times the compute of the
cached path, for identical answers. For more than a few thousand galaxies,
install the repository build as well:

```console
pip install 'lazy-photoz[tabfm]'
pip install 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm'
```

On the release build, `TabFMHistogram` warns at `fit` with a
`TabFMPerformanceWarning`. `lazy.models._icl_stream.streaming_available()`
returns `True` when the fast path is in use.

## Pretrained checkpoints

Weights are not bundled. They are downloaded from the Hugging Face Hub the
first time a model is fitted or predicts, and cached from then on. None of the
repositories is gated, so no Hugging Face account or token is needed.

| Backend  | Versions        | Size           | Licence of the weights                          |
| -------- | --------------- | -------------- | ----------------------------------------------- |
| `tabfm`  | `v1.0`          | ~6.6 GB        | Google, **non-commercial**                      |
| `tabicl` | `v2`            | ~100 MB        | BSD-3-Clause                                    |
| `tabpfn` | `v2` to `v3.5`  | 41 MB – 880 MB | Prior Labs, **non-commercial**, except `v2` (Apache-2.0 with attribution) |

Each version is pinned to a fixed revision; {data}`lazy.CHECKPOINTS` lists them,
with the licence text for each. To download ahead of time, for example before
moving to a machine without a network:

```python
import lazy

lazy.download_checkpoint("tabpfn", "v3.5")
lazy.is_cached("tabpfn", "v3.5")   # True
```

See {doc}`guide/clusters` for caches and running offline.

## From source

```console
git clone https://github.com/biprateep/lazy-photoz
cd lazy-photoz
uv sync --extra all
```

This installs the package in editable mode with every dependency pinned by
`uv.lock`, including TabFM's repository build. See {doc}`contributing` for the
development workflow.
