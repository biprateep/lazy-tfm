# Installation

`lazy-tfm` needs Python 3.12 or newer. The core package has no
deep-learning dependencies; each foundation-model backend is an optional extra:

```console
pip install lazy-tfm              # the grid, metrics, plots and datasets
pip install 'lazy-tfm[tabpfn]'    # + the TabPFN backend (v2 to v3.5)
pip install 'lazy-tfm[tabicl]'    # + the TabICLv2 backend
pip install 'lazy-tfm[tabfm]'     # + the TabFM backend
pip install 'lazy-tfm[limix]'     # + the LimiX-2 backend's dependencies
pip install 'lazy-tfm[all]'       # + all four
```

With [uv](https://docs.astral.sh/uv/), use `uv add 'lazy-tfm[tabpfn]'` in a
project or `uv pip install` in an environment. The wheel is pure Python, so it
installs the same way on Linux, macOS and Windows.

## One `lazy` per environment

The package is installed as `lazy-tfm` and imported as `lazy`. An unrelated
PyPI package, [`lazy`](https://pypi.org/project/lazy/) (lazy attributes for
Python objects), also installs a top-level `lazy`, and two packages cannot
share an import name in one environment: the one installed last overwrites
the other's files. Aliasing the import (`import lazy as lz`) does not help,
since there is only one `lazy` on disk. If both are installed, `import lazy`
raises an `ImportNameWarning`, or, when the other package was installed
last, `lazy.LazyModel` does not exist at all. Either way, use an environment
without the other package:

```console
pip uninstall lazy lazy-tfm
pip install lazy-tfm
```

## PyTorch and GPUs

The backends run on PyTorch, which the extras pull in. The default PyTorch
wheel from PyPI suits most Linux machines with an NVIDIA GPU. For a specific
CUDA version, or ROCm, install PyTorch first by following
[pytorch.org](https://pytorch.org/get-started/locally/), then install
`lazy-tfm`.

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
pip install 'lazy-tfm[tabfm]'
pip install 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm'
```

On the release build, `TabFMHistogram` warns at `fit` with a
`TabFMPerformanceWarning`. `lazy.models._icl_stream.streaming_available()`
returns `True` when the fast path is in use.

## LimiX-2: install its code separately

LimiX's code is not on PyPI, and PyPI does not let a package depend on a
repository, so the `limix` extra installs only its dependencies. Install the
code itself at the commit `lazy` was validated on:

```console
pip install 'lazy-tfm[limix]'
pip install 'LimiX @ git+https://github.com/limix-ldm-ai/LimiX@516bf396333feb3198cf7aff8a6c10421f218e24'
```

or clone the repository and set `LAZY_LIMIX_SRC` to the checkout. `lazy` loads
two parts of it under private module names, so LimiX's generic top-level
package names (`model`, `utils`, ...) never shadow anything. Another commit
works with a `LimiXSourceWarning`. LimiX's network imports `triton`, which
arrives with PyTorch's Linux wheels.

## Pretrained checkpoints

Weights are not bundled. They are downloaded from the Hugging Face Hub the
first time a model is fitted or predicts, and cached from then on. None of the
repositories is gated, so no Hugging Face account or token is needed.

| Backend  | Versions        | Size           | Licence of the weights                          |
| -------- | --------------- | -------------- | ----------------------------------------------- |
| `limix`  | `v2`            | ~1.6 GB        | Stable AI License 1.0 (Apache-2.0 with attribution: "Built with StableAI LimiX") |
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
`uv.lock`, including TabFM's repository build (LimiX's code still comes from
`LAZY_LIMIX_SRC` or a separate install). See {doc}`contributing` for the
development workflow.
