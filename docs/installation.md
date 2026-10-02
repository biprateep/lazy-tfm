# Installation

`lazy-tfm` needs Python 3.12 or newer. The core package has no deep-learning
dependencies, and each foundation-model backend is installed as an optional
extra:

```console
pip install lazy-tfm              # the grid, metrics, plots and datasets
pip install 'lazy-tfm[tabpfn]'    # + the TabPFN backend (v2 to v3.5)
pip install 'lazy-tfm[tabicl]'    # + the TabICLv2 backend
pip install 'lazy-tfm[tabfm]'     # + the TabFM backend
pip install 'lazy-tfm[limix]'     # + the LimiX-2 backend's dependencies
pip install 'lazy-tfm[qp]'        # + qp interoperability (to_qp / from_qp, for RAIL)
pip install 'lazy-tfm[all]'       # + everything above
```

With [uv](https://docs.astral.sh/uv/), use `uv add 'lazy-tfm[tabpfn]'` in a
project or `uv pip install` in an environment. The wheel is pure Python, so it
installs the same way on Linux, macOS and Windows.

## One `lazy` per environment

The package is installed as `lazy-tfm` and imported as `lazy`. An unrelated
PyPI package, [`lazy`](https://pypi.org/project/lazy/) (lazy attributes for
Python objects), also installs a top-level `lazy`, and two packages cannot
share an import name in one environment, since the one installed last
overwrites the files of the other. Aliasing the import (`import lazy as lz`)
does not help, as there is only one `lazy` on disk. If both are installed,
`import lazy` raises an `ImportNameWarning`, or, when the other package was
installed last, `lazy.LazyModel` does not exist at all. In either case, the
solution is an environment without the other package:

```console
pip uninstall lazy lazy-tfm
pip install lazy-tfm
```

## PyTorch and GPUs

The backends run on PyTorch, which the extras pull in. The default PyTorch
wheel from PyPI suits most Linux machines with an NVIDIA GPU. For a specific
CUDA version, or for ROCm, install PyTorch first by following the instructions
on [pytorch.org](https://pytorch.org/get-started/locally/), and then install
`lazy-tfm`.

Every model chooses its device when it is fitted (`device="auto"` selects
CUDA if it is available and the CPU otherwise), and passing `device="cuda"`
explicitly on a machine without CUDA raises an error rather than quietly
running on the CPU. Most of the models need a GPU to be practical. TabPFN-3.5,
the default, is slow on a CPU and, like every TabPFN from v3 on, refuses
contexts larger than 5,000 rows there, so `lazy` warns at `fit` when it finds
no GPU. On a laptop, we recommend TabICL, which is small, BSD-licensed and
quick on a CPU:

```python
# pip install 'lazy-tfm[tabicl]'
model = lazy.LazyModel("tabicl")
```

or the smaller and faster checkpoint of TabPFN, TabPFN-3.5-fast, which runs
on a CPU at more than twice the speed of TabPFN-3.5, though less accurately,
with a context of at most 5,000 rows:

```python
model = lazy.LazyModel("tabpfn", version="v3.5-fast")
```

Alternatively, TabPFN-3.5 can be given a context of at most 5,000 rows drawn
at random, or run with `ignore_pretraining_limits=True` at the cost of a long
wait.
{doc}`models/index` gives the hardware needs of each model.

## TabFM: install the repository build too

The PyPI release of TabFM works, but it has no KV-cache API, so every chunk
of query rows re-encodes the whole training context. This costs about 26
times the compute of the cached path, for identical answers. For more than a
few thousand query rows, we therefore recommend installing the repository
build as well:

```console
pip install 'lazy-tfm[tabfm]'
pip install --force-reinstall --no-deps 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm@fbb665569425fd2f490c6576b3af967876fe11ff'
```

The `--force-reinstall` flag is necessary, because the repository build calls
itself 1.0.1, like the release, and without the flag pip keeps the release
and changes nothing.

On the release build, `TabFMHistogram` warns at `fit` with a
`TabFMPerformanceWarning`, and `lazy.models._icl_stream.streaming_available()`
returns `True` when the fast path is in use.

## LimiX-2: install its code separately

The code of LimiX is not on PyPI, and PyPI does not allow a package to depend
on a repository, so the `limix` extra installs only its dependencies. The
code itself should be installed at the commit `lazy` was validated on:

```console
pip install 'lazy-tfm[limix]'
pip install 'LimiX @ git+https://github.com/limix-ldm-ai/LimiX@516bf396333feb3198cf7aff8a6c10421f218e24'
```

or the repository can be cloned and `LAZY_LIMIX_SRC` set to the checkout.
`lazy` loads two parts of it under private module names and never imports
the generic top-level packages of LimiX (`model`, `inference`, `utils`,
`config`). However, a `pip install` still puts those names into the
environment, where they can shadow the modules of another package, so in a
shared environment we recommend the clone and `LAZY_LIMIX_SRC`. Another commit
works with a `LimiXSourceWarning`. The LimiX network imports `triton`, which
only has Linux wheels, so the backend is Linux-only for now.

## Pretrained checkpoints

Weights are not bundled with the package. They are downloaded from the Hugging
Face Hub the first time a model is fitted or predicts, and cached from then
on. None of the repositories is gated, so no Hugging Face account or token is
needed.

| Backend  | Versions        | Size           | License of the weights                          |
| -------- | --------------- | -------------- | ----------------------------------------------- |
| `limix`  | `v2`            | ~1.6 GB        | StableAI LimiX, **non-commercial** (with the attribution "Built with StableAI LimiX") |
| `tabfm`  | `v1.0`          | ~6.6 GB        | Google, **non-commercial**                      |
| `tabicl` | `v2`            | ~100 MB        | BSD-3-Clause                                    |
| `tabpfn` | `v2` to `v3.5`  | 41 MB – 880 MB | Prior Labs, **non-commercial**, except `v2` (Apache-2.0 with attribution) |

Each version is pinned to a fixed revision, and
{data}`lazy.CHECKPOINTS <lazy.models.CHECKPOINTS>` lists them with the license
text for each. To download the weights ahead of time (e.g., before moving to a
machine without a network):

```python
import lazy

lazy.download_checkpoint("tabpfn", "v3.5")
lazy.is_cached("tabpfn", "v3.5")   # True
```

See {doc}`guide/clusters` for caches and running offline.

## From source

```console
git clone https://github.com/biprateep/lazy-tfm
cd lazy-tfm
uv sync --extra all
```

This installs the package in editable mode with every dependency pinned by
`uv.lock`, including the repository build of TabFM (the code of LimiX still
comes from `LAZY_LIMIX_SRC` or a separate install). See {doc}`contributing`
for the development workflow.
