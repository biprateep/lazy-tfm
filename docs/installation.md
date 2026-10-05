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

:::{note}
The package is installed as `lazy-tfm` and imported as `lazy`. An unrelated
PyPI package, [`lazy`](https://pypi.org/project/lazy/), also installs a
top-level `lazy`, and the one installed last overwrites the files of the
other, so keep it out of the environment. If both were installed, run
`pip uninstall lazy lazy-tfm` and then `pip install lazy-tfm`.
:::

## PyTorch and GPUs

The backends run on PyTorch, which the extras pull in. The default PyTorch
wheel from PyPI suits most Linux machines with an NVIDIA GPU. However, make sure to check the compatibility of the PyTorch version with your system before installing.

Every model chooses its device when it is fitted (`device="auto"` selects
CUDA if it is available and the CPU otherwise), and passing `device="cuda"`
explicitly on a machine without CUDA will raise an error. 

## Specific backend installation notes

PyPI cannot install everything two of the backends need. After installing
their extras, one command completes them, and is safe to run again:

```console
lazy setup
```

- **TabFM.** The extra installs TabFM's PyPI release, which has no KV-cache
  API and takes about 26 times the compute for the same answers. `lazy setup`
  replaces it with the repository build, at the commit `lazy` was validated
  on. Without it, `fit` warns with a `TabFMPerformanceWarning`. In a uv
  project, `uv sync` would put the release back, so add the build to the
  project instead, in the same command as `lazy-tfm`; `lazy setup` is then
  not needed for TabFM:

  ```console
  uv add 'lazy-tfm[tabfm]' 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm@fbb665569425fd2f490c6576b3af967876fe11ff'
  ```

  In a uv project that already has `lazy-tfm`, `lazy setup` prints the
  `uv add` command to run rather than installing a build `uv sync` would
  undo.
- **LimiX-2.** The code of LimiX is not on PyPI, so `lazy` downloads it
  (~10 MB) at the validated commit on first use, into its cache, without
  installing it. `lazy setup` or `lazy.download_checkpoint("limix")` does
  this ahead of time, for a machine without a network (see
  {doc}`guide/clusters`). To use your own checkout instead, set
  `LAZY_LIMIX_SRC` to it. The backend is Linux-only, because LimiX needs
  `triton`.

## Installation from source

```console
git clone https://github.com/biprateep/lazy-tfm
cd lazy-tfm
uv sync --extra all
```

This installs the package in editable mode with every dependency pinned by
`uv.lock`, the repository build of TabFM included. See {doc}`contributing`
for the development workflow.
