# Clusters, caches and offline use

No cache location is baked into the package, and the device is chosen at run
time, so a laptop and a cluster node run the same code. The table lists where
each kind of file is cached by default and the environment variables that
override it:

| What                 | Default location                                        | Override                             |
| -------------------- | ------------------------------------------------------- | ------------------------------------ |
| Pretrained weights   | `~/.cache/huggingface/hub`                              | `HF_HOME`                            |
| Benchmark catalogs   | `$XDG_CACHE_HOME/lazy-tfm`, else `~/.cache/lazy-tfm` | `LAZY_DATA_HOME`, else `XDG_CACHE_HOME` |
| LimiX's source       | `limix/` in the catalogs' directory                     | `LAZY_LIMIX_SRC` (a checkout), else as the catalogs |

Compute nodes often have no outbound network access. In that case, we
recommend warming the caches on a login node first and then running with
downloads disabled, so that a missing file raises an immediate error rather
than causing a hang:

```python
from lazy import download_checkpoint, is_cached
from lazy.datasets import fetch_dc1, fetch_hsc_grid

download_checkpoint("tabpfn", "v3.5")   # on the login node
fetch_dc1()
fetch_hsc_grid()                        # only for fetch_dc1_biased

is_cached("tabpfn", "v3.5")             # on the compute node: True
fetch_dc1(download_if_missing=False)
```

For LimiX-2, `download_checkpoint("limix")` also fetches LimiX's source,
which is not on PyPI, and `is_cached("limix")` checks for both. Setting
`HF_HUB_OFFLINE=1` on the compute node has the same effect for the weights,
and stops the source from being downloaded too. The checkpoint that a model loads is exactly the one
{func}`~lazy.models.download_checkpoint` fetches, because the pinned revision
is passed to the backend rather than left to its own default. Therefore,
warming the cache cannot prefetch the wrong file.
