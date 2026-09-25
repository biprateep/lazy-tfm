# Clusters and offline use

Nothing about where things are cached is baked into the package, and the device
is chosen at run time, so a laptop and a cluster node run the same code.

| What                 | Default location                                        | Override                             |
| -------------------- | ------------------------------------------------------- | ------------------------------------ |
| Pretrained weights   | `~/.cache/huggingface/hub`                              | `HF_HOME`                            |
| Benchmark catalogues | `$XDG_CACHE_HOME/lazy-tfm`, else `~/.cache/lazy-tfm` | `LAZY_DATA_HOME`, else `XDG_CACHE_HOME` |

Compute nodes often have no outbound network. Warm both caches on a login node
first, then run with downloads disabled, so that a missing file is an immediate
error rather than a hang:

```python
from lazy import download_checkpoint, is_cached
from lazy.datasets import fetch_dc1, fetch_hsc_grid

download_checkpoint("tabpfn", "v3.5")   # on the login node
fetch_dc1()
fetch_hsc_grid()                        # only for fetch_dc1_biased

is_cached("tabpfn", "v3.5")             # on the compute node: True
fetch_dc1(download_if_missing=False)
```

Setting `HF_HUB_OFFLINE=1` on the compute node has the same effect for the
weights. The checkpoint a model loads is exactly the one
{func}`~lazy.download_checkpoint` fetches: the pinned revision is passed to the
backend rather than left to its own default, so warming the cache cannot
prefetch the wrong file.
