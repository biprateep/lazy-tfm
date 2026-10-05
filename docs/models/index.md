# Supported models

Every model in LAZY is a pretrained tabular foundation model used in context
(i.e., `fit` stores the labeled rows and each prediction is a forward pass
over them). The models differ a great deal in size, speed and hardware
requirements.

| Model | `LazyModel(...)` | Weights | Parameters | License of the weights | GPU | CPU-friendly |
| ----- | ---------------- | ------: | ---------: | ---------------------- | --- | ------------ |
| TabPFN-3.5 (**default**) | `"tabpfn"` | 0.88 GB | 219 M | Prior Labs, non-commercial | recommended | no |
| TabPFN-3.5-fast | `"tabpfn", version="v3.5-fast"` | 0.33 GB | 84 M | Prior Labs, non-commercial | optional | ≤5,000 rows |
| TabPFN-3 | `"tabpfn", version="v3"` | 0.23 GB | 58 M | Prior Labs, non-commercial | recommended | no |
| TabPFN-2.6 | `"tabpfn", version="v2.6"` | 0.05 GB | 13 M | Prior Labs, non-commercial | optional | ≤1,000 rows |
| TabPFN-2.5 | `"tabpfn", version="v2.5"` | 0.04 GB | 10 M | Prior Labs, non-commercial | optional | ≤1,000 rows |
| TabPFN-2 | `"tabpfn", version="v2"` | 0.04 GB | 11 M | Apache-2.0 + attribution | optional | ≤1,000 rows |
| TabICLv2 | `"tabicl"` | 0.11 GB | 29 M | BSD-3-Clause | optional | **yes** |
| LimiX-2 | `"limix"` | 1.6 GB | 406 M | StableAI LimiX, non-commercial | yes | no (and Linux only) |
| TabFM v1.0 | `"tabfm"` | 6.6 GB | 1.64 B | Google, non-commercial | yes | no |

The weights are downloaded at the first `fit` and cached from then on. The
licenses are summarized here and quoted in `lazy.CHECKPOINTS`; please check
the license of each backend at its original source before use.
{doc}`../guide/clusters` describes caches and offline use.

{doc}`../guide/choosing` describes which model to use for which purpose and
{doc}`../guide/scaling` gives the measured time and memory of each. All models
take the same parameters and give the same outputs; {doc}`../guide/interface`
describes how each implements them.

```{toctree}
:maxdepth: 1

tabpfn
tabicl
tabfm
limix
```
