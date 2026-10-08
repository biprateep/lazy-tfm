# Supported models

Every model in LAZY is a pretrained tabular foundation model used in context
(i.e., `fit` stores the labeled rows and each prediction is a forward pass
over them). The models differ a great deal in size, speed and hardware
requirements.

<!-- The table is written by docs/conf.py from lazy.CHECKPOINTS: see the
"Tables of the models" section there. -->

```{include} _overview.md
```

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
