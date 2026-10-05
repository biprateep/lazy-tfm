# TabPFN

TabPFN, from Prior Labs, is a prior-data fitted network: a transformer
pretrained on synthetic datasets drawn from a prior built on structural causal
models, which predicts the targets of the query rows from the context in a
forward pass. For regression it predicts the probability of each of 5,000
buckets of the standardized target (a bar distribution), which LAZY keeps as
the model's native distribution. LAZY supports six checkpoints, from TabPFN v2
to TabPFN-3.5 and its fast variant, and uses TabPFN-3.5 by default.

```python
model = lazy.LazyModel("tabpfn")                        # TabPFN-3.5
model = lazy.LazyModel("tabpfn", version="v3.5-fast")   # or any version below
```

**Papers:** TabPFN v1 ([Hollmann et al.
2023](https://arxiv.org/abs/2207.01848)), TabPFN v2 ([Hollmann et al. 2025,
Nature](https://doi.org/10.1038/s41586-024-08328-6)), TabPFN-2.5 ([Grinsztajn
et al. 2025](https://arxiv.org/abs/2511.08667)), TabPFN-3 ([Grinsztajn et al.
2026](https://arxiv.org/abs/2605.13986)) and TabPFN-3.5 ([Jäger et al.
2026](https://arxiv.org/abs/2609.17895)). **Code:**
[PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN). **Documentation:**
[docs.priorlabs.ai](https://docs.priorlabs.ai). **Weights:** [Prior-Labs on
Hugging Face](https://huggingface.co/Prior-Labs).

## Versions

Each version is a separately pinned checkpoint with its own pretraining
limits, which TabPFN enforces unless `ignore_pretraining_limits=True`. On a
CPU it also refuses contexts above 1,000 rows (v2 to v2.6) or 5,000 rows (v3
and later).

| `version` | Layers | Max. context rows | Max. features | License of the weights |
| --------- | -----: | ----------------: | ------------: | ---------------------- |
| `"v2"`        | 12 | 10,000    | 500    | Apache-2.0 with attribution |
| `"v2.5"`      | 18 | 50,000    | 2,000  | non-commercial |
| `"v2.6"`      | 24 | 100,000   | 2,000  | non-commercial |
| `"v3"`        | 24 | 1,000,000 | 2,000  | non-commercial |
| `"v3.5"`      | 24 | 1,000,000 | 20,000 | non-commercial |
| `"v3.5-fast"` | 8  | 1,000,000 | 20,000 | non-commercial |

The limits are the ones stored in each checkpoint. The model cards and reports
quote smaller ones for some versions (e.g., 50,000 rows for v2.6).

## Defaults

The table lists the creators' defaults for TabPFN-3.5 in their
`TabPFNRegressor` and the checkpoint's own inference configuration, next to
LAZY's. Each TabPFN version carries its own preprocessing recipe, and under
`transforms="auto"` LAZY writes each one out explicitly rather than inheriting
it from the installed `tabpfn` package.

| Setting | Creators' default (v3.5) | LAZY default |
| ------- | ------------------------ | ------------ |
| Ensemble members | 8 (4 for v3.5-fast), raised automatically up to 32 for wide tables | `n_estimators=8`, exactly, for every version |
| Feature preprocessing | the version's recipe (v3.5: none, features shuffled) | the same recipe (`transforms="auto"`) |
| Feature shuffling | on | on (`feature_shuffle=True`) |
| Softmax temperature | 1.0 for v3.5 and v3.5-fast, 0.9 before | the same (`softmax_temperature="auto"`) |
| Outlier clipping | soft clip at 12 standard deviations for v3.5, none before | the same (`outlier_threshold="auto"`) |
| Target transforms | none for half the members, a safe power transform for the other half | the same |
| Categorical columns | detected automatically | off: every column is numeric |
| Key/value cache | off; int8 when on | on, at full precision (`kv_cache=True`) |
| Precision | autocast on a GPU, sometimes bfloat16 on a CPU | autocast on a GPU, float32 on a CPU |
| Random seed | 0 | `random_state=0` |
| Row subsampling | none | none (`bag_size=None`) |

## Built-in standardizations

TabPFN standardizes both the features and the target. The target is z-scored
with the mean and standard deviation of the context, and the predicted bucket
probabilities are mapped back to the original units. Before the network, each
ensemble member removes constant columns, applies its version's feature
transform (none for v3.5, quantile or squashing transforms for earlier
versions), adds a "fingerprint" feature that identifies duplicate rows,
shuffles the columns and soft-clips outliers. Inside the network, missing and
infinite values get indicator features and are imputed with the context mean,
and every feature is z-scored on the context and clipped at 100 standard
deviations. TabPFN-3.5 also adds each value's empirical rank within the
context as an extra input. LAZY keeps all of these.

## Where LAZY differs

LAZY runs every version with the same parameters and the same defaults as the
other models, so a few settings differ from the creators'. We run exactly
`n_estimators` members, eight by default, including for TabPFN-3.5-fast, whose
checkpoint asks for four, and we never raise the count for wide tables as
TabPFN does, so that a member count means the same thing for every model. We
also turn off TabPFN's automatic detection of categorical columns (which
treats a numeric column with few distinct values as categorical), so every
column is numeric, as in every other model. We cache the context's keys and
values at full precision, which makes repeated predictions faster and changes
them only by rounding, and we always compute in float32 on a CPU.
LAZY's density is also constant within each bucket, while TabPFN gives its two
outermost buckets half-normal tails; this only matters about 128 standard
deviations from the mean of the context targets. When `transforms` is set
explicitly rather than to `"auto"`, LAZY also turns off the fingerprint
feature, the dimensionality reduction and the target transform, so that the
named transforms are the only ones applied.
