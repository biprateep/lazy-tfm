# Saving and loading

A fitted model saves with {mod}`pickle` or `joblib`, and the loaded model
gives the same predictions:

```python
import joblib

joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

A `copy.deepcopy` goes through the same steps. Loading a pickle can run any
code, so load only files from a source you trust.

## What the file holds

The file holds the parameters and every fitted attribute, `provenance_`
included. It also holds the context: the rows that each member sees, with
the preprocessing that each member fitted on them. A file therefore carries
the training data, and can go only where that data can go.

The file does not hold the network's weights. After loading, the model
reads them from the pinned checkpoint before it first predicts (TabPFN as
it loads), and downloads the checkpoint if this machine lacks it. To load a
file without a network connection, warm the cache first (see
{doc}`clusters`). A LimiX-2 file also needs LimiX's source, which LAZY loads
before the file (see {doc}`../installation`).

What happens to the key/value cache depends on the backend:

| Backend | The key/value cache |
| ------- | ------------------- |
| TabPFN | In the file. |
| TabICL, LimiX-2, Kumo Tabular | Left out, and rebuilt from the context before the first prediction, on the device where the model loads. LimiX-2 rebuilds within that device's free memory, so some members may predict without the cache, and it issues `PerformanceWarning`. |
| TabFM | None to keep: TabFM fills its cache inside every `predict` call. |

## Loading on another device

A model fitted with `device="auto"` chooses its device again when it loads,
so a model fitted on a GPU predicts on the CPU of a machine without one.
`provenance_` then records the new device and precision. A model fitted with
an explicit device, such as `device="cuda"`, keeps it. A CPU and a GPU can
give different answers for the same inputs and seed (see
[Across devices](reproducibility.md#across-devices)).

## Across versions

Loading does not check which versions of LAZY and of the backend's package
wrote the file; `provenance_` records both. A file written by one version is
not guaranteed to load, or to predict the same, in another. {doc}`../changelog`
lists each change to what a file holds.
