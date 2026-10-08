# Adding a backend

A backend is one pretrained in-context model behind LAZY's shared interface.
It lives in its own module in `src/lazy/models/` as a subclass of
{class}`~lazy.models.ContextEnsembleEstimator`, which already does everything
the models have in common: it validates the parameters, fetches the pinned
checkpoint, plans the ensemble members, draws their bags, applies the
transforms a model lacks, chunks the queries, shows the progress bar and
combines the members' answers. The subclass declares what its model does
natively, as class attributes, and implements four methods. This page
describes the contract those methods must meet, gives a skeleton module, and
lists everything else a new backend needs, all of which the test suite
checks.

## The contract

A backend declares its model's capabilities as class attributes
(`native_output`, `native_transforms`, `auto_tokens`,
`supports_native_bagging`, `member_combination`, `kv_cache_modes`,
`has_softmax`, `native_outlier_clipping` and the rest, documented on
{class}`~lazy.models.ContextEnsembleEstimator`) and implements
`_import_backend`, `_fit_group`, `_predict_group` and `_native_grid`. Its
constructor takes every uniform parameter as a keyword with the shared
default, stores each one verbatim, and takes nothing else positionally, so
that `cls()` builds a working estimator; `register` refuses a class that falls
short of this.

The base class splits the ensemble into groups of members that one call of
the model can serve together (a member with a transform the model does not
implement, or a bag the model cannot draw itself, is a group of its own).
`_fit_group` receives one group's prepared context and returns a *handle*,
which can be any object: the base class keeps it in `handles_` and passes it
back, unchanged, to `_predict_group` for every chunk of queries. A fit may
also set fitted attributes of its own (TabPFN records its bucket borders,
TabFM resolves its bins and its inference path), lower `kv_cache_` to the mode
that actually ran, and add keys to `provenance_`. It may not change the
members, their rows or their seeds, which the base class has already planned
and recorded. What a backend must do around the whole fit goes in
`_before_fit`, which runs once the package is imported and before any
parameter is validated (TabFM clips the context targets to its bins' support
there), and `_after_fit`, which runs once the fit has succeeded and
`provenance_` is checked (LimiX records each member's pipeline there), rather
than in an override of `_fit`.

`_predict_group` receives a chunk of prepared queries and must return a
{mod}`lazy.distributions` object of the kind `native_output` declares (a
{class}`~lazy.distributions.HistogramDistribution`, or a mixture of them, for
`"histogram"`; a {class}`~lazy.distributions.QuantileDistribution` for
`"quantiles"`), with exactly one row per query. The base class checks both on
every call and raises a `TypeError` or `RuntimeError` naming the backend
otherwise. A query's answer must not depend on the other queries in its
chunk, and a prediction must leave the handle as it found it, so that the
cache and chunking tests can compare answers bit for bit.

`provenance_` records which weights, which code and which ensemble produced a
set of numbers. The default `_load_checkpoint` sets it from
`Checkpoint.provenance` (in `lazy.models._hub`), the base class adds the
resolved uniform settings, and a fit that ends without every key in
`lazy.models._ensemble.REQUIRED_PROVENANCE` raises. A backend that overrides
`_load_checkpoint`, for instance to load its weights lazily, must still
record the checkpoint's keys.

A fitted estimator must survive pickling and give the same predictions
afterwards. A network shared between estimators through `lazy.models._weights`
should not travel with it: a backend that loads its network outside
`_fit_group` keeps it through `_memoized_network`, whose reference the base
class drops on pickling and which loads the network again on first use, so a
pickle does not carry gigabytes of weights (TabFM and LimiX do this). The base
class resolves `device` into `device_` (`"cpu"`, `"cuda"`, `"cuda:<index>"` or
`"mps"`) before the checkpoint is loaded, and sets `mixed_precision_` only on
CUDA; a backend runs on `device_`, in float32 unless `mixed_precision_` is set,
and raises a clear error on a device its model does not support. No current
backend is tested on `"mps"`.

## Hooks for the rest of the package

Code shared by every backend never names one. Where a backend needs more than
the contract above from the rest of the package, it overrides a hook that the
package calls on every registered class, and each hook does nothing by
default. The class attribute `cpu_note` is a clause that the warning about
running on a CPU appends after "runs slowly on CPU" (TabPFN's says how many
context rows it accepts there). The classmethod `prefetch` fetches what the
model needs beside its weights, and {func}`~lazy.models.download_checkpoint` and
{func}`~lazy.models.is_cached` call it, so that a job on a cluster can find
everything in the cache (LimiX downloads its source, which is not on PyPI).
`clear_upstream_caches` empties the caches the upstream package keeps of its
own, and {func}`~lazy.models.clear_model_cache` calls it (TabPFN keeps the last
checkpoint it read in memory). `setup` completes an install that the pip extra
cannot, and `lazy setup` runs it for every backend: it prints a line saying
what it did, skips a backend whose extra is not installed, and returns whether
the backend is complete (TabFM installs its repository build, and LimiX
downloads its source).

## A skeleton

The module below is a starting point for a model called MyModel, whose
package `mymodel` is assumed to offer a regressor that predicts bucket
probabilities. Every name from `mymodel` is a stand-in for the real upstream
API.

```python
"""Target distributions from MyModel's bucket probabilities."""

from __future__ import annotations

from collections.abc import Mapping
import types
from typing import Any

from lazy import _typing
from lazy import distributions
from lazy import grid as grid_lib
from lazy.models import _ensemble
from lazy.models import _members
from lazy.models import _progress

#: Every upstream setting that changes a prediction, pinned per version so
#: that an upgrade of mymodel cannot change what "auto" runs.
_AUTO_RECIPES: dict[str, dict[str, Any]] = {
    "v1": {"softmax_temperature": 0.9, "n_buckets": 1000},
}


class MyModelHistogram(_ensemble.ContextEnsembleEstimator):
    """Target distributions from MyModel's bucket probabilities."""

    backend = "mymodel"
    display_name = "MyModel"
    method_note = "Bucket masses of MyModel's regression head."
    extra = "mymodel"
    native_output = "histogram"
    native_transforms = {"none": "none"}
    auto_tokens = ("none",)

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        version: str = "v1",
        n_estimators: int = 8,
        transforms: str | tuple[str, ...] = "auto",
        feature_shuffle: bool = True,
        bag_size: int | float | None = None,
        kv_cache: bool = True,
        y_grid: grid_lib.GridLike = None,
        device: str = "auto",
        random_state: int | None = 0,
        chunk_size: int = 8_192,
        softmax_temperature: float | str = "auto",
        mixed_precision: bool = True,
        outlier_threshold: float | str | None = "auto",
        progress: _progress.Progress = "auto",
        verbose: bool = False,
    ):
        self.version = version
        self.n_estimators = n_estimators
        self.transforms = transforms
        self.feature_shuffle = feature_shuffle
        self.bag_size = bag_size
        self.kv_cache = kv_cache
        self.y_grid = y_grid
        self.device = device
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.softmax_temperature = softmax_temperature
        self.mixed_precision = mixed_precision
        self.outlier_threshold = outlier_threshold
        self.progress = progress
        self.verbose = verbose

    def _import_backend(self) -> types.ModuleType:
        # Names the extra when mymodel is missing.
        return _ensemble.import_extra(
            "mymodel", needed_by=type(self).__name__, extra=self.extra
        )

    @classmethod
    def _pinned_recipe(cls, version: str) -> Mapping[str, Any]:
        if version not in _AUTO_RECIPES:
            return super()._pinned_recipe(version)
        return types.MappingProxyType(_AUTO_RECIPES[version])

    def _auto_softmax_temperature(self) -> float:
        return self._pinned_recipe(self.version)["softmax_temperature"]

    def _fit_group(
        self,
        X: _typing.FloatArray,
        y: _typing.FloatArray,
        group: _members.MemberGroup,
    ) -> Any:
        recipe = self._pinned_recipe(self.version)
        regressor = self._import_backend().Regressor(
            checkpoint=self.checkpoint_,
            device=self.device_,
            n_estimators=group.n_members,
            seed=group.seed,
            softmax_temperature=self._temperature(),
            n_buckets=recipe["n_buckets"],
        )
        return regressor.fit(X, y)

    def _predict_group(
        self, handle: Any, X: _typing.FloatArray
    ) -> distributions.Distribution:
        edges, probabilities = handle.predict_buckets(X)
        return distributions.HistogramDistribution(edges, probabilities)

    def _native_grid(self) -> grid_lib.Grid:
        return grid_lib.Grid.from_edges(
            self.handles_[0].bucket_edges, normalization="histogram"
        )
```

The module is registered at the end of `src/lazy/models/registry.py`, with
`register("mymodel", mymodel.MyModelHistogram)`, and its checkpoint is pinned
in `src/lazy/models/_hub.py`, as an entry in `CHECKPOINTS` with a full commit
hash as its `revision` and an entry in `DEFAULT_VERSIONS`. Registering is what
puts the backend in `LazyModel`, `LazyEnsembleModel`, `list_estimators` and
both conformance suites. It also puts the backend in the tables of the models
on {doc}`models/index` and {doc}`guide/interface`, which `docs/conf.py` writes
at every build from each checkpoint's description (`display_name`,
`parameters`, `size_bytes`, `license_name`, `gpu`, `cpu` and the size that
starts its `size_note`) and from the class's `method_note`, so those tables
are never edited by hand.

## What else a backend needs

The table lists every other piece a new backend needs, where it goes, and
which test fails until it is there.
`tests/lazy/test_backend_completeness.py` checks the registered backends for
all of them at once and lists what each one still lacks, so it is the place to
start.

| What | Where | Checked by |
| ---- | ----- | ---------- |
| A pinned checkpoint, with a full commit hash, for every version | `CHECKPOINTS` and `DEFAULT_VERSIONS` in `src/lazy/models/_hub.py` | `register`, `test_backend_completeness.py`, `test_registry.py` |
| A pinned recipe for every version | the backend's `_pinned_recipe` | `test_backend_completeness.py` |
| A pip extra, also named in the `all` extra | `[project.optional-dependencies]` in `pyproject.toml` | `test_backend_completeness.py`, `test_packaging.py` |
| The upstream package among the optional modules mypy may miss | `[[tool.mypy.overrides]]` in `pyproject.toml` | mypy, without the extras installed |
| A description of every checkpoint for the docs' tables | the fields above, in `CHECKPOINTS` in `src/lazy/models/_hub.py` | `test_backend_completeness.py` |
| A one-sentence account of the method for the docs' tables | the backend's `method_note` | `test_backend_completeness.py` |
| A model page, in the models toctree | `docs/models/<name>.md`, `docs/models/index.md` | `test_backend_completeness.py` |
| Golden densities, their settings and tolerances | `tests/lazy/golden_data.py`, `tests/lazy/golden/<name>.npz` (written by `tests/lazy/record_golden.py <name>`), `RTOL` and `ATOL` in `tests/lazy/test_golden.py` | `test_backend_completeness.py`, `test_golden.py` |
| Small CPU settings for the checkpoint tests | `tests/lazy/backend_settings.py` | `test_backend_completeness.py`, `test_conformance_backends.py` |
| A fake upstream, with `install` and `SETTINGS` | `tests/lazy/fakes/<name>.py` | `test_backend_completeness.py`, `test_conformance.py`, `test_bagging.py` |
| A real-weights entry in the shared-network test | `_REAL` in `tests/lazy/test_weights_cache.py` | `test_backend_completeness.py` |
| A row in the hand-written model lists | `README.md`, `docs/installation.md` | review |
| A line in the changelog | `CHANGELOG.md`, under `[Unreleased]` | review |

The fake replaces what the backend needs from outside the package (the
upstream model, its network, the checkpoint download) with stand-ins whose
answers depend on each query and its member's context alone, so that the
behavioral conformance tests in `tests/lazy/test_conformance.py` (seeds,
chunking, the cache, transforms, bagging, pickling) run the backend's own code
in the default suite, on a CPU and in seconds; `tests/lazy/fakes/__init__.py`
describes what a fake provides, and the four existing ones are the models to
follow. A fake that needs the upstream package skips without it.

The golden file is recorded once, on a CPU, and never regenerated afterwards:
it is the guard on the published numbers. The tests that load a checkpoint run
only with `LAZY_RUN_CHECKPOINT_TESTS=1`, so a new backend is finished when
both the default suite and that run pass.
