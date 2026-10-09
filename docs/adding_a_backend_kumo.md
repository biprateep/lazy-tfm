# Worked example: Kumo Tabular

{doc}`adding_a_backend` describes the contract a backend meets and the
checklist of what else it needs. This page follows one backend through that
process from start to finish: NVIDIA's Kumo Tabular. The commits below first
move what an existing backend already had into shared code, and then add the
backend itself. We describe what we read
in the upstream package, what we decided from it, and how each piece of the
contract was met, with excerpts from `src/lazy/models/kumo.py`.

| Commit | What it does |
| ------ | ------------ |
| [5c37bd2](https://github.com/biprateep/lazy-tfm/commit/5c37bd2) | Moves TabICL's quantile-head helpers into `lazy.models._quantile_head` |
| [107aa27](https://github.com/biprateep/lazy-tfm/commit/107aa27) | Adds the trimmed mean of the members' quantiles to the base class |
| [b07585c](https://github.com/biprateep/lazy-tfm/commit/b07585c) | Moves TabFM's repository install into `lazy.models._install` |
| [bd0bb9e](https://github.com/biprateep/lazy-tfm/commit/bd0bb9e) | Adds the backend, its checkpoints, tests and documentation |

## Reading the upstream model

We started from NVIDIA's announcement and the code of its
`structured-data-models` package (imported as `sdm`). For regression,
`sdm.models.KumoTabular` returns 999 quantiles of the target per query row, at
the levels 0.001 to 0.999, so the backend declares
`native_output = "quantiles"`, as TabICL does. Every member of its ensemble
runs through a *recipe* of preprocessing steps
(`sdm.models.kumo.tabular.recipe.default_recipe`), which cycles the members
through four feature transforms, shuffles and sign-flips their columns and
soft-clips outliers, and the members' quantiles are combined by a mean that
drops the highest and lowest fifth of the members at every level. The weights
are three files in one Hugging Face repository,
`{small,medium,large}/regressor.pt`, at a tagged commit, which upstream
downloads itself. Since the column-wise attention lets each query see only the
context rows, we expected a query's answer not to depend on the other queries,
and on a CPU the cached and the uncached forward passes, and the queries
predicted at once or seven at a time, gave bit-identical answers. This allows
the backend to declare `exact_chunking = True`, which the conformance suite
checks on a CPU (on a GPU under mixed precision every backend agrees only to
float16 rounding). The package itself cannot come from PyPI, which holds only a
placeholder of the name, so the backend needs a `setup` hook, as TabFM and
LimiX do.

From this reading we decided that the three sizes become three versions
(`v1-large`, the default as it is upstream, `v1-medium` and `v1-small`), since
a version in LAZY is a pinned checkpoint and the sizes share one recipe, and
that the trimmed mean stays, since it is part of the creators' recipe, but it is taken
over every member of the ensemble rather than over each call to the model, so
that a bagged ensemble, whose members run one at a time, is combined the same
way as one that runs in a single call.

## Sharing before adding

Some of what Kumo needed already existed in another backend, and we moved it
into shared code, a commit for each piece, before writing the backend, so that
there is one copy of each rather than two. TabICL's native grid, its warning
about probability outside that grid, its quantile levels and its
double-precision target scaling became the `QuantileHead` mixin in
`lazy.models._quantile_head`, which TabICL and Kumo both inherit. TabFM's
install of a repository build became `lazy.models._install.install`, which
installs with pip or uv, or prints the `uv add` to run in a uv project. The
trimmed mean is new to the package, so it went into the base class as a third
`member_combination`, `"quantile_trimmed_average"`, with a hook,
`_predict_members`, that returns each member's quantiles separately so that
the base class can trim over all of them at once. A refactor of an existing
backend must not move its predictions, so we ran TabICL's checkpoint tests on
a CPU, its golden file included, which compare the answers to float rounding,
and the `lazy setup` tests, which check that TabFM's install commands and
messages are unchanged.

## The module

### Declaring what the model does

A backend states what its model does natively as class attributes, and the
base class scaffolds the rest. For Kumo, only `none` maps onto one of the
model's own transforms, since its power, robust and rank transforms are fitted
on z-scored and clipped columns and differ from the uniform transforms of the
same names, which LAZY then applies itself. The model has its own outlier clip
and context cache but cannot run members on different rows in one call, so
bagging is scaffolded, one member per call:

```python
class KumoQuantile(
    _quantile_head.QuantileHead, _ensemble.ContextEnsembleEstimator
):
    backend = "kumo"
    display_name = "Kumo Tabular"
    extra = "kumo"
    native_output = "quantiles"
    native_transforms = {"none": "identity"}
    auto_tokens = AUTO_TRANSFORMS
    supports_native_bagging = False
    member_combination = "quantile_trimmed_average"
    member_trim_fraction = AUTO_TRIM_FRACTION
    kv_cache_modes = (True, False)
    has_softmax = False
    native_outlier_clipping = True
    recommended_max_context = 60_000
    exact_chunking = True
    cpu_friendly_versions = ("v1-small",)
```

The constructor takes the uniform parameters with the shared defaults and
stores them verbatim, as in the skeleton on {doc}`adding_a_backend`, with
`version="v1-large"`.

### Pinning the recipe

Upstream's recipe is written out in the module rather than taken from the
installed package (`AUTO_TRANSFORMS`, `AUTO_OUTLIER_THRESHOLD`,
`AUTO_TRIM_FRACTION` and `_PINNED_RECIPE`, assembled by `_recipe`), so that an
upgrade of `sdm` cannot change what `transforms="auto"` runs, and
`_pinned_recipe` returns these settings for every version. The recipe maps
onto the uniform parameters: the members' transforms follow `transforms`, the
Latin-square shuffle is present only when `feature_shuffle=True`, and the
two-pass clip only when `outlier_threshold` resolves to a number. Its output
step sorts each member's quantiles and does not average them, since the
average is LAZY's. A test compares the text of the pinned recipe with
upstream's numerical branch, step by step, so that a copy that drifts from
upstream fails.

### Fitting a group

The base class splits the ensemble into groups of members that one call of the
model serves, and `_fit_group` fits one of them. We standardize the target in
double precision first, because the network reads it in single precision,
which would round away a narrow spread about a large offset. With the cache
on, the group is fitted on the shared network, and the context upstream
records is moved from the network into the group's handle, since other fits
use the same network:

```python
offset, scale = _quantile_head.target_scaling(y)
handle = _KumoGroup(
    n_members=group.n_members,
    seed=group.seed,
    recipe=self._upstream_recipe(group),
    offset=offset,
    scale=scale,
)
context = (X, ((y - offset) / scale).astype(np.float32))
if not self.kv_cache:
    handle.context = context
    return handle
network = self._network()
x, target = self._tensors(*context)
with self._autocast():
    network.fit(
        x=x,
        y=target,
        recipe=handle.recipe,
        num_estimators=handle.n_members,
        estimator_batch_size=_PINNED_RECIPE["estimator_batch_size"],
        generator=self._generator(handle.seed),
    )
handle.cache, network._cache = network._cache, None
return handle
```

The group's seed, which the base class draws for its first member, seeds a
`torch.Generator` on the device, from which upstream draws the members' column
shuffles and sign flips. With the cache off, the handle keeps the prepared
context, and every prediction reruns the whole forward pass with a generator
seeded the same way, so that the members are the same in every chunk.

### Predicting

Because the members combine by a trimmed mean, the base class calls
`_predict_members` rather than `_predict_group`, and the backend returns one
distribution per member. The fitted context is lent to the shared network for
the duration of one prediction (`_replaying`), and the quantiles are mapped
back to the units of the target:

```python
with self._autocast():
    if handle.cache is None:
        x, target = self._tensors(*handle.context)
        out = network(
            x_context=x,
            y_context=target,
            x_query=queries,
            recipe=handle.recipe,
            num_estimators=handle.n_members,
            estimator_batch_size=_PINNED_RECIPE["estimator_batch_size"],
            generator=self._generator(handle.seed),
        )
    else:
        with _replaying(network, handle.cache):
            out = network.predict(queries)
quantiles = out.numerical.double().cpu().numpy()
levels = _quantile_head.quantile_levels(quantiles.shape[-1])
return [
    distributions.QuantileDistribution(
        levels, member * handle.scale + handle.offset
    )
    for member in quantiles
]
```

The base class checks every member's answer against the contract (a
`QuantileDistribution` with one row per query) and trims over all of them.
`_predict_group` is still implemented, as the contract requires, as the
trimmed mean of the group's own members.

### The network

`_load_network` builds `sdm.models.KumoTabular` for the version's size on
PyTorch's meta device, which allocates no memory, and assigns the tensors of
the pinned checkpoint file to it, as upstream does for its own download. The
network is held in `lazy.models._weights` under a key of the backend, version,
file, revision, device and precision, so every fit with the same weights runs
on one network, and the estimator keeps its reference through
`_memoized_network`, which pickling drops. The weights stay in single
precision, and `mixed_precision` only switches on upstream's float16 autocast,
as NVIDIA runs the model.

### Installing the package

The `kumo` extra in `pyproject.toml` lists the dependencies of
`structured-data-models` (PyPI refuses a git URL in an extra), and joins the
`all` extra. The `setup` hook skips the backend when the extra is missing,
does nothing when `sdm` is already installed, and otherwise installs the
pinned commit from GitHub through `lazy.models._install.install`. Since `sdm`
ships no type information and is absent from CI, it also joins the modules
mypy may miss.

## Registering and pinning

One line in `src/lazy/models/registry.py` registers the class, which puts it
in `LazyModel`, `LazyEnsembleModel` and both conformance suites, and the class
is exported from `lazy` and `lazy.models` beside the others. The three
checkpoints are pinned in `src/lazy/models/_hub.py`, one entry per size, all
at the commit the repository's `v1.0.1` tag points to:

```python
Checkpoint(
    backend="kumo",
    version="v1-large",
    repo_id="nvidia/Kumo-Tabular",
    package="structured-data-models",
    filename="large/regressor.pt",
    revision="3c3e10bbdb590ace29e7026847f92db3c603096d",
    license_note=_KUMO_LICENSE,
    size_note="~860 MB. Pretrained on tables of up to 60,000 rows.",
    display_name="Kumo Tabular Large",
    parameters=215_683_543,
    size_bytes=862_909_259,
    license_name="OpenMDW-1.1",
    gpu="recommended",
    cpu="no",
),
```

The parameter counts are summed over the tensors of each checkpoint file, and
the sizes are those the Hub reports at the pinned revision. The `gpu` and
`cpu` fields come from timing upstream on the 20-core Arm CPU of an NVIDIA
GB10, where eight members on 1,000 context rows and 1,000 queries took 9 s
with the small size against 28 s and 33 s with the medium and large ones. We
therefore mark only `v1-small` as CPU-friendly.
`DEFAULT_VERSIONS` gains `"kumo": "v1-large"`.

## Tests

`tests/lazy/test_backend_completeness.py` lists everything a registered
backend still lacks, and we worked through its list until it passed. The fake
upstream, `tests/lazy/fakes/kumo.py`, subclasses upstream's own `ICLModel`
with a forward pass that answers from the processed features alone, so that
the backend's code and upstream's recipe, ensembling and context cache all run
as they do on the real model, on a CPU and in seconds. It records each group's
rows and standardized targets for the bagging tests, and the behavioral
conformance suite (seeds, chunking, the cache, transforms, bagging, pickling)
then runs on Kumo with no checkpoint. The fake needs `sdm` and skips without
it. The golden file, `tests/lazy/golden/kumo.npz`, was recorded once on a CPU
with `tests/lazy/record_golden.py kumo`, at the small size with five members,
the fewest from which the trimmed mean drops one at each end. The CPU
settings, the golden tolerance and the real-weights entry of the
shared-network test are one line each. `tests/lazy/test_kumo.py` holds what is
particular to Kumo: the pinned recipe against upstream's, and, with the
checkpoint, the backend's answer against upstream's own call on the same
context and seed. The `lazy setup` tests gained Kumo's install.

## Documentation

The model page, {doc}`models/kumo`, follows the others, with the creators'
defaults next to LAZY's, the standardizations the model applies, and where
LAZY differs. It joins the toctree of {doc}`models/index`, whose table of the
models, like the table of backends in {doc}`guide/interface`, is written by
`docs/conf.py` from the checkpoints' fields and the class's `method_note`, so
neither was edited by hand. The install commands in the README and on
{doc}`installation`, the license notes, the list of concrete classes and the
changelog were updated by hand.

## Checking the result

The default suite passes, as does the suite of checkpoint tests on a CPU
(`LAZY_RUN_CHECKPOINT_TESTS=1`), and the documentation builds with warnings
treated as errors. On a CPU, LAZY's answer with the small size matches
upstream's own call to about $3\times 10^{-7}$ in the units of the target on
our test problem, which we attribute to upstream taking its trimmed mean in
single precision and LAZY in double precision. On a GPU with mixed precision,
the cached and uncached answers of the large size agree to about $1.3\times
10^{-3}$ on the same problem, which is of the size we expect from float16
rounding.
