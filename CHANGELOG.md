# Changelog

All notable changes to `lazy-tfm` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). Until 1.0, a minor release may
change the API.

## [Unreleased]

Every backend now takes one set of parameters with one set of defaults, and no
upstream default decides a prediction unseen. **Default predictions change**
(below); to reproduce 0.1.x numbers, pin `random_state` (TabPFN and TabICL 42,
TabFM 1) and, on TabFM, `n_estimators=4`.

### Added

- `softmax_temperature`, `mixed_precision` and `outlier_threshold` on every
  backend. `softmax_temperature="auto"` is the checkpoint's calibrated value
  (TabICL, with no softmax, takes only `"auto"`); `mixed_precision=True` uses
  each model's reduced-precision path on a GPU and float32 everywhere else;
  `outlier_threshold` is the soft clip TabPFN, TabICL and TabFM share, applied
  by `lazy` on LimiX-2, with `"auto"` meaning the model's own clip under
  `transforms="auto"` and none under an explicit recipe. `provenance_` records
  the values used.
- The registry enforces every shared parameter's default, not only some.
- TabPFN-3.5-fast (`LazyModel("tabpfn", version="v3.5-fast")`) is treated as
  CPU-friendly: it fits on a CPU without the slow-CPU warning, which now
  suggests it beside TabICL. A backend can declare such versions with
  `cpu_friendly_versions`.

### Changed

- One default per parameter on every backend: `n_estimators=8` (TabFM had 4),
  `random_state=0` (TabPFN and TabICL had 42, TabFM 1), `chunk_size=8192`
  (TabPFN, TabICL and TabFM had 16,384).
- `transforms="auto"` is each model's recipe written out in `lazy` for every
  version, so an upgrade of tabpfn, tabicl or tabfm cannot change it; with
  numeric features TabPFN, TabFM (in bfloat16) and LimiX-2 (8 or 16 members)
  predict exactly as before.
- An explicit `transforms` value is all the model sees: TabPFN's fingerprint
  feature, polynomial features, SVD components and target transforms, and
  TabICL's and TabFM's 4-sigma clip, are off unless asked for. A transform
  name runs on a model's own implementation only where that is the shared
  definition; elsewhere `lazy` applies it. Explicit recipes such as `"limix"`
  therefore predict differently than in 0.1.x.
- Every column is numeric on every model: TabPFN no longer takes a column with
  fewer than four distinct values for a category.
- TabICL uses mixed precision on a GPU at every context size (upstream only
  from 1,024 context rows or 60 features), keeps its features in float64
  through its preprocessing, and averages its members in a fixed order, so a
  seed gives the same answer in every process.
- TabFM runs in float32 on a CPU (bfloat16 before) and in bfloat16 on a GPU.
- LimiX-2 shuffles a bagged member's columns as upstream does, after its
  pipeline, rather than permuting the inputs.

### Removed

- TabPFN's `n_estimators="auto"`.
- TabFM's `decode_chunk_rows`: `chunk_size` bounds both of its paths, and `0`
  means one pass on both.

### Fixed

- `n_estimators` is exactly the number of members on every model: TabICL ran
  fewer when its column orders ran out (six of eight with three features), and
  LimiX-2 with four members or fewer ran only its quantile pipeline.
- TabFM's `bag_size` could give a member one row too many, and its provenance
  listed bag rows that TabFM, which draws its own, never used.
- `feature_shuffle=False` held on TabFM only up to 500 features.
- The docs' exactness claims for chunking and the cache now hold: exact on a
  CPU, to the rounding of the precision under mixed precision on a GPU.

## [0.1.1] - 2026-09-29

### Fixed

- The README, quickstart and docs landing page passed the grid to `evaluate`
  positionally, where it is read as `method`, and failed; they now pass
  `z_grid=`. They also scored the first 20,000 DC1 test rows, which the file
  sorts by redshift (median 0.29 against 0.64), and now take a random 20,000.

## [0.1.0] - 2026-09-29

The first release. Everything below is new.

### Added

- Photometric-redshift densities from pretrained tabular foundation models,
  behind the scikit-learn estimator API: `fit`, `predict_proba` (densities on a
  redshift grid), `predict` (point estimates), `score` (negative CDE loss) and
  `evaluate` (the full metric table).
- Three backends, each an optional extra: `tabfm` (TabFM v1.0, a hierarchy of
  in-context classifiers), `tabicl` (TabICLv2, quantile regression) and
  `tabpfn` (TabPFN v2 to v3.5, the bar distribution). `LazyModel(name, ...)`
  selects one by name.
- Pinned checkpoint revisions for every backend version, `provenance_` on every
  fitted model, and `download_checkpoint` / `is_cached` for offline use.
- `Grid`: output binning, normalisation and exact, mass-conserving
  rebinning. The grid is chosen at prediction time, so one fitted model answers
  on any grid.
- `lazy.metrics`: the LSST DESC PZ Data Challenge point and PDF metrics, and
  `summarize`.
- `lazy.datasets`: the DC1 catalogue (download, checksum, cache), feature views
  through `Catalog.features`, and `fetch_dc1_biased`, a spectroscopically
  selected training split.
- `lazy.selection`: the HSC spectroscopic selection function, ported from RAIL.
- `lazy.plotting`: diagnostic figures and a publication style.
- Progress bars while predicting (`progress="auto"`), and exact chunking of
  query rows to bound memory (`chunk_size`).
- A `TabFMPerformanceWarning` when TabFM falls back to its uncached path.
- Support for Python 3.12, 3.13 and 3.14.
- `Grid(normalization="histogram")`: densities constant across each
  bin, normalised as `sum(p * widths) == 1`; the metrics (`cde_loss`,
  `summarize`, `grid_point_estimates`, ...) take `bin_edges=` to score such
  grids exactly, and estimators use it automatically. Existing grids keep the
  trapezoid (DC1) convention, byte for byte.
- A uniform feature layer for every foundation-model backend
  (`lazy.models.ContextEnsembleEstimator`): the same parameters mean the same
  thing on every model -- `kv_cache`, `n_estimators`, `feature_shuffle`,
  `transforms` (a shared vocabulary: `none`, `power`, `quantile`,
  `quantile_uniform`, `quantile_rtdl`, `robust`, each optionally
  `+original`, plus recipes such as `"limix"`; `"auto"` keeps each model's
  own), and `bag_size` (per-member row subsets) -- translated to each model's
  own machinery where it has it and scaffolded where it does not. The default
  output grid is the model's native grid. `ContextSizeWarning` when a context
  exceeds what a model handles without bagging.
- `lazy.models.registry.register(name, cls)`: backends join the registry only
  through it, and it refuses a class that lacks the uniform features. A
  two-tier conformance suite runs every registered backend: parameter and
  planning checks always, and behavioural checks (cache, chunking, bagging,
  transforms, quantiles, native grid) with the checkpoints.
- `predict_distribution(X)` (the model's native distributions) and
  `predict_quantiles(X, quantiles)` (exact, from the native distribution) on
  every estimator; `z_grid="native"` asks a model for its own grid.
- `lazy.distributions`: per-galaxy redshift distributions in each model's
  native form (`HistogramDistribution`, `QuantileDistribution`,
  `MixtureDistribution`) with exact `pdf`, `cdf`, `ppf`, `sf`, `rvs`, `mean`,
  `median`, `mode`, `std`, `var`, `interval` and `histogramize`, following
  scipy.stats and LSST DESC's qp; `to_qp()` / `from_qp()` with the new `qp`
  extra.
- A fourth backend, `limix` (`LimiXBarDistribution`): LimiX-2's 5,000-bucket
  densities, on the uniform feature layer. LimiX is not on PyPI; the
  `limix` extra installs its dependencies and the backend loads its code from
  a checkout (`$LAZY_LIMIX_SRC`) or a `pip install` of the repository, under
  private module names. Its feature preprocessing is fitted on the context
  alone and its positional embedding has its own generator, so a query's
  answer no longer depends on the other queries in its chunk (upstream's does:
  its global generator is advanced by an amount set by the chunk's size). A
  ported key/value cache (`kv_cache`, default on) runs the context through the
  network once, at fit. `ContextSizeWarning` above 20,000 unbagged context
  rows. Built with StableAI LimiX.
- An `ImportNameWarning` at `import lazy` when the unrelated PyPI package
  `lazy`, which installs the same import name, is in the environment too.
- Features may be given as NumPy arrays, structured or record arrays, pandas
  DataFrames, astropy Tables or any object with `to_pandas()`. Missing values
  are `NaN` and pass through to each model's own handling; infinities and
  non-numeric columns are rejected with the offending columns named.
- An introductory tutorial notebook, rendered in the docs with buttons to open
  it in Colab or on GitHub. Its executed copy lives on the `tutorials` branch.
- A **Supported models** page in the docs: each model's weights, parameter
  count, licence, measured cost on GPU and CPU, and hardware needs.
- `lazy.PerformanceWarning` at `fit` when `device="auto"` finds no GPU for a
  model that is slow on a CPU (TabPFN, LimiX-2, TabFM), suggesting TabICL.
- Every `lazy.metrics` function that takes `z_grid` also takes a `Grid`, and
  then scores it by its own normalisation. New `metrics.per_object_scores`
  and `metrics.summarize_scores`, for building the summary from pieces.
- `scale="1+z" | "none"` on `point_metrics`, `summarize` and `evaluate`: the
  default keeps the photo-z convention (and DC1's numbers); `"none"` uses
  plain residuals, for other targets. Summary tables gain a `scale` column.
- `random_state=None` draws a fresh seed at `fit` and records it as
  `random_state_` and in `provenance_`.

- `diagnostic_panel`, `plot_nz` and `plot_pdfs` take a `Grid` (or
  `bin_edges=`) and draw native histogram grids exactly, framed on the data;
  `plot_residuals(scale="none")` for targets other than redshift.
### Changed

- The docs use the Read the Docs theme.
- TabICL's native grid extends 25% of the training range on each side (1,500
  bins), no longer clips at zero, and `predict_proba` on it warns when a row
  puts more than 1% of its probability outside it.
- TabPFN's bucket borders and TabICL's target scaling are computed in float64,
  so a narrow target on a large offset keeps its resolution. Densities move at
  the float32-rounding level (at most 3e-5 relative for TabPFN, 5e-4 for
  TabICL); no peak moves.
- TabFM no longer claims bit-exact chunking and caching on a GPU, where it
  runs in bfloat16 and densities can differ by a few per cent of the peak.
- A missing LimiX dependency (torch, einops, kditransform, nvtx, triton)
  raises an `ImportError` naming it and both install steps; triton's says it
  is Linux-only.
- `use_style` leaves `figure.dpi` alone (figures save at 300 dpi still), and
  `verify_style` warns rather than raises when the paper font is missing
  (`strict_font=True` to raise).
- Files already cached under the old `~/.cache/lazy-photoz` are used, with a
  one-time warning suggesting you move them.
- `LazyModel()` defaults to TabPFN, and TabPFN to v3.5.
- A cached checkpoint loads without contacting the Hugging Face Hub, so there
  is no network round trip and no "unauthenticated requests" notice; the
  TabICL checkpoint is now pinned to a Hub commit like the others.
- `predict`, `score` and `evaluate` work through the rows a block at a time
  (about 256 MB of densities per block), so memory stays bounded on large
  native grids; results are unchanged.
- The package is for any continuous target, not only redshift, and its
  generic API says so: `RedshiftGrid` is now `Grid` and `BasePhotoZEstimator`
  is `BaseDensityRegressor`; parameters that took redshifts take `values`,
  `plot_pdfs(n_galaxies=)` is `n_objects=`, and progress bars count rows.
  Docstrings speak of the target, keeping redshift for the photo-z tools
  (`datasets`, `selection`, and the point metrics' `1 + z` scaling).
- The distribution is named `lazy-tfm` (`pip install lazy-tfm`); the import
  name stays `lazy`. The benchmark-catalogue cache moved from
  `~/.cache/lazy-photoz` to `~/.cache/lazy-tfm` (move an existing cache there,
  or point `LAZY_DATA_HOME` at it).
- `LazyModel` builds its backend at `fit`, as the fitted attribute
  `estimator_`, rather than in `__init__`. An unknown backend name or parameter
  is reported at `fit`, and `set_params` rejects unknown parameter names.
- `TabICLQuantile` runs on the uniform feature layer: its key/value cache is
  on by default (`kv_cache`, exact), `transforms` map onto its
  `norm_methods`, `bag_size` is scaffolded (one regressor per bag, quantile
  functions averaged), and its default grid is its native one (1,000 bins over
  the training redshifts).
- `TabPFNBarDistribution` runs on the uniform feature layer. `kv_cache`
  (default on) replaces `fit_mode` and caches at full precision, so it is
  exact; upstream's int8 cache, which shifts densities by up to about half a
  per cent and which earlier `fit_mode="fit_with_cache"` runs used, is
  `kv_cache="int8"`. `transforms` map onto its `PREPROCESS_TRANSFORMS`,
  `bag_size` onto its per-member `SUBSAMPLE_SAMPLES`. Its default grid is its
  native one: the bar distribution's buckets, in full.
- `TabFMHistogram` runs on the uniform feature layer. `kv_cache` (default on:
  the streaming prefill/decode path, falling back with a warning) replaces
  `inference`; `transforms` map onto its `norm_methods` (others scaffolded as
  hierarchies of their own, so `transforms="limix"` reproduces the paper's
  TabFM-with-LimiX-transforms runs), `bag_size` onto its `max_num_rows`.
  Its equal-mass bins now span the constructor grid's range or the training
  redshifts', not the grid a prediction happens to be asked on; pass
  `z_grid=DC1_GRID` to reproduce earlier runs. Dithers form a mixture, and the
  default grid is the union of their edges.
- The code follows the Google Python Style Guide, with Google-style
  docstrings throughout; scikit-learn's `X` and `y` keep their names.
- `feature_names_in_` follows scikit-learn: it is set only when the features
  have string column names, and a named/unnamed mismatch at prediction time is
  a warning (columns used by position) rather than an error.
- `Catalog.build_features` is now the module-level function
  `lazy.datasets.build_features`, with the same arguments.
- One-letter public parameters have descriptive names:
  `Grid.bin_index(values)`, `tabfm.quantile_edges(values, ...)`,
  `plotting.plot_pdfs(..., n_objects=...)`, `style.set_palette(n_colors=...)`,
  `style.better_step(bin_edges, heights, ...)`,
  `style.binned_quantiles(x_values, y_values, ..., percentiles=...)` and
  `style.running_median(ax, x_values, y_values, ...)`.

### Fixed

- TabFM fits tied, discrete, zero-inflated and constant targets (it raised
  `IndexError`), and clips training targets outside its `z_grid` with a
  warning. TabPFN fits a constant target.
- Repeated entries in `transforms` keep their weight on TabPFN and TabFM;
  TabICL rejects them, as upstream would run them as identical members.
- `kv_cache="int8"` or `"fp8"` on TabPFN v2.x raises instead of silently
  running at full precision.
- The "pip install lazy-tfm[...]" hint is given only when the backend package
  itself is missing; other import errors come through unchanged.
- A fitted LimiX-2 model unpickles in a fresh process, re-resolving
  `device="auto"` and its checkpoint; its key/value-cache memory check no
  longer counts existing caches twice, and members whose cache fits keep it.
  A target whose spread float64 cannot resolve against its mean raises a
  clear error.
- Parallel downloads into one data home no longer collide, downloads time out
  instead of hanging, and a cached file of the wrong size is fetched again.
- `build_features` keeps the input's index.
- `plot_zphot_ztrue` works for negative and large targets and small samples;
  `plot_pdfs` draws a single PDF; `save(fig, "model.v2")` writes
  `model.v2.png`.
- `Grid.from_centers` and `as_grid` refuse unevenly spaced centres, which they
  used to move silently; use `Grid.from_edges` for those.
- An unfitted model raises scikit-learn's `NotFittedError` from every method.
- `score` and `evaluate` check the targets (finite, one per row) and the
  point-estimate `method` before running the model; NaN targets used to give a
  plausible, wrong score.
- Zero-row inputs: `fit` raises scikit-learn's error; predictions return
  correctly shaped empty results.
- `LazyModel.set_params(model=...)` keeps the settings the new backend also
  takes, so a `GridSearchCV` over backends no longer resets them, and
  attributes read after `set_params` show the new values.
- `n_estimators`, `bag_size`, `random_state` and `chunk_size` accept NumPy
  numbers; `device` accepts a `torch.device` and rejects unknown values
  clearly; an unknown `version` raises `ValueError` listing the pinned ones.
- `ContextSizeWarning` also fires when bags are larger than a model's limit,
  and a warning flags bags under 10 rows (as from `bag_size=1` meaning `1.0`).
- The `power` transform passes constant and all-NaN columns through; indexing
  a distribution with `ancil` by an integer works; `Grid.rebin` validates its
  input; `HistogramDistribution.ppf` at 0 and 1 stays inside the support;
  `Grid`'s hash agrees with its equality; feature-name warnings point at the
  caller's line.
- TabFM's `inference="predict_proba"` path (the PyPI `tabfm` release) now
  honours `chunk_size`. It used to pass every query row to upstream
  `predict_proba` at once, so peak memory grew with the query set. The
  streaming path is bounded by `query_block_rows` and `decode_chunk_rows`, and
  its documentation now says so. The chunking test checks that chunks are really
  formed.

[Unreleased]: https://github.com/biprateep/lazy-tfm/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/biprateep/lazy-tfm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/biprateep/lazy-tfm/releases/tag/v0.1.0
