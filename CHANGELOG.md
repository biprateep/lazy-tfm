# Changelog

All notable changes to `lazy-tfm` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). Until 1.0, a minor release may
change the API.

## [Unreleased]

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

### Changed

- The package is for any continuous target, not only redshift, and its
  generic API says so: `RedshiftGrid` is now `Grid` and `BasePhotoZEstimator`
  is `BaseDensityRegressor`.
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
  `plotting.plot_pdfs(..., n_galaxies=...)`, `style.set_palette(n_colors=...)`,
  `style.better_step(bin_edges, heights, ...)`,
  `style.binned_quantiles(x_values, y_values, ..., percentiles=...)` and
  `style.running_median(ax, x_values, y_values, ...)`.

### Fixed

- TabFM's `inference="predict_proba"` path (the PyPI `tabfm` release) now
  honours `chunk_size`. It used to pass every query row to upstream
  `predict_proba` at once, so peak memory grew with the query set. The
  streaming path is bounded by `query_block_rows` and `decode_chunk_rows`, and
  its documentation now says so. The chunking test checks that chunks are really
  formed.

[Unreleased]: https://github.com/biprateep/lazy-photoz/commits/main
