# Changelog

All notable changes to `lazy-photoz` are recorded here. The format follows
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
- `RedshiftGrid`: output binning, normalisation and exact, mass-conserving
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
- `lazy.distributions`: per-galaxy redshift distributions in each model's
  native form (`HistogramDistribution`, `QuantileDistribution`,
  `MixtureDistribution`) with exact `pdf`, `cdf`, `ppf`, `sf`, `rvs`, `mean`,
  `median`, `mode`, `std`, `var`, `interval` and `histogramize`, following
  scipy.stats and LSST DESC's qp; `to_qp()` / `from_qp()` with the new `qp`
  extra.
- Features may be given as NumPy arrays, structured or record arrays, pandas
  DataFrames, astropy Tables or any object with `to_pandas()`. Missing values
  are `NaN` and pass through to each model's own handling; infinities and
  non-numeric columns are rejected with the offending columns named.

### Changed

- `LazyModel` builds its backend at `fit`, as the fitted attribute
  `estimator_`, rather than in `__init__`. An unknown backend name or parameter
  is reported at `fit`, and `set_params` rejects unknown parameter names.
- The code follows the Google Python Style Guide, with Google-style
  docstrings throughout; scikit-learn's `X` and `y` keep their names.
- `feature_names_in_` follows scikit-learn: it is set only when the features
  have string column names, and a named/unnamed mismatch at prediction time is
  a warning (columns used by position) rather than an error.
- `Catalog.build_features` is now the module-level function
  `lazy.datasets.build_features`, with the same arguments.
- One-letter public parameters have descriptive names:
  `RedshiftGrid.bin_index(redshifts)`, `tabfm.quantile_edges(redshifts, ...)`,
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
