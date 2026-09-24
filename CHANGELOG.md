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

### Changed

- `LazyModel` builds its backend at `fit`, as the fitted attribute
  `estimator_`, rather than in `__init__`. An unknown backend name or parameter
  is reported at `fit`, and `set_params` rejects unknown parameter names.

[Unreleased]: https://github.com/biprateep/lazy-photoz/commits/main
