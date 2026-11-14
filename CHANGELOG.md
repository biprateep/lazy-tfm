# Changelog

All notable changes to `lazy-tfm` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). Until 1.0, a minor release may
change the API.

## [Unreleased]

### Changed

- Kumo Tabular now runs on `device="mps"` instead of raising a `ValueError`.
  MPS has no float64, so there its features reach upstream in float32 and its
  recipe runs in float32, as upstream's own `Cast` does on MPS. Outputs on
  CUDA and the CPU are unchanged.

### Removed

- `TabFMHistogram` no longer takes `prior_shift`, and `lazy.models.tabfm`
  no longer exports `prior_shift_em` or sets the fitted attribute
  `bin_prior_`. The label-shift correction ran on TabFM alone, and it made
  a row's answer depend on the other rows predicted with it, so `predict`
  and `predict_proba` could disagree. Passing `prior_shift` is now a
  `TypeError`, and unpickling a model fitted with `prior_shift="em"` raises
  a `ValueError` rather than predicting without the correction. Predictions
  with the default, `prior_shift=None`, are unchanged. A correction that
  every backend shares is planned in
  [#300](https://github.com/biprateep/lazy-tfm/issues/300).

### Fixed

- TabPFN now predicts on `device="mps"`. The bucket softmax ran in float64 on
  the device, which MPS lacks, so every prediction raised a `TypeError`.
  Outputs on CUDA and the CPU are unchanged.

- TabFM now predicts on `device="mps"`. Its streamed prediction moved float64
  arrays to the device before casting them, which MPS refuses. Outputs on CUDA
  and the CPU are unchanged.

## [0.6.0] - 2026-11-07

A release that makes each module simpler and faster, and predictions safe
to run from several threads. Outputs are unchanged except where an entry
under Changed or Fixed says otherwise, as for LimiX-2 and Kumo Tabular on a
GPU and for a constant target. Several names and behaviours are deprecated
and change in 0.7.0; each warns with a `FutureWarning` now.

### Added

- `Checkpoint`, the record that `lazy.get_checkpoint` returns, is now
  exported from `lazy` and `lazy.models` and documented in the API
  reference. The page of each package with an `__all__` (`lazy`,
  `lazy.models` and `lazy.plotting`) lists the names it exports, and the
  reference builds with no unresolved cross-reference.
- `lazy.LazyWarning`, a `UserWarning` that is the base of lazy's own
  warning classes, and `lazy.GridCoverageWarning`, issued when densities are
  renormalized because the native grid misses part of their probability.
  `lazy.ImportNameWarning`, `lazy.LimiXSourceWarning` and
  `lazy.TabFMPerformanceWarning` are now exported from `lazy`.
- `lazy.UnknownNameError`, the error for an unknown backend name or model
  version. It is both a `KeyError` and a `ValueError`, so existing `except`
  clauses still catch it, and its message names the valid choices.
- `lazy models` lists every pinned checkpoint, its default version and
  whether it is cached, and `lazy download NAME[:VERSION]` downloads the
  weights of the checkpoints named, with the license notice, so the cache
  can be warmed from the command line.
- The first download of every model's weights issues a
  `lazy.LicenseWarning`. It names the license, which using the weights
  accepts, and asks that the model be cited as well as LAZY, with a link to
  the model's docs page, which lists its references.
- Each pinned checkpoint has a `commercial_use` flag, which says whether the
  license of its weights allows commercial use. `lazy models` and the docs'
  table of the models show it.
- The package is marked typed (`py.typed`), so type checkers read its
  annotations.
- The source distribution includes the tests and the docs pages they read.
- `lazy.metrics.grid_point_estimate` computes one point estimate, where
  `grid_point_estimates` computes all four. `predict`, `evaluate`,
  `evaluate_grid_pdfs` and `summarize` now compute only the one they return,
  with the same results to the bit.
- `lazy.metrics.per_object_metrics` returns the CDE terms, PIT values, CRPS
  and NLL of each row from one normalization, where `per_object_scores`,
  `per_object_crps` and `per_object_nll` normalize once each. `evaluate`
  uses it, with the same results to the bit.
- The user guide has four new pages: a glossary, how the models work with
  scikit-learn's tools, saving and loading a fitted model, and every warning
  that lazy issues. The scaling page states how `n_jobs` uses the model cache
  and the GPU, and how `chunk_size` multiplies the cost of the context when
  `kv_cache=False`.
- Each tutorial page notes when its notebook may not match the source
  shown, and, once the notebook is re-executed, the lazy version and commit
  it ran on. The docs of a
  release show the notebooks executed for it, a fork's docs build fetches
  the notebooks from `biprateep/lazy-tfm`, and an offline local build shows
  the tutorials without outputs instead of failing.
- The user guide has a page that compares LAZY with each model's own
  package, TabTune and AutoGluon-Tabular, and says when to use each instead.
- The user guide has four concept figures, drawn by
  `docs/figures/make_figures.py` without a model: the point estimates of a
  bimodal density, the three pools of an ensemble, chunks and bags of the
  context, and a grid that ends inside a distribution.
- LimiX-2 runs without `triton`, so on a CPU it runs on macOS and Windows
  too. LimiX imports `triton` but runs its kernels only on CUDA, so without
  `triton` lazy gives that import a stand-in. Without `triton`, a CUDA
  device raises a `RuntimeError` when the network loads. Linux results are
  unchanged.

### Changed

- On Windows, the default cache of datasets and LimiX's source is
  `%LOCALAPPDATA%\lazy-tfm`, the platform's usual place for caches,
  instead of `~/.cache/lazy-tfm`. Files cached in the old folder are not
  read: lazy downloads them again, unless `LAZY_DATA_HOME` names the old
  folder. Otherwise the old folder, up to about 1 GB with DC1, can be
  deleted. `XDG_CACHE_HOME`, when set, still decides the folder on every
  platform. Weights are unaffected; they stay in the Hugging Face cache.
- Under scikit-learn's metadata routing, `y_grid` is no longer metadata.
  Once requested with `set_score_request(y_grid=True)`, a grid with as many
  values as the rows given to a router such as `cross_validate` was split
  between the folds, so each fold was scored on part of the grid. A grid
  passed through a router now raises `TypeError`, whatever its length; set
  `y_grid` in the constructor instead. The models no longer have
  `set_predict_proba_request` or `set_score_request`, so calling either
  raises `AttributeError`. `set_predict_request` remains, for `method`.
- LimiX-2 on a GPU draws its feature positional embedding as on a CPU, with
  a CPU generator seeded per member, and moves it to the GPU. LAZY drew it
  with a GPU generator, which gives another stream for the same seed, so the
  same `random_state` gave different densities on the two devices. Densities
  on a GPU change; on a CPU they are unchanged.
- A constant target, every training value the same, now gets a point mass
  at that value on every backend, as scikit-learn's regressors answer it,
  and no network runs. Its quantiles, intervals and point estimates are the
  value, `predict_proba` puts all of the probability in the grid bin that
  holds it, and the native grid has three bins centred on it.
  `LazyEnsembleModel` gives the same point mass under every pooling.
  Before, the backends disagreed: LimiX-2 raised `ValueError` at `y = 0.3`
  and gave quantiles 1.4 units wide at `y = 7.0`, TabICL gave quantiles
  0.04 units wide, and TabPFN and TabFM gave one narrow bucket or bin. The
  fitted model's `constant_target_` records the value. On a backend,
  `provenance_` then holds only `backend`, `version` and
  `constant_target`. Results for any other target are unchanged, and so
  are the golden densities.
- The documentation uses the PyData Sphinx theme, which has a dark mode,
  and the favicon comes as an SVG and as PNG files of 64 and 180 pixels.
  A new brand page lists the spellings of the name, the logo files and the
  colors. Prose spells the project LAZY.
- The package metadata lists both authors, as `CITATION.cff` does. Its
  summary and keywords describe conditional density estimation in general,
  with photometric redshifts and astronomy as one use.
- Kumo Tabular draws its members' random sign flips and column shuffles on a
  CPU generator on every device, and moves them to the device, so a CPU and a
  GPU give the same ensemble for the same `random_state`. NVIDIA's package
  draws them on the device, where the two give different draws
  (NVIDIA/structured-data-models#1339). CPU results are unchanged, and so
  are the golden densities. On a GPU, results change for the same seed: on
  50 `california_housing` rows, the quantiles moved by up to 0.28 of a row's
  5% to 95% range, and now agree with the CPU's to 6.6e-6. A Kumo model
  saved before this release still draws on the device it runs on, so it
  repeats its answers on the same kind of device.
- A pickled TabICL model no longer holds its key/value caches; its first
  prediction after unpickling records them again, with the same results to
  the bit. Kumo Tabular and LimiX already left theirs out; all three now do
  it the same way, and an unpickled LimiX model with one member group has
  `regressor_` again. With 2 members and 3 features on a 4-core Xeon CPU
  (2.8 GHz, 4 torch threads, torch 2.14.0, tabicl 2.2.0), a pickle of 300
  context rows fell from 35.06 MB to 0.04 MB under `kv_cache=True` and from
  6.78 MB to 0.04 MB under `"repr"`; of 1,200 rows, from 123.60 MB and
  10.54 MB to 0.11 MB. Loading it in a fresh process and predicting 200 rows
  took 3.92 s instead of 3.54 s at 300 rows, and 4.71 s instead of 3.73 s
  at 1,200 rows, under `kv_cache=True` (medians of 3 interleaved runs).
  A `copy.deepcopy` also leaves the caches out and rebuilds them on first
  use. Pickles of earlier versions still load, with the caches they hold.
- A pickled TabPFN model no longer holds its network. Unpickling reattaches
  the network, so it needs the pinned checkpoint on that machine: it is
  downloaded through the hub if missing, and `lazy download` fetches it in
  advance. A pickle made by this version does not load in earlier versions
  of lazy; pickles made by earlier versions still load. A v3.5 model with 8
  members and 500 context rows, fitted on a CPU, pickled to 935.6 MB before
  and 59.4 MB now. Loading it in a fresh process took a median of 2.77 s
  before and 3.59 s now, since the network is read from the checkpoint (5
  interleaved runs, 4 torch threads, 4-core Xeon at 2.8 GHz, torch 2.14.0,
  tabpfn 9.0.0).
- `predict_proba` normalizes each row's density once instead of twice, and
  `predict`, `score` and `evaluate` twice instead of three times, so that
  they still equal the metrics run on `predict_proba`'s output to the bit.
  Results move by float rounding only: densities by at most 1e-14 of a
  row's peak, point estimates and scores by at most 3.1e-15, with modes
  unchanged, on the test stand-ins and fakes of every backend. On 500 rows
  of 5,000 buckets, the median time of `predict_proba` fell from 0.047 s to
  0.034 s, its peak memory from 60 MiB to 41 MiB, and the median time of
  `predict` from 0.075 s to 0.048 s.
- The error for a non-numeric feature table names only the columns that
  fail to convert, not every column of an array.
- `lazy.datasets` reports each download at level INFO to the
  `lazy.datasets` logger instead of printing it, as scikit-learn's fetchers
  do; `logging.basicConfig(level=logging.INFO)` shows it. A cached copy of
  the wrong size now warns, with a `lazy.LazyWarning`, before it is fetched
  again.
- The warning for rows with probability outside the native grid, a plain
  `UserWarning` before, is the new `lazy.GridCoverageWarning`. Its text starts
  with "Densities are renormalised on the native grid" and gives the row count
  and the grid at the end. Every `UserWarning` class of lazy, and the warnings
  that were plain `UserWarning`s, now subclass `lazy.LazyWarning`, so
  `filterwarnings("ignore", category=UserWarning)` still catches them.
- `get_estimator`, `get_checkpoint`, `download_checkpoint`, `LazyModel`,
  `LazyEnsembleModel` and a backend's `fit` raise the new
  `lazy.UnknownNameError` for an unknown backend name or model version.
  They raised `KeyError` or `ValueError`, with messages of several formats.
- `fit` reports an unknown model version before a missing backend package.
  A missing package is still reported before any other bad parameter.
- `predict_cdf` returns each row's native CDF at the bin centers, exactly,
  as `predict_quantiles` and `predict_pit` do. It computed the CDF from the
  densities on the grid, which drops the probability outside the grid and
  adds a discretization error. On the test fakes and stand-ins, on 50 and
  500 bins over a range partly outside the support, values moved by up to
  0.32, mostly from the probability outside the grid; the median move was
  at most 0.006.
- `lazy.plotting.diagnostics.diagnostic_panel` computes the point estimates
  once instead of twice, with the same figure to the bit. On 5,000 rows and
  500 bins its median time fell from 0.59 s to 0.54 s.
- `import lazy` no longer imports h5py or scikit-learn's dataset loaders.
  `lazy.datasets` and `lazy.selection` load when first accessed, as
  attributes of `lazy` or by import, so code that uses them is unchanged.
- On its own buckets (the default grid of an unbagged TabPFN or LimiX model,
  and TabFM's `"grid"` bins with one dither), a histogram's densities are each
  bucket's mass over its width, computed directly instead of through
  differences of cumulative sums, which lost the relative precision of small
  bins. `predict_proba`, `predict` and `score` on that grid are 2.2 to 2.8
  times faster for a 5,000-bucket model (`benchmarks/cpu.py`), and their
  results move slightly toward the exact values: by up to 2.8e-11 of a row's
  peak on 1,000 rows of 5,000 uneven buckets, and more for very narrow buckets
  carrying little mass, which the old sums could round to zero.
- Normalizing densities builds fewer temporary arrays, so a 5,000-bucket
  model's `predict_proba`, `predict` and `score` are 1.7 to 2.0 times faster,
  and their peak memory falls from 88 MiB to 60 to 76 MiB
  (`benchmarks/cpu.py`), with the same results to the bit.
- A histogram's CDF builds fewer temporary arrays. On the CPU benchmark,
  `LazyEnsembleModel`'s peak memory falls from 119 to 99 MiB, and a bagged
  5,000-bucket model's `predict_quantiles` from 301 to 269 MiB, with the
  same results to the bit.
- `LazyEnsembleModel` pools its members one at a time, into one running sum,
  so it holds one member's distribution of a chunk instead of all of them.
  With two members on the CPU benchmark, its peak memory falls from 99 to
  96 MiB; the saving grows with the number of members. The results are the
  same to the bit.
- `predict_quantiles`, `predict_interval` and `predict_pit` reduce each of
  the model's `chunk_size` chunks of rows as it is predicted, instead of every
  row's distribution at once, so their peak memory no longer grows with the
  number of rows (TabFM, which predicts all its queries together, is
  unchanged): on 20,000 rows of a 5,000-bucket model it falls from 3.0 GiB to
  1.2 GiB with the default `chunk_size`, as measured. Their results are
  unchanged, but for float rounding: at most 1e-14 relatively, measured in
  one row of 120.
- A bagged model's quantiles and intervals, and the `mode` of its
  `predict_distribution`, build the histogram on its members' combined bucket
  edges for a block of rows at a time, instead of for every row at once, with
  identical results. With eight members of 5,000 buckets, the peak memory of
  `predict_quantiles` on 311 rows falls from 1,069 MiB to 295 MiB, and its
  time from 5.6 s to 1.8 s, as measured.
- `MixtureDistribution.to_histogram` and `to_qp` also build that histogram
  a block of rows at a time, with identical results. With eight components
  of 5,000 buckets on 1,000 rows, the peak memory of `to_histogram` falls
  from 1,298 MiB to 631 MiB, about twice its 305 MiB result, and its time
  rises from 1.11 s to 1.25 s, as measured.
- The minimum dependency versions are raised to ones that install and work
  together: scipy 1.13, scikit-learn 1.6, pandas 2.2.2, matplotlib 3.8.4 and
  h5py 3.11. The old floors allowed pandas and h5py builds for NumPy 1, with
  which `import lazy` crashed. CI now tests the floors.
- Every model is a scikit-learn regressor (`is_regressor` is now `True`), so
  `VotingRegressor`, `StackingRegressor` and `partial_dependence` accept it.
  Its tags now declare that it accepts NaN features, so selectors such as
  `SequentialFeatureSelector` pass NaN through, and that it requires a
  target. `score` is still the negative CDE loss.
- The logo sets "LAZY" as the row of a table: each letter in its own column,
  under a header row, between a top and a bottom rule. The letters and the
  square icon are unchanged.
- The `kumo` extra installs NVIDIA's `structured-data-models` 0.2.1 or
  later from PyPI, so Kumo Tabular no longer needs `lazy setup kumo`. The
  release's Kumo Tabular code is identical to the commit LAZY was validated
  on. Without a name, `lazy setup` does nothing for Kumo Tabular.
- On its native grid, TabICL and Kumo Tabular take each row's probability
  outside the grid from the CDF that tabulating the density already
  evaluates at the grid's ends, instead of interpolating every row's
  quantiles a second time. Densities, scores and the warning about that
  probability are unchanged to the bit.
- Densities of a histogram or a mixture of histograms on a grid other than
  their own buckets now come from differencing the distribution's CDF at the
  grid's edges, as quantile distributions' already did. They move by float
  rounding only. On the test fakes of every backend, the test stand-ins,
  the `LazyEnsembleModel` pools and the stand-ins of `benchmarks/cpu.py`,
  densities moved by at most 3.3e-13 of a row's peak (8.7e-12 absolute),
  and medians and scores by at most 5.2e-13 and 4.3e-13, both on the bagged
  benchmark stand-in. Densities on a model's own buckets, and those of
  TabICL and Kumo Tabular, are unchanged to the bit. The native grid of a
  bagged TabPFN or LimiX model, or of TabFM with more than one dither, joins
  the members' edges, so its densities move by rounding too. Where a bucket
  spans several bins of equal density, rounding can now pick another of
  those tied bins as the mode. With `benchmarks/cpu.py --repeats 5` (500
  query rows, a 4-CPU Linux machine, four threads, NumPy 2.5.3), the median
  time of `predict_proba` fell from 0.33 s to 0.27 s and its peak memory
  from 229 MiB to 172 MiB for four bagged members of 5,000 buckets, and from
  0.0132 s to 0.0117 s for 999 quantiles.
- `predict`, `score` and `evaluate` on a model that chunks its queries (every
  backend but TabFM, alone or through `LazyModel`) show one progress bar for
  the whole call; each member of a `LazyEnsembleModel` keeps its own. They
  used to show one bar per block of about 256 MB of densities, so several in a
  row for a large query set on a fine grid. Their results are unchanged.

### Deprecated

- `lazy.is_cached` returns `False` for an unknown backend or version and
  warns with a `FutureWarning`. From 0.7.0 it raises
  `lazy.UnknownNameError`.
- Features given as an array or list of strings that hold numbers (`str` or
  `bytes`) are still converted to float64, but warn with a `FutureWarning`.
  A DataFrame column of strings is still accepted without a warning, as in
  scikit-learn. From 0.7.0 they raise `ValueError`, as scikit-learn does.
- A DataFrame whose column names mix strings with other types is still read
  as having no feature names, but warns with a `FutureWarning`. From 0.7.0
  it raises `TypeError`, as scikit-learn does.
- Predicting on an `X` with no rows warns with a `FutureWarning` in
  `predict`, `predict_proba`, `predict_pdf`, `predict_cdf`,
  `predict_quantiles`, `predict_interval`, `predict_pit` and
  `predict_distribution`, which still return empty results. From 0.7.0 they
  raise `ValueError`, as scikit-learn does and as `fit`, `score` and
  `evaluate` already do.
- `lazy setup kumo` warns with a `FutureWarning` and will be removed in
  0.7.0. It still installs `structured-data-models` when
  `sdm.models.KumoTabular` does not import. The `requested` argument of the
  backends' `setup` hook, which only this path uses, goes with it.
  `lazy.models.kumo.REPOSITORY_BUILD`, the git requirement it used to
  install, also warns and will be removed in 0.7.0.
- `Grid.rebin` and `Grid.from_quantiles` warn with a `FutureWarning` and
  will be removed in 0.7.0, since nothing in LAZY uses them now.
  `HistogramDistribution(edges, probs).on_grid(grid)` and
  `QuantileDistribution(levels, values).on_grid(grid)` replace them, but
  they normalize the densities by the grid's convention, and the second adds
  tails beyond the outermost levels.
- The `bin_edges` argument of the `lazy.metrics` functions warns with a
  `FutureWarning` and will be removed in 0.7.0. Results are unchanged. Pass
  `lazy.Grid.from_edges(bin_edges, normalization="histogram")` as the grid
  instead.
- `lazy.plotting.style.better_step`, `stacked_hist`, `running_median`,
  `side_colorbar` and `grid_figsize` (also `lazy.plotting.grid_figsize`) warn
  with a `FutureWarning` and will be removed in 0.7.0. Nothing in lazy uses
  them, and they still work.

### Removed

- `lazy.datasets` no longer reads files cached in `~/.cache/lazy-photoz`,
  the directory used before the package's rename in 0.1.0, which has warned
  since then. Files left there are downloaded again into `data_home()`.
- The backend class attribute `setup_needs_name`, and the `then` argument
  of `lazy.models._ensemble.import_extra`, are removed, since no backend
  needs them now. `lazy setup --help` no longer lists backends that
  install only when named.
- The `backends` dependency group. A source checkout gets every backend
  with `uv sync --extra all`.

### Fixed

- LimiX-2 fits a bagged context in which one bag's targets are all equal.
  It raised `ValueError` when such a bag's standard deviation rounded to
  about 1e-17 rather than to 0; the bag is now left unscaled, as TabICL and
  Kumo Tabular leave it.
- With `tabpfn` 9.2 or later installed, TabPFN passes
  `enable_torch_compile=False` instead of leaving it to upstream's default,
  so a later default cannot change its densities. Outputs do not change.
- The docs state each measured figure with its setup, and the figures that
  their sources or re-measurements contradict are corrected. TabFM's
  key/value cache saves a ratio of 1.6 in time at the defaults (30.1 s
  against 47.4 s), not 26 times the compute, and `TabFMPerformanceWarning`
  now says so. LimiX-2's cache takes about 3 GB per member at 20,000 rows,
  not 2 GB. A refit on a shared
  network takes 0.03 s for TabPFN-3.5 and 0.1 s for LimiX-2, from 0.7 s and
  1.3 s. TabFM in bfloat16 moves a typical row's density by about 2% of its
  peak and single bins by up to 15%, and the NLL on DC1 by at most 0.002;
  TabPFN's int8 cache moves a row by up to 3% of its peak. Smaller
  corrections: the rms error of extrapolation on the chirp reaches 2.5, Kumo
  Tabular matches NVIDIA's code to 2e-7 and its cached GPU path to 0.6% of a
  row's quantile spread, and TabFM's `n_dither=3` costs 1.8 to 2.8 times a
  prediction.
- `load_dataset("bike_sharing")` and `load_dataset("kings_county")` load
  under pandas 3.1, where they raised a `TypeError` from scikit-learn's pandas
  ARFF parser. These two datasets are now read with scikit-learn's
  `liac-arff` parser. Their categories are unchanged. Their floats can differ
  by one unit in the last place from what the pandas parser gave under
  pandas 3.0, and `kings_county`'s integer columns are now float64.
- `on_grid`, and so `predict_proba`, keeps the mass of a zero-width bucket of
  a histogram, which `cdf`, `ppf` and `mean` already counted as a point mass.
  It dropped that mass before: masses `[0.5, 0.5]` in buckets with edges
  `[0, 1, 1]` gave bin masses `[0.5, 0.5, 0, 0]` on the edges
  `[0, 0.5, 1, 1.5, 2]`; they are now `[0.25, 0.25, 0.5, 0]`. Such buckets
  arise from TabPFN's float32 borders, which `bucket_masses` returns without
  the regressor.
- A point mass on a bin edge, from a zero-width bucket or from tied
  quantiles, now falls in the bin above the edge, and one on the grid's top
  edge in the last bin, as `Grid.bin_index` assigns a true value there for
  the scores. `histogramize` counted it in the bin below, and dropped one on
  the grid's bottom edge, as did the probability outside the grid that the
  native-grid warning counts. Distributions without point masses are
  unchanged to the bit. `Grid.rebin` now computes its densities as
  `HistogramDistribution` does, so it gets this fix too.
- A TabPFN model fitted on a GPU can now be unpickled on a machine without
  one. Its pickle held the regressors' CUDA tensors, which torch refuses to
  load there.
- A LimiX or TabFM model fitted on a GPU with `device="auto"` and unpickled
  on a machine without one now runs on the CPU with mixed precision off, and
  its `provenance_` says so. LimiX already moved to the CPU but kept mixed
  precision on in `mixed_precision_` and `provenance_`, and TabFM tried to
  load its weights on the missing GPU. In the other direction, a TabFM or
  TabPFN model fitted on the CPU with `device="auto"` and unpickled on a
  machine with a GPU now runs on that GPU, as TabICL and Kumo Tabular
  already did.
- An unpickled `LazyEnsembleModel` now reports its members' devices as they
  are after unpickling in `provenance_["members"]`, not as they were when
  pickled.
- An unpickled TabICL model predicting from two threads at once could load
  a second copy of its network from the checkpoint and then raise an
  `AttributeError`. The network is now attached once, under a lock.
- An unpickled Kumo, LimiX or TabICL model predicting from two threads at
  once could read the checkpoint path of the machine that made the pickle,
  which on another machine does not exist. The path is now found again
  before the model marks it found.
- LimiX-2 predicting from two threads at once could leave `sys.stdout` on
  a discarded buffer, so that later prints vanished. The threads now take
  turns to silence upstream's prints.
- An unpickled Kumo model predicting from two threads at once recorded each
  key/value cache twice, one recording after the other, and briefly held
  two caches on the device. It now records each cache once, under a lock.
- An unpickled LimiX model predicting from two threads at once could fail
  with a `TypeError`, or leave members uncached depending on timing: the
  second thread used the key/value caches before the first had rebuilt
  them. The second thread now waits for the rebuild.
- Threads predicting on one TabFM model changed its `bin_prior_`, which
  every prediction wrote. `fit` now sets it. Its value is the same: the bin
  fractions of the last member group's context, at the last dither. A model
  pickled without it gets it when unpickled.
- The warning that a cached checkpoint snapshot may be incomplete names the
  user's line, as lazy's other warnings do, instead of a line inside lazy.
- TabPFN no longer claims that chunking the queries gives the same bits on a
  CPU (`exact_chunking` is now `False`), and Kumo Tabular's docs no longer
  claim a bit-for-bit cache there: on a 4-core x86 CPU both agree only to
  float rounding. Kumo Tabular's `kv_cache_rtol` is 1e-3, above the largest
  difference measured there, 2.2e-4 of a density.
- LimiX-2's `provenance_["kv_cache"]` is `False` when no member's cache fit
  in the device's memory, as `kv_cache_` already was. It said `True`. The
  warning about uncached members now also says that a smaller `bag_size` or
  fewer `n_estimators` needs less memory and runs faster, but changes the
  answers.
- A row slice or a concatenation of a `MixtureDistribution` keeps its
  weights bit for bit, instead of normalizing them again. For a bagged model
  of 6, 7, 13 or some other numbers of members, a second normalization moved
  them by one unit in the last place, so `predict_distribution(X).ppf` and
  a slice of it differed from `predict_quantiles` in the last bits. They now
  agree exactly, and `predict_distribution` returns weights that differ from
  before by at most one unit in the last place.
- `MixtureDistribution` and `QuantileDistribution.average` raise a
  `ValueError` for weights that are not finite, are negative, or do not have
  a positive finite sum. All-zero or infinite weights used to give NaN
  weights, and so NaN results, with only a `RuntimeWarning`.
- A `Grid` copies the edges it is given, instead of making the caller's
  array read-only.
- Two grids compare equal when their edges agree to 1e-12 of the largest
  edge's magnitude and to 1e-6 of the narrowest bin's width, instead of to
  1e-12 absolutely. A grid rebuilt from its
  own centers far from zero (from 10⁷) was unequal to itself, and grids whose
  edges are all below 1e-12 were equal however they differed. The metrics
  check a `bin_edges` passed with a `Grid` by the same rule, and score with
  the grid's own edges.
- An empty row of a `HistogramDistribution` (all masses zero) put
  probability on its zero-width buckets, which made a point mass at each.
  It is now uniform over the buckets of positive width only.
- On Windows, two processes downloading the same dataset file into a
  shared cache no longer fail with a `PermissionError`: the one that finishes
  second keeps the copy already in place.
- The API reference describes every public constant. Most showed as empty
  entries, since the descriptions were comments that it does not read.
- A target with several columns raises a `ValueError`; one with half as
  many rows as X was flattened and fitted on interleaved values. A column
  vector is flattened with a `DataConversionWarning`, as in scikit-learn,
  which points at the caller's line.
- Sparse features raise a `TypeError` that names the dense alternative, and
  complex features or targets raise instead of losing their imaginary
  part. The errors for `y=None` and for a wrong or zero feature count use
  scikit-learn's wording.
- TabFM's `n_dither` and `keep_cache_on_device`, and TabPFN's
  `ignore_pretraining_limits`, are validated at `fit` with a `ValueError`
  that names the argument. A float or string `n_dither` failed deep inside,
  and `n_dither=True` was taken as 1.
- `LazyModel.set_params(model=...)` with a value that is not a backend name,
  such as an array, no longer raises; as for any unknown name, `fit` refuses
  it. scikit-learn's `check_do_not_raise_errors_in_init_or_set_params` now
  passes.
- A pandas Series target works on pandas 2.x: `fit`, `score`, `evaluate`
  and `predict_pit` raised `TypeError` for it, since NumPy's masked-array
  helper read the Series' internal block manager instead of its values.
- A TabFM build that has `TabFM.prefill` and `TabFM.decode` but lacks
  another API that lazy's KV-cache path calls, such as
  `EnsembleGenerator.transform_context_only`, now falls back to upstream
  `predict_proba` with a `TabFMPerformanceWarning`. Prediction raised
  `AttributeError` before. `lazy setup` now installs the pinned repository
  build over such a build.
- The documentation no longer calls TabPFN-3.5-fast a smaller copy of
  TabPFN-3.5, which it is not, and no longer gives a CDE loss for it that
  disagrees with the scaling page's table. Several facts that pages
  repeated, such as TabPFN's pretraining limits on the limits page, now
  link to the page that holds them, and the pages no longer describe plans,
  past behaviour or the test suite. The citation's BibTeX entry has a plain
  key and keeps the title's capitals.

### Security

- LimiX's source, which `lazy` downloads into its cache, is checked against
  a pinned hash of the validated commit's code before it runs, and runs
  from a private copy. Before, code or bytecode planted in a shared
  `LAZY_DATA_HOME` ran in every LimiX user's `fit`. A checkout named by
  `LAZY_LIMIX_SRC` is still run as it is.
- `fetch_dc1` checks the cached DC1 test file's SHA-256 before it unpickles
  it, once per process. Its catalog has an object column, so loading it
  unpickles, and so runs, what the file holds. Before, any cached file of
  the right size was trusted, so anyone able to write to a shared
  `LAZY_DATA_HOME` could plant code there.
- The other DC1 arrays load without pickle.

## [0.5.1] - 2026-10-28

A citation release, which archives lazy-tfm on Zenodo with its new
citation metadata.

### Added

- lazy-tfm is citable: every release is archived on Zenodo, under the
  concept DOI 10.5281/zenodo.23751881, and `CITATION.cff` gives both authors
  with their ORCIDs and the ML4PS 2026 paper as the preferred citation, which
  GitHub shows as "Cite this repository". The "Citing" page gives Zenodo's
  BibTeX entry, and the README carries the DOI badge.

## [0.5.0] - 2026-10-28

A documentation release: a tutorial on feature importance, a guide page on
Apple silicon and AMD GPUs, and a README that points to the documentation.

### Added

- The tutorials gain "Feature importance", which uses scikit-learn's
  permutation importance and SHAP values to show which features drive a
  model's prediction and its uncertainty.
- The User guide gains "Apple silicon and AMD GPUs", on what to expect of
  each model on hardware other than NVIDIA GPUs, which LAZY has not been
  tested on, and what to try there.

### Changed

- The README keeps only its opening and points to the documentation, which
  holds everything it used to repeat.
- "Adding a backend" and its worked example sit under "Contributing to
  lazy-tfm" in the documentation's sidebar.

## [0.4.0] - 2026-10-17

### Added

- NVIDIA's Kumo Tabular joins as a backend: `LazyModel("kumo")`, or
  `lazy.KumoQuantile`, in three sizes (`version="v1-large"`, the default,
  `"v1-medium"` and `"v1-small"`), with its weights under the OpenMDW License
  1.1, which permits commercial use. `pip install 'lazy-tfm[kumo]'` installs
  its dependencies and `lazy setup kumo` installs NVIDIA's
  `structured-data-models` from GitHub, since it is not on PyPI. Its own recipe, its trimmed mean of
  the members' quantiles included, is pinned as `transforms="auto"`.
- The documentation gains "Adding a backend", a contributor's page with the
  contract a new backend meets, a skeleton module and a checklist, and
  "Worked example: Kumo Tabular", which follows the Kumo backend through it.
- Backends gain class hooks for what shared code used to do by name:
  `cpu_note`, `prefetch`, `clear_upstream_caches` and `setup`, which
  `lazy setup` now runs for every registered backend, or for the backends it
  names (`lazy setup kumo`).
- Each pinned checkpoint describes its model (`display_name`, `parameters`,
  `size_bytes`, `license_name`, `gpu`, `cpu`), and each backend its method
  (`method_note`). The documentation's tables of the models are written from
  them at every build.
- `QuantileDistribution.trimmed_average` combines quantile functions by a
  trimmed mean, and a backend can combine its members that way
  (`member_combination="quantile_trimmed_average"`, with
  `member_trim_fraction` and a `_predict_members` hook), with the trim taken
  over every member of the ensemble at once.
- Backends gain two class attributes: `context_limit_note`, which says in the
  `ContextSizeWarning` what is known of a model's context limit (LimiX-2
  degrades beyond it, while Kumo Tabular was only pretrained up to it), and
  `setup_needs_name`, from which `lazy setup --help` lists the backends it
  installs only when they are named.
- A `backends` dependency group gives a checkout of the repository Kumo
  Tabular's `structured-data-models` at the validated commit
  (`uv sync --extra all --group backends`). It is not a default group, since
  it brings PyTorch, so a plain `uv sync` and CI stay light, and in a checkout
  `lazy setup` prints that command instead of a `uv add`.

### Changed

- The probability beyond the outermost quantile of a `QuantileDistribution`
  (TabICL, Kumo Tabular) is spread evenly over one quantile spacing past it,
  rather than sitting as a point mass on it, which made a one-bin spike on a
  fine grid that could become the mode and raise the CDE loss. Tied
  outermost quantiles use the nearest spacing that is not zero, and only a
  constant row keeps its point mass. Densities between the outermost
  quantiles are unchanged, `ppf` below the first level or above the last now
  falls inside the tails, `to_qp` passes the tails' ends to qp, and the
  TabICL and Kumo Tabular golden densities were recorded again.
- Kumo Tabular combines its members by a plain mean under an explicit
  `transforms`, which drops every model's recipe extras, and by the creators'
  20% trimmed mean only under `transforms="auto"`; `trim_fraction_` and
  `provenance_` record the trim used. Its random sign flips of the columns and
  the target stay under any `transforms`.
- TabICL and Kumo Tabular set `n_quantiles_` at `fit`, and a prediction no
  longer changes any attribute of the estimator.
- Kumo Tabular's `provenance_` records each group's transforms and column
  shuffle, and an unpickled model's records follow the device it now runs on.

- The `EnsembleSizeWarning` message no longer quotes measured losses; it
  points to "Ensembles of models" in the documentation, which gives them.
- `lazy-tfm` now requires NumPy 2.0, which the metrics already needed.
- A fit that fails, or whose inputs are refused, leaves the model unfitted
  rather than keeping a half-updated earlier fit.
- TabICL and Kumo Tabular no longer claim bit-for-bit chunking: a one-row
  chunk differs from a longer one by float rounding (about 1e-6 on a CPU).
- `datasets.load_trainz` returns both files on the DC1 grid, as densities.
- `SelectionSplit` keeps its parent catalog, so that `summary` can describe it
  in any band and depth.

### Fixed

- Distributions: a CDF rounding just above 1 made `histogramize`, the
  mixtures' quantiles and mode, and the ensemble's pooling raise, and PIT
  could exceed 1; a value on a histogram's last edge counted as outside it
  (density and NLL); concatenating mixtures with different weights gave
  wrong answers; variances lost their precision far from zero.
- Threads: fits and predictions sharing a TabICL network could answer from
  each other's context, and overlapping threaded fits could deadlock on the
  network cache or leave TabPFN's loader replaced.
- TabICL pickles no longer carry the network or a fixed checkpoint path, so
  they load on another machine and a bagged model shares one network.
- Masked astropy columns and arrays are read with their masked entries as
  missing, instead of their raw values (a -99 placeholder became a target).
- The soft outlier clip no longer turns a column with one observed context
  value into missing values.
- LimiX drops a column that is constant on the context apart from missing
  values, which made a query's answer depend on the rest of its chunk.
- Bagged TabFM keeps the probability of a coarse bin that no bag reaches.
- A constant context target on TabICL or Kumo Tabular is scaled by one, found
  by its exact range. Before, the rounding of its standard deviation decided
  between a nearly point-like answer (a target of 0.1) and one of full width
  (a target of 0.3).
- The warning of TabICL and Kumo Tabular about probability outside the native
  grid counts the exact probability there, tails included.
- Kumo Tabular refuses `device="mps"`, which has no double precision, with a
  clear error before it loads the weights.
- `lazy setup` decides whether it runs in a uv project from the environment it
  runs in, not from the working directory, so it installs into a pip or conda
  environment used from inside a uv project's directory. In a uv project's
  own environment it prints the `uv add` to run (with `--project` when run
  from elsewhere) and says why it exits with an error, under `--dry-run` too.
- A partly downloaded snapshot checkpoint (TabFM's) no longer counts as
  cached.
- TabPFN's `bucket_masses` returns the float64 bucket borders the backend
  uses, given the regressor.
- Datasets: downloaded files are readable by other users of a shared data
  directory; `holdout_rows` are validated and a biased training set is never
  empty; the PIT plots and `binned_quantiles` drop non-finite and
  out-of-range values; `SelectionSplit.summary` uses the band and depth it is
  given for the catalog row.

## [0.3.0] - 2026-10-08

A cross-model ensemble, `LazyEnsembleModel`, pools the densities of several
models fitted on the same context.

### Added

- `lazy.LazyEnsembleModel` fits several models on the same context and pools
  their densities with equal weights: `pooling="geometric"` (the default, a
  renormalized product), `"linear"` (a mixture) or `"quantile"` (averaged
  quantile functions), on a fine internal grid (`native_grid_`, `pool_bins`).
  Members are backend names or configured models, tunable as
  `<member>__<parameter>`; `lazy.EnsembleSizeWarning` is raised when two or
  more members are fitted on fewer than 3,000 context rows. The User guide
  gains "Ensembles of models", and the tutorials an ensemble example.

### Changed

- The figures of "What are Tabular Foundation Models?" are drawn as SVG, and
  the page's overview is revised.

## [0.2.0] - 2026-10-05

Every backend now takes one set of parameters with one set of defaults, and no
upstream default decides a prediction unseen. **Default predictions change**
(below); to reproduce 0.1.x numbers, pin `random_state` (TabPFN and TabICL 42,
TabFM 1) and, on TabFM, `n_estimators=4`.

### Added

- `lazy setup` (also `python -m lazy setup`) finishes the installs PyPI
  cannot: it installs TabFM's repository build, with its KV-cache API, over
  the PyPI release, with pip or uv, and downloads LimiX's source. In a uv
  project it prints the `uv add` that keeps the build instead of installing
  one `uv sync` would undo. `--dry-run` says what it would do.
- LimiX's source no longer needs installing: when neither `LAZY_LIMIX_SRC`
  nor an installed `LimiX` provides it, the archive of the validated commit is
  downloaded once into `limix/<commit>` under the catalogs' cache directory
  and loaded from there. `download_checkpoint("limix")` fetches it too,
  `is_cached("limix")` checks for it, and `HF_HUB_OFFLINE=1` prevents the
  download.
- TabFM's opt-in `n_coarse_bins="grid"`, `n_fine_bins="grid"` takes its
  classes from the constructor's `y_grid` (at most 100 bins), so the density
  lives on that grid with no rebinning; empty bins get zero probability.
  `provenance_["bins"]` records the layout, and new fitted attributes `bins_`,
  `n_coarse_bins_`, `n_fine_bins_` and `bin_grid_` expose it.
- Loaded networks are shared across fits in one process: a later fit with the
  same backend, version, checkpoint (file and pinned revision), device and
  weight precision reuses the network instead of reading the checkpoint again,
  so a grid search or a sweep loads each network once (a one-member refit on
  500 rows on a GB10: TabPFN-3.5 0.8 s to under 0.1 s, LimiX-2 1.5 s to
  0.2 s). Predictions are bit-identical with and without sharing.
  `lazy.clear_model_cache()` frees the networks, and
  `lazy.set_model_cache(False)` or `LAZY_MODEL_CACHE=0` turns sharing off.
  TabFM's backbone goes through this cache instead of TabFM's own, which could
  not be emptied.
- `Distribution.pit(values)` gives each row's CDF at its own value, shape
  `(n_rows,)`, exactly and without the `(n, n)` table `cdf` would build, and
  `predict_pit(X, y)` gives exact PIT values from the native distribution.
- A logo: "LAZY" with the L, A and Y drawn as data and the Z as predictions
  with error bars, in the docs, the README and the favicon. It is drawn by
  `docs/logo/make_logo.py`.
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
- `predict_interval(X, coverage=0.95)` on every model: the central interval
  holding `coverage` of each row's probability, shape `(n_samples, 2)`,
  computed exactly from the native distribution.
- `lazy.datasets.load_dataset(name)` and `list_datasets()`: one loader for
  the demonstration datasets (California housing, insurance, diamonds, King
  County, bike sharing, wine quality, Ames, concrete, energy, kin8nm,
  protein, yacht, Year Prediction MSD, all pinned to an OpenML data id, and
  DC1), downloaded once and cached under `data_home()`, and `"chirp"`, a
  one-feature toy problem generated on the spot (a noisy chirp with chunks cut
  out as the test set, its size, noise and gaps set by keywords).
  `split=True` returns the train/test split of DC1 (the challenge's) and of
  the chirp (its gaps), as two `Dataset`s or, with `return_X_y=True`, as
  `X_train, X_test, y_train, y_test`.
- Three proper scores in `lazy.metrics`: `crps` (the continuous ranked
  probability score), `nll` (the negative log predictive density, floored at
  `NLL_DENSITY_FLOOR`) and `pinball_loss` (for `predict_quantiles`), with
  per-object `per_object_crps` and `per_object_nll`. `summarize`, `evaluate`
  and `PDFMetrics` gain `crps` and `nll`, appended after the existing columns,
  which are unchanged.
- New documentation pages: the landing page's demo, "What are Tabular
  Foundation Models?", and a page for each supported model under "Supported
  models". The quickstart is gone; its advice for machines without a GPU is
  now in the installation page. The figures are drawn by
  `docs/figures/make_figures.py`.
- The User guide is rebuilt as explanation and reference: what tabular
  foundation models are, from model output to distribution (with the
  metrics), one interface for every model, choosing a model, scaling and
  performance, limits and pitfalls, reproducibility, clusters and offline
  use, the demo datasets, and an FAQ. Its worked examples move to the
  tutorials.
- The Tutorials page is a thumbnail gallery, generated by `docs/conf.py`
  from each tutorial's title, optional `gallery: description:` metadata and
  the figure cell tagged `thumbnail`, which `execute.sh` cuts into
  `thumbnails/<name>.png` on the `tutorials` branch.

### Changed

- TabFM's `n_coarse_bins` and `n_fine_bins` default to `"auto"`:
  floor(sqrt(n/5)) bins per level (2 to 10), about five context rows per bin.
  Contexts of 500 rows or more keep 10x10 and give bit-identical predictions;
  on yacht's 277 rows CRPS goes from 0.631 to 0.517 and NLL from 1.28 to 0.50.
  TabFM now fits contexts of any size under `"auto"`, and `n_dither=3` is
  documented as the setting to use when accuracy matters.
- `load_dataset("diamonds")` returns cut, color and clarity as ordered
  categoricals from worst to best (cut Fair to Ideal, color J to D, clarity I1
  to IF), so `.cat.codes` gives their natural order; it raises `ValueError` if
  the source's categories change.
- Bagging is the same on every backend. `lazy` draws every member's bag (the
  paper's draw), and each member fits its transform, outlier clip and target
  standardization on its own bag. TabPFN runs one regressor per bag, sharing
  one loaded network, instead of its own row subsampling, which standardized
  the target and placed its buckets on the whole context. TabFM's members are
  handed `lazy`'s bags (coarse level: the bag; fine level `j`: the bag's rows
  in coarse bin `j`) instead of drawing their own through `max_num_rows`, with
  their preprocessing refitted per member; bagged TabFM members run one at a
  time. Under bagging the native grid of TabPFN and LimiX-2 is the union of
  every member's buckets, and TabICL's spans every context target.
- Member seeds come from `SeedSequence([random_state, i])`, and a group is
  seeded by its first member, replacing `random_state + i` and
  `random_state + 1000 * g`, which gave neighboring seeds overlapping members.
  **Default predictions change** by about the scatter between seeds; the
  golden files are re-recorded.
- The point metrics and plots measure plain residuals, `y_pred - y_true`, by
  default: `scale` defaults to `"none"` in `evaluate`, `summarize`,
  `summarize_scores`, `point_metrics`, `evaluate_grid_pdfs`, `plot_residuals`
  and `diagnostic_panel` (new there). Pass `scale="1+y"` for the
  photometric-redshift convention and DC1's numbers. Likewise
  `plot_actual_vs_predicted` draws DC1's outlier boundary only with
  `outlier_lines=True`.
- A model without a native grid defaults to 200 bins over its training
  targets' range, widened by 5% on each side, instead of DC1's grid;
  `as_grid(None)` raises, since only a model knows its default. `fit`
  records the targets' range as `y_range_`.
- `lazy-tfm[all]` installs everything, the `qp` extra included; a test keeps
  every future extra in it.
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

### Renamed

The general-purpose API no longer uses redshift vocabulary; it follows
scikit-learn's `y` for the target. Only `lazy.datasets` and `lazy.selection`,
which load astronomical data, keep it. There are no aliases: replace

| 0.1 | 0.2 |
| --- | --- |
| `z_grid` (every estimator, `LazyModel`, `predict_proba`, `predict`, `score`, `evaluate`, metrics, plots) | `y_grid` |
| `z_true`, `z_pred` (metrics, plots) | `y_true`, `y_pred` |
| `Grid.z_min`, `Grid.z_max`, `Grid.linear(z_min, z_max, n_bins)` | `Grid.y_min`, `Grid.y_max`, `Grid.linear(y_min, y_max, n_bins)` |
| point estimates `"z_peak"`, `"z_weight"`, `"z_mean"`, `"z_median"` (`predict(method=)`, `POINT_ESTIMATORS`, `grid_point_estimates` keys, `point=`) | `"mode"`, `"peak_mean"`, `"mean"`, `"median"` |
| `metrics.z_peak`, `metrics.z_weight` | `metrics.mode`, `metrics.peak_mean` |
| `plotting.plot_zphot_ztrue` | `plotting.plot_actual_vs_predicted` |
| `plotting.plot_nz` | `plotting.plot_stacked_pdfs` |
| `scale="1+z"` | `scale="1+y"` |
| summary column `median_abs_ez` | `median_abs_error` |
| `lazy.DC1_GRID`, `lazy.grid.DC1_GRID` | `lazy.datasets.DC1_GRID` |

### Removed

- TabPFN's `n_estimators="auto"`.
- TabFM's `decode_chunk_rows`: `chunk_size` bounds both of its paths, and `0`
  means one pass on both.
- TabFM's `provenance_["bag_rows_drawn_by"]` and `_members.GROUP_SEED_STRIDE`.

### Fixed

- TabFM's bfloat16 rounding on a GPU is documented at its measured size, about
  3% of a typical row's peak density and up to a quarter of the peak in single
  bins on any path, with CRPS and NLL moving about as much as with a change of
  seed; the docs had claimed a few percent between the cached and uncached
  paths.
- Chunked TabPFN predictions on a CPU are bit-identical to unchunked ones
  again: histogram-grid normalization sums each row on its own rather than in
  one BLAS matrix-vector product, whose rounding depended on how many rows came
  with it.
- `Distribution.on_grid(grid)` returns exactly what `predict_proba(X, grid)`
  returns for histogram and mixture distributions too. It dropped probability
  outside the grid without renormalizing and ignored the grid's normalization,
  so on `DC1_GRID` it was off by up to about 18% for low-redshift objects.
  `predict_proba` is unchanged.
- The docs say what `fit` costs: it trains no weights, but with `kv_cache=True`
  it runs the context through the network once per member (TabFM defers that
  to `predict`). Limits and pitfalls gains a section on output grids finer than
  a model's own bins, which can show spurious peaks.
- With LimiX's code from `LAZY_LIMIX_SRC`, `provenance_["package"]` reads
  "LimiX (source checkout at <commit>)" instead of "LimiX (not installed)".
- The `plot_pit_qq` and `plot_pit` docstrings describe the PIT shapes
  correctly: an S-shape means densities too narrow or too wide, and a curve
  wholly below or above the diagonal (a sloped histogram) means predictions
  biased high or low.
- TabPFN checks its size limits on each member's bag, so a `bag_size` within
  the limit runs a larger context (above the CPU cap included) without
  `ignore_pretraining_limits`, and the error names the `bag_size` and
  `n_estimators` that fix it. Bagged TabPFN no longer loads one network per
  member (peak memory 8.05 GB against 4.34 GB unbagged on a 43,730-row
  context; 1.90 GB now).
- LimiX-2's `ContextSizeWarning` is documented as firing for bags larger than
  20,000 rows too.
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

[Unreleased]: https://github.com/biprateep/lazy-tfm/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/biprateep/lazy-tfm/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/biprateep/lazy-tfm/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/biprateep/lazy-tfm/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/biprateep/lazy-tfm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/biprateep/lazy-tfm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/biprateep/lazy-tfm/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/biprateep/lazy-tfm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/biprateep/lazy-tfm/releases/tag/v0.1.0
