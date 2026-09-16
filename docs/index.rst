LAZY
====

**Lazy but Accurate photo-Z for Yinz**

Photometric redshift PDFs from pretrained tabular foundation models. The models
are never fine-tuned: you hand them labelled galaxies as *context* and they
answer queries in one forward pass, so there is no training loop, no
hyper-parameter search and no per-survey retraining. Hence lazy.

.. warning::

   Pre-release. The API is still moving and the numbers in the paper are not
   final. Pin a commit if you depend on it.

Installation
------------

.. code-block:: console

   $ pip install lazy-photoz              # the grid, metrics, plots and datasets
   $ pip install 'lazy-photoz[tabfm]'     # + the TabFM backend
   $ pip install 'lazy-photoz[tabicl]'    # + the TabICLv2 backend
   $ pip install 'lazy-photoz[tabpfn]'    # + the TabPFN-3 backend
   $ pip install 'lazy-photoz[all]'       # + all three

The backends are optional so that the parts of the library that need no deep
learning stack -- the metrics, the plots, the grid machinery -- install in
seconds and run anywhere.

Pretrained weights are **not** bundled. They are fetched from the Hugging Face
Hub the first time you predict and cached in the usual HF cache
(``~/.cache/huggingface/hub``) from then on. TabFM's classification checkpoint
is about 6.6 GB and is released by Google under a **non-commercial** licence;
TabICLv2's is about 100 MB; the TabPFN checkpoints run from 41 MB to 880 MB and
are **non-commercial** from Prior Labs, apart from ``v2``, which is Apache-2.0
with an attribution clause. Every licence is quoted in
:data:`lazy.CHECKPOINTS`. To warm the cache before running somewhere without a
network:

.. code-block:: python

   from lazy import download_checkpoint

   download_checkpoint("tabfm")

Quickstart
----------

.. code-block:: python

   from lazy import LazyModel, RedshiftGrid
   from lazy.datasets import fetch_dc1

   train, test = fetch_dc1(split=True)
   X_train, X_test = train.features("mag-color"), test.features("mag-color")

   model = LazyModel("tabfm", n_estimators=4, n_dither=3)
   model.fit(X_train, train.redshift)

   pdfs = model.predict_proba(X_test, RedshiftGrid.linear(0, 2, 200))
   z = model.predict(X_test, method="z_peak")
   print(model.evaluate(X_test, test.redshift))

``fetch_dc1`` downloads the LSST DESC PZ Data Challenge catalogue (about 1 GB,
checksummed, cached) in one call: ``split=True`` hands back the challenge's own
train/test split, and the default hands back both files concatenated, with
``source`` marking where each row came from, for when you want to make your own
splits.

:meth:`~lazy.datasets.Catalog.features` builds the tabular view the model sees
-- ``"mag"`` for the magnitudes and their errors, ``"mag-color"`` (the default)
for the reference magnitude and the adjacent colours with errors propagated in
quadrature, ``"all"`` for every magnitude, every colour and all their errors. It
is photometry-specific rather than general, which is why it lives on the
catalogue; it acts on a plain :class:`~pandas.DataFrame`, so a catalogue you
built yourself goes through the same code via
:meth:`~lazy.datasets.Catalog.from_frame` or
:meth:`~lazy.datasets.Catalog.build_features`. Any tabular features work -- the
DC1 helpers are a convenience, not a requirement.

The API
-------

It is scikit-learn's, with one difference. A photo-z model's natural output is a
density rather than a number, so
:meth:`~lazy.base.BasePhotoZEstimator.predict_proba` returns a density on a
redshift grid rather than class probabilities, and ``predict`` is a documented
reduction of it:

=================================  =====================================================
``fit(X, y)``                      Store the labelled context. No weights are updated.
``predict_proba(X, z_grid)``       ``(n_samples, n_bins)`` densities on ``z_grid``.
``predict_pdf(X, z_grid)``         Alias, for when "pdf" reads better than "proba".
``predict_cdf(X, z_grid)``         The same, cumulative.
``predict(X, method="z_peak")``    One redshift per row. Also ``z_weight``, ``z_mean``,
                                   ``z_median``.
``score(X, y)``                    Negative CDE loss, so higher is better.
``evaluate(X, y)``                 A one-row table of every diagnostic metric.
=================================  =====================================================

The returned arrays are **densities**, normalised to unit trapezoid mass over
the bin centres -- the convention every metric in :mod:`lazy.metrics` uses.
Multiply by ``grid.widths`` for per-bin probability masses.

``get_params`` / ``set_params`` / :func:`sklearn.base.clone` all work, so the
models drop into scikit-learn pipelines and search objects unmodified;
``LazyModel`` flattens its backend's parameters into its own, so
``GridSearchCV(model, {"n_dither": [1, 3]})`` needs no prefix.

The grid belongs to the prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing about fitting depends on the output binning: these models place their
internal bins by the quantiles of the *context* redshifts, and the grid only
enters at the final, exact rebinning step. So one fitted model answers on as
many grids as you like, without refitting:

.. code-block:: python

   model.predict_proba(X_test, RedshiftGrid.linear(0.0, 3.0, 300))
   model.predict_proba(X_test, np.linspace(0.005, 2.995, 300))   # or bin centres

The constructor still takes ``z_grid`` as a per-model default for when every
call would pass the same thing; a grid given at call time wins, and ``None`` at
both levels means :data:`~lazy.grid.DC1_GRID`.

Choosing a backend
------------------

===========  ==============================  =========  =======================================
Name         Method                          Weights    Best for
===========  ==============================  =========  =======================================
``tabfm``    Hierarchy of in-context          ~6.6 GB    Sharpest densities; multimodal PDFs.
             classifiers over equal-mass
             redshift bins.
``tabicl``   Quantiles of an in-context       ~100 MB    Fast baselines; modest hardware.
             regression head, differenced
             onto the grid.
``tabpfn``   Bucket masses of the bar        41-880 MB  Sharp densities; small checkpoint.
             distribution, rebinned onto
             the grid.
===========  ==============================  =========  =======================================

No backend needs a pinned or patched dependency. Peak memory is bounded by
``chunk_size`` on all of them, and chunking is exact: the in-context stage
builds its keys and values from the context rows alone, so a query row's answer
never depends on which other query rows share its chunk. The test suite asserts
the outputs are bit-identical rather than assuming it.

.. note::

   A ``tabfm`` build carrying the KV-cache API is detected and used
   automatically (``inference="auto"``). It prefills each ensemble member's
   context once instead of once per chunk, so it is faster on large query sets
   -- a speed optimisation, not a correctness requirement. The PyPI release does
   not have it; a build from the repository does::

      $ pip install 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm'

They all write onto whatever :class:`~lazy.grid.RedshiftGrid` you ask for --
any number of bins, any spacing, any range:

.. code-block:: python

   import numpy as np
   from lazy import LazyModel, RedshiftGrid

   model = LazyModel("tabfm", z_grid=RedshiftGrid.linear(0.0, 3.0, 300))
   pdfs = model.fit(X, z).predict_proba(X_test, np.linspace(0.005, 2.995, 300))

TabFM's classifier is limited to ten classes, but that limit applies to each
level of the internal bin hierarchy, never to the output grid.

Which model, exactly
--------------------

A backbone is a family, not a model, so every backend takes a ``version`` and
each version is a separately pinned checkpoint:

.. code-block:: python

   lazy.list_versions("tabpfn")
   # ['v2', 'v2.5', 'v2.6', 'v3', 'v3.5', 'v3.5-fast']

   model = LazyModel("tabpfn", version="v2.5")
   model.name_          # 'tabpfn:v2.5' -- the label evaluate() puts on its row

Sweeping ``version`` compares model versions on equal terms, and a results table
then says which one produced each row rather than only which family. What
actually answered is recorded on the fitted model, ready to be written out
beside the numbers:

.. code-block:: python

   model.fit(X_train, z_train).provenance_
   # {'backend': 'tabpfn', 'version': 'v3',
   #  'repo_id': 'Prior-Labs/tabpfn_3',
   #  'filename': 'tabpfn-v3-regressor-v3_default.ckpt',
   #  'revision': '24a16a89d245878b846555110985634aa2e656d7',
   #  'package': 'tabpfn 9.0.0', 'lazy': 'lazy-photoz 0.1.0.dev0',
   #  'device': 'cuda'}

It holds the weights *and* the code that read them, and no local paths, so it
means the same thing on another machine and survives a trip through JSON.
Settings are not in it -- ``get_params()`` has those.

:data:`lazy.CHECKPOINTS` is keyed ``"backend:version"`` and is the authority on
what each name loads, so :func:`~lazy.download_checkpoint` warms exactly the
file a run will read:

.. code-block:: python

   lazy.download_checkpoint("tabpfn", "v2.5")
   lazy.get_checkpoint("tabpfn", "v2.5").license_note

Versions are not interchangeable. TabPFN ``v2`` is pretrained for at most 10,000
context rows and ``v2.5`` for 50,000, against a million for ``v3``; ``v2`` is
also the only one under a commercial-use licence.

Caches, and running without a network
-------------------------------------

Nothing about where things are cached is baked into the package, and nothing in
it is platform-specific: the wheel is pure Python (``py3-none-any``), the
lockfile resolves for macOS, Linux and Windows on Python 3.12 to 3.14, and the
device is chosen at run time (``device="auto"``), never at install time.

Three caches, each redirectable, so a laptop and a cluster node run the same
code:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - What
     - Default location
     - Override
   * - Pretrained weights
     - ``~/.cache/huggingface/hub``
     - ``HF_HOME``
   * - Benchmark catalogues
     - ``$XDG_CACHE_HOME/lazy-photoz``, else ``~/.cache/lazy-photoz``
     - ``LAZY_DATA_HOME``, else ``XDG_CACHE_HOME``

Compute nodes often have no outbound network. Warm both caches on a login node
first, then run with downloads disabled so a missing file is an immediate,
explicit error rather than a hang:

.. code-block:: python

   from lazy import download_checkpoint, is_cached
   from lazy.datasets import fetch_dc1

   download_checkpoint("tabicl")          # on the login node
   fetch_dc1()

   is_cached("tabicl")                    # on the compute node: True
   fetch_dc1(download_if_missing=False)

The checkpoint a model loads is the one :func:`~lazy.download_checkpoint`
fetches -- the pinned revision in :data:`lazy.models.CHECKPOINTS` is passed to
the backend rather than left to its own default, so warming the cache cannot
prefetch the wrong file.

Memory is bounded by default on every backend: ``chunk_size`` (16384 query rows)
caps the largest intermediate, and chunking is numerically exact, so the default
costs nothing but a little repeated context work. Lower it on a small machine;
set it to ``0`` for a single pass.

Metrics and figures
-------------------

The LSST DESC PZ Data Challenge metric definitions are implemented in
:mod:`lazy.metrics`, read off the challenge's own scripts rather than
paraphrased, so the numbers are directly comparable with Schmidt et al. (2020):

.. code-block:: python

   import pandas as pd
   from lazy.metrics import summarize

   table = pd.concat([
       summarize(z_true, grid.centers, pdfs_a, label="TabFM"),
       summarize(z_true, grid.centers, pdfs_b, label="TabICLv2"),
       summarize(z_true, grid.centers, pdfs_c, label="TabPFN-3"),
   ])

:mod:`lazy.plotting` carries the publication figure style and the standard
diagnostics -- accuracy, calibration and N(z) -- drawn identically for every
method so that comparisons hold up by eye:

.. code-block:: python

   from lazy.plotting import use_style, diagnostic_panel

   use_style()
   fig = diagnostic_panel(z_true, grid.centers, pdfs, label="TabFM")

Contributing
------------

.. code-block:: console

   $ git clone https://github.com/biprateep/lazy-photoz
   $ cd lazy-photoz
   $ uv sync                  # locked environment, including dev tooling
   $ uv run pre-commit install
   $ uv run pytest

``uv sync --locked`` is what CI runs; if it fails, ``uv lock`` and commit the
result. Tests need neither a GPU nor a checkpoint -- the backends are imported
lazily and the estimator contract is covered by a backend-free stand-in.

.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
