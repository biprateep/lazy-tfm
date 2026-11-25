# Demo datasets

To try a model on something other than photometry,
{func}`~lazy.datasets.load_dataset` loads a regression benchmark by name. The
data are downloaded on first use and cached under
{func}`~lazy.datasets.data_home`, so later calls need no network. It returns a
{class}`~lazy.datasets.Dataset` holding the features as a
{class}`pandas.DataFrame` (as the source gives them, with categorical columns
keeping the `category` dtype), the target as a float64 array, and the source,
license and citation of the data. With `return_X_y=True` it returns only the
features and the target. The one change to the features is that categorical
columns with a natural order are ordered from worst to best, so that
`.cat.codes` gives integers that keep their order. For `diamonds` these are the
cut (Fair to Ideal), the color (J to D) and the clarity (I1 to IF).
{func}`~lazy.datasets.list_datasets` lists every dataset with its size and
license without downloading anything:

```python
from lazy import datasets

X, y = datasets.load_dataset("california_housing", return_X_y=True)
datasets.list_datasets()
```

| Name | Rows | Features | Target | What it is | Source | License |
|---|---:|---:|---|---|---|---|
| `california_housing` | 20,640 | 8 | `medianHouseValue` | Median house values of California census block groups | [OpenML 44977](https://www.openml.org/d/44977) | Public |
| `insurance` | 1,338 | 6 | `charges` | Medical insurance charges of US individuals | [OpenML 46931](https://www.openml.org/d/46931) | DbCL v1.0 |
| `diamonds` | 53,940 | 9 | `price` | Prices of round-cut diamonds, from ggplot2 | [OpenML 46923](https://www.openml.org/d/46923) | MIT |
| `kings_county` | 21,613 | 21 | `price` | House sale prices in King County, WA, 2014-2015 | [OpenML 44989](https://www.openml.org/d/44989) | CC0 |
| `bike_sharing` | 17,379 | 11 | `count` | Hourly bike rental counts in Washington, DC | [OpenML 44063](https://www.openml.org/d/44063) | Public |
| `wine_quality` | 6,497 | 12 | `median_wine_quality` | Quality scores of red and white Portuguese wines | [OpenML 46964](https://www.openml.org/d/46964) | CC BY 4.0 |
| `ames_housing` | 2,930 | 80 | `Sale_Price` | House sale prices in Ames, Iowa, 2006-2010 | [OpenML 43926](https://www.openml.org/d/43926) | Public |
| `concrete` | 1,030 | 8 | `strength` | Compressive strength of concrete from its mixture and age | [OpenML 44959](https://www.openml.org/d/44959) | CC BY 4.0 |
| `energy` | 768 | 8 | `heating_load` | Heating load of simulated residential buildings | [OpenML 44960](https://www.openml.org/d/44960) | CC BY 4.0 |
| `kin8nm` | 8,192 | 8 | `y` | Simulated forward kinematics of an 8-link robot arm | [OpenML 44980](https://www.openml.org/d/44980) | Public |
| `protein` | 45,730 | 9 | `RMSD` | Deviation of protein tertiary structures, from CASP 5-9 | [OpenML 44963](https://www.openml.org/d/44963) | CC BY 4.0 |
| `yacht` | 308 | 6 | `Residuary.resistance` | Residuary resistance of sailing yacht hulls | [OpenML 42370](https://www.openml.org/d/42370) | CC0 |
| `year` | 515,345 | 90 | `year` | Release years of songs from their audio timbre | [OpenML 44027](https://www.openml.org/d/44027) | CC BY 4.0 |
| `dc1` | 434,476 | 12 | `redshift` | LSST DESC photo-z Data Challenge 1 galaxy photometry | [Zenodo 10975874](https://zenodo.org/records/10975874) | CC BY 4.0 |
| `plasticc_ddf` | 32,699 | 11 | `true_z` | PLAsTiCC simulated LSST light curves of extragalactic objects in the Deep Drilling Fields | [Zenodo 2539456](https://zenodo.org/records/2539456) | CC BY 4.0 |
| `chirp` | 1,000 | 1 | `y` | A noisy chirp with chunks cut out, generated on the spot | Generated | MIT |

Every dataset except DC1, PLAsTiCC and the chirp comes from
[OpenML](https://www.openml.org) and is pinned to its OpenML id, so a later
upload under the same name never changes what is loaded. DC1 is
{func}`~lazy.datasets.fetch_dc1` with both of its files concatenated and the
`"mag-color"` features. The chirp is generated on the spot.

## Train and test splits

DC1 and the chirp come with a train/test split of their own, which
`split=True` returns: DC1's challenge split of 43,486 training and 390,990
test galaxies, and for the chirp, the chunks cut out of the curve. The split
is returned as two {class}`~lazy.datasets.Dataset` objects, or with
`return_X_y=True` as four arrays in the order of
{func}`sklearn.model_selection.train_test_split`. Without `split`, the
training and test rows are concatenated, in that order. The OpenML datasets
have no split of their own, and `split=True` raises a `ValueError` for them.

```python
train, test = datasets.load_dataset("dc1", split=True)
X_train, X_test, y_train, y_test = datasets.load_dataset(
    "chirp", split=True, return_X_y=True, random_state=0
)
```

## MulTaBench

[MulTaBench](https://arxiv.org/abs/2605.10616) is a benchmark of tables with
columns of text or images. Its 40 regression datasets load by name, as
`"multabench_<name>"`, from the
[Hugging Face Hub](https://huggingface.co/multabench), each pinned to the commit
that the `multabench` package 0.2.1 pins. They need the `multabench` extra,
which reads their parquet tables:

```console
pip install 'lazy-tfm[multabench]'
```

20 of them have columns of text, and 20 have a column of images, often with
text too. `list_datasets()` lists them with the others: MulTaBench's tier
(`core` or `extended`) and kind are in the description. The image column holds
the paths of the extracted image files, which {mod}`lazy.image` reads, and
{attr}`Dataset.modalities <lazy.datasets.Dataset.modalities>` names each
column of text or images, as MulTaBench types them. That map goes straight
to {class}`lazy.Tabularizer`:

```python
train, test = datasets.load_dataset("multabench_khaadi_clothes", split=True)
model = lazy.make_pipeline(lazy.Tabularizer(train.modalities), lazy.LazyModel())
model.fit(train.X, train.y)
model.score(test.X, test.y)
```

`split=True` gives MulTaBench's own split, TabArena's: three folds repeated
twice with seed 42, chosen by `fold` (0 to 5, default 0). By default,
`size="10k"` keeps at most 10,000 training and 5,000 test rows of each split,
as MulTaBench's results do; `size="full"` keeps them all. The rows keep their
labels in the whole table.

The tables total 210 MB. The images of a dataset take from 10 MB to 9.3 GB,
24 GB for all 20. They download with the table and are extracted once under
{func}`~lazy.datasets.data_home`; `images=False` skips them, and the image
column then holds the files' names and is left out of `modalities`. Each dataset keeps the licence of its
source, which the `license` column links. If you use them, cite MulTaBench:
Arazi et al. 2026, *MulTaBench: Benchmarking Multimodal Tabular Learning with
Text and Image*, NeurIPS, arXiv:2605.10616.

## PLAsTiCC

The Photometric LSST Astronomical Time-Series Classification Challenge
(PLAsTiCC) simulated three years of LSST light curves of transient and
variable objects. `"plasticc_ddf"` is its test objects in the Deep Drilling
Fields (DDF), the fields that LSST revisits most often, from the unblinded
data on [Zenodo 2539456](https://zenodo.org/records/2539456). The first file
of test light curves holds exactly the 32,926 DDF objects. The 227 galactic
objects have no redshift, so the dataset keeps the 32,699 extragalactic ones,
from z = 0.04 to 3.35.

The target is the true redshift, `true_z`. Each row has 11 features:

| Column | What it holds |
|---|---|
| `hostgal_photoz`, `hostgal_photoz_err` | The photometric redshift of the host galaxy, and its error |
| `mwebv` | The Milky Way's reddening E(B-V) towards the object |
| `ra`, `decl` | The position, in degrees |
| `flux_u` to `flux_y` | One column per LSST band (`ugrizy`); each cell is a 1-D float64 array of the band's fluxes in order of time |

The six flux columns are time series. A median object has 56 points in each
band and 70 in `u`. {attr}`Dataset.modalities <lazy.datasets.Dataset.modalities>`
maps them to `"time"`, so {class}`lazy.Tabularizer` encodes them with
{mod}`lazy.time` encoders:

```python
dataset = datasets.load_dataset("plasticc_ddf")
model = lazy.make_pipeline(
    lazy.Tabularizer(dataset.modalities), lazy.LazyModel()
)
```

The loader drops every column that gives away the answer or is not an
observation: the other `true_*` columns, the spectroscopic redshift
`hostgal_specz`, the distance modulus `distmod`, the class `target`, the
true fluxes `tflux_*`, `ddf_bool`, `libid_cadence` and `object_id`. The
fluxes' errors and detection flags are left out too. The dataset has no
train/test split of its own.

The two files total 306 MB. The first load downloads them and parses them,
which takes about 10 seconds; it caches the parsed table, 89 MB, beside them,
and later loads read it in under a second. The data are licensed CC BY 4.0. If
you use them, cite the PLAsTiCC team et al. 2018, *The Photometric LSST
Astronomical Time-series Classification Challenge (PLAsTiCC): Data set*,
arXiv:1810.00001, and Kessler et al. 2019, *Models and Simulations for the
Photometric LSST Astronomical Time Series Classification Challenge
(PLAsTiCC)*, PASP 131, 094501. Hložek et al. 2023, ApJS 267, 25, report the
challenge's results.

## The chirp

The chirp is a toy problem with one feature, `x`, and the target
y = A(x) sin(φ(x)) on 0 ≤ x ≤ 10, where the amplitude A(x) = 1 + 0.2x grows
from 1 to 3 and the frequency φ′(x)/2π = 0.2 + 0.03x from 0.2 to 0.5 cycles
per unit of x. Gaussian noise is added to the training rows, and chunks of x
are cut out as the test rows without noise, so the true curve is known where
a model has to interpolate. It is the only dataset that takes options, passed
to {func}`~lazy.datasets.load_dataset` as keywords:

| Option | Default | What it sets |
|---|---|---|
| `n_samples` | 1000 | Points evenly spaced over 0 ≤ x ≤ 10, before the gaps are cut |
| `noise` | 0.2 | The standard deviation of the noise on the training targets |
| `n_gaps` | 3 | How many chunks to cut out; 0 for none |
| `gap_width` | 0.04 | Each chunk's width, as a fraction of the range of x |
| `random_state` | `None` | A seed or a NumPy Generator, for the noise and where the gaps fall |

The gaps never overlap: the range is cut into `n_gaps` equal segments, and
each segment holds one gap, placed at random.
