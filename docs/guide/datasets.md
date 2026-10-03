# Demo datasets

To try a model on something other than photometry,
{func}`~lazy.datasets.load_dataset` loads a regression benchmark by name,
downloading it on first use and caching it under
{func}`~lazy.datasets.data_home`, so later calls read the cached copy and need
no network. It returns a {class}`~lazy.datasets.Dataset` holding the features as
a {class}`pandas.DataFrame`, as the source gives them (categorical columns keep
the `category` dtype), the target as a float64 array, and the source, license and
citation of the data, while `return_X_y=True` gives only the features and the
target. The one change to the features is that categorical columns with a
natural order come ordered from worst to best, which for `diamonds` are the cut
(Fair to Ideal), the color (J to D) and the clarity (I1 to IF), so that
`.cat.codes` turns them into integers that keep their order.
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
| `chirp` | 1,000 | 1 | `y` | A noisy chirp with chunks cut out, generated on the spot | Generated | MIT |

Every dataset except DC1 and the chirp comes from
[OpenML](https://www.openml.org) and is pinned to its OpenML id, so a later
upload under the same name never changes what is loaded. DC1 is
{func}`~lazy.datasets.fetch_dc1` with both of its files concatenated and the
`"mag-color"` features, and the chirp is generated on the spot rather than
downloaded.

## Train and test splits

DC1 and the chirp come with a train/test split of their own, which
`split=True` returns: DC1's challenge split of 43,486 training and 390,990
test galaxies, and for the chirp, the chunks cut out of the curve. The split
comes as two {class}`~lazy.datasets.Dataset` objects, or with
`return_X_y=True` as four arrays in the order of
{func}`sklearn.model_selection.train_test_split`. Without `split`, the
training and test rows come concatenated, in that order. The OpenML datasets
have no split of their own, and `split=True` raises a `ValueError` for them.

```python
train, test = datasets.load_dataset("dc1", split=True)
X_train, X_test, y_train, y_test = datasets.load_dataset(
    "chirp", split=True, return_X_y=True, random_state=0
)
```

## The chirp

The chirp is a toy problem with one feature, `x`, and the target
y = A(x) sin(φ(x)) on 0 ≤ x ≤ 10, where the amplitude A(x) = 1 + 0.2x grows
from 1 to 3 and the frequency φ′(x)/2π = 0.2 + 0.03x from 0.2 to 0.5 cycles
per unit of x. Gaussian noise is added to the training rows, and chunks of x
are cut out as the test rows without noise, so the true curve is known where
a model has to interpolate. It is the only dataset that takes options, which
are passed to {func}`~lazy.datasets.load_dataset` as keywords:

| Option | Default | What it sets |
|---|---|---|
| `n_samples` | 1000 | Points evenly spaced over 0 ≤ x ≤ 10, before the gaps are cut |
| `noise` | 0.2 | The standard deviation of the noise on the training targets |
| `n_gaps` | 3 | How many chunks to cut out; 0 for none |
| `gap_width` | 0.04 | Each chunk's width, as a fraction of the range of x |
| `random_state` | `None` | A seed or a NumPy Generator, for the noise and where the gaps fall |

The gaps never overlap: the range is cut into `n_gaps` equal segments, and
each segment holds one gap, placed at random.
