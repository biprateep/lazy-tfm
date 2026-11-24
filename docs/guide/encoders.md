# Encoders

A model takes a table of numbers. An *encoder* turns a column of items that
are not numbers (time series, images, texts) into numeric columns, one row
per item, which the model takes as features. {class}`lazy.Tabularizer` puts
the encoders in front of the model: it names the encoder of each column of
items, and passes the numeric columns through.

```python
import lazy

tab = lazy.Tabularizer({"light_curve": "time", "notes": "text"})
model = lazy.make_pipeline(tab, lazy.LazyModel())
model.fit(table, y)
low, median, high = model.predict_quantiles(new_table, [0.16, 0.5, 0.84]).T
```

An encoder is frozen: an item's row does not depend on the items that the
encoder was fitted on, and `fit` only checks its input. A missing item (`None`
or `NaN`) gets a row of `NaN`, which the models read as missing features.
{func}`lazy.encoders.list_encoders` lists the encoders by name, and
{func}`lazy.encoders.get_encoder` builds one from its name.

The encoders on this page are *classical*: fixed computations with no weights
and no download. They are baselines, and they work offline.

| Name | Class | Columns |
| ---- | ----- | ------- |
| `"time_statistics"` | {class}`lazy.time.Statistics` | 13 |
| `"time_spectrum"` | {class}`lazy.time.Spectrum` | `n_octaves + 1` (9) |
| `"image_statistics"` | {class}`lazy.image.Statistics` | `8 * n_channels` (24) |
| `"image_pixels"` | {class}`lazy.image.Pixels` | `size**2 * n_channels` (192) |
| `"text_statistics"` | {class}`lazy.text.Statistics` | 8 |
| `"text_hashing"` | {class}`lazy.text.Hashing` | `n_features` (128) |

The count in parentheses is that of the default parameters. The models read
at most a few hundred features well ({doc}`limits`), so keep the total small.

## The Tabularizer

{class}`lazy.Tabularizer` takes a map from a column to its encoder:

- a modality, `"image"`, `"text"` or `"time"`, for the default encoder of
  that modality ({data}`lazy.encoders.DEFAULT_ENCODERS`);
- the name of an encoder, such as `"time_spectrum"`;
- an encoder, such as `lazy.text.Hashing(n_features=32)`.

An embedding can have many columns, such as the 192 of `image_pixels`. The
Tabularizer reduces each embedding wider than `n_components` (default 8): it
standardizes each column of the embedding and keeps the first `n_components`
principal components. Both are fitted on the rows given to `fit`, the context,
as the model is. A missing item gives a row of `NaN`, which is left out of the
fit and stays `NaN`; the context needs more items that are not missing than
`n_components`. `n_components=None` keeps every column.

The other columns pass through, unless `remainder` says otherwise, and
`get_feature_names_out` names every output column. A search tunes the
Tabularizer as any step, for example over `tabularizer__n_components`.

One setting, `n_components`, serves every column. For another reduction, or
a search over an encoder's own parameters, build a
{class}`~sklearn.compose.ColumnTransformer` of encoders and any other steps:
an encoder is a scikit-learn transformer on its own, and
{func}`~lazy.encoders.get_encoder` builds the default encoder of a modality,
as in `get_encoder("text")`.

## Time series

A time series is a row of a numeric array, or a sequence of numbers of any
length. Values that are not finite are dropped, and the rest is read as evenly
sampled. We write $x_1, \dots, x_n$ for the values that remain, $m$ for their
mean and $s$ for their standard deviation.

{class}`lazy.time.Statistics` gives 13 columns:

| Column | Value |
| ------ | ----- |
| `mean`, `std` | $m$ and $s$ |
| `min`, `q25`, `median`, `q75`, `max` | the quartiles |
| `skewness` | the mean of $((x_t - m)/s)^3$ |
| `kurtosis` | the mean of $((x_t - m)/s)^4$, minus 3 |
| `slope` | the least-squares slope of $x_t$ on $t$, per sample |
| `autocorrelation` | $\sum_t (x_t - m)(x_{t+1} - m) \,/\, \sum_t (x_t - m)^2$ |
| `roughness` | the mean of $\lvert x_{t+1} - x_t \rvert$ |
| `length` | $n$ |

A statistic that a series does not define, such as the skewness of a constant
series, is 0.

{class}`lazy.time.Spectrum` describes the periodic content. The discrete
Fourier transform $\hat x_k$ of the series, less its mean, splits its variance
among the frequencies $f_k = k/n$ cycles per sample, for $0 < f_k \le 1/2$:

$$
s^2 = \sum_{0 < f_k \le 1/2} P_k, \qquad
P_k = \frac{w_k \lvert \hat x_k \rvert^2}{n^2},
$$

with $w_k = 2$, except $w_k = 1$ at $f_k = 1/2$. Octave $j$ holds the
frequencies

$$
2^{-j-2} < f_k \le 2^{-j-1},
$$

and the last octave holds every frequency below. The columns are each octave's
share of the variance, from `octave0` (the fastest variations) downwards, and
`peak`, the frequency of the largest $P_k$. The octaves do not depend on
$n$, so series of different lengths compare. A constant series gives zeros.

## Images

An image is an array of shape (height, width) or (height, width, channels),
the path of an image file, or a Pillow image. An array keeps its values and
may have any number of channels. A file is read as 8-bit values with 1 (grey),
3 (RGB) or 4 (RGBA) channels, except a file of one channel of more than 8
bits, such as a 16-bit or floating-point TIFF, which keeps its values and needs
`n_channels=1`. Every image of a column has `n_channels`
channels (3 by default), and every pixel is finite.

{class}`lazy.image.Statistics` gives 8 columns per channel: the mean, the
standard deviation, the minimum, the three quartiles and the maximum of the
channel's values, and `gradient`, the mean absolute difference between
neighbouring pixels. It ignores where things are in the image.

{class}`lazy.image.Pixels` keeps where things are. It cuts the image into
`size` by `size` equal cells, and gives the mean of each channel over each
cell. A pixel that straddles two cells counts in each by the area inside it,
so images of different sizes give the same columns, and the mean of the image
is kept.

## Text

A text is one `str`.

{class}`lazy.text.Statistics` gives 8 columns: the numbers of `characters` and
of `words`, the mean `word_length`, the share of the words that are
`distinct`, and the shares of the characters that are `digits`, `uppercase`,
`punctuation` (ASCII's) and `whitespace`. A word is a run of characters
between whitespace.

{class}`lazy.text.Hashing` is scikit-learn's
{class}`~sklearn.feature_extraction.text.HashingVectorizer`. Each n-gram of a
lower-cased text adds or subtracts 1 in the column that its hash names, and
each row is scaled to unit length. By default the n-grams are of 2 to 4
characters inside each word, which suits short texts and tolerates spelling
variants; `analyzer="word"` hashes words. Hashing keeps no vocabulary, so the
encoder stays frozen. A method that learns its vocabulary from the data, such
as TF-IDF, is not an encoder; use scikit-learn's transformer as a pipeline
step for that.

## The cache

Every encoder takes `cache`. With `cache=True`, the row of each item is filed
on disk and never computed twice ({doc}`../adding_an_encoder` describes the
files). The classical encoders default to `cache=False`: computing a row costs
less than reading its file.
