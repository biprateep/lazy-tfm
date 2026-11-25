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

Each modality has a *pretrained* encoder, a frozen network whose weights are
pinned to a commit on the Hugging Face Hub, and *classical* encoders, fixed
computations with no weights and no download. A modality's name picks its
pretrained encoder, as `LazyModel()` picks TabPFN.

| Name | Class | Columns |
| ---- | ----- | ------- |
| `"text_pretrained"` (`"text"`) | {class}`lazy.text.Pretrained` | by `model` |
| `"image_pretrained"` (`"image"`) | {class}`lazy.image.Pretrained` | by `model` |
| `"time_pretrained"` (`"time"`) | {class}`lazy.time.Pretrained` | by `model` |
| `"time_statistics"` | {class}`lazy.time.Statistics` | 13 |
| `"time_spectrum"` | {class}`lazy.time.Spectrum` | `n_octaves + 1` (9) |
| `"image_statistics"` | {class}`lazy.image.Statistics` | `8 * n_channels` (24) |
| `"image_pixels"` | {class}`lazy.image.Pixels` | `size**2 * n_channels` (192) |
| `"text_statistics"` | {class}`lazy.text.Statistics` | 8 |
| `"text_hashing"` | {class}`lazy.text.Hashing` | `n_features` (128) |

The count in parentheses is that of the default parameters. The models read
at most a few hundred features well ({doc}`limits`), so keep the total small.

## Pretrained encoders

The pretrained encoders need the `encoders` extra, which installs PyTorch,
Hugging Face `transformers` and `chronos-forecasting`:

```console
pip install 'lazy-tfm[encoders]'
```

Each modality's pretrained encoder runs one of several frozen networks, which
its `model` parameter names, as a backend's `version` does:

```python
lazy.text.Pretrained()  # e5-small-v2
lazy.text.Pretrained(model="gte-modernbert-base")  # a larger network
lazy.Tabularizer(
    {"cutout": lazy.image.Pretrained(model="clip-vit-base-patch16")}
)
```

Each network's weights are pinned to a commit on the Hugging Face Hub
({data}`lazy.encoders.CHECKPOINTS`), and its recipe (pooling, prompt, context)
is pinned in the encoder's class. Installing `lazy` downloads no weights. An
encoder downloads its network on first use, with a {class}`~lazy.LicenseWarning`
that names the network and its licence, and loads it once per process.
{func}`lazy.download_checkpoint` fetches one ahead of time, as for a model, by
modality and network: `lazy.download_checkpoint("text")` for the default,
`lazy.download_checkpoint("image:dinov2-large")` for another. So does the
command `lazy download text image:dinov2-large`, and `lazy models` lists them
all. Every network here has a licence that allows commercial use, and runs on
a CPU; each takes `device` as the models do.

```{include} _encoders.md
```

A row depends on its own item alone, up to float rounding: a batch pads its
items to one shape, and the items are batched in order of length. The
pretrained encoders cache their embeddings by default (see
[The cache](#the-cache)).

### How the networks were chosen

The networks were chosen from published evaluations of frozen embeddings as
features, and of embeddings fed to tabular foundation models (TFMs). No study
yet measures them in LAZY's own pipeline, which predicts a distribution; the
defaults will be confirmed on the MulTaBench benchmark.

- **Text.** On tables, a text encoder's rank on the MTEB leaderboard predicts
  its value poorly: on MulTaBench text regression and on STRABLE, TF-IDF
  features match or beat sentence encoders. The default, E5-small-v2, is the
  text encoder of MulTaBench and of the TabPFN-3.5 report, so results compare
  directly with theirs. BGE-small-en-v1.5 scores higher on MTEB at the same
  size, and GTE-ModernBERT-base is the strongest MTEB model near 150 million
  parameters and reads 8,192 tokens.
- **Images.** DINOv2-small with registers matches DINOv3-small, the
  MulTaBench encoder, on ImageNet (87.3 against 87.0 ImageNet-ReaL linear
  accuracy); DINOv3 requires a sign-up to download. SigLIP 2 base is the
  strongest frozen network of its size there. CLIP ViT-B/16 is the strongest
  general network in a study of galaxy redshifts from images (R² 0.43 on HSC
  and 0.47 on JWST, against 0.38 and 0.44 for DINOv3-base). A network trained
  on galaxies does better still, but each needs its own package.
- **Time series.** On light curves of variable stars from ZTF, Chronos-T5
  embeddings beat Chronos-Bolt's on every classification metric, though
  features made for light curves beat every pretrained network. Chronos-T5
  reads one value per token. No general network reads observation times or
  uncertainties, so an irregular light curve is read as evenly sampled.

Issue [#344](https://github.com/biprateep/lazy-tfm/issues/344) gives the
sources and the figures.

### Images that are not 8-bit

The image networks read 8-bit RGB. An image of another type, such as a cutout
of fluxes, is stretched band by band first. With $p_1$ and $p_{99}$ the 1st and
99th percentiles of a band's values $v$,

$$
x = \frac{v - p_1}{p_{99} - p_1}, \qquad
u = \min\!\left(1, \max\!\left(0,
    \frac{\operatorname{asinh}(20 x)}{\operatorname{asinh}(20)}\right)\right),
$$

and the network reads $255\,u$. This is the stretch that studies of galaxy
images with general networks use. It keeps an image's contrast but not its
brightness or its colour, so a flux or a colour that matters belongs in a
table column of its own. A network reads one band (grey) or three.

### Rotated and flipped views

`views` rotates and flips each image before the network reads it, and averages
the embeddings of the views. Each view is one more forward pass. The views are
the eight rotations and flips of a square, named in {data}`lazy.image.VIEWS`
after Pillow's transposes. With $g$ a view, $f$ the network and $V$ the views,
the embedding of an image $x$ is

$$
e(x) = \frac{1}{|V|} \sum_{g \in V} f(g\,x).
$$

`views="d4"` takes all eight, so the embedding of every rotation or flip of an
image is the same, up to float rounding; the networks themselves are not
invariant. One view, such
as `views=["rotate_90"]`, gives one member of a pool its own view of the
images:

```python
views = ["identity", "rotate_90", "rotate_180", "rotate_270"]
model = lazy.LazyEnsembleModel(
    [
        lazy.make_pipeline(
            lazy.Tabularizer({"cutout": lazy.image.Pretrained(views=[view])}),
            lazy.LazyModel(),
        )
        for view in views
    ]
)
```

The default, `None`, embeds the image as it is.

### Citing

If you use a pretrained encoder, cite its network as well as LAZY:

- E5: Wang, L., et al. (2022). Text Embeddings by Weakly-Supervised
  Contrastive Pre-training. arXiv:2212.03533.
- BGE: Xiao, S., et al. (2024). C-Pack: Packed Resources for General Chinese
  Embeddings. SIGIR.
- all-MiniLM-L6-v2: Wang, W., et al. (2020). MiniLM. NeurIPS. Reimers, N.,
  and Gurevych, I. (2019). Sentence-BERT. EMNLP.
- GTE-ModernBERT: Li, Z., et al. (2023). Towards General Text Embeddings with
  Multi-stage Contrastive Learning. arXiv:2308.03281.
- DINOv2: Oquab, M., et al. (2024). DINOv2: Learning Robust Visual Features
  without Supervision. TMLR. Darcet, T., et al. (2024). Vision Transformers
  Need Registers. ICLR.
- SigLIP 2: Tschannen, M., et al. (2025). SigLIP 2. arXiv:2502.14786.
- CLIP: Radford, A., et al. (2021). Learning Transferable Visual Models From
  Natural Language Supervision. ICML.
- Chronos: Ansari, A. F., et al. (2024). Chronos: Learning the Language of
  Time Series. TMLR.

## The Tabularizer

{class}`lazy.Tabularizer` takes a map from a column to its encoder:

- a modality, `"image"`, `"text"` or `"time"`, for the default encoder of
  that modality ({data}`lazy.encoders.DEFAULT_ENCODERS`);
- the name of an encoder, such as `"time_spectrum"`;
- an encoder, such as `lazy.text.Hashing(n_features=32)`.

An embedding can have many columns, such as the 192 of `image_pixels`. The
Tabularizer reduces each embedding wider than `n_components` (default 30): it
standardizes each column of the embedding and keeps the first `n_components`
principal components. Both are fitted on the rows given to `fit`, the context,
as the model is. A missing item gives a row of `NaN`, which is left out of the
fit and stays `NaN`. A context with no more items that are not missing than
`n_components` keeps one component fewer than it has such items, with a
warning. `n_components=None` keeps every column.

The default follows the published pipelines (MulTaBench, TabPFN's own text
features, STRABLE), which all keep 30 components per column. STRABLE measured
15 as worse than 30, and 60 or 120 as no better and slower.

The other columns pass through, unless `remainder` says otherwise. The output
is a DataFrame with the index of the input, and its columns carry the names of
`get_feature_names_out`, such as `notes__pc0` or `remainder__price`, so the
model after it knows each column by name. The encoded columns are float64,
and a column that passes through keeps its type. A search tunes the
Tabularizer as any step, for example over `tabularizer__n_components`.

One setting, `n_components`, serves every column. For another reduction, or
a search over an encoder's own parameters, build a
{class}`~sklearn.compose.ColumnTransformer` of encoders and any other steps:
an encoder is a scikit-learn transformer on its own, and
{func}`~lazy.encoders.get_encoder` builds the default encoder of a modality,
as in `get_encoder("text")`.

## Classical encoders

The classical encoders are baselines that need no download, and work offline.

### Time series

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

### Images

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

### Text

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
files). The pretrained encoders default to `cache=True`, so a cross-validation
or a second session runs each network once per item. The classical encoders
default to `cache=False`: computing a row costs less than reading its file.
