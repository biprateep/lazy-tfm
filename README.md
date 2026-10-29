# Reproducing the paper

This branch of [lazy-tfm](https://github.com/biprateep/lazy-tfm) holds the
code that produces every result and figure of the paper on tabular foundation
models as photometric redshift estimators, on the LSST DESC DC1 catalogues.
It shares no history with `main`.

Every piece of data handling goes through the `lazy` package (lazy-tfm): the
DC1 catalogues and the HSC-like spectroscopic selection (`lazy.datasets`),
the common redshift grid of 4,000 bins on [0, 2] (`lazy.grid`), the metrics
(`lazy.metrics`) and the three foundation models, TabPFN-3.5, LimiX-2 and
Kumo Tabular (`lazy.LazyModel`). The three bespoke estimators (FlexZBoost,
CMNN and GPz) are fitted with flexcode, our own CMNN and RAIL's GPz, on the
same training sets and scored with the same `lazy` code.

## Environment

One [uv](https://docs.astral.sh/uv/) environment holds everything, on Python
3.12; `uv.lock` pins every package, lazy-tfm at its 0.4.0 release.

```bash
git clone --branch paper-reproduce --single-branch https://github.com/biprateep/lazy-tfm.git fm4pz
cd fm4pz
uv sync                 # the locked environment, in .venv
uv run lazy setup       # LimiX-2's source, at the commit lazy-tfm pins
```

The catalogues and the model checkpoints download on first use. The
foundation models need an NVIDIA GPU; the bespoke estimators run on CPUs.

## Running it

The experiments come in two directories, one per training regime, each in
three steps run in order. Each step writes its products under `results/`
(never committed), where the next step reads them, and skips what is already
done, so an interrupted step resumes where it stopped.

| Step | `representative/` | `biased/` |
|---|---|---|
| 1. `tune.py` | hyperparameter search per method and training size (1k to 43,486) | the same, on the biased training sets (1k to 35,011) |
| 2. `produce.py` | the chosen settings refitted on 5 subsamples, scored on the DC1 test file | the same on the representative hold-out, plus the calibration matrix (0 to 10k calibration galaxies added) |
| 3. `plot.py` | Figure 1: `figures/unbiased_scaling.png` | the appendix scaling figure and the five calibration matrices |

```bash
uv run python representative/tune.py
uv run python representative/produce.py
uv run python representative/plot.py

uv run python biased/tune.py
uv run python biased/produce.py
uv run python biased/plot.py
```

Every script takes `--methods` (and steps 1 and 2 take `--sizes`; step 2
`--seeds`) to run part of a step, for instance to split the work between
machines; `--help` lists the options. `FM4PZ_RESULTS` moves the results
directory.

The paper's runs took several days on two GPU nodes. The searches dominate:
each is 5-fold cross-validation on the training set, never the test set, with
the trial budgets in `src/fm4pz/config.py` (300 trials for CMNN, 200 or 100
for FlexZBoost, 100 for GPz, and until 40 trials complete for each foundation
model). The paper's settings came from two such rounds, the second
warm-started from the first; one round with the final search spaces, as here,
lands on settings of the same quality, but parallel searches are not
repeatable trial for trial, so the numbers may differ slightly from the
paper's. Given the same settings, steps 2 and 3 reproduce them exactly.

## Layout

| Path | Contents |
|---|---|
| `representative/`, `biased/` | the three steps of each regime |
| `src/fm4pz/` | the code they share: training sets (`splits`), grid and metrics (`scoring`), searches (`tune`), production runs (`produce`), the bespoke estimators (`baselines/`), the foundation models (`tfms/`), the steps (`pipeline`) and the figures (`figures`) |
| `figures/` | the final figures, as they appear in the paper |
| `results/` | every intermediate product (not committed) |

## License

MIT; see [LICENSE](LICENSE).
