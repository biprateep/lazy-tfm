# Contributing to lazy-tfm

Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/biprateep/lazy-tfm/issues).

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/), and `uv.lock`
pins every dependency, so everyone works in the same environment:

```bash
git clone https://github.com/biprateep/lazy-tfm
cd lazy-tfm
uv sync                     # the package, editable, plus the dev tools
uv sync --extra all         # ...and every backend
uv run pre-commit install   # lint and format on every commit
```

A plain `uv sync` installs no extra, so it and CI stay light, without PyTorch.

The hooks run ruff, mypy and the style checker from this environment, at the
versions in `uv.lock`, so run `uv sync` again after the lockfile changes.

`uv sync` also removes packages that are not in the lockfile. If you keep
extra packages in the environment, use `uv sync --inexact`.

## Code style

Code follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
checked by ruff (formatting and lint, configured in `pyproject.toml`), mypy,
and the checker of the code-style skill (`.claude/skills/code-style/`), which
covers the rules ruff cannot express. In practice, this means 80-column lines,
imports of modules rather than of their members, Google docstrings (`Args:`,
`Returns:`, `Raises:`) that give shapes and units, and an annotation on every
signature. The one deliberate exception is scikit-learn's vocabulary, in which
the feature matrix is `X` and the target `y`, as scikit-learn requires.

## Checks

```bash
uv run pytest               # unit tests and doctests; no GPU or checkpoint needed
uv run pre-commit run --all-files   # ruff format, ruff check, mypy, style
```

Tests that download and load a pinned checkpoint are marked `checkpoint` and
are skipped by default. To run them:

```bash
LAZY_RUN_CHECKPOINT_TESTS=1 uv run pytest
```

The checkpoint-tests workflow runs them on a CPU on request: from the Actions
tab, optionally with a `pytest -k` expression, or by adding the label
`checkpoints` to a pull request; removing the label and adding it back runs
it again.

The docs-checks workflow checks, on request, what the docs build does not.
Run it from the Actions tab ("Docs checks" -> Run workflow), or by adding the
label `docs-checks` to a pull request; removing the label and adding it back
runs it again. One job checks that every external link resolves
(`sphinx-build -b linkcheck`). The other runs `docs/snippets/*.py` on a CPU,
with the locked backend packages, and downloads the TabPFN v3.5-fast weights.

The golden tests compare each backend's densities with files in
`tests/lazy/golden/`, at four torch threads. TabPFN's file was recorded on a
GitHub runner and matches only there: another processor moves TabPFN by a few
percent of the peak. To re-record golden files, which needs Biprateep's
approval, run the workflow by hand with `record` naming the backends, and
commit those backends' files from its `golden` artifact.

On every pull request, CI runs the tests on Python 3.12 to 3.14 against the
lockfile, on Linux, with Python 3.12 on macOS and Windows too. It also runs
them with every backend package on a CPU, and at the lowest dependency
versions `pyproject.toml` allows. It runs the pre-commit
hooks, a docs build with warnings treated as errors, and a build of the package
itself. Nothing runs on a schedule. The smoke-test workflow, run by hand from
the Actions tab, tests against the newest resolvable dependencies, backend
packages included, to warn of an upstream release that breaks `lazy`. It also
tries Python 3.15, which `lazy` does not support yet, without failing the run.
The upstream-check workflow, also run by hand, reports what upstream has
published since `lazy` pinned it: newer checkpoint revisions, new checkpoint
repositories and newer backend packages.

## Benchmarks

A change expected to affect speed or memory reports a benchmark against
`main`. `benchmarks/cpu.py` times lazy's own prediction code (densities, point
estimates, quantiles, scores, `evaluate` and ensemble pooling) on test
stand-ins sized like the real models, with no network or checkpoint:

```bash
git fetch origin main
uv run python benchmarks/cpu.py --against origin/main --repeats 5
```

It runs each case in a fresh process, alternating this checkout and `main`,
and prints the setup, each side's median time and peak memory, and the ratio.
The stand-ins' own work is the same on both sides, so a ratio understates
lazy's own speed-up. To see how much the ratio varies on your machine, compare
the checkout with itself: `--against HEAD`.

## How changes are reviewed

Changes reach `main` through pull requests; a fix of a few lines of prose or
formatting may go straight to `main`. A pull request asks one or both
maintainers for a review, and waits for it before merging, when it changes
something users rely on (an imported name, the `lazy` command, a `LAZY_*`
variable, a saved-file format, or a signature, default, return value, error
or warning), changes outputs for the same inputs and seed, or changes a core
design choice. Its Review section names the reason. Other pull requests, such
as internal changes, new features, speed-ups and tests, merge once CI passes
and their tests cover the change. Waiting for a review never holds up other
work.

## Adding a backend

A new foundation model joins LAZY as a backend: a subclass of
`ContextEnsembleEstimator`, registered by name. [Adding a
backend](docs/adding_a_backend.md) describes the contract it must meet, gives
a skeleton module, and lists everything else it needs (a pinned checkpoint and
recipe for every version, a pip extra, a docs page, golden densities and test
fakes). `tests/lazy/test_backend_completeness.py` checks every registered
backend for all of them. [Worked example: Kumo
Tabular](docs/adding_a_backend_kumo.md) follows one backend through every step.

## Changing dependencies

Edit `pyproject.toml`, then run `uv lock` and commit both files. CI runs
`uv sync --locked`, which fails if the lockfile is out of date.

The published metadata cannot contain direct URLs, since PyPI rejects them.
To use a development build of a dependency, add it under `[tool.uv.sources]`,
following the example of `tabfm`.

## Documentation

The docs are written in MyST Markdown, built with Sphinx and published on Read
the Docs. To build them locally:

```bash
uv sync --group docs
uv run sphinx-build -W -b html docs docs/_build/html
```

The API reference is generated from the docstrings, which follow the Google
style (rendered by napoleon).

### The logo

`docs/logo/make_logo.py` draws the logo, its white version, the square icon
and the favicons (PNG at 64 and 180 pixels, and an SVG), and writes them
into `docs/_static/`, where the docs and the README pick them up. The
[brand page](docs/brand.md) lists the files and the colors. To change the
logo, edit the script and rerun it:

```bash
uv run python docs/logo/make_logo.py
```

The output is deterministic, so a rerun with no change leaves the files
untouched.

### The figures

`docs/figures/make_figures.py` draws every figure of the landing page and the
user guide, and writes them as SVG into `docs/figures/figs/`. The figures of a
model's output run TabPFN-3.5-fast on a CPU, so the script needs the `tabpfn`
extra and the TabPFN checkpoint, even to redraw only a concept figure.
Rerun it after a change to a model's output or to the landing snippet
`docs/snippets/quickstart.py`:

```bash
uv run --extra tabpfn python docs/figures/make_figures.py
```

The output is deterministic, so a rerun with no change leaves the files
untouched.

### Tutorials

Tutorials are py:percent scripts in `docs/tutorials/`, and main tracks only
those. Since they need a GPU, neither CI nor Read the Docs runs them. The
executed notebooks, with their outputs and figures, live on the orphan branch
`tutorials`, from which the docs build fetches them and the Colab and GitHub
buttons open them. This keeps megabytes of images out of main's history.
After changing a tutorial, re-execute it on a GPU machine and commit the
notebook, with the gallery thumbnail that `execute.sh` cuts from its figure
cell tagged `thumbnail`, to that branch. A tutorial that uses LimiX-2
downloads LimiX's source on first use, or takes a checkout from
`LAZY_LIMIX_SRC` (see the installation page):

```bash
docs/tutorials/execute.sh basic_usage   # writes the notebook and thumbnails/basic_usage.png (git-ignored)
git worktree add ../lazy-tutorials tutorials
cp docs/tutorials/basic_usage.ipynb ../lazy-tutorials/
mkdir -p ../lazy-tutorials/thumbnails
cp docs/tutorials/thumbnails/basic_usage.png ../lazy-tutorials/thumbnails/
git -C ../lazy-tutorials add basic_usage.ipynb thumbnails/basic_usage.png
git -C ../lazy-tutorials commit -m "Re-execute basic_usage"
git push origin tutorials
```

`execute.sh` stamps each notebook's metadata with the lazy version and
commit it ran on and the SHA-256 of its script. The docs show that version and
commit, and note a notebook whose script has changed since it ran.

A local docs build uses the notebook and thumbnail in `docs/tutorials/` if
there are any, and fetches them from the branch otherwise: from `origin`, or,
in a fork with no such branch, from `biprateep/lazy-tfm`. If neither fetch
works, a local build writes each missing notebook from its script, without
outputs, and the next build that can fetch replaces it; on Read the Docs or in
CI (`READTHEDOCS=True` or `CI=true`), it fails. The
docs of a release tag `v<version>` use the tag `tutorials/v<version>`, if it
exists, in place of the branch. The gallery on the
tutorials page is generated by `docs/conf.py` from each tutorial's title, its
optional `gallery: description:` metadata and its thumbnail, in the order of
`TUTORIAL_ORDER`. Every user-visible change also gets
a line in `CHANGELOG.md`, under `[Unreleased]`.

## Making a release

Releases go to PyPI from GitHub, by trusted publishing, and nobody uploads
from a laptop.

Before the first release, there is a one-time setup. On
[PyPI](https://pypi.org/manage/account/publishing/) and
[TestPyPI](https://test.pypi.org/manage/account/publishing/), add a trusted
publisher with owner `biprateep`, repository `lazy-tfm`, workflow
`publish-to-pypi.yml`, and environment `pypi` or `testpypi` respectively.
Then, in the GitHub repository settings, create the environments `pypi` and
`testpypi`. We recommend requiring a reviewer on `pypi`.

Each release starts by making sure CI is green on `main`. Next, set the
version, for example with `uv version 0.1.0`, which updates `pyproject.toml`
and `uv.lock`. In `CHANGELOG.md`, rename `[Unreleased]` to the version and
date, add a fresh empty `[Unreleased]` section, and commit both changes. As an
optional dry run, run the **Publish** workflow by hand from the Actions tab,
which uploads to TestPyPI, and check that the result installs with
`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ lazy-tfm`.
To keep the release's docs on the notebooks executed for it, tag the
`tutorials` branch too:
`git tag tutorials/v0.1.0 origin/tutorials && git push origin tutorials/v0.1.0`.
Then create a GitHub release with the tag `v<version>`, for example `v0.1.0`.
Publishing it runs the workflow, which checks that the tag matches the
version, builds and checks the package, runs the test suite on the tagged
commit, and uploads it to PyPI only if all of them pass. Finally, move
to the next development version, for example `uv version 0.2.0.dev0`, and
commit.
