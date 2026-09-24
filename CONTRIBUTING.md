# Contributing to lazy-photoz

Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/biprateep/lazy-photoz/issues).

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/), and `uv.lock`
pins every dependency, so everyone works in the same environment:

```bash
git clone https://github.com/biprateep/lazy-photoz
cd lazy-photoz
uv sync                     # the package, editable, plus the dev tools
uv sync --extra all         # ...and all three foundation-model backends
uv run pre-commit install   # lint and format on every commit
```

`uv sync` also removes packages that are not in the lockfile. If you keep extra
packages in the environment, use `uv sync --inexact`.

## Code style

Code follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
checked by ruff (formatting and lint, configured in `pyproject.toml`) and mypy.
In short: 80-column lines; import modules, not their members; Google
docstrings (`Args:`, `Returns:`, `Raises:`) that give shapes and units; every
signature annotated. The one deliberate exception is scikit-learn's vocabulary:
the feature matrix is `X` and the target `y`, as scikit-learn requires.

## Checks

```bash
uv run pytest               # unit tests and doctests; no GPU or checkpoint needed
uv run pre-commit run --all-files   # ruff format, ruff check, mypy
```

Tests that need a GPU and a downloaded checkpoint are marked `gpu` and skipped
by default. To run them:

```bash
LAZY_RUN_CHECKPOINT_TESTS=1 uv run pytest
```

CI runs the tests on Python 3.12 to 3.14 against the lockfile, a daily job
against the newest resolvable dependencies, the pre-commit hooks, a docs build
with warnings treated as errors, and a build of the package itself.

## Changing dependencies

Edit `pyproject.toml`, then run `uv lock` and commit both files. CI runs
`uv sync --locked`, which fails if the lockfile is out of date.

The published metadata cannot contain direct URLs, since PyPI rejects them. To
use a development build of a dependency, add it under `[tool.uv.sources]`;
see how `tabfm` is done.

## Documentation

The docs are Sphinx with MyST Markdown, published on Read the Docs. To build
them locally:

```bash
uv sync --group docs
uv run sphinx-build -W -b html docs docs/_build/html
```

The API reference is generated from the docstrings, which use the numpydoc
format. Add a line to `CHANGELOG.md`, under `[Unreleased]`, for every
user-visible change.

## Making a release

Releases go to PyPI from GitHub, by trusted publishing; nobody uploads from a
laptop.

One-time setup, before the first release:

1. On [PyPI](https://pypi.org/manage/account/publishing/) and
   [TestPyPI](https://test.pypi.org/manage/account/publishing/), add a trusted
   publisher: owner `biprateep`, repository `lazy-photoz`, workflow
   `publish-to-pypi.yml`, and environment `pypi` or `testpypi` respectively.
2. In the GitHub repository settings, create the environments `pypi` and
   `testpypi`. Requiring a reviewer on `pypi` is recommended.

Each release:

1. Make sure CI is green on `main`.
2. Set the version, for example `uv version 0.1.0`, which updates
   `pyproject.toml` and `uv.lock`.
3. In `CHANGELOG.md`, rename `[Unreleased]` to the version and date, and add a
   fresh empty `[Unreleased]` section. Commit both changes.
4. Optional dry run: in the Actions tab, run the **Publish** workflow by hand.
   It uploads to TestPyPI. Check that the result installs with
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ lazy-photoz`.
5. Create a GitHub release with the tag `v<version>`, for example `v0.1.0`.
   Publishing it runs the workflow, which checks that the tag matches the
   version, builds and checks the package, and uploads it to PyPI.
6. Move to the next development version, for example
   `uv version 0.2.0.dev0`, and commit.
