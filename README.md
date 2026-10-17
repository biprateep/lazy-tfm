# fm4pz: the paper's experiments

This orphan branch of [lazy-tfm](https://github.com/biprateep/lazy-tfm) holds
everything needed to reproduce the paper on tabular foundation models as
photometric redshift estimators. It shares no history with `main`.

What is committed here, and what is not:

- **Committed:** the Python sources, the environment (`pyproject.toml`,
  `uv.lock`, `.python-version`) and the final figures in `figures/`.
- **Never committed:** intermediate data products. That covers downloaded
  catalogues, training splits, model outputs, per-run records, tables and logs.
  Running the scripts regenerates every one of them, and `.gitignore` keeps
  them out.

Every foundation-model prediction goes through lazy-tfm. The baselines
(FlexZBoost, GPz, CMNN) run through
[RAIL](https://github.com/LSSTDESC/rail). Both import the same
`lazy.datasets`, `lazy.grid` and `lazy.metrics`, so the training rows, the
redshift grid and the metric definitions are the same code on both sides.

## Environment

One [uv](https://docs.astral.sh/uv/) environment holds everything, on
Python 3.12. `uv.lock` pins every package. lazy-tfm and NVIDIA's
structured-data-models (Kumo Tabular) are pinned to a git commit in
`[tool.uv.sources]`.

```bash
git clone --branch paper --single-branch https://github.com/biprateep/lazy-tfm.git fm4pz
cd fm4pz
uv sync                 # the locked environment, in .venv
uv run lazy setup       # LimiX-2's source, at the commit lazy-tfm pins
```

The model checkpoints download from Hugging Face on first use. The DC1
catalogues download through `lazy.datasets` the same way.

To move the paper to a newer lazy-tfm, change its `rev` in `pyproject.toml`,
then run `uv lock` and `uv sync`, and commit both files.

## Layout

| Path | Contents |
|---|---|
| `src/fm4pz/` | code the experiment scripts share, installed into the environment |
| `figures/` | the final figures, as they appear in the paper |
| `results/` | every intermediate product the scripts write (not committed) |

The code follows the Google Python Style Guide. Lint it with
`uv run ruff check` and type-check it with `uv run mypy src`; the
configuration is in `ruff.toml` and `mypy.ini`.

## License

MIT; see [LICENSE](LICENSE).
