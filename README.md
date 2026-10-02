# lazy-tfm tutorials (executed)

This orphan branch holds the executed tutorial notebooks of
[lazy-tfm](https://github.com/biprateep/lazy-tfm), outputs and figures
included, and in `thumbnails/` the image of each one's card in the docs'
tutorial gallery. It shares no history with `main`, so the images never weigh
on the package's repository.

- The source of each tutorial is a py:percent script in `docs/tutorials/` on
  `main`; edit that, not the notebook here.
- `docs/tutorials/execute.sh` on `main` writes both the notebook and the
  thumbnail (the figure of the cell tagged `thumbnail`, else the last one).
- The docs build on Read the Docs fetches both from this branch, and the
  tutorials' "Open in Colab" and "View on GitHub" buttons open the notebooks
  here.
- To update one, see "Tutorials" in `CONTRIBUTING.md` on `main`.

| Tutorial | Colab |
| -------- | ----- |
| [Introduction](introduction.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biprateep/lazy-tfm/blob/tutorials/introduction.ipynb) |
