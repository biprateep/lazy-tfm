<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/biprateep/lazy-tfm/main/docs/_static/lazy-logo-white.svg">
    <img src="https://raw.githubusercontent.com/biprateep/lazy-tfm/main/docs/_static/lazy-logo.svg" alt="LAZY" width="440">
  </picture>
</h1>

**LAZY: Lazy but Accurate Z\* for Yinz** gives full conditional distributions
of a continuous target from pretrained tabular foundation models.

\*Where Z is whatever you want to predict from tabular features, such as a
redshift, a metallicity, a mass or a yield.

[![PyPI](https://img.shields.io/pypi/v/lazy-tfm)](https://pypi.org/project/lazy-tfm/)
[![Python](https://img.shields.io/pypi/pyversions/lazy-tfm)](https://pypi.org/project/lazy-tfm/)
[![Unit test and code coverage](https://github.com/biprateep/lazy-tfm/actions/workflows/testing-and-coverage.yml/badge.svg)](https://github.com/biprateep/lazy-tfm/actions/workflows/testing-and-coverage.yml)
[![Documentation](https://readthedocs.org/projects/lazy-tfm/badge/?version=latest)](https://lazy-tfm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/biprateep/lazy-tfm/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23751881.svg)](https://doi.org/10.5281/zenodo.23751881)

LAZY is a unified, scikit-learn-compatible wrapper around several pretrained
tabular foundation models (TabPFN, TabICL, TabFM, LimiX-2 and Kumo Tabular).
The models are pretrained and never fine-tuned: we hand them labeled rows as
*context*, and they answer queries in one forward pass, with no training loop,
no hyperparameter search and no per-dataset retraining. This is the sense in
which the package is lazy.

Everything else is in the documentation at
[lazy-tfm.readthedocs.io](https://lazy-tfm.readthedocs.io): how to install
the package and each model, a quick start, the tutorials, the user guide, the
supported models with their licenses, the API reference, how to contribute or
add a backend, and how to cite LAZY. The changes in each release are in
[CHANGELOG.md](https://github.com/biprateep/lazy-tfm/blob/main/CHANGELOG.md).
