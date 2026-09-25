# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Sphinx configuration for the lazy-tfm documentation.

Pages are MyST Markdown (via myst-nb, which also renders notebooks when the
tutorials arrive). The API reference is generated from the docstrings by
sphinx-autoapi, public names only. Options:
https://www.sphinx-doc.org/en/master/usage/configuration.html

Typical usage example:

  uv run sphinx-build -W -b html docs docs/_build/html
"""

# Sphinx reads its configuration as this module's global names, so they are
# public, lowercase and (for lists and dicts) mutable by design.
# ruff: noqa: GS004

from importlib import metadata

project = "lazy-tfm"
author = "Biprateep Dey"
copyright = "2026, Biprateep Dey"  # noqa: A001 - a Sphinx setting.
release = metadata.version("lazy-tfm")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
root_doc = "index"

# -- MyST and notebooks ------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
# Notebooks will be committed with their outputs (they need a GPU), so the docs
# build never executes them.
nb_execution_mode = "off"

# -- API reference -------------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../src"]
autoapi_root = "autoapi"
autoapi_add_toctree_entry = False
autoapi_member_order = "groupwise"
# No private members or modules: the reference documents the public API only.
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
add_module_names = False
# Attributes: sections render as fields, not as a second copy of the
# attributes autoapi already documents (dataclass fields would appear twice).
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# -- Copy button: strip console prompts from copied commands -----------------
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- HTML ---------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "lazy-tfm"
html_show_sourcelink = False
html_theme_options = {
    "github_url": "https://github.com/biprateep/lazy-tfm",
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_context = {
    "github_user": "biprateep",
    "github_repo": "lazy-tfm",
    "github_version": "main",
    "doc_path": "docs",
}
