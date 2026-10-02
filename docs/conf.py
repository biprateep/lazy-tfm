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
import pathlib
import subprocess
from typing import Any

from sphinx import application

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
    "sphinx_rtd_theme",
]

exclude_patterns = ["_build", "build", "**.ipynb_checkpoints"]
root_doc = "index"

# -- MyST and notebooks ------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
# The tutorials need a GPU, so the docs build never executes them. main tracks
# their py:percent source; the executed notebooks, outputs included, live on
# the orphan branch `tutorials` (docs/tutorials/execute.sh writes them), and
# are fetched from there when the working tree has none.
nb_execution_mode = "off"
TUTORIALS = pathlib.Path(__file__).parent / "tutorials"
TUTORIALS_BRANCH = "tutorials"


def _fetch_tutorials() -> None:
    """Writes each tutorial's executed notebook from the ``tutorials`` branch.

    A notebook already in the working tree (a fresh ``execute.sh`` run) is
    kept, so a local build shows the local outputs.

    Raises:
        RuntimeError: If a notebook is neither present nor on the branch.
    """
    missing = [
        script.with_suffix(".ipynb")
        for script in sorted(TUTORIALS.glob("*.py"))
        if not script.with_suffix(".ipynb").exists()
    ]
    if not missing:
        return
    git = ["git", "-C", str(TUTORIALS)]
    fetch = subprocess.run(
        [*git, "fetch", "--quiet", "--depth=1", "origin", TUTORIALS_BRANCH],
        capture_output=True,
        text=True,
        check=False,
    )
    if fetch.returncode:
        raise RuntimeError(
            f"cannot fetch the executed tutorials from the {TUTORIALS_BRANCH!r}"
            f" branch: {fetch.stderr.strip()}"
        )
    for notebook in missing:
        notebook.write_bytes(
            subprocess.run(
                [*git, "show", f"FETCH_HEAD:{notebook.name}"],
                capture_output=True,
                check=True,
            ).stdout
        )


_fetch_tutorials()

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
    "imported-members",
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
html_theme = "sphinx_rtd_theme"
html_title = "lazy-tfm"
# The logo and favicon are drawn by docs/logo/make_logo.py; the white logo
# sits on the theme's blue sidebar header.
html_static_path = ["_static"]
html_logo = "_static/lazy-logo-white.svg"
html_favicon = "_static/favicon.png"
html_show_sourcelink = False
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "style_external_links": True,
    "prev_next_buttons_location": "both",
    "logo_only": True,
}
# The "Edit on GitHub" link at the top of each page.
html_context = {
    "display_github": True,
    "github_user": "biprateep",
    "github_repo": "lazy-tfm",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


def _skip_public_reexports(  # Sphinx's event signature.
    app: application.Sphinx,  # noqa: ARG001
    what: str,  # noqa: ARG001
    name: str,  # noqa: ARG001
    obj: Any,  # An autoapi mapper object.
    skip: bool,
    options: object,  # noqa: ARG001
) -> bool:
    """Documents a re-exported name only where it has no public home.

    ``lazy`` and ``lazy.models`` re-export names from private modules (the
    checkpoint registry, the warnings) and from public ones (the backends,
    the grid). The first have no page of their own, so they are documented
    where they are re-exported; the second already have one, and a second
    copy would make every cross-reference to them ambiguous.
    """
    original = getattr(obj, "obj", {}).get("original_path") or ""
    if getattr(obj, "imported", False) and original:
        module = original.rpartition(".")[0]
        if not any(part.startswith("_") for part in module.split(".")):
            return True
    return skip


def _edit_tutorial_source(  # Sphinx's event signature.
    app: application.Sphinx,  # noqa: ARG001
    pagename: str,
    templatename: str,  # noqa: ARG001
    context: dict[str, Any],
    doctree: object,  # noqa: ARG001
) -> None:
    """Points a tutorial's "Edit on GitHub" link at its py:percent source.

    The page is built from the executed notebook, which is not on main.
    """
    if pagename.startswith("tutorials/"):
        context["meta"] = {
            **(context.get("meta") or {}),
            "github_url": (
                "https://github.com/biprateep/lazy-tfm/blob/main/docs/"
                f"{pagename}.py"
            ),
        }


def setup(app: application.Sphinx) -> None:  # noqa: D103 - Sphinx's extension hook.
    app.connect("autoapi-skip-member", _skip_public_reexports)
    app.connect("html-page-context", _edit_tutorial_source)
