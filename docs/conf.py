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
import inspect
import pathlib
import re
import subprocess
from typing import Any

import jupytext
from sphinx import application

import lazy

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

# _gallery.md is a fragment that tutorials/index.md includes, not a page, and
# so are the tables of the models (below).
exclude_patterns = [
    "_build",
    "build",
    "**.ipynb_checkpoints",
    "tutorials/_gallery.md",
    "models/_overview.md",
    "guide/_backends.md",
]
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
THUMBNAILS = TUTORIALS / "thumbnails"


def _fetch_tutorials() -> None:
    """Writes each tutorial's notebook and thumbnail from ``tutorials``.

    The branch holds ``<name>.ipynb`` and ``thumbnails/<name>.png``, both
    written by ``execute.sh``. A file already in the working tree (a fresh
    ``execute.sh`` run) is kept, so a local build shows the local outputs.
    A thumbnail missing from the branch is not an error: its gallery card
    shows a placeholder.

    Raises:
        RuntimeError: If a notebook is neither present nor on the branch.
    """
    scripts = sorted(TUTORIALS.glob("*.py"))
    missing = [
        path
        for script in scripts
        for path in (
            script.with_suffix(".ipynb"),
            THUMBNAILS / f"{script.stem}.png",
        )
        if not path.exists()
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
        if any(path.suffix == ".ipynb" for path in missing):
            raise RuntimeError(
                "cannot fetch the executed tutorials from the"
                f" {TUTORIALS_BRANCH!r} branch: {fetch.stderr.strip()}"
            )
        return  # Only thumbnails are missing: the cards show placeholders.
    for path in missing:
        show = subprocess.run(
            [*git, "show", f"FETCH_HEAD:{path.relative_to(TUTORIALS)}"],
            capture_output=True,
            check=path.suffix == ".ipynb",
        )
        if not show.returncode:
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(show.stdout)


_fetch_tutorials()

# -- Tutorial gallery ----------------------------------------------------------
# docs/tutorials/index.md includes _gallery.md, which this writes at import:
# one sphinx-design card per tutorial (thumbnail, title, description, link)
# and the hidden toctree. Each card is read from the tutorial's py:percent
# source, so adding a tutorial needs no edit here beyond its place in the
# order: the title is its first Markdown heading, the description the
# optional `gallery: description:` key of its jupytext header, and the
# thumbnail thumbnails/<name>.png (execute.sh documents the syntax). A
# missing thumbnail gives a placeholder, and a missing description none.
# Tutorials are shown in TUTORIAL_ORDER, then any others alphabetically.
TUTORIAL_ORDER = [
    "basic_usage",
    "multimodal",
    "skewed_targets",
    "discrete_targets",
    "messy_inputs",
    "choosing_a_model",
    "ensembles",
    "tuning",
    "large_data",
    "distribution_shift",
    "photo_z",
]
GALLERY = TUTORIALS / "_gallery.md"
PLACEHOLDER = "/_static/tutorial-placeholder.svg"


def _tutorial_card(script: pathlib.Path) -> str:
    """Returns the gallery card of one tutorial, as MyST Markdown.

    Args:
        script: The tutorial's py:percent source.

    Returns:
        A ``grid-item-card`` directive linking to the tutorial's page.
    """
    notebook = jupytext.read(script)
    title = script.stem.replace("_", " ").capitalize()
    for cell in notebook.cells:
        heading = re.search(r"^#\s+(.+)$", cell.source, re.MULTILINE)
        if cell.cell_type == "markdown" and heading:
            title = heading.group(1).strip()
            break
    gallery = notebook.metadata.get("gallery") or {}
    description = " ".join(str(gallery.get("description", "")).split())
    thumbnail = THUMBNAILS / f"{script.stem}.png"
    image = (
        f"/tutorials/thumbnails/{thumbnail.name}"
        if thumbnail.exists()
        else PLACEHOLDER
    )
    return (
        f":::{{grid-item-card}} {title}\n"
        f":img-top: {image}\n"
        f":img-alt: {script.stem}\n"
        f":link: /tutorials/{script.stem}\n"
        ":link-type: doc\n"
        ":shadow: none\n"
        ":class-card: lazy-gallery-card\n"
        f"\n{description}\n"
        ":::\n"
    )


def _write_gallery() -> None:
    """Writes the tutorial cards and toctree to ``tutorials/_gallery.md``.

    The file is rewritten only when it changes, so an incremental build does
    not re-read the gallery page every time.
    """
    scripts = sorted(
        TUTORIALS.glob("*.py"),
        key=lambda script: (
            TUTORIAL_ORDER.index(script.stem)
            if script.stem in TUTORIAL_ORDER
            else len(TUTORIAL_ORDER),
            script.stem,
        ),
    )
    text = "\n".join(
        [
            "<!-- Written by docs/conf.py from the tutorials; do not edit. -->",
            "",
            "::::{grid} 1 2 3 3",
            ":gutter: 3",
            "",
            *(_tutorial_card(script) for script in scripts),
            "::::",
            "",
            "```{toctree}",
            ":hidden:",
            "",
            *(script.stem for script in scripts),
            "```",
            "",
        ]
    )
    if not GALLERY.exists() or GALLERY.read_text() != text:
        GALLERY.write_text(text)


_write_gallery()

# -- Tables of the models ------------------------------------------------------
# docs/models/index.md and docs/guide/interface.md include the tables below,
# which this writes at import from lazy.CHECKPOINTS and the registry: one row
# per pinned checkpoint in the first, from the fields that describe it, and
# one per backend in the second, from its class's method_note and its
# checkpoints' size notes. A new backend or version therefore needs no edit
# here. The overview lists LazyModel's default first and the rest by the size
# of their default checkpoint; the guide follows INTERFACE_ORDER, the order
# its prose takes, and then any other backend alphabetically. The "yes" of a
# model that runs on a CPU at any size is in bold.
MODEL_TABLES = {
    "overview": pathlib.Path(__file__).parent / "models" / "_overview.md",
    "interface": pathlib.Path(__file__).parent / "guide" / "_backends.md",
}
INTERFACE_ORDER = ["tabpfn", "limix", "tabicl", "tabfm"]


def _version_key(version: str) -> list[tuple[int, int | str]]:
    """Returns a key that sorts ``"v2.6"`` before ``"v10"``."""
    return [
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", version)
        if part
    ]


def _gigabytes(size_bytes: int) -> str:
    """Returns ``"0.88 GB"``, or ``"6.6 GB"`` from one gigabyte on."""
    size = size_bytes / 1e9
    return f"{size:.2f} GB" if size < 1 else f"{size:.1f} GB"


def _parameter_count(parameters: int) -> str:
    """Returns ``"219 M"``, or ``"1.64 B"`` from a billion on."""
    if parameters >= 1e9:
        return f"{parameters / 1e9:.2f} B"
    return f"{parameters / 1e6:.0f} M"


def _size_note(checkpoint: Any, *, approximate: bool = True) -> str:
    """Returns the download size a checkpoint's note quotes.

    Args:
        checkpoint: A checkpoint from ``lazy.CHECKPOINTS``.
        approximate: Whether to keep the note's ``"~"``.

    Returns:
        The size, such as ``"~1.6 GB"``, or ``"1.6 GB"`` if not approximate.

    Raises:
        ValueError: If the note starts with no size.
    """
    found = re.match(r"~?[\d.,]+ [kMGT]?B\b", checkpoint.size_note)
    if not found:
        raise ValueError(
            f"{checkpoint.key}'s size_note does not start with its size: "
            f"{checkpoint.size_note!r}"
        )
    return found.group() if approximate else found.group().lstrip("~")


def _markdown_table(
    header: list[str], align: list[str], rows: list[list[str]]
) -> list[str]:
    """Returns a Markdown table as lines, its columns padded to one width.

    Args:
        header: The column titles.
        align: Each column's alignment, ``"left"`` or ``"right"``.
        rows: The cells, row by row.

    Returns:
        The header, the rule and the rows, one line each.
    """
    widths = [
        max(len(cell) for cell in column) for column in zip(header, *rows)
    ]
    rule = [
        "-" * (width - 1) + (":" if side == "right" else "-")
        for width, side in zip(widths, align)
    ]
    lines = []
    for cells in (header, rule, *rows):
        padded = (
            cell.rjust(width) if side == "right" else cell.ljust(width)
            for cell, width, side in zip(cells, widths, align)
        )
        lines.append("| " + " | ".join(padded) + " |")
    return lines


def _overview_table() -> list[str]:
    """Returns docs/models/index.md's table, one row per checkpoint."""
    default_backend = (
        inspect.signature(lazy.LazyModel).parameters["model"].default
    )
    backends = sorted(
        (name for name in lazy.ESTIMATORS if lazy.list_versions(name)),
        key=lambda name: (
            name != default_backend,
            lazy.get_checkpoint(name).size_bytes,
            name,
        ),
    )
    rows = []
    for name in backends:
        default = lazy.DEFAULT_VERSIONS[name]
        others = sorted(
            (v for v in lazy.list_versions(name) if v != default),
            key=_version_key,
            reverse=True,
        )
        for version in (default, *others):
            checkpoint = lazy.get_checkpoint(name, version)
            title = checkpoint.display_name
            call = f'`"{name}"`'
            if version != default:
                call = f'`"{name}", version="{version}"`'
            elif name == default_backend:
                title += " (**default**)"
            cpu = "**yes**" if checkpoint.cpu == "yes" else checkpoint.cpu
            rows.append(
                [
                    title,
                    call,
                    _gigabytes(checkpoint.size_bytes),
                    _parameter_count(checkpoint.parameters),
                    checkpoint.license_name,
                    checkpoint.gpu,
                    cpu,
                ]
            )
    return _markdown_table(
        [
            "Model",
            "`LazyModel(...)`",
            "Weights",
            "Parameters",
            "License of the weights",
            "GPU",
            "CPU-friendly",
        ],
        ["left", "left", "right", "right", "left", "left", "left"],
        rows,
    )


def _interface_table() -> list[str]:
    """Returns docs/guide/interface.md's table, one row per backend."""
    backends = sorted(
        (name for name in lazy.ESTIMATORS if lazy.list_versions(name)),
        key=lambda name: (
            INTERFACE_ORDER.index(name)
            if name in INTERFACE_ORDER
            else len(INTERFACE_ORDER),
            name,
        ),
    )
    rows = []
    for name in backends:
        checkpoints = sorted(
            (lazy.get_checkpoint(name, v) for v in lazy.list_versions(name)),
            key=lambda checkpoint: checkpoint.size_bytes,
        )
        weights = _size_note(checkpoints[0])
        if len(checkpoints) > 1:
            weights = (
                f"{_size_note(checkpoints[0], approximate=False)} – "
                f"{_size_note(checkpoints[-1], approximate=False)}"
            )
        rows.append([f"`{name}`", lazy.ESTIMATORS[name].method_note, weights])
    return _markdown_table(
        ["Name", "Method", "Weights"], ["left", "left", "left"], rows
    )


def _write_model_tables() -> None:
    """Writes the tables of the models, each to its include file.

    A file is rewritten only when it changes, as the gallery is.
    """
    tables = {
        "overview": _overview_table(),
        "interface": _interface_table(),
    }
    for key, lines in tables.items():
        text = "\n".join(
            [
                "<!-- Written by docs/conf.py from lazy.CHECKPOINTS and the "
                "registry; do not edit. -->",
                "",
                *lines,
                "",
            ]
        )
        path = MODEL_TABLES[key]
        if not path.exists() or path.read_text() != text:
            path.write_text(text)


_write_model_tables()

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
html_css_files = ["gallery.css"]  # The tutorial cards (tutorials/index.md).
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
    # Not the gallery page, tutorials/index.md, which is its own source.
    if (TUTORIALS.parent / f"{pagename}.py").exists():
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
