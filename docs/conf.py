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

import hashlib
from importlib import metadata
import inspect
import os
import pathlib
import re
import subprocess
from typing import Any

import jupytext
import nbformat
from sphinx import application
from sphinx.ext import intersphinx
from sphinx.util import nodes

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
]

# _gallery.md is a fragment that tutorials/index.md includes, not a page, and
# so are the tables of the models (below).
exclude_patterns = [
    "_build",
    "build",
    "**.ipynb_checkpoints",
    "_templates",
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
# A fork's `origin` has no `tutorials` branch, so the upstream one is next.
REMOTES = ("origin", "https://github.com/biprateep/lazy-tfm.git")
THUMBNAILS = TUTORIALS / "thumbnails"


def _sha256(script: pathlib.Path) -> str:
    """Returns the hex SHA-256 of a tutorial's source, as execute.sh does.

    Args:
        script: The tutorial's py:percent source.

    Returns:
        The digest, 64 hex digits.
    """
    return hashlib.sha256(script.read_bytes()).hexdigest()


def _fetch_branch(git: list[str]) -> list[str]:
    """Fetches the executed tutorials into ``FETCH_HEAD``.

    It tries each of ``REMOTES`` in turn. A Read the Docs build of a release
    tag ``v<version>`` first tries the tag ``tutorials/v<version>``, which
    marks the notebooks executed for that release, then the branch.

    Args:
        git: The git command, run in the tutorials directory.

    Returns:
        Each failed fetch's error, or an empty list once a fetch succeeds.
    """
    refs = [f"refs/heads/{TUTORIALS_BRANCH}"]
    if os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag":
        tag = os.environ["READTHEDOCS_GIT_IDENTIFIER"]  # The tag, not the slug.
        refs.insert(0, f"refs/tags/{TUTORIALS_BRANCH}/{tag}")
    errors = []
    for ref in refs:
        for remote in REMOTES:
            fetch = subprocess.run(
                [*git, "fetch", "--quiet", "--depth=1", remote, ref],
                capture_output=True,
                text=True,
                check=False,
            )
            if not fetch.returncode:
                return []
            errors.append(f"{remote} {ref}: {fetch.stderr.strip()}")
    return errors


def _unexecuted(path: pathlib.Path) -> bool:
    """Whether a notebook is one that an offline build wrote unexecuted."""
    if path.suffix != ".ipynb":
        return False
    # A search of the bytes, not a parse of megabytes of outputs.
    return b'"unexecuted": true' in path.read_bytes()


def _on_ci() -> bool:
    """Whether the build runs on Read the Docs or in CI."""
    return (
        os.environ.get("READTHEDOCS") == "True"
        or os.environ.get("CI") == "true"
    )


def _write_unexecuted(path: pathlib.Path) -> None:
    """Writes a tutorial's notebook from its script, without outputs."""
    notebook = jupytext.read(path.with_suffix(".py"))
    notebook.metadata["lazy"] = {"unexecuted": True}
    jupytext.write(notebook, path)
    print(f"cannot fetch the tutorials: wrote {path.name} unexecuted")


def _fetch_tutorials() -> None:
    """Writes each tutorial's notebook and thumbnail from ``tutorials``.

    The branch holds ``<name>.ipynb`` and ``thumbnails/<name>.png``, both
    written by ``execute.sh``. A file already in the working tree (a fresh
    ``execute.sh`` run) is kept, so a local build shows the local outputs.
    A thumbnail missing from the branch is not an error: its gallery card
    shows a placeholder. If no fetch succeeds, a local build writes each
    missing notebook from its script, without outputs and marked
    ``unexecuted``; the next build that can fetch replaces it.

    Raises:
        RuntimeError: If a notebook is neither present nor fetched, on Read
            the Docs or in CI.
    """
    missing = []
    for script in sorted(TUTORIALS.glob("*.py")):
        for path in (
            script.with_suffix(".ipynb"),
            THUMBNAILS / f"{script.stem}.png",
        ):
            if not path.exists() or _unexecuted(path):
                missing.append(path)
    if not missing:
        return
    git = ["git", "-C", str(TUTORIALS)]
    errors = _fetch_branch(git)
    if errors:
        notebooks = [path for path in missing if path.suffix == ".ipynb"]
        if notebooks and _on_ci():
            raise RuntimeError(
                f"cannot fetch the executed tutorials: {'; '.join(errors)}"
            )
        for path in notebooks:
            _write_unexecuted(path)
        return
    for path in missing:
        show = subprocess.run(
            [*git, "show", f"FETCH_HEAD:{path.relative_to(TUTORIALS)}"],
            capture_output=True,
            check=path.suffix == ".ipynb" and _on_ci(),
        )
        if not show.returncode:
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(show.stdout)
        elif path.suffix == ".ipynb":  # A tutorial not yet on the branch.
            _write_unexecuted(path)


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
    "feature_importance",
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


def _release_key(version: str) -> list[int]:
    """Returns the release numbers of a version: ``"v1-small"`` gives [1]."""
    release = re.match(r"v?([\d.]+)", version)
    if release is None:
        return []
    return [int(part) for part in release.group(1).split(".") if part]


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
        # Newest first, and among the sizes of one release (Kumo Tabular's
        # v1-small, v1-medium and v1-large) the largest first.
        others = sorted(
            (v for v in lazy.list_versions(name) if v != default),
            key=lambda version: (
                _release_key(version),
                lazy.get_checkpoint(name, version).parameters,
            ),
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
                    "yes" if checkpoint.commercial_use else "no",
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
            "Commercial use",
            "GPU",
            "CPU-friendly",
        ],
        ["left", "left", "right", "right", "left", "left", "left", "left"],
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
autoapi_template_dir = "_templates/autoapi"
add_module_names = False
# Else napoleon's escaping of a trailing underscore shows through, and the
# fitted attributes (grid_, provenance_, ...) render without it.
strip_signature_backslash = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Every reference must resolve, or the build says which one does not. The
# exceptions are names the reference leaves out on purpose, or that no
# inventory documents.
nitpicky = True
nitpick_ignore_regex = [
    # The reference documents public names only, so a reference to a private
    # name (one with a part that starts with "_") renders as code.
    ("py:.*", r"(.*\.)?_[A-Za-z]\w*(\..*)?"),
    # Type aliases that their modules leave out of __all__, so that autoapi
    # does not document them; a signature spells their names.
    ("py:class", r"Ancil|FeatureMode|Member|Normalization|Pooling"),
    # Not in matplotlib's inventory.
    ("py:class", r"matplotlib\.typing\.RcKeyType"),
]

# -- Copy button: strip console prompts from copied commands -----------------
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- HTML ---------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "LAZY"
# The logo and favicons are drawn by docs/logo/make_logo.py (docs/brand.md
# lists them).
html_static_path = ["_static"]
# gallery.css styles the tutorial cards (tutorials/index.md). The favicon is
# the SVG; the PNG sizes are extra link tags, which Sphinx writes from
# html_css_files, since the theme has no documented favicons option.
html_favicon = "_static/favicon.svg"
html_css_files = [
    "gallery.css",
    ("favicon.png", {"rel": "icon", "sizes": "64x64", "type": "image/png"}),
    (
        "apple-touch-icon.png",
        {
            "rel": "apple-touch-icon",
            "sizes": "180x180",
            "type": "image/png",
        },
    ),
]
html_show_sourcelink = False
html_theme_options = {
    # The banner at the top of every page. The theme fetches it from main, so
    # editing the file changes it on every version of the docs, and an empty
    # file hides it. A fetched banner gets the theme's close button.
    "announcement": (
        "https://raw.githubusercontent.com/biprateep/lazy-tfm/main/docs/"
        "_templates/announcement.html"
    ),
    "logo": {
        "image_light": "_static/lazy-logo.svg",
        "image_dark": "_static/lazy-logo-white.svg",
        "alt_text": "LAZY",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/biprateep/lazy-tfm",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/lazy-tfm/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navigation_depth": 3,
    "show_toc_level": 2,
    "navbar_align": "left",
    "use_edit_page_button": True,
}
# The "Edit on GitHub" button of each page.
html_context = {
    "github_user": "biprateep",
    "github_repo": "lazy-tfm",
    "github_version": "main",
    "doc_path": "docs",
}


def _first_private(path: str) -> int | None:
    """Returns the index of the first part of ``path`` that starts with ``_``.

    Args:
        path: A dotted name.

    Returns:
        The index into ``path.split(".")``, or ``None`` when the path is
        public.
    """
    parts = path.split(".")
    return next((i for i, p in enumerate(parts) if p.startswith("_")), None)


def _resolve_alias(
    app: application.Sphinx,
    env: Any,  # The build environment.
    node: Any,  # The unresolved pending_xref.
    contnode: Any,  # Its text.
) -> Any:
    """Resolves a reference to a re-exported name or to numpy typing.

    ``:func:`lazy.is_cached``` gives the path a user imports, and a ``Bases:``
    line gives the path of the private module that defines the name. Neither
    is documented (see ``_skip_public_reexports``), so the reference follows
    autoapi's record of each import to where the name is documented. A
    private path is first read without its private module, and a
    bare name (from a summary on a package page) as one in that package. numpy
    lists ``numpy.typing.ArrayLike`` as data, and a signature references it
    as a class.

    Returns:
        A reference node, or ``None`` when the target is neither.
    """
    if node.get("refdomain") != "py":
        return None
    target = node["reftarget"]
    if target.startswith("npt."):  # numpy.typing, as the module imports it.
        target = "numpy.typing." + target.removeprefix("npt.")
    if target.startswith("numpy.typing."):
        data = node.deepcopy()
        data["reftype"] = "data"
        data["reftarget"] = target
        return intersphinx.missing_reference(app, env, data, contnode)
    if "." not in target and node.get("py:module"):
        target = f"{node['py:module']}.{target}"  # A name in the same module.
    if not target.startswith("lazy."):
        return None
    parts = target.split(".")
    private = _first_private(target)
    # A name inside a private module, not the private module itself.
    if private is not None and private + 1 < len(parts):
        target = ".".join([*parts[:private], *parts[private + 1 :]])
    documented = env.get_domain("py").objects
    seen: set[str] = set()
    while target not in documented:
        seen.add(target)
        imported = env.autoapi_all_objects.get(target)
        target = imported and imported.obj.get("original_path")
        if not target or target in seen:
            return None
    entry = documented[target]
    return nodes.make_refnode(
        app.builder, node["refdoc"], entry.docname, entry.node_id, contnode
    )


def _skip_public_reexports(  # Sphinx's event signature.
    app: application.Sphinx,  # noqa: ARG001
    what: str,
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

    An attribute without a docstring of its own is skipped too: its class's
    ``Attributes:`` section already documents it, and autoapi would add an
    empty entry for it.
    """
    if what == "attribute" and not getattr(obj, "docstring", ""):
        return True
    original = getattr(obj, "obj", {}).get("original_path") or ""
    if getattr(obj, "imported", False) and original:
        if _first_private(original.rpartition(".")[0]) is None:
            return True
    return skip


def _edit_tutorial_source(  # Sphinx's event signature.
    app: application.Sphinx,  # noqa: ARG001
    pagename: str,
    templatename: str,  # noqa: ARG001
    context: dict[str, Any],
    doctree: object,  # noqa: ARG001
) -> None:
    """Points a tutorial's edit button at its py:percent source.

    The page is built from the executed notebook, which is not on main.
    """
    # Not the gallery page, tutorials/index.md, which is its own source.
    if (TUTORIALS.parent / f"{pagename}.py").exists():
        context["page_source_suffix"] = ".py"
    elif pagename.startswith("autoapi/"):  # Generated: no file to edit.
        context["theme_use_edit_page_button"] = False


def _note_tutorial_source(  # Sphinx's event signature.
    app: application.Sphinx,
    docname: str,
    source: list[str],
) -> None:
    """Prepends to a tutorial's notebook the code version it ran on.

    The ``lazy`` metadata that ``execute.sh`` stamps holds the lazy version,
    the commit and the script's SHA-256. A notebook whose stamp is missing,
    or whose script has changed since, gets a note that its outputs may not
    match the code; one that an offline build wrote gets a note that it has
    no outputs. Only the text Sphinx parses changes, never the file.
    """
    script = TUTORIALS.parent / f"{docname}.py"
    if not script.exists():
        return
    app.env.note_dependency(str(script))  # An edit to it updates the note.
    notebook = nbformat.reads(source[0], as_version=4)
    stamp = notebook.metadata.get("lazy", {})
    lines = []
    if stamp.get("unexecuted"):
        lines.append(
            "This build could not fetch the executed notebook, so this page"
            " has no outputs."
        )
    elif stamp.get("source_sha256") != _sha256(script):
        lines.append(
            "This notebook was not executed from the source shown, so its"
            " outputs may not match the code."
        )
    if "version" in stamp:
        lines.append(
            f"Executed with lazy {stamp['version']}"
            f" at commit {stamp['commit'][:12]}"
            + (
                ", with uncommitted changes."
                if "dirty" in stamp["commit"]
                else "."
            )
        )
    if lines:
        text = "\n\n".join(lines)
        cell = nbformat.v4.new_markdown_cell(f":::{{note}}\n{text}\n:::")
        notebook.cells.insert(1, cell)  # After the title cell.
        source[0] = nbformat.writes(notebook)


def setup(app: application.Sphinx) -> None:  # noqa: D103 - Sphinx's extension hook.
    app.connect("autoapi-skip-member", _skip_public_reexports)
    app.connect("missing-reference", _resolve_alias)
    app.connect("html-page-context", _edit_tutorial_source)
    app.connect("source-read", _note_tutorial_source)
