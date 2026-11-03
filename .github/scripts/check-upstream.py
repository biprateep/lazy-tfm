# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Reports what upstream has published since lazy pinned it.

Prints a Markdown report with three sections:

1. Each pinned checkpoint against its repository's latest revision: by the
   pinned file's content where one file is pinned, else by the revision.
2. Repositories of a pinned model's family created after its newest pin: a
   new checkpoint, such as a ``tabfm-1.1.0``.
3. Each backend package's newest release or commit against the lockfile and
   ``[tool.uv.sources]``, and LimiX's source against its pinned commit.

It changes nothing and fails only when a lookup fails, after printing the
sections before it. Run it with lazy installed:

    uv run --no-sync python .github/scripts/check-upstream.py
"""

import functools
import json
import pathlib
import re
import subprocess
import tomllib
import urllib.request

import huggingface_hub

import lazy
from lazy.models import _limix_source

ROOT = pathlib.Path(__file__).resolve().parents[2]
"""The repository root."""


def _family(repo_id: str) -> str:
    """The model family a repository name starts with, lower case."""
    return re.split(r"[-_.]", repo_id.split("/")[1])[0].lower()


def _pypi_version(package: str) -> str:
    """The newest release of ``package`` on PyPI."""
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)["info"]["version"]


def _git_head(url: str) -> str:
    """The commit at the head of the default branch of the repository."""
    out = subprocess.run(
        ["git", "ls-remote", url, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    return out.split()[0]


def _file_hash(
    api: huggingface_hub.HfApi, repo_id: str, filename: str, revision: str
) -> str | None:
    """The content hash of one file at ``revision``; None if it is absent."""
    found = api.get_paths_info(repo_id, [filename], revision=revision)
    if not found:
        return None
    return found[0].lfs.sha256 if found[0].lfs else found[0].blob_id


def _canonical(name: str) -> str:
    """A package name as uv.lock writes it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _row(*cells: object) -> str:
    """One Markdown table row."""
    return f"| {' | '.join(str(cell) for cell in cells)} |"


def _status(pinned: str, latest: str) -> str:
    """``current`` when the two agree, else a bold ``differs``."""
    return "current" if pinned == latest else "**differs**"


def _checkpoints(api: huggingface_hub.HfApi) -> list[str]:
    """The pinned checkpoints' table.

    A checkpoint that pins one file is compared by that file's content, so
    that an edit elsewhere in its repository does not count; a snapshot is
    compared by the repository's revision.

    Args:
        api: A Hugging Face Hub client.

    Returns:
        The report's lines.
    """
    lines = [
        "## Pinned checkpoints",
        "",
        _row("checkpoint", "repository", "pinned", "latest", "status"),
        _row(*["---"] * 5),
    ]
    head_of = functools.cache(lambda repo_id: api.model_info(repo_id).sha)
    for key, checkpoint in sorted(lazy.CHECKPOINTS.items()):
        repo_id, revision = checkpoint.repo_id, checkpoint.revision
        head = head_of(repo_id)
        if revision is None:
            lines.append(_row(key, repo_id, "none", head[:7], "**unpinned**"))
            continue
        status = _status(revision, head)
        if checkpoint.filename and revision != head:
            latest = _file_hash(api, repo_id, checkpoint.filename, head)
            if latest is None:
                status = "**file removed at head**"
            elif latest == _file_hash(
                api, repo_id, checkpoint.filename, revision
            ):
                status = "other files differ"
        lines.append(_row(key, repo_id, revision[:7], head[:7], status))
    return lines


def _new_repositories(api: huggingface_hub.HfApi) -> list[str]:
    """Repositories of a pinned model's family created after its newest pin.

    Args:
        api: A Hugging Face Hub client.

    Returns:
        The report's lines.
    """
    pinned = {c.repo_id for c in lazy.CHECKPOINTS.values()}
    found = []
    for author, family in sorted(
        {(r.split("/")[0], _family(r)) for r in pinned}
    ):
        listed = api.list_models(
            author=author, search=family, expand=["createdAt"]
        )
        models = [m for m in listed if _family(m.id) == family and m.created_at]
        dates = [m.created_at for m in models if m.id in pinned]
        # A pinned repository missing from the listing (renamed, private)
        # leaves no date to compare with: list every repository then.
        newest = max(dates) if dates else None
        found += [
            m.id
            for m in models
            if m.id not in pinned and (newest is None or m.created_at > newest)
        ]
    lines = ["## New repositories of a pinned model", ""]
    return lines + ([f"- {r}" for r in sorted(found)] or ["None."])


def _packages() -> list[str]:
    """The backend packages' table.

    A package installed from git, and LimiX's source, are compared by the
    head commit, which may differ from the pin without being ahead of it.

    Returns:
        The report's lines.
    """
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = {_canonical(p["name"]): p["version"] for p in lock["package"]}
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    git = {
        _canonical(name): (source["git"], source["rev"])
        for name, source in pyproject["tool"]["uv"]["sources"].items()
        if "git" in source
    }
    backends = {_canonical(c.package) for c in lazy.CHECKPOINTS.values()}
    pypi = sorted((backends & locked.keys()) - git.keys())
    git["LimiX source"] = (_limix_source.REPOSITORY, _limix_source.LIMIX_COMMIT)
    lines = [
        "## Backend packages",
        "",
        _row("package", "pinned", "latest", "status"),
        _row(*["---"] * 4),
    ]
    for package in pypi:
        latest = _pypi_version(package)
        status = _status(locked[package], latest)
        lines.append(_row(package, locked[package], latest, status))
    for package, (url, rev) in sorted(git.items()):
        head = _git_head(url)
        lines.append(_row(package, rev[:7], head[:7], _status(rev, head)))
    return lines


def main() -> None:
    """Prints the report, a section at a time.

    Raises:
        Exception: Whatever a failed lookup raised, after the sections before
            it and a line saying the report stops there.
    """
    api = huggingface_hub.HfApi()
    sections = (
        functools.partial(_checkpoints, api),
        functools.partial(_new_repositories, api),
        _packages,
    )
    for section in sections:
        try:
            lines = section()
        except Exception as error:
            print(f"**The report stops here: a lookup failed ({error}).**")
            raise
        print("\n".join(lines) + "\n", flush=True)


if __name__ == "__main__":
    main()
