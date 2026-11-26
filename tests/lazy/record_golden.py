#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Records the golden densities of every backend on the fixed small problem.

Runs on CPU and needs every backend's checkpoint in the local cache. Only
run it to deliberately move the reference; the golden test compares against
the files it writes. ``encoders`` records the embeddings of the pinned
encoder models that the encoders' file lacks instead, which needs the
``encoders`` extra; delete the file to record them all again.

Typical usage example:

  python tests/lazy/record_golden.py tabpfn tabicl
  python tests/lazy/record_golden.py encoders
"""

import argparse
from collections.abc import Sequence
import json
import os
import platform
import subprocess
import sys
from typing import Any

import golden_data
import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    """Records the requested backends (all, by default) and returns 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    every = [*sorted(golden_data.RECORDED_PARAMS), "encoders"]
    parser.add_argument(
        "backends",
        nargs="*",
        choices=every,
        help="the backends to record, or encoders; all when none is named",
    )
    args = parser.parse_args(argv)
    _, _, X_test = golden_data.problem()
    golden_data.GOLDEN_DIR.mkdir(exist_ok=True)
    with golden_data.pinned_threads():
        recorded = _recording()
        for name in args.backends or every:
            if name == "encoders":
                _record_encoders(recorded)
                continue
            model = golden_data.fit(name)
            pdfs = model.predict_proba(X_test, golden_data.GRID)
            path = golden_data.path(name)
            np.savez_compressed(
                path,
                pdfs=pdfs,
                edges=golden_data.GRID.edges,
                provenance=json.dumps(
                    {**recorded, "model": model.provenance_}, default=str
                ),
            )
            print(f"{name}: {pdfs.shape} -> {path}")
    return 0


def _record_encoders(recorded: dict[str, object]) -> None:
    """Records the embeddings of the encoder models the file lacks.

    The embeddings already in the file are kept as they are, unless their
    model is no longer pinned (dropped) or is pinned to another revision
    (recorded again). The provenance lists every model; a recording that
    adds to the file is listed under ``"added"``, with the models it
    recorded.

    Args:
        recorded: What :func:`_recording` returned.
    """
    import lazy  # noqa: PLC0415 - imported by golden_data in any case.

    path = golden_data.ENCODERS_PATH
    kept: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {**recorded, "models": {}}
    if path.exists():
        with np.load(path) as golden:
            kept = {name: golden[name] for name in golden.files}
        provenance = json.loads(str(kept.pop("provenance")))
    checkpoints = lazy.encoders.CHECKPOINTS
    # A model no longer pinned is dropped; one pinned to another revision
    # is recorded again.
    kept = {
        name: rows
        for name, rows in kept.items()
        if (key := name.replace("__", ":")) in checkpoints
        and provenance["models"][key]["revision"] == checkpoints[key].revision
    }
    provenance["models"] = {
        key: model
        for key, model in provenance["models"].items()
        if key.replace(":", "__") in kept
    }
    keys = [
        key for key in sorted(checkpoints) if key.replace(":", "__") not in kept
    ]
    embeddings = {key: golden_data.embed(key) for key in keys}
    for key in keys:
        provenance["models"][key] = checkpoints[key].provenance(device="cpu")
    if kept and keys:
        provenance.setdefault("added", []).append({**recorded, "keys": keys})
    np.savez_compressed(
        path,
        **kept,
        **{key.replace(":", "__"): rows for key, rows in embeddings.items()},
        provenance=json.dumps(provenance, default=str),
    )
    for key, rows in embeddings.items():
        print(f"{key}: {rows.shape} -> {path}")


def _recording() -> dict[str, object]:
    """What records the files: the commit, machine, threads and versions.

    The commit ends in ``-dirty`` when the checkout has uncommitted changes.
    """
    import torch  # noqa: PLC0415 - pytest imports this module, without torch.

    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
        cwd=golden_data.GOLDEN_DIR,
    ).stdout.strip()
    return {
        "commit": commit,
        "machine": platform.platform(),
        "cpus": os.cpu_count(),
        # TabPFN's CPU densities differ between processors.
        "cpu_vendor": golden_data.cpu_vendor(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_threads": torch.get_num_threads(),
        "thread_variables": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


if __name__ == "__main__":
    sys.exit(main())
