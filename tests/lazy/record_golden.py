#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Records the golden densities of every backend on the fixed small problem.

Runs on CPU and needs every backend's checkpoint in the local cache. Only
run it to deliberately move the reference; the golden test compares against
the files it writes. ``encoders`` records the embeddings of every pinned
encoder model instead, which needs the ``encoders`` extra.

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
    """Records the embeddings of every pinned encoder model, in one file.

    Args:
        recorded: What :func:`_recording` returned.
    """
    import lazy  # noqa: PLC0415 - imported by golden_data in any case.

    keys = sorted(lazy.encoders.CHECKPOINTS)
    embeddings = {key: golden_data.embed(key) for key in keys}
    models = {
        key: lazy.encoders.CHECKPOINTS[key].provenance(device="cpu")
        for key in keys
    }
    np.savez_compressed(
        golden_data.ENCODERS_PATH,
        **{key.replace(":", "__"): rows for key, rows in embeddings.items()},
        provenance=json.dumps({**recorded, "models": models}, default=str),
    )
    for key, rows in embeddings.items():
        print(f"{key}: {rows.shape} -> {golden_data.ENCODERS_PATH}")


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
