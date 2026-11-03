#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Records the golden densities of every backend on the fixed small problem.

Runs on CPU and needs every backend's checkpoint in the local cache. Only
run it to deliberately move the reference; the golden test compares against
the files it writes.

Typical usage example:

  python tests/lazy/record_golden.py tabpfn tabicl
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
    parser.add_argument(
        "backends", nargs="*", default=sorted(golden_data.RECORDED_PARAMS)
    )
    args = parser.parse_args(argv)
    _, _, X_test = golden_data.problem()
    golden_data.GOLDEN_DIR.mkdir(exist_ok=True)
    recorded = _recording()
    for name in args.backends:
        model = golden_data.fit(name)
        pdfs = model.predict_proba(X_test, golden_data.GRID)
        path = golden_data.GOLDEN_DIR / f"{name}.npz"
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
