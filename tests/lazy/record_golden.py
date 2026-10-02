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
    for name in args.backends:
        model = golden_data.fit(name)
        pdfs = model.predict_proba(X_test, golden_data.GRID)
        path = golden_data.GOLDEN_DIR / f"{name}.npz"
        np.savez_compressed(path, pdfs=pdfs, edges=golden_data.GRID.edges)
        print(f"{name}: {pdfs.shape} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
