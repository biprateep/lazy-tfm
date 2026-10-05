# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""``python -m lazy``: the ``lazy`` command (see :mod:`lazy._cli`)."""

import sys

from lazy import _cli

if __name__ == "__main__":
    sys.exit(_cli.main())
