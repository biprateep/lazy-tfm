# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Type aliases shared across the package.

Typical usage example:

  from lazy import _typing

  def normalize(densities: _typing.FloatArray) -> _typing.FloatArray: ...
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

#: A NumPy array of float64; shapes and units go in the docstring (§3.8.3).
FloatArray: TypeAlias = npt.NDArray[np.float64]
#: A NumPy array of platform integers, such as bin indices.
IntArray: TypeAlias = npt.NDArray[np.intp]
#: A NumPy array of booleans, such as a selection mask.
BoolArray: TypeAlias = npt.NDArray[np.bool_]
