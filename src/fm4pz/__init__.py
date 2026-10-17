# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Shared code for the paper's experiments.

The experiment scripts import this package for what they have in common, and
it is installed into both environments. Every foundation-model prediction goes
through lazy-tfm; nothing here wraps a model itself.
"""
