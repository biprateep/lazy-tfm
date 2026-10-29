# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""Shared code for the paper's experiments.

The scripts under ``representative/`` and ``biased/`` import this package for
what they have in common: the training and test sets (:mod:`fm4pz.splits`,
built on ``lazy.datasets``), the common redshift grid and metrics
(:mod:`fm4pz.scoring`, on ``lazy.grid`` and ``lazy.metrics``), the
hyperparameter search (:mod:`fm4pz.tune`), the production runs
(:mod:`fm4pz.produce`), the steps run in order (:mod:`fm4pz.pipeline`) and the
figures (:mod:`fm4pz.figures`). Every foundation-model prediction goes through
lazy-tfm; nothing here wraps a model itself.
"""
