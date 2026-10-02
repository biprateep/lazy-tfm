# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Publication figure style and the standard diagnostic plots.

``lazy.plotting.style`` holds journal geometry, house rcParams and figure
helpers for any figure. ``lazy.plotting.diagnostics`` holds the accuracy,
calibration and sample-distribution figures a conditional density estimate is
judged by. Both are re-exported here.
"""

# The re-exports below define the package's public API, so they name members.
# ruff: noqa: GS001

from lazy.plotting.diagnostics import diagnostic_panel
from lazy.plotting.diagnostics import plot_actual_vs_predicted
from lazy.plotting.diagnostics import plot_coverage
from lazy.plotting.diagnostics import plot_pdfs
from lazy.plotting.diagnostics import plot_pit
from lazy.plotting.diagnostics import plot_pit_qq
from lazy.plotting.diagnostics import plot_residuals
from lazy.plotting.diagnostics import plot_stacked_pdfs
from lazy.plotting.style import ASPECTS
from lazy.plotting.style import COLUMN_WIDTH
from lazy.plotting.style import figsize
from lazy.plotting.style import grid_figsize
from lazy.plotting.style import Journal
from lazy.plotting.style import journal
from lazy.plotting.style import JOURNALS
from lazy.plotting.style import one_to_one
from lazy.plotting.style import RC_PARAMS
from lazy.plotting.style import register_journal
from lazy.plotting.style import save
from lazy.plotting.style import set_palette
from lazy.plotting.style import TEXT_WIDTH
from lazy.plotting.style import use_style
from lazy.plotting.style import verify_style

__all__ = [
    "ASPECTS",
    "COLUMN_WIDTH",
    "JOURNALS",
    "RC_PARAMS",
    "TEXT_WIDTH",
    "Journal",
    "diagnostic_panel",
    "figsize",
    "grid_figsize",
    "journal",
    "one_to_one",
    "plot_coverage",
    "plot_stacked_pdfs",
    "plot_pdfs",
    "plot_pit",
    "plot_pit_qq",
    "plot_residuals",
    "plot_actual_vs_predicted",
    "register_journal",
    "save",
    "set_palette",
    "use_style",
    "verify_style",
]
