"""Publication figure style and the standard photo-z diagnostic plots.

``lazy.plotting.style`` is general-purpose: journal geometry, house rcParams and
figure helpers for any figure in the paper. ``lazy.plotting.diagnostics`` is
photo-z specific: the accuracy, calibration and N(z) figures every method in the
benchmark is judged by. Both are re-exported here.
"""

from lazy.plotting.diagnostics import (
    diagnostic_panel,
    plot_coverage,
    plot_nz,
    plot_pdfs,
    plot_pit,
    plot_pit_qq,
    plot_residuals,
    plot_zphot_ztrue,
)
from lazy.plotting.style import (
    ASPECTS,
    COLUMN_WIDTH,
    JOURNALS,
    RC_PARAMS,
    TEXT_WIDTH,
    Journal,
    figsize,
    grid_figsize,
    journal,
    one_to_one,
    register_journal,
    save,
    set_palette,
    use_style,
    verify_style,
)

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
    "plot_nz",
    "plot_pdfs",
    "plot_pit",
    "plot_pit_qq",
    "plot_residuals",
    "plot_zphot_ztrue",
    "register_journal",
    "save",
    "set_palette",
    "use_style",
    "verify_style",
]
