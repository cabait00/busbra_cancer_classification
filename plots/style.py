"""Shared figure style of the statistics plots: surface colours, axis styling, seed markers and the group labels / colours of the seven final groups.

The statistics plots draw only numbers from stored result tables. The one exception is plot_gradcam_test_interpretation.py, which reads the few selected target images and masks for local previews.
Importing this module selects the non-interactive Agg backend.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
SEED_STYLE = {7: ("#2a78d6", "o"), 21: ("#eb6834", "s"), 42: ("#1baf7a", "^"), 123: ("#0a6e4c", "D"), 2026: ("#6250d6", "v")}  # seed -> (colour, marker)

GROUPS = ["dummy", "logreg_l2", "small_cnn", "full_frozen", "full_finetuned", "expert_roi_finetuned", "auto_roi_finetuned"]
LABEL = {"dummy": "Dummy\n(class prior)", "logreg_l2": "LogReg L2\n64 px", "small_cnn": "Small CNN\n(3F)", "full_frozen": "Full image\nfrozen", "full_finetuned": "Full image\nfine-tuned",
         "expert_roi_finetuned": "Expert ROI\n(manual mask)", "auto_roi_finetuned": "Automatic\nROI"}
COLOR = {"dummy": "#999999", "logreg_l2": "#CC79A7", "small_cnn": "#E69F00", "full_frozen": "#56B4E9", "full_finetuned": "#0072B2", "expert_roi_finetuned": "#009E73", "auto_roi_finetuned": "#D55E00"}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
