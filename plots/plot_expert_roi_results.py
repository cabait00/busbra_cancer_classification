# %% [plot] Statistics figures of the expert-mask ROI comparison (step 4E), drawn from the STORED tables only
#
# Reads reports/tables/expert_roi_paired_differences.csv, expert_roi_summary_validation.csv, expert_roi_paired_summary.csv, expert_roi_threshold_curves.csv, expert_roi_operating_points.csv and
# expert_roi_operating_point_differences.csv / ..._summary.csv. It imports neither TensorFlow nor Keras, opens no image and no mask, loads no model and starts no training; the figures can be
# re-created at any time from the tables.
#
# INFORMATION CONDITION: the ROI branch uses an EXPERT MASK to localise the lesion (a manual mask is needed at application time). Its curves are not a pure image-to-score performance
# without human localisation; this is written into every figure.
#
# Figure 1 (expert_roi_validation_auc.png): validation ROC-AUC of the four groups (frozen and extended fine-tuning 1e-4, full image 4B and expert ROI), every run with its own colour AND
#   marker shape, the mean (bar) and sample SD (whiskers, ddof=1); right: the paired ROC-AUC difference ROI minus full image within the same seed.
# Figure 2 (expert_roi_sensitivity_vs_fp.png): image sensitivity against the number of false positives for the extended fine-tuning, one panel per seed (full image 4B and expert ROI), markers for
#   the threshold 0.5 and the planned sensitivity targets 0.90 and 0.95; sixth panel: the paired difference of the false alarms (ROI minus full image) at the two targets, every seed with its own
#   marker, the mean as a dashed line. No best seed, no pooled curve, no significance marks: five paired seeds show a tendency, not a proof.
#
# Usage (CPU only, from the project root BUSBRA):
#   .venv/bin/python plots/plot_expert_roi_results.py

# %% Imports and paths
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "plots")]
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from style import SEED_STYLE, SURFACE, INK, INK_2, GRID, style  # noqa: E402
from study_figures import draw_auc_and_pairs  # noqa: E402

GROUPS = [("mobilenetv2_frozen_160_aug", "Frozen\nfull image (4B)"), ("mobilenetv2_frozen_160_roi", "Frozen\nexpert ROI"),
          ("mobilenetv2_higher_lr_160_aug", "Extended 1e-4\nfull image (4B)"), ("mobilenetv2_higher_lr_160_roi", "Extended 1e-4\nexpert ROI")]
PAIRS = [("mobilenetv2_frozen_160_roi minus mobilenetv2_frozen_160_aug", "Frozen"), ("mobilenetv2_higher_lr_160_roi minus mobilenetv2_higher_lr_160_aug", "Extended fine-tuning 1e-4")]
BRANCH_COLOUR = {"full": "#2a78d6", "roi": "#eb6834"}
NOTE = "Expert-mask ROI: the mask localises the lesion; a manual mask is needed at application time (not a pure image-to-score performance). Explorative, same validation images, five seeds."

# %% Read the stored tables and check that they describe the complete design
paired = pd.read_csv(TABLES_DIR / "expert_roi_paired_differences.csv")
paired_summary = pd.read_csv(TABLES_DIR / "expert_roi_paired_summary.csv")
summary = pd.read_csv(TABLES_DIR / "expert_roi_summary_validation.csv")
curves = pd.read_csv(TABLES_DIR / "expert_roi_threshold_curves.csv", float_precision="round_trip")
points = pd.read_csv(TABLES_DIR / "expert_roi_operating_points.csv", float_precision="round_trip")
point_diff = pd.read_csv(TABLES_DIR / "expert_roi_operating_point_differences.csv", float_precision="round_trip")
point_summary = pd.read_csv(TABLES_DIR / "expert_roi_operating_point_summary.csv", float_precision="round_trip")

# %% Figure 1: ROC-AUC per group and paired differences (shared drawing of the study comparisons)
fig = draw_auc_and_pairs(summary, paired, paired_summary, GROUPS, PAIRS, title_left="ROC-AUC: full image (4B) and expert-mask ROI (five seeds, same validation images)",
                         ylabel_pairs="validation ROC-AUC: expert ROI minus full image (same seed)", height=6.6, headroom=0.14, tight_top=0.94,
                         suptitle="Step 4E: expert-mask ROI (contract roi-expert-mask-v1) against the 4B full-image pipeline at 160 x 160 px\n" + NOTE)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES_DIR / "expert_roi_validation_auc.png", dpi=130, facecolor=SURFACE)
plt.close(fig)

# %% Figure 2: sensitivity against false positives (extended fine-tuning) and the paired false-alarm differences at the two targets
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.8), facecolor=SURFACE)
flat_axes = list(axes.ravel())
POINT_STYLE = {"0.5": ("o", "threshold 0.5"), "target_0.90": ("^", "target 0.90"), "target_0.95": ("D", "target 0.95")}
GROUP_OF = {"full": "4B_finetune", "roi": "ROI_finetune"}
n_negative = int(curves["tn"].iloc[0] + curves["fp"].iloc[0])
n_positive = int(curves["tp"].iloc[0] + curves["fn"].iloc[0])
for ax, seed in zip(flat_axes[:5], sorted(SEED_STYLE)):
    style(ax)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlim(-4, n_negative + 4)
    ax.set_ylim(0.0, 1.02)
    for level in (0.90, 0.95):
        ax.axhline(level, color=INK_2, linewidth=0.7, linestyle=(0, (2, 3)), zorder=1)
    ax.set_title(f"seed {seed}", color=INK, fontsize=11, loc="left")
    ax.set_xlabel(f"false positives (benign images classified malignant, of {n_negative})", color=INK_2, fontsize=9)
    ax.set_ylabel(f"image sensitivity (of {n_positive} malignant images)", color=INK_2, fontsize=9)
    for branch in ("full", "roi"):
        curve = curves[(curves["group"] == GROUP_OF[branch]) & (curves["seed"] == seed)]
        ax.plot(curve["fp"], curve["sensitivity"], color=BRANCH_COLOUR[branch], linewidth=1.7, zorder=3)
        for label, (marker, _) in POINT_STYLE.items():
            row = points[(points["group"] == GROUP_OF[branch]) & (points["seed"] == seed) & (points["point_label"] == label)].iloc[0]
            ax.plot([row["fp"]], [row["sensitivity"]], marker=marker, markersize=8.5, color=BRANCH_COLOUR[branch], markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=5)
ax = flat_axes[5]
style(ax)
targets = [("target_0.90", "target 0.90\n(113 of 125 detected)"), ("target_0.95", "target 0.95\n(119 of 125 detected)")]
for k, (label, text) in enumerate(targets):
    part = point_diff[(point_diff["stage"] == "finetune") & (point_diff["point"] == label) & (point_diff["level"] == "image") & (point_diff["metric"] == "fp")].sort_values("seed")
    stats = point_summary[(point_summary["stage"] == "finetune") & (point_summary["point"] == label) & (point_summary["level"] == "image") & (point_summary["metric"] == "fp")].iloc[0]
    for offset, row in zip(np.linspace(-0.3, 0.3, 5), part.itertuples()):
        colour, marker = SEED_STYLE[int(row.seed)]
        ax.scatter(k + offset, row.difference_roi_minus_full, s=80, color=colour, marker=marker, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.hlines(stats["mean_difference"], k - 0.4, k + 0.4, color=INK_2, linestyle="--", linewidth=1.6, zorder=3)
    ax.text(k, 0.98, f"mean {stats['mean_difference']:+.1f}, SD {stats['sd_ddof1']:.1f}\nnegative/positive/zero:\n{int(stats['n_negative'])}/{int(stats['n_positive'])}/{int(stats['n_zero'])}", ha="center", va="top", fontsize=9, color=INK,
            transform=ax.get_xaxis_transform(), bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.0})
ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
ax.set_xticks(range(len(targets)), [t for _, t in targets])
low, high = point_diff[(point_diff["stage"] == "finetune") & point_diff["point"].isin([t for t, _ in targets]) & (point_diff["level"] == "image") & (point_diff["metric"] == "fp")]["difference_roi_minus_full"].agg(["min", "max"])
ax.set_ylim(min(low, 0.0) - 32, max(high, 0.0) + 55)
ax.set_xlim(-0.65, len(targets) - 0.35)
ax.set_ylabel("false alarms (of 260 benign): expert ROI minus full image", color=INK_2, fontsize=9)
ax.set_title("Paired false-alarm differences at the planned targets", color=INK, fontsize=10, loc="left")
ax.legend(handles=[plt.Line2D([], [], marker=SEED_STYLE[s][1], color=SEED_STYLE[s][0], linestyle="none", markersize=7, label=f"seed {s}") for s in sorted(SEED_STYLE)], frameon=False, fontsize=8, labelcolor=INK_2,
          loc="lower left", ncol=3, bbox_to_anchor=(0.0, 0.0))
handles = [plt.Line2D([], [], color=BRANCH_COLOUR["full"], linewidth=2.2, label="full image (4B)"), plt.Line2D([], [], color=BRANCH_COLOUR["roi"], linewidth=2.2, label="expert-mask ROI")]
handles += [plt.Line2D([], [], color=INK_2, marker=m, markersize=8, linestyle="none", markeredgecolor=SURFACE, label=name) for m, name in POINT_STYLE.values()]
handles += [plt.Line2D([], [], color=INK_2, linewidth=0.8, linestyle=(0, (2, 3)), label="sensitivity 0.90 / 0.95"), plt.Line2D([], [], color=INK_2, linestyle="--", linewidth=1.6, label="mean of the five paired differences")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Step 4E: image sensitivity against false positives, extended fine-tuning 1e-4: full image (4B) vs expert-mask ROI, same seed, same 385 validation images\n" + NOTE + "\n"
             "The sensitivity-target thresholds come from the same images (explorative); no best seed, no pooled curve, no final threshold.", fontsize=9, color=INK_2, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.92))
fig.savefig(FIGURES_DIR / "expert_roi_sensitivity_vs_fp.png", dpi=130, facecolor=SURFACE)
plt.close(fig)
print(f"written reports/figures/expert_roi_validation_auc.png and expert_roi_sensitivity_vs_fp.png from stored tables; TensorFlow imported: {'tensorflow' in sys.modules}")
