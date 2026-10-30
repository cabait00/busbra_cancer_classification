# %% [plot] Statistics figures of step 4F-B (automatic ROI), drawn from the STORED tables only
#
# Reads reports/tables/auto_roi_run_metrics.csv, auto_roi_paired_differences.csv, auto_roi_operating_points.csv, auto_roi_operating_point_differences.csv, auto_roi_crop_categories.csv and
# auto_roi_box_definitions.csv. It imports neither TensorFlow nor Keras nor project modules, opens no image and no mask, loads no model and starts no inference; the figures can be re-created at any time
# from the tables.
#
# Figure 1 (auto_roi_validation_auc.png): validation ROC-AUC of the three inputs per stage (frozen head, extended fine-tuning 1e-4): full image (4B, stored), expert-mask ROI (4E, stored), automatic ROI
#   (4F-B, same unchanged 4E checkpoints), every seed with its own colour AND marker, mean (bar) and sample SD (whiskers, ddof = 1); right: the paired differences within the same seed.
# Figure 2 (auto_roi_false_alarms_at_targets.png): benign false alarms of 260 at the 4C sensitivity targets 0.90 and 0.95 (at least 113 / 119 of 125 malignant images detected), per seed and input, and the paired
#   differences automatic ROI minus expert ROI and minus full image. No best seed, no pooled value, no significance marks: five seeds show a tendency, not a proof.
# Figure 3 (auto_roi_crop_coverage.png): what the ACTUAL automatic crops do to the validation expert masks: categories by class and by device (counts, fallback images listed on their own, never counted as
#   coverage), and the four box definitions (all predicted pixels, with and without 20 %, largest component, the actual crop) judged on the images without fallback.
#
# The automatic ROI needs no manual mask at inference, but the 4E classifiers were trained on expert-mask ROIs and the U-Net was chosen with the validation images: explorative, not an independent assessment.
#
# Usage (CPU only, from the project root BUSBRA):
#   .venv/bin/python plots/plot_auto_roi_results.py

# %% Imports and paths
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "plots")]
PROJECT_ROOT = ROOT
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from style import SEED_STYLE, SURFACE, INK, INK_2, style  # noqa: E402

SEEDS = [7, 21, 42, 123, 2026]
STAGES = [("frozen", "Frozen head", ["4B_frozen", "4E_frozen", "4FB_frozen"]), ("finetune", "Extended fine-tuning 1e-4", ["4B_finetune", "4E_finetune", "4FB_finetune"])]
INPUT_LABELS = ["Full image\n(4B, stored)", "Expert ROI\n(4E, stored)", "Automatic ROI\n(4F-B)"]
CATEGORY_COLOURS = {"covers_all_expert_pixels": "#1baf7a", "covers_90_to_below_100_percent": "#9fd6b8", "covers_more_than_0_to_below_90_percent": "#eb9a34", "covers_no_expert_pixel": "#c83a2e", "fallback_full_image": "#8a8a85"}
CATEGORY_LABELS = {"covers_all_expert_pixels": "crop covers all expert pixels", "covers_90_to_below_100_percent": "covers 90 % to < 100 %", "covers_more_than_0_to_below_90_percent": "covers > 0 to < 90 %",
                   "covers_no_expert_pixel": "covers no expert pixel", "fallback_full_image": "fallback: whole image (empty prediction)"}
NOTE = ("Automatic ROI: the stored 4F-A U-Net localises the lesion without a manual mask, but the 4E classifiers were trained on expert-mask ROIs and the U-Net was chosen with the validation images.\n"
        "Explorative, same 385 validation images, one segmentation seed, five classifier seeds (not five cohorts).")



def seed_legend(fig) -> None:
    handles = [plt.Line2D([], [], marker=SEED_STYLE[s][1], color=SEED_STYLE[s][0], linestyle="", markersize=7, label=f"seed {s}") for s in SEEDS]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))


# %% Read the stored tables and check that they describe the complete design
runs = pd.read_csv(TABLES_DIR / "auto_roi_run_metrics.csv")
paired = pd.read_csv(TABLES_DIR / "auto_roi_paired_differences.csv")
points = pd.read_csv(TABLES_DIR / "auto_roi_operating_points.csv", float_precision="round_trip")
point_diffs = pd.read_csv(TABLES_DIR / "auto_roi_operating_point_differences.csv")
categories = pd.read_csv(TABLES_DIR / "auto_roi_crop_categories.csv")
definitions = pd.read_csv(TABLES_DIR / "auto_roi_box_definitions.csv")
if len(runs) != 30 or sorted(runs["seed"].unique()) != SEEDS:
    raise SystemExit(f"the run table must hold 6 groups x 5 seeds, found {len(runs)} rows")


def scatter_group(ax, x, values_by_seed: dict, width=0.34) -> None:
    for k, seed in enumerate(SEEDS):
        colour, marker = SEED_STYLE[seed]
        ax.scatter(x + (k - 2) * width / 4, values_by_seed[seed], s=46, color=colour, marker=marker, zorder=3, edgecolor=SURFACE, linewidth=0.7)
    values = np.array([values_by_seed[s] for s in SEEDS], dtype=float)
    ax.hlines(values.mean(), x - width * 0.55, x + width * 0.55, color=INK, linewidth=1.6, zorder=2)
    sd = values.std(ddof=1)
    ax.vlines(x, values.mean() - sd, values.mean() + sd, color=INK, linewidth=1.0, zorder=2)


# %% Figure 1: validation ROC-AUC and the paired differences
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.5, 1]})
for row, (stage, title, groups) in enumerate(STAGES):
    ax = axes[row, 0]
    style(ax)
    for position, group in enumerate(groups):
        part = runs[runs["group"] == group].set_index("seed")["roc_auc"]
        scatter_group(ax, position, {s: float(part[s]) for s in SEEDS})
    ax.set_xticks(range(3))
    ax.set_xticklabels(INPUT_LABELS, fontsize=9, color=INK_2)
    ax.set_ylabel("validation ROC-AUC (385 images)", color=INK_2, fontsize=9)
    ax.set_title(f"{title}: ROC-AUC by input (bar = mean, whiskers = sample SD)", color=INK, fontsize=10.5, loc="left")
    ax = axes[row, 1]
    style(ax)
    for position, (pair, label) in enumerate((("auto_minus_expert", "Automatic ROI\nminus expert ROI"), ("auto_minus_full", "Automatic ROI\nminus full image"))):
        part = paired[(paired["pair"] == f"{pair} ({stage})") & (paired["metric"] == "roc_auc")].set_index("seed")["difference_b_minus_a"]
        scatter_group(ax, position, {s: float(part[s]) for s in SEEDS}, width=0.5)
    ax.axhline(0, color=INK_2, linewidth=1.0, linestyle=(0, (3, 3)))
    ax.set_xticks(range(2))
    ax.set_xticklabels(["Automatic ROI\nminus expert ROI", "Automatic ROI\nminus full image"], fontsize=9, color=INK_2)
    ax.set_ylabel("paired ROC-AUC difference within the seed", color=INK_2, fontsize=9)
    ax.set_title(f"{title}: paired differences", color=INK, fontsize=10.5, loc="left")
fig.suptitle("Step 4F-B: validation ROC-AUC of the ten unchanged 4E classifiers on three inputs\n" + NOTE, fontsize=9.2, color=INK_2, x=0.01, ha="left")
seed_legend(fig)
fig.tight_layout(rect=(0, 0.04, 1, 0.92))
fig.savefig(FIGURES_DIR / "auto_roi_validation_auc.png", dpi=130, facecolor=SURFACE)
plt.close(fig)

# %% Figure 2: benign false alarms at the sensitivity targets
fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 1.15]})
TARGET_NEEDED = {"0.90": 113, "0.95": 119}
for row, (stage, title, groups) in enumerate(STAGES):
    for column, target in enumerate(("0.90", "0.95")):
        ax = axes[row, column]
        style(ax)
        for position, group in enumerate(groups):
            part = points[(points["group"] == group) & (points["point_label"] == f"target_{target}")].set_index("seed")["fp"]
            scatter_group(ax, position, {s: float(part[s]) for s in SEEDS})
        ax.set_xticks(range(3))
        ax.set_xticklabels(INPUT_LABELS, fontsize=8.5, color=INK_2)
        ax.set_ylabel("benign false alarms (of 260)", color=INK_2, fontsize=9)
        ax.set_title(f"{title}: target {target} (>= {TARGET_NEEDED[target]} of 125 malignant)", color=INK, fontsize=10, loc="left")
    ax = axes[row, 2]
    style(ax)
    labels, position = [], 0
    for target in ("0.90", "0.95"):
        for pair, label in (("auto_minus_expert", "auto − expert"), ("auto_minus_full", "auto − full")):
            part = point_diffs[(point_diffs["pair"] == pair) & (point_diffs["stage"] == stage) & (point_diffs["point"] == f"target_{target}") & (point_diffs["level"] == "image") & (point_diffs["metric"] == "fp")].set_index("seed")["difference_auto_minus_reference"]
            scatter_group(ax, position, {s: float(part[s]) for s in SEEDS}, width=0.5)
            labels.append(f"{label}\ntarget {target}")
            position += 1
    ax.axhline(0, color=INK_2, linewidth=1.0, linestyle=(0, (3, 3)))
    ax.set_xticks(range(position))
    ax.set_xticklabels(labels, fontsize=8, color=INK_2)
    ax.set_ylabel("paired difference of the false alarms (negative = fewer)", color=INK_2, fontsize=9)
    ax.set_title(f"{title}: paired differences", color=INK, fontsize=10, loc="left")
fig.suptitle("Step 4F-B: benign false alarms at the 4C sensitivity targets (largest threshold reaching the target on the same validation images; explorative, no threshold is chosen)\n" + NOTE, fontsize=9.2, color=INK_2, x=0.01, ha="left")
seed_legend(fig)
fig.tight_layout(rect=(0, 0.04, 1, 0.92))
fig.savefig(FIGURES_DIR / "auto_roi_false_alarms_at_targets.png", dpi=130, facecolor=SURFACE)
plt.close(fig)

# %% Figure 3: what the actual crops do to the expert masks
fig = plt.figure(figsize=(15, 8.6), facecolor=SURFACE)
grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1], hspace=0.62, wspace=0.28)
order = ["covers_all_expert_pixels", "covers_90_to_below_100_percent", "covers_more_than_0_to_below_90_percent", "covers_no_expert_pixel", "fallback_full_image"]
for slot, (group_type, names, title) in enumerate((("class", ["benign", "malignant"], "By class"), ("device", None, "By device"))):
    ax = fig.add_subplot(grid[slot, 0])
    style(ax)
    part = categories[categories["group_type"] == group_type]
    groups = names or sorted(part["group"].unique())
    left = np.zeros(len(groups))
    for category in order:
        values = np.array([float(part[(part["group"] == g) & (part["category"] == category)]["share_of_images"].iloc[0]) for g in groups])
        counts = [int(part[(part["group"] == g) & (part["category"] == category)]["n_images"].iloc[0]) for g in groups]
        ax.barh(range(len(groups)), values, left=left, color=CATEGORY_COLOURS[category], height=0.62, label=CATEGORY_LABELS[category])
        for k, (v, c, l) in enumerate(zip(values, counts, left)):
            if c and v >= 0.03:
                ax.text(l + v / 2, k, str(c), ha="center", va="center", fontsize=8, color="#fff" if category in ("covers_no_expert_pixel", "fallback_full_image", "covers_all_expert_pixels") else INK)
        left += values
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{g.split(' @')[0]}\n(n = {int(part[part['group'] == g]['denominator_images'].iloc[0])}, {int(part[part['group'] == g]['denominator_cases'].iloc[0])} cases)" for g in groups], fontsize=8.5, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the validation images of the group (numbers: images, shown for segments of at least 3 %; all counts are in the table)", color=INK_2, fontsize=8)
    ax.set_title(f"Actual automatic crop against the expert mask: {title.lower()}", color=INK, fontsize=10.5, loc="left")
    if slot == 0:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
ax = fig.add_subplot(grid[:, 1])
style(ax)
names = list(definitions["definition"])
labels = ["all predicted pixels\n(4F-A box)", "all predicted pixels\n+ 20 %", "largest component\n(tight box)", "largest component\n+ 20 % (ACTUAL crop)"]
width = 0.26
n_base = int(definitions["n_images_without_fallback"].iloc[0])
for k, (column, label, colour) in enumerate((("n_below_100_percent", "expert mask not fully inside", "#eb9a34"), ("n_below_90_percent", "less than 90 % inside", "#c83a2e"), ("n_zero", "no expert pixel inside", "#6250d6"))):
    ax.bar(np.arange(len(names)) + (k - 1) * width, definitions[column], width=width * 0.92, color=colour, label=label)
    for x, v in zip(np.arange(len(names)) + (k - 1) * width, definitions[column]):
        ax.text(x, v + 0.5, str(int(v)), ha="center", va="bottom", fontsize=8, color=INK_2)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(labels, fontsize=8.5, color=INK_2)
ax.set_ylabel(f"validation images (of {n_base} without fallback)", color=INK_2, fontsize=9)
ax.set_title("Four box definitions on the images without fallback", color=INK, fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
fig.suptitle("Step 4F-B: how well the frozen automatic crops contain the validation expert masks (diagnosis after the freeze; the crop rule is not adapted)\n" + NOTE, fontsize=9.2, color=INK_2, x=0.01, ha="left")
fig.savefig(FIGURES_DIR / "auto_roi_crop_coverage.png", dpi=130, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("wrote reports/figures/auto_roi_validation_auc.png, auto_roi_false_alarms_at_targets.png and auto_roi_crop_coverage.png", flush=True)
sys.exit(0)
