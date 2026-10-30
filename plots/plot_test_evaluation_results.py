# %% Statistics figures of step 5C (CPU, no TensorFlow, no inference): drawn ONLY from the aggregate tables of the finished study (numbers, no image data)
#
# Reads reports/tables/test_evaluation_*.csv and final_validation_metrics.csv and writes four figures to reports/figures/:
#   test_auc_by_group.png            image and case ROC-AUC on the test subset: the single seeds, the seed mean +- sample SD, and the validation seed mean of 5A for orientation
#   test_main_comparisons.png        the three main comparisons: point estimate and two-sided 95 % percentile bootstrap interval (AUC, and sensitivity / specificity at the frozen 0.90 / 0.95 points, image and case level)
#   test_operating_points.png        sensitivity reached and benign false alarms at the frozen image-level 0.90 / 0.95 thresholds, per seed, next to the validation target
#   test_segmentation_quality.png    Dice of the raw U-Net prediction (4F-A definition) and the crop categories of the actual automatic crops (4F-B definition), fallbacks separate
# The figures describe a fixed test subset, fixed models and fixed thresholds. They are no ranking, no significance statement and no clinical claim.
#
# Usage (CPU only, from the project root BUSBRA):  .venv/bin/python plots/plot_test_evaluation_results.py
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "plots")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from style import COLOR, GROUPS, LABEL  # noqa: E402

from busbra import results  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
NOTE = textwrap.fill("Fixed test subset (376 images / 212 cases), fixed models and frozen validation thresholds; descriptive, no ranking, no significance statement, no clinical claim.", 120)
STAGE_NOTE = "Full image = augmented 4B pipelines; expert ROI = 4E classifiers on crops from a manual mask; automatic ROI = U-Net (4F-A) crops fed to the same 4E classifiers. Main comparisons use the fine-tuned groups."
COMPARISON_LABEL = {"expert_minus_full": "expert ROI minus full image", "auto_minus_expert": "automatic ROI minus expert ROI", "auto_minus_full": "automatic ROI minus full image"}
COMPARISON_COLOR = {"expert_minus_full": "#009E73", "auto_minus_expert": "#D55E00", "auto_minus_full": "#0072B2"}


def load_tables() -> dict:
    names = ("metrics", "operating_points", "bootstrap_summary", "segmentation_dice_histogram", "segmentation_crop_categories", "segmentation_summary", "validation_vs_test")
    tables = {name: results.table(f"test_evaluation_{name}", dtype={"seed": "Int64"}) for name in names}
    tables["validation_metrics_5a"] = results.table("final_validation_metrics", dtype={"seed": "Int64"})
    return tables


def figure_auc(tables: dict) -> Path:
    metrics, validation = tables["metrics"], tables["validation_metrics_5a"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    rng = np.random.default_rng(3)
    for ax, level in zip(axes, ("image", "case")):
        for k, group in enumerate(GROUPS):
            test = metrics[(metrics["group_id"] == group) & (metrics["level"] == level)]
            values = test["roc_auc"].to_numpy(dtype=float)
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(values)), values, color=COLOR[group], s=28, alpha=0.85, zorder=3, label="single seeds (test)" if k == 0 else None)
            mean = values.mean()
            sd = values.std(ddof=1) if len(values) > 1 else None
            ax.errorbar(k + 0.28, mean, yerr=sd, color="black", marker="D", markersize=6, capsize=4, linewidth=1.4, zorder=4, label="test seed mean +- sample SD" if k == 0 else None)
            val = validation[(validation["group_id"] == group) & (validation["level"] == level)]["roc_auc"].mean()
            ax.scatter(k - 0.28, val, facecolor="none", edgecolor="#555555", marker="s", s=46, linewidth=1.3, zorder=3, label="validation seed mean (5A, for orientation)" if k == 0 else None)
        ax.axhline(0.5, color="#777777", linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(GROUPS)), [LABEL[g] for g in GROUPS], fontsize=7.5)
        ax.set_title(f"{level}-level ROC-AUC ({'376 images' if level == 'image' else '212 cases; case score = maximum of the image scores'})", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("ROC-AUC (malignant = positive)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Final evaluation: ROC-AUC of the seven groups\n" + textwrap.fill(STAGE_NOTE, 150) + "\n" + NOTE, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    path = FIGURES / "test_auc_by_group.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_comparisons(tables: dict) -> Path:
    summary = tables["bootstrap_summary"]
    panels = [("ROC-AUC difference", [("auc_image", "image"), ("auc_case", "case")]),
              ("image level, frozen image thresholds", [(f"{k}_image_{t}", f"{'sensitivity' if k == 'sensitivity' else 'specificity'} at target {t}") for t in ("0.90", "0.95") for k in ("sensitivity", "specificity")]),
              ("case level, frozen case thresholds", [(f"{k}_case_{t}", f"{'sensitivity' if k == 'sensitivity' else 'specificity'} at target {t}") for t in ("0.90", "0.95") for k in ("sensitivity", "specificity")])]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    for ax, (title, metrics) in zip(axes, panels):
        y = 0
        ticks, labels = [], []
        for metric, name in metrics:
            for comparison in COMPARISON_LABEL:
                row = summary[(summary["comparison"] == comparison) & (summary["metric"] == metric)].iloc[0]
                ax.errorbar(row["point_estimate_seed_mean_difference"], y, xerr=[[row["point_estimate_seed_mean_difference"] - row["ci95_low"]], [row["ci95_high"] - row["point_estimate_seed_mean_difference"]]], color=COMPARISON_COLOR[comparison],
                            marker="o", markersize=6, capsize=4, linewidth=1.6, label=COMPARISON_LABEL[comparison] if (metric == metrics[0][0]) else None)
                y -= 1
            ticks.append(y + 2)
            labels.append(name)
            y -= 0.8
        ax.axvline(0, color="#777777", linewidth=0.9)
        ax.set_yticks(ticks, labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("difference (other minus reference), 95 % percentile interval", fontsize=8)
        ax.grid(axis="x", alpha=0.25)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle("Three main comparisons: paired case-stratified cluster bootstrap, 2,000 draws, RNG seed 2026\n" + textwrap.fill("Point estimate = mean over the five seeds of the paired differences; the interval describes test-sample uncertainty only. " + STAGE_NOTE, 150) + "\n" + NOTE, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))
    path = FIGURES / "test_main_comparisons.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_operating_points(tables: dict) -> Path:
    points = tables["operating_points"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True)
    rng = np.random.default_rng(5)
    for c, target in enumerate(("0.90", "0.95")):
        for r, (metric, ylabel) in enumerate((("sensitivity", "sensitivity reached on the test subset"), ("fp", "benign false alarms (of the benign test images)"))):
            ax = axes[r][c]
            for k, group in enumerate(GROUPS):
                part = points[(points["group_id"] == group) & (points["level"] == "image") & (points["point"] == f"target_{target}")]
                values = part[metric].to_numpy(dtype=float)
                ax.scatter(k + rng.uniform(-0.12, 0.12, len(values)), values, color=COLOR[group], s=28, alpha=0.85, zorder=3)
                ax.errorbar(k + 0.28, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else None, color="black", marker="D", markersize=5.5, capsize=3.5, linewidth=1.3, zorder=4)
            if metric == "sensitivity":
                ax.axhline(float(target), color="#777777", linestyle="--", linewidth=1.0, label=f"validation target {target}")
                ax.legend(fontsize=8, frameon=False, loc="lower left")
            ax.set_title(f"{'sensitivity reached' if metric == 'sensitivity' else 'benign false alarms'} at the frozen image-level threshold for target {target}", fontsize=9.5)
            ax.set_ylabel(ylabel, fontsize=8.5)
            ax.grid(axis="y", alpha=0.25)
    for ax in axes[1]:
        ax.set_xticks(range(len(GROUPS)), [LABEL[g] for g in GROUPS], fontsize=7.5)
    fig.suptitle("Frozen operating points on the test subset: single seeds (dots) and seed mean +- sample SD (diamonds)\nThe target names the VALIDATION selection; the test sensitivity is reported as reached.\n" + NOTE, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = FIGURES / "test_operating_points.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_segmentation(tables: dict) -> Path:
    histogram, categories, summary = tables["segmentation_dice_histogram"], tables["segmentation_crop_categories"], tables["segmentation_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    centres = (histogram["bin_low"] + histogram["bin_high"]) / 2
    ax.bar(centres, histogram["n_images"], width=0.045, color="#0072B2")
    dice = summary[(summary["group"] == "all") & (summary["metric"] == "dice_original")].iloc[0]
    ax.axvline(dice["mean"], color="#D55E00", linewidth=1.5, label=f"mean {dice['mean']:.3f}")
    ax.axvline(dice["median"], color="#009E73", linewidth=1.5, linestyle="--", label=f"median {dice['median']:.3f}")
    ax.set_xlabel("Dice of ALL predicted pixels vs the expert mask (original pixel grid, no padding)")
    ax.set_ylabel("number of test images")
    ax.set_title("Raw U-Net prediction (4F-A definition)", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    part = categories[categories["group_type"] == "all"]
    order = ["covers_all_expert_pixels", "covers_90_to_below_100_percent", "covers_more_than_0_to_below_90_percent", "covers_no_expert_pixel", "fallback_full_image"]
    names = ["covers all\nexpert pixels", "covers 90 to\n<100 %", "covers >0 to\n<90 %", "covers no expert\npixel (lesion missed)", "full-image fallback\n(empty prediction)"]
    counts = [int(part[part["category"] == c].iloc[0]["n_images"]) for c in order]
    colors = ["#009E73", "#56B4E9", "#E69F00", "#D55E00", "#999999"]
    ax.bar(range(len(order)), counts, color=colors)
    for k, v in enumerate(counts):
        ax.text(k, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(order)), names, fontsize=8)
    ax.set_ylabel("number of test images")
    ax.set_title("Actual automatic crops against the expert mask\n(4F-B definition; fallbacks listed separately)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Segmentation and localisation quality on the test subset, evaluated after the automatic ROIs were frozen\n" + NOTE, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = FIGURES / "test_segmentation_quality.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> int:
    tables = load_tables()
    FIGURES.mkdir(parents=True, exist_ok=True)
    made = [figure_auc(tables), figure_comparisons(tables), figure_operating_points(tables), figure_segmentation(tables)]
    print(f"wrote {len(made)} figures to reports/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
