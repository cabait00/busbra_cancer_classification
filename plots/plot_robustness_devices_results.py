# %% Statistics figures of step 5E (CPU, no TensorFlow, no inference): drawn ONLY from the aggregate tables of the finished study (numbers, no image data)
#
# Reads reports/tables/robustness_test_*.csv and device_test_*.csv and writes four figures to reports/figures/:
#   robustness_test_auc_ap.png        image / case ROC-AUC and average precision over epsilon: seed values (thin lines) and seed mean +- sample SD, two conditions
#   robustness_test_errors.png        false positives / false negatives (image level) at the frozen thresholds 0.5 / target 0.90 / target 0.95 over epsilon: seed values and seed mean +- SD
#   device_test_clean.png             unperturbed image-level ROC-AUC, sensitivity and specificity at the frozen target-0.95 threshold by device for the seven groups (seed mean, SD where defined) with the device denominators
#   device_test_attacked.png          image-level ROC-AUC and false-negative / false-positive counts by device over epsilon for the two attacked conditions (seed mean)
# The figures describe a fixed test subset, fixed models, frozen thresholds and a one-step digital perturbation. They are no ranking, no significance statement, no robustness proof, no fairness statement and no clinical claim.
#
# Usage (CPU only, from the project root BUSBRA):  .venv/bin/python plots/plot_robustness_devices_results.py
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
from busbra import results  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
COND = {"full": "Full image (4B, fine-tuned)", "auto": "Automatic ROI (4E classifier)"}
COLOR = {"full": "#0072B2", "auto": "#D55E00"}
EPS = [0, 1, 2, 4]
EPS_LABEL = ["0", "1/255", "2/255", "4/255"]
GROUP_LABEL = {"dummy": "Dummy", "logreg_l2": "LogReg", "small_cnn": "Small CNN", "full_frozen": "Full frozen", "full_finetuned": "Full fine-tuned", "expert_roi_finetuned": "Expert ROI", "auto_roi_finetuned": "Auto ROI"}
GROUP_COLOR = {"dummy": "#999999", "logreg_l2": "#CC79A7", "small_cnn": "#E69F00", "full_frozen": "#56B4E9", "full_finetuned": "#0072B2", "expert_roi_finetuned": "#009E73", "auto_roi_finetuned": "#D55E00"}
NOTE = textwrap.fill("Fixed test subset (376 images / 212 cases), fixed models and frozen validation thresholds; one-step digital FGSM with true labels; descriptive, no ranking, no significance statement, no robustness proof, no clinical claim.", 130)


def load_tables() -> dict:
    names = ["robustness_test_metrics", "robustness_test_operating_points", "robustness_test_summary_metrics", "robustness_test_summary_operating_points", "device_test_assignment", "device_test_clean_summary", "device_test_attacked_summary"]
    return {name: results.table(name, dtype={"seed": "Int64"}) for name in names}


def seed_plot(ax, frame, value, condition, color, offset):
    """Seed values (thin lines) and seed mean +- sample SD over the four epsilons for one condition."""
    part = frame[frame["condition"] == condition]
    for _, g in part.groupby("seed"):
        g = g.sort_values("epsilon_255")
        ax.plot(np.arange(4) + offset, g[value], color=color, alpha=0.3, lw=0.9, marker="o", ms=2.5)
    agg = part.groupby("epsilon_255")[value].agg(["mean", "std"]).reindex(EPS)
    ax.errorbar(np.arange(4) + offset, agg["mean"], yerr=agg["std"], color=color, lw=2.2, marker="D", ms=5, capsize=3, label=COND[condition])


def style(ax, title, ylabel, hline=None):
    ax.set_xticks(range(4), EPS_LABEL)
    ax.set_xlabel("FGSM epsilon (classifier input scale [0, 1])", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    if hline is not None:
        ax.axhline(hline, color="grey", lw=0.8, ls="--")


def figure_auc_ap(t) -> Path:
    m = t["robustness_test_metrics"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for r, metric in enumerate(("roc_auc", "average_precision")):
        for c, level in enumerate(("image", "case")):
            ax = axes[r][c]
            part = m[m["level"] == level]
            for condition, off in (("full", -0.03), ("auto", 0.03)):
                seed_plot(ax, part, metric, condition, COLOR[condition], off)
            style(ax, f"{level}-level {'ROC-AUC' if metric == 'roc_auc' else 'average precision'}", "ROC-AUC" if metric == "roc_auc" else "average precision", 0.5 if metric == "roc_auc" else None)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("FGSM one-step stress test on the frozen test subset: discrimination over epsilon\nThin lines: the five seeds; thick line: seed mean +- sample SD (image level 376 images, case level 212 cases; case score = maximum of the image scores)\n" + NOTE, fontsize=9)
    fig.subplots_adjust(top=0.84, bottom=0.07, left=0.07, right=0.98, hspace=0.38, wspace=0.22)
    path = FIGURES / "robustness_test_auc_ap.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_errors(t) -> Path:
    core = t["robustness_test_operating_points"]
    core = core[core["level"] == "image"]
    points = [("0.5", "reference 0.5"), ("target_0.90", "frozen target 0.90"), ("target_0.95", "frozen target 0.95")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for r, (metric, label) in enumerate((("fp", "benign images classified malignant (FP, of 254)"), ("fn", "malignant images classified benign (FN, of 122)"))):
        for c, (point, title) in enumerate(points):
            ax = axes[r][c]
            for condition, off in (("full", -0.03), ("auto", 0.03)):
                seed_plot(ax, core[core["point"] == point], metric, condition, COLOR[condition], off)
            style(ax, title, label)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle("Error counts at the FROZEN image-level thresholds (no threshold is re-chosen on clean or attacked scores)\nThin lines: seeds; thick line: seed mean +- sample SD. The targets name the VALIDATION selection.\n" + NOTE, fontsize=9)
    fig.subplots_adjust(top=0.85, bottom=0.07, left=0.06, right=0.99, hspace=0.38, wspace=0.28)
    path = FIGURES / "robustness_test_errors.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def device_axis(assign):
    devices = [g for g in assign["group"] if g not in ("all", "mixed devices")]
    names = [f"{d.split(' @')[0]}\n{int(assign.set_index('group').loc[d, 'n_images'])} img ({int(assign.set_index('group').loc[d, 'n_images_malignant'])} mal.)" for d in devices]
    return devices, names


def figure_device_clean(t) -> Path:
    assign, s = t["device_test_assignment"], t["device_test_clean_summary"]
    devices, names = device_axis(assign)
    img = s[s["level"] == "image"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    spec = [("roc_auc", "", "ROC-AUC (image level)"), ("sensitivity", "target_0.95", "sensitivity at the frozen target-0.95 threshold"), ("specificity", "target_0.95", "specificity at the frozen target-0.95 threshold")]
    groups = list(GROUP_LABEL)
    for ax, (metric, point, title) in zip(axes, spec):
        for k, g in enumerate(groups):
            part = img[(img["group_id"] == g) & (img["metric"] == metric) & (img["point"].fillna("") == point)].set_index("group").reindex(devices)
            x = np.arange(len(devices)) + (k - 3) * 0.11
            ax.errorbar(x, part["mean"], yerr=part["sd_ddof1"], fmt="o", color=GROUP_COLOR[g], ms=4, capsize=2, label=GROUP_LABEL[g], lw=1)
        ax.set_xticks(range(len(devices)), names, fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.05, 1.12)
        ax.tick_params(labelsize=8)
        if metric == "roc_auc":
            ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    axes[0].legend(fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle("Unperturbed test results by acquisition device (from the stored 5C scores and the global frozen thresholds): seed mean +- sample SD per group\nDescriptive: small and unbalanced device groups (an undefined value is not drawn), no device-specific thresholds, no fairness statement, no device-cause claim.\n" + NOTE, fontsize=9)
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.05, right=0.99, wspace=0.22)
    path = FIGURES / "device_test_clean.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_device_attacked(t) -> Path:
    assign, s = t["device_test_assignment"], t["device_test_attacked_summary"]
    devices, names = device_axis(assign)
    img = s[s["level"] == "image"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6))
    cmap = ["#000000", "#E69F00", "#009E73", "#CC79A7"]
    for r, condition in enumerate(("full", "auto")):
        part = img[img["condition"] == condition]
        for c, (metric, point, title) in enumerate((("roc_auc", "", "ROC-AUC"), ("fn", "target_0.95", "FN at target 0.95 (count)"), ("fp", "target_0.95", "FP at target 0.95 (count)"))):
            ax = axes[r][c]
            for k, d in enumerate(devices):
                g = part[(part["group"] == d) & (part["metric"] == metric) & (part["point"].fillna("") == point)].set_index("epsilon_255").reindex(EPS)
                ax.errorbar(np.arange(4) + (k - 1.5) * 0.04, g["mean"], yerr=g["sd_ddof1"], marker="o", ms=4, color=cmap[k], capsize=2, lw=1.2, label=names[k].replace("\n", ", "))
            style(ax, f"{'Full image (4B)' if condition == 'full' else 'Automatic ROI'}: {title}", title, 0.5 if metric == "roc_auc" else None)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Perturbed test results by device (image level, seed mean +- sample SD; an undefined value is not drawn)\nDescriptive: the smallest device group has a handful of images; counts are per device group, not rates.\n" + NOTE, fontsize=9)
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.06, right=0.99, hspace=0.38, wspace=0.28)
    path = FIGURES / "device_test_attacked.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> int:
    tables = load_tables()
    FIGURES.mkdir(parents=True, exist_ok=True)
    made = [figure_auc_ap(tables), figure_errors(tables), figure_device_clean(tables), figure_device_attacked(tables)]
    print(f"wrote {len(made)} figures to reports/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
