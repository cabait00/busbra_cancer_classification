# %% Statistics figures of the exploratory add-on 5B-R (CPU, no TensorFlow, no training): drawn ONLY from the aggregate tables (numbers, no image data)
#
# Reads reports/tables/adversarial_roi_run_metrics.csv, _summary.csv and _paired_differences.csv and writes three figures to reports/figures/:
#   adversarial_roi_clean_performance.png    image ROC-AUC and average precision on UNCHANGED images, per seed (paired lines) and seed mean, three states, two input conditions
#   adversarial_roi_fgsm_curves.png          image ROC-AUC and BCE against the FGSM strength epsilon (0 to 4/255), seed mean with sample SD and the five seeds
#   adversarial_roi_paired_differences.png   paired differences within the seed: fgsm minus clean continuation and the two continuations minus the original, per seed
# The figures describe five training seeds on the same validation holdout. They are no ranking, no significance statement and no evidence of general robustness.
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGURES = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
STATES = ["original", "clean_continuation", "fgsm_continuation"]
LABEL = {"original": "original 4E state", "clean_continuation": "clean continuation (5 epochs)", "fgsm_continuation": "FGSM continuation (5 epochs)"}
COLOR = {"original": "#4d4d4d", "clean_continuation": "#0072B2", "fgsm_continuation": "#D55E00"}
MARKER = {"original": "s", "clean_continuation": "o", "fgsm_continuation": "^"}
CONDITION = {"expert": "expert-mask ROI (primary)", "auto": "frozen automatic ROI (transfer check)"}
EPSILONS = [0, 1, 2, 4]
NOTE = "Exploratory validation add-on, five seeds on the same holdout (385 images / 213 cases); no ranking, no significance statement, no evidence of general robustness."


def main() -> int:
    runs = pd.read_csv(TABLES / "adversarial_roi_run_metrics.csv")
    summary = pd.read_csv(TABLES / "adversarial_roi_summary.csv")
    differences = pd.read_csv(TABLES / "adversarial_roi_paired_differences.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ---- 1. clean performance
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for r, metric in enumerate(["roc_auc", "average_precision"]):
        for c, condition in enumerate(["expert", "auto"]):
            ax = axes[r][c]
            part = runs[(runs["condition"] == condition) & (runs["epsilon_255"] == 0) & (runs["level"] == "image")]
            for seed, seed_part in part.groupby("seed"):
                values = [seed_part[seed_part["state"] == s][metric].iloc[0] for s in STATES]
                ax.plot(range(3), values, color="#bbbbbb", linewidth=0.9, zorder=1)
                for k, s in enumerate(STATES):
                    ax.scatter(k, values[k], color=COLOR[s], marker=MARKER[s], s=26, zorder=3)
            for k, s in enumerate(STATES):
                row = summary[(summary["state"] == s) & (summary["condition"] == condition) & (summary["epsilon_255"] == 0) & (summary["level"] == "image") & (summary["metric"] == metric)].iloc[0]
                ax.errorbar(k + 0.18, row["mean"], yerr=row["sd_ddof1"], color=COLOR[s], marker="D", markersize=7, capsize=4, linewidth=1.6, zorder=4)
            ax.set_xticks(range(3), ["original", "clean\ncontinuation", "FGSM\ncontinuation"])
            ax.set_title(f"{CONDITION[condition]}: {'image ROC-AUC' if metric == 'roc_auc' else 'image average precision'}", fontsize=10)
            ax.set_ylabel("validation value, unchanged images")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Unchanged validation images: the three states per seed (grey lines connect one seed; diamonds: seed mean ± sample SD)\n" + NOTE, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "adversarial_roi_clean_performance.png", dpi=130)
    plt.close(fig)

    # ---- 2. FGSM curves
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True)
    for r, metric in enumerate(["roc_auc", "bce"]):
        for c, condition in enumerate(["expert", "auto"]):
            ax = axes[r][c]
            for s in STATES:
                for seed, seed_part in runs[(runs["state"] == s) & (runs["condition"] == condition) & (runs["level"] == "image")].groupby("seed"):
                    ordered = seed_part.sort_values("epsilon_255")
                    ax.plot(ordered["epsilon_255"], ordered[metric], color=COLOR[s], alpha=0.25, linewidth=0.8)
                rows = summary[(summary["state"] == s) & (summary["condition"] == condition) & (summary["level"] == "image") & (summary["metric"] == metric)].sort_values("epsilon_255")
                ax.errorbar(rows["epsilon_255"], rows["mean"], yerr=rows["sd_ddof1"], color=COLOR[s], marker=MARKER[s], markersize=6, capsize=3, linewidth=1.8, label=LABEL[s])
            ax.set_xticks(EPSILONS, ["0", "1/255", "2/255", "4/255"])
            ax.set_title(f"{CONDITION[condition]}: {'image ROC-AUC' if metric == 'roc_auc' else 'image BCE (from the logit)'}", fontsize=10)
            ax.set_ylabel("validation value")
            ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("FGSM strength epsilon (one step from the clean input, generated against the evaluated model)", fontsize=8)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("FGSM stress test on the validation images: seed mean ± sample SD (bold), the five seeds (thin)\n" + NOTE, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(FIGURES / "adversarial_roi_fgsm_curves.png", dpi=130)
    plt.close(fig)

    # ---- 3. paired differences
    pairs = [("fgsm_continuation minus clean_continuation", "#D55E00"), ("clean_continuation minus original", "#0072B2"), ("fgsm_continuation minus original", "#009E73")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for c, condition in enumerate(["expert", "auto"]):
        ax = axes[c]
        positions, labels_ = [], []
        for e, k in enumerate([0, 2, 4]):
            for p, (pair, color) in enumerate(pairs):
                part = differences[(differences["pair"] == pair) & (differences["condition"] == condition) & (differences["epsilon_255"] == k) & (differences["level"] == "image") & (differences["metric"] == "roc_auc")]
                x = e * 4 + p
                ax.scatter(np.full(len(part), x), part["difference_b_minus_a"], color=color, s=30, zorder=3, label=pair if e == 0 else None)
                ax.hlines(part["difference_b_minus_a"].mean(), x - 0.3, x + 0.3, color="black", linewidth=1.6, zorder=4)
            positions.append(e * 4 + 1)
            labels_.append(f"epsilon {'0' if k == 0 else str(k) + '/255'}")
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(positions, labels_)
        ax.set_title(CONDITION[condition], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if c == 0:
            ax.set_ylabel("difference of the image ROC-AUC within the seed (black bar = mean)", fontsize=9)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Paired differences in the image ROC-AUC (state B minus state A within the same seed)\n" + NOTE, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(FIGURES / "adversarial_roi_paired_differences.png", dpi=130)
    plt.close(fig)
    print("wrote 3 figures to reports/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
