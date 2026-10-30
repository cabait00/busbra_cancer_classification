# %% [plot] Statistics figures of the multi-seed comparison (step 3F), drawn from the STORED tables only
#
# Reads reports/tables/multiseed_run_metrics.csv, multiseed_summary_validation.csv, multiseed_paired_differences.csv and
# multiseed_paired_summary.csv. It imports neither TensorFlow nor Keras, opens no image, loads no model and starts no
# training; the figures can therefore be re-created at any time from the tables.
#
# Figure 1: validation ROC-AUC of the five neural pipelines. Every single run (five seeds) is visible with its own colour AND
#   marker shape (identity is never colour alone); the mean (bar) and the sample standard deviation (whiskers, ddof=1) are
#   printed in the plot. Dummy and the logistic regression are fixed references (no seed spread); the earlier single runs are
#   hollow diamonds for orientation only.
# Figure 2: the paired validation ROC-AUC difference extended fine-tuning 1e-4 minus extended fine-tuning 1e-5 for each seed,
#   with the zero line and the descriptive mean. No significance marks: five seeds show a tendency, not a proof.
#
# Usage (CPU only, from the project root BUSBRA):
#   .venv/bin/python plots/plot_multiseed_results.py

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

PIPELINES = ["small_cnn_160", "mobilenetv2_frozen_160", "mobilenetv2_finetuned_160", "mobilenetv2_extended_160", "mobilenetv2_higher_lr_160"]
LABELS = {"small_cnn_160": "Small CNN", "mobilenetv2_frozen_160": "MobileNetV2\nfrozen", "mobilenetv2_finetuned_160": "Limited\nfine-tuning 1e-5",
          "mobilenetv2_extended_160": "Extended\nfine-tuning 1e-5", "mobilenetv2_higher_lr_160": "Extended\nfine-tuning 1e-4"}

# %% Read the stored tables and check that they describe the complete design
runs = pd.read_csv(TABLES_DIR / "multiseed_run_metrics.csv")
summary = pd.read_csv(TABLES_DIR / "multiseed_summary_validation.csv")
paired = pd.read_csv(TABLES_DIR / "multiseed_paired_differences.csv")
paired_summary = pd.read_csv(TABLES_DIR / "multiseed_paired_summary.csv")
validation = runs[runs["subset"] == "validation"]
seeds = sorted(validation["seed"].unique())
if seeds != sorted(SEED_STYLE) or any(len(validation[validation["pipeline"] == p]) != len(seeds) for p in PIPELINES):
    sys.exit("The stored run table does not hold five seeds for each of the five pipelines; no figure drawn.")



# %% Figure 1: validation ROC-AUC per pipeline, all five seeds
fig, ax = plt.subplots(figsize=(12.5, 6.6), facecolor=SURFACE)
style(ax)
auc = summary[summary["metric"] == "roc_auc"].set_index("pipeline")
for k, name in enumerate(PIPELINES):
    rows = validation[validation["pipeline"] == name].sort_values("seed")
    offsets = np.linspace(-0.27, 0.27, len(rows))
    for offset, row in zip(offsets, rows.itertuples()):
        colour, marker = SEED_STYLE[int(row.seed)]
        ax.scatter(k + offset, row.roc_auc, s=70, color=colour, marker=marker, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    mean, sd = auc.loc[name, "mean"], auc.loc[name, "sd_ddof1"]
    ax.hlines(mean, k - 0.36, k + 0.36, color=INK, linewidth=2.2, zorder=3)
    ax.vlines(k + 0.40, mean - sd, mean + sd, color=INK_2, linewidth=1.6, zorder=3)
    ax.hlines([mean - sd, mean + sd], k + 0.37, k + 0.43, color=INK_2, linewidth=1.6, zorder=3)
    top = max(rows["roc_auc"].max(), mean + sd)
    ax.text(k, top + 0.012, f"mean {mean:.3f}\nSD {sd:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    hist = auc.loc[name, "historical_single_seed_value"]
    ax.scatter(k - 0.40, hist, s=48, facecolor="none", edgecolor=INK_2, marker="d", linewidth=1.3, zorder=3)
for label, model, linestyle in (("Dummy (fixed reference)", "dummy_prior", ":"), ("Logistic regression L2 (fixed reference)", "logreg_l2", "--")):
    value = float(summary[(summary["pipeline"] == model) & (summary["metric"] == "roc_auc")]["mean"].iloc[0])
    ax.axhline(value, color=INK_2, linestyle=linestyle, linewidth=1.1, zorder=2)
    ax.text(len(PIPELINES) - 0.45, value + 0.004, f"{label}: {value:.3f}", ha="right", va="bottom", fontsize=8.5, color=INK_2)
ax.set_xticks(range(len(PIPELINES)), [LABELS[p] for p in PIPELINES])
ax.set_ylabel("validation ROC-AUC (385 images, 213 cases)", color=INK_2)
low = min(0.45, validation["roc_auc"].min() - 0.03)
ax.set_ylim(low, max(validation["roc_auc"].max(), auc["mean"].max()) + 0.085)
ax.set_xlim(-0.6, len(PIPELINES) - 0.4)
handles = [plt.Line2D([], [], marker=SEED_STYLE[s][1], color=SEED_STYLE[s][0], linestyle="none", markersize=8, label=f"seed {s}") for s in seeds]
handles += [plt.Line2D([], [], color=INK, linewidth=2.2, label="mean of five runs"),
            plt.Line2D([], [], color=INK_2, linewidth=1.6, marker="_", markersize=9, label="± sample SD (ddof=1)"),
            plt.Line2D([], [], marker="d", markerfacecolor="none", markeredgecolor=INK_2, linestyle="none", markersize=8, label="earlier single run (reference)")]
ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
fig.suptitle("Validation ROC-AUC over five training seeds on the same fixed validation images (training variation only; no new splits, no significance statement)",
             fontsize=9.5, color=INK_2, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIGURES_DIR / "multiseed_validation_auc.png", dpi=130, facecolor=SURFACE)
plt.close(fig)

# %% Figure 2: paired difference extended 1e-4 minus extended 1e-5, per seed
pair_name = "mobilenetv2_higher_lr_160 minus mobilenetv2_extended_160"
diff = paired[(paired["pair"] == pair_name) & (paired["metric"] == "roc_auc")].sort_values("seed")
stats = paired_summary[(paired_summary["pair"] == pair_name) & (paired_summary["metric"] == "roc_auc")].iloc[0]
fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=SURFACE)
style(ax)
xs = np.arange(len(diff))
for x, row in zip(xs, diff.itertuples()):
    colour, marker = SEED_STYLE[int(row.seed)]
    ax.vlines(x, 0, row.difference_b_minus_a, color=colour, linewidth=2.0, zorder=3)
    ax.scatter(x, row.difference_b_minus_a, s=95, color=colour, marker=marker, edgecolor=SURFACE, linewidth=1.2, zorder=4, label=f"seed {int(row.seed)}")
    ax.text(x, row.difference_b_minus_a + (0.004 if row.difference_b_minus_a >= 0 else -0.004), f"{row.difference_b_minus_a:+.4f}", ha="center",
            va="bottom" if row.difference_b_minus_a >= 0 else "top", fontsize=9, color=INK)
ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
ax.axhline(stats["mean_difference"], color=INK_2, linestyle="--", linewidth=1.4, zorder=2)
ax.text(len(diff) - 0.55, stats["mean_difference"], f"mean {stats['mean_difference']:+.4f}\nSD of the differences {stats['sd_ddof1']:.4f}", ha="left", va="center", fontsize=9, color=INK,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.0})
ax.set_xticks(xs, [f"seed {int(s)}" for s in diff["seed"]])
ax.set_xlim(-0.6, len(diff) + 0.9)
span = max(abs(diff["difference_b_minus_a"]).max(), abs(stats["mean_difference"])) * 1.45
ax.set_ylim(-span, span)
ax.set_ylabel("validation ROC-AUC: extended 1e-4 minus extended 1e-5", color=INK_2)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
fig.suptitle(f"Paired differences within the same seed (positive in {int(stats['n_positive'])}, negative in {int(stats['n_negative'])}, equal in {int(stats['n_tie'])} of {int(stats['n_seeds'])} seeds); "
             "descriptive, no significance statement", fontsize=9.5, color=INK_2, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIGURES_DIR / "multiseed_lr_paired_differences.png", dpi=130, facecolor=SURFACE)
plt.close(fig)
print("written reports/figures/multiseed_validation_auc.png and reports/figures/multiseed_lr_paired_differences.png from stored tables; "
      f"TensorFlow imported: {'tensorflow' in sys.modules}")
