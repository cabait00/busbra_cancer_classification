# %% [plot] Statistics figures of the first segmentation (step 4F-A), drawn from the STORED tables only
#
# Reads reports/tables/segmentation/unet_seed7/history.csv, dice_histogram.csv and metrics_summary.csv. It imports neither TensorFlow nor Keras, opens no image and no mask, loads no model and starts
# no training; the figures can be re-created at any time from the tables.
#
# Figure 1 (segmentation_learning_curve.png): loss (binary cross-entropy + 1 - soft Dice) and soft Dice per epoch for train and validation, the best epoch (smallest validation loss) marked. The train
#   curves are computed during training (flip augmentation, BatchNorm batch statistics) and are not comparable with inference-mode values.
# Figure 2 (segmentation_dice_distribution.png): distribution of the per-image Dice on the ORIGINAL pixel grid (prediction mapped back, no padding): left validation by class, right train against
#   validation, with the medians; the share of images per bin (bins of 0.05) and the number of images are shown.
# The validation expert masks steer the validation loss and the checkpoint choice: the validation values are NOT an independent final assessment (written into the figures).
#
# Usage (CPU only, from the project root BUSBRA):
#   .venv/bin/python plots/plot_segmentation_results.py

# %% Imports and paths
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "plots")]
PROJECT_ROOT = ROOT
RUN_DIR = PROJECT_ROOT / "reports" / "tables" / "segmentation" / "unet_seed7"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from style import SURFACE, INK, INK_2, style  # noqa: E402

SUBSET_COLOUR = {"train": "#2a78d6", "validation": "#eb6834"}
CLASS_COLOUR = {"benign": "#2a78d6", "malignant": "#eb6834"}
NOTE = ("First segmentation, one run (seed 7). The validation expert masks steer the validation loss and the checkpoint choice:\n"
        "the validation values are explorative, NOT an independent final assessment.")

# %% Read the stored tables
history = pd.read_csv(RUN_DIR / "history.csv")
histogram = pd.read_csv(RUN_DIR / "dice_histogram.csv")
summary = pd.read_csv(RUN_DIR / "metrics_summary.csv")
best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])  # first minimum, as in the training



# %% Figure 1: learning curve
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE)
for ax, (column, title, ylabel) in zip(axes, (("loss", "Loss: BCE + (1 - soft Dice)", "loss (mean per image)"), ("dice_soft", "Soft Dice on the 256 x 256 model grid", "soft Dice"))):
    style(ax)
    ax.plot(history["epoch"], history[column], color=SUBSET_COLOUR["train"], linewidth=1.8, marker="o", markersize=3.5, label="train (augmented, BatchNorm batch statistics)")
    ax.plot(history["epoch"], history[f"val_{column}"], color=SUBSET_COLOUR["validation"], linewidth=1.8, marker="s", markersize=3.5, label="validation (inference mode)")
    ax.axvline(best_epoch, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(best_epoch, ax.get_ylim()[1], f" best epoch {best_epoch}", ha="left", va="top", fontsize=8.5, color=INK_2)
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center" if column == "loss" else "lower center", bbox_to_anchor=(0.38, 0.9 if column == "loss" else 0.03))
fig.suptitle(f"Step 4F-A: learning curve ({len(history)} epochs, best epoch {best_epoch} = smallest validation loss)\n{NOTE}", fontsize=9.5, color=INK_2, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(FIGURES_DIR / "segmentation_learning_curve.png", dpi=130, facecolor=SURFACE)
plt.close(fig)

# %% Figure 2: distribution of the per-image Dice on the original pixel grid
def shares(subset: str, group: str) -> tuple[np.ndarray, np.ndarray, int]:
    part = histogram[(histogram["subset"] == subset) & (histogram["group"] == group)].sort_values("bin_left")
    n = int(part["n_images_in_group"].iloc[0])
    return part["bin_left"].to_numpy(), part["n_images"].to_numpy() / n, n


def median_of(subset: str) -> float:
    row = summary[(summary["subset"] == subset) & (summary["space"] == "original_area") & (summary["metric"] == "dice")].iloc[0]
    return float(row["median"])


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE, sharey=True)
width = 0.05
style(axes[0])
for group in ("benign", "malignant"):
    left, share, n = shares("validation", group)
    axes[0].step(np.append(left, 1.0), np.append(share, share[-1]), where="post", color=CLASS_COLOUR[group], linewidth=1.8, label=f"{group} (n = {n} images)")
axes[0].set_title("Validation, by class", color=INK, fontsize=11, loc="left")
style(axes[1])
for subset in ("train", "validation"):
    left, share, n = shares(subset, "all")
    axes[1].step(np.append(left, 1.0), np.append(share, share[-1]), where="post", color=SUBSET_COLOUR[subset], linewidth=1.8, label=f"{subset} (n = {n} images, median {median_of(subset):.3f})")
    axes[1].axvline(median_of(subset), color=SUBSET_COLOUR[subset], linewidth=1.0, linestyle=(0, (4, 3)))
axes[1].set_title("Train against validation (dashed: medians)", color=INK, fontsize=11, loc="left")
for ax in axes:
    ax.set_xlabel("per-image Dice, original pixel grid (mapped back, no padding; threshold 0.5)", color=INK_2, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
axes[0].set_ylabel("share of images per bin (width 0.05)", color=INK_2, fontsize=9)
fig.suptitle(f"Step 4F-A: distribution of the per-image Dice (images of one case are dependent)\n{NOTE}", fontsize=9.5, color=INK_2, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(FIGURES_DIR / "segmentation_dice_distribution.png", dpi=130, facecolor=SURFACE)
plt.close(fig)
print("wrote reports/figures/segmentation_learning_curve.png and reports/figures/segmentation_dice_distribution.png", flush=True)
sys.exit(0)
