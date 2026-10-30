# %% [plot] Readable re-plot of the limited fine-tuning learning curves (step 3C) from STORED values only
#
# Purpose
# -------
# The original 3C figure placed the labels of the start reference line and of the best epoch where they overlap curves
# and each other. This small CPU script draws the same stored values again, with the explanations in a legend below each
# panel instead of text inside the curve area. It does NOT train, load a model, import TensorFlow/Keras, open an image or
# change any number: the inputs are the stored 3C history table and the start values of the frozen checkpoint (config.FROZEN_START_VALIDATION).
# The original figure and tables stay untouched; the new file is an additional figure:
#   reports/figures/mobilenetv2_finetuned_learning_curves_readable.png
#
# Interpretation
# --------------
# Train curves are computed with flip and dropout active, validation curves in inference mode, so the two are not
# directly comparable. The dotted reference line is the exact evaluation of the loaded frozen start model on the
# validation set before any update (loss, ROC-AUC); the Keras AUC curve is an approximation and is only comparable in
# trend. The best epoch is the smallest validation loss among the new fine-tuning epochs.
#
# Usage (CPU only, from the project root BUSBRA):
#   .venv/bin/python plots/plot_fine_tuning_history.py

# %% Imports and paths
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "plots")]
PROJECT_ROOT = ROOT

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from busbra import config  # noqa: E402
from style import GRID, INK, INK_2, SURFACE, style  # noqa: E402

VARIANT = config.VARIANTS["mobilenetv2_finetuned"]
HISTORY_PATH = config.run_paths(VARIANT, config.HISTORICAL_SEED)["history"]
FIGURES_DIR = config.FIGURES_DIR
OUTPUT_PATH = FIGURES_DIR / "mobilenetv2_finetuned_learning_curves_readable.png"


# %% Read the stored values (and refuse to plot anything else)
def curve_inputs() -> dict:
    """History table, best epoch (first smallest validation loss), epoch count, stop reason and the exact start values; all from the stored history and config.FROZEN_START_VALIDATION."""
    history = pd.read_csv(HISTORY_PATH)
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    n_epochs = len(history)
    stop_reason = "early_stopping" if n_epochs < VARIANT.max_epochs else "epoch_limit_reached"
    return {"history": history, "best_epoch": best_epoch, "n_epochs": n_epochs, "stop_reason": stop_reason, "start": config.FROZEN_START_VALIDATION}


# %% Figure: explanations in the legend below each panel, nothing printed inside the curve area
COLOR = "#0a6e4c"  # same colour as the limited fine-tuning candidate in the comparison figures


def main() -> None:
    data = curve_inputs()
    history, best_epoch, n_epochs, stop_reason, start = (data[k] for k in ("history", "best_epoch", "n_epochs", "stop_reason", "start"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), facecolor=SURFACE)
    for ax, (train_col, val_col, title, start_value) in zip(axes, [
            ("loss", "val_loss", "Loss (binary cross-entropy)", start["loss"]),
            ("auc", "val_auc", "ROC-AUC (Keras, approximate)", start["roc_auc"])]):
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)  # the curves also need the vertical grid
        ax.plot(history["epoch"], history[train_col], color=COLOR, linestyle="--", linewidth=2, label="train (flip + dropout active)", zorder=3)
        ax.plot(history["epoch"], history[val_col], color=COLOR, linewidth=2, label="validation (inference mode)", zorder=3)
        ax.axhline(start_value, color=INK_2, linewidth=1, linestyle=":", zorder=2,
                   label=f"start: validation, frozen epoch 20 without update ({start_value:.4f}, exact evaluation)")
        ax.axvline(best_epoch, color=INK_2, linewidth=1, linestyle="-.", zorder=2, label=f"best epoch {best_epoch} (smallest validation loss)")
        ax.plot([best_epoch], [history.loc[history["epoch"] == best_epoch, val_col].iloc[0]], marker="o", markersize=7, color=COLOR,
                markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=4)
        ax.set_xlabel("fine-tuning epoch (counted from 1)", color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    fig.suptitle(f"MobileNetV2 limited fine-tuning ({n_epochs} epochs, stop: {stop_reason}); train and validation curves are computed under "
                 "different conditions", fontsize=9.5, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f"written {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}; input: {HISTORY_PATH.name} (read only) and config.FROZEN_START_VALIDATION")


if __name__ == "__main__":
    main()
