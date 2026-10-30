# %% [plot] Statistics figure of the augmentation comparison (step 4B), drawn from the STORED tables only
#
# Reads reports/tables/augmentation160_summary_validation.csv, augmentation160_paired_differences.csv and augmentation160_paired_summary.csv. It imports neither TensorFlow nor Keras, opens no
# image, loads no model and starts no training; the figure can be re-created at any time from the tables.
#
# Left panel: validation ROC-AUC of the four groups (frozen and extended fine-tuning 1e-4, each flip only (study 3F, read) and with rotation + zoom (step 4B)), every single run
#   (five seeds) with its own colour AND marker shape, the mean (bar) and the sample standard deviation (whiskers, ddof=1) printed in the plot.
# Right panel: the paired validation ROC-AUC difference (augmented minus 3F) within the same seed for both pipelines, with the zero line and the descriptive mean.
# No significance marks: five paired seeds show a tendency, not a proof.
#
# Usage (CPU only, from the project root BUSBRA):
#   .venv/bin/python plots/plot_augmentation160_results.py

# %% Imports and paths
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "plots")]
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from style import SURFACE  # noqa: E402
from study_figures import draw_auc_and_pairs  # noqa: E402

STUDY = "augmentation160"
GROUPS = [("mobilenetv2_frozen_160", "Frozen\nflip only (3F)"), ("mobilenetv2_frozen_160_aug", "Frozen\nflip + rotation + zoom"),
          ("mobilenetv2_higher_lr_160", "Extended 1e-4\nflip only (3F)"), ("mobilenetv2_higher_lr_160_aug", "Extended 1e-4\nflip + rotation + zoom")]
PAIRS = [("mobilenetv2_frozen_160_aug minus mobilenetv2_frozen_160", "Frozen"), ("mobilenetv2_higher_lr_160_aug minus mobilenetv2_higher_lr_160", "Extended fine-tuning 1e-4")]

# %% Read the stored tables and draw (the shared figure stops if they do not describe the complete design)
summary, paired, paired_summary = (pd.read_csv(TABLES_DIR / f"{STUDY}_{name}.csv") for name in ("summary_validation", "paired_differences", "paired_summary"))
fig = draw_auc_and_pairs(summary, paired, paired_summary, GROUPS, PAIRS, title_left="ROC-AUC with and without rotation + zoom (five seeds, same validation images)", ylabel_pairs="validation ROC-AUC: augmented minus 3F (same seed)",
                      suptitle="Step 4B: course augmentation (rotation at most 10 degrees, zoom at most 10 %, training only) vs 3F at 160 x 160 px\n"
                               "Training variation on the same fixed validation images; no significance statement, no model choice")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES_DIR / "augmentation160_validation_auc.png", dpi=130, facecolor=SURFACE)
plt.close(fig)
print(f"written reports/figures/augmentation160_validation_auc.png from stored tables; TensorFlow imported: {'tensorflow' in sys.modules}")
