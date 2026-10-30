# %% [07] Validation operating points of the 27 combinations (CPU, stored validation scores only)
#
# For every combination (seven groups, five seeds for the five trained groups) the thresholds are chosen on the VALIDATION subset only:
# the fixed reference 0.5 and the highest threshold that reaches at least 90 % / 95 % sensitivity (image level and case level; the case score
# is the maximum of the image scores). The thresholds are stored as decimal text plus IEEE-754 hex (reports/tables/final_operating_points.csv)
# and are the frozen thresholds of the test evaluation (09). They are never re-chosen with a test label.
#
# Actions (the default recomputes everything in memory from the stored validation scores and compares it with the stored tables; nothing is written):
#   .venv/bin/python analysis/07_operating_points.py
#   .venv/bin/python analysis/07_operating_points.py --write    write the five tables; stops if one of them already exists

# %% Imports
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from busbra import config, data, final_evaluation as fe  # noqa: E402

CORE_COLUMNS = ["combo_id", "group_id", "seed", "level", "point", "point_kind", "selection_rule", "target_sensitivity", "needed_tp", "threshold", "threshold_hex",
                "n_units_with_score_equal_to_threshold", "n_units", "n_positive", "n_negative", "n_predicted_positive", "tn", "fp", "fn", "tp", "sensitivity", "specificity",
                "precision", "npv", "f1", "accuracy", "predicted_positive_share", "target_met", "near_all_positive"]
CASE_VIEW_COLUMNS = ["combo_id", "group_id", "seed", "point", "threshold_origin", "target_sensitivity", "threshold", "threshold_hex", "n_units_with_score_equal_to_threshold", "n_units",
                     "n_positive", "n_negative", "n_predicted_positive", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "precision", "npv", "f1", "accuracy", "predicted_positive_share"]
METRIC_COLUMNS = ["combo_id", "group_id", "seed", "level", "n_units", "n_positive", "n_negative", "roc_auc", "average_precision", "n_distinct_scores", "largest_tie_group"]
FILES = {"final_operating_points": "final_operating_points.csv", "final_validation_metrics": "final_validation_metrics.csv",
         "final_case_metrics_at_image_thresholds": "final_case_metrics_at_image_thresholds.csv", "final_validation_metrics_summary": "final_validation_metrics_summary.csv",
         "final_operating_point_summary": "final_operating_point_summary.csv"}


# %% Computation
def compute() -> dict[str, pd.DataFrame]:
    manifest = data.read_manifest()
    data.check_split(manifest)
    validation = data.subset_frame(manifest, "validation")
    core, metrics, case_view, problems = [], [], [], []
    for entry in fe.combinations():
        c, m, v, p = fe.compute_combination(entry, fe.read_validation_scores(entry, validation))
        core, metrics, case_view, problems = core + c, metrics + m, case_view + v, problems + p
    if problems:
        raise ValueError("threshold rule check failed: " + "; ".join(problems[:5]))
    core_frame, metrics_frame, case_frame = pd.DataFrame(core)[CORE_COLUMNS], pd.DataFrame(metrics)[METRIC_COLUMNS], pd.DataFrame(case_view)[CASE_VIEW_COLUMNS]
    if not (len(core_frame) == 162 and len(metrics_frame) == 54 and len(case_frame) == 81 and core_frame["combo_id"].nunique() == fe.N_COMBINATIONS):
        raise ValueError("unexpected number of result rows")
    return {"final_operating_points": core_frame, "final_validation_metrics": metrics_frame, "final_case_metrics_at_image_thresholds": case_frame,
            "final_validation_metrics_summary": fe.summarize_metrics(metrics_frame), "final_operating_point_summary": fe.summarize_operating_points(core_frame)}


def to_csv(frame: pd.DataFrame, path: Path) -> None:
    if "seed" in frame.columns:
        frame = frame.assign(seed=frame["seed"].astype("Int64"))
    if "needed_tp" in frame.columns:
        frame = frame.assign(needed_tp=frame["needed_tp"].astype("Int64"))
    frame.to_csv(path, index=False, lineterminator="\n", na_rep="")


def same(new: pd.DataFrame, path: Path) -> bool:
    """Value-wise equality with a stored table (exact for text and integers, 1e-12 for floats; undefined values equal undefined values)."""
    stored = pd.read_csv(path, dtype={"threshold": str, "threshold_hex": str}, float_precision="round_trip")
    if list(stored.columns) != list(new.columns) or len(stored) != len(new):
        return False
    new = pd.read_csv(pd.io.common.StringIO(new.assign(**({"seed": new["seed"].astype("Int64")} if "seed" in new else {})).to_csv(index=False, na_rep="")),
                      dtype={"threshold": str, "threshold_hex": str}, float_precision="round_trip")
    for column in stored.columns:
        a, b = stored[column], new[column]
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            if not np.allclose(a.astype(float), b.astype(float), atol=1e-12, rtol=0, equal_nan=True):
                return False
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            return False
    return True


# %% Run
def main(argv: list[str]) -> None:
    tables = compute()
    if "--write" in argv:
        clash = [config.TABLES_DIR / f for f in FILES.values() if (config.TABLES_DIR / f).exists()]
        if clash:
            raise FileExistsError(f"already present: {[p.name for p in clash]}; nothing was changed")
        for name, file in FILES.items():
            to_csv(tables[name], config.TABLES_DIR / file)
        print("wrote", ", ".join(FILES.values()))
        return
    for name, file in FILES.items():
        path = config.TABLES_DIR / file
        print(f"{file:52s} {len(tables[name]):4d} rows  " + ("identical to the stored table" if path.is_file() and same(tables[name], path) else "DIFFERENT or not stored"))


if __name__ == "__main__":
    main(sys.argv[1:] if "__file__" in globals() else [])
