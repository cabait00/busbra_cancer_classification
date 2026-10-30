# %% [12] Sensitivity-oriented threshold analysis on the validation images (explorative, from stored predictions only)
#
# Question: the 160 px full-image pipeline with extended fine-tuning (Adam 1e-4) detects 75-94 of the 125 malignant validation images at the fixed threshold 0.5, depending on the
# training seed. How do detected malignant findings (true positives) and false alarms trade off when the threshold moves, and how does the pipeline with rotation and zoom (study 4B)
# compare with the same pipeline without them (study 3F, same seeds)?
#
# Pure evaluation of the STORED predictions of ten finished runs (five seeds x two groups): nothing is trained, no model is run, no image is opened and the test subset is not accessible
# (the prediction files contain train and validation rows only; train rows are discarded first). The validation images were already used repeatedly for model and epoch development, so the
# operating points are EXPLORATIVE, not an independent performance estimate; the sensitivity targets 0.80 / 0.90 / 0.95 are illustrative, not clinical requirements and not a final threshold
# choice. The scores are not calibrated probabilities. Rules (src/busbra/thresholds.py): a unit is positive if score >= threshold; equal scores form one step of the curve; a target threshold is
# the highest threshold that reaches at least ceil(q * positives) detected positives; the case score is the maximum of the image scores; a rate with a zero denominator is undefined.
#
# Actions (the default recomputes the tables in memory and compares them with the stored ones; nothing is written):
#   .venv/bin/python analysis/12_threshold_analysis.py
#   .venv/bin/python analysis/12_threshold_analysis.py --write    write the tables that do not exist yet; stops if one exists
# The sensitivity-versus-false-positives figure is drawn separately from the stored tables.

# %% Imports
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from busbra import config, data, operating as op, results, thresholds as th  # noqa: E402

TABLES = config.TABLES_DIR
GROUPS = [("3F", "multiseed", "mobilenetv2_higher_lr_160"), ("4B", "augmentation160", "mobilenetv2_higher_lr_160_aug")]  # group, study table prefix, pipeline
SEEDS = config.SEEDS
EXPECTED = {"images": 385, "cases": 213, "malignant_images": 125, "malignant_cases": 69}
COMPARISONS = [("reference_0.5", "0.5", "same numeric threshold 0.5"), *[(f"target_{t}", f"target_{t}", f"sensitivity target {t}: pipeline-specific thresholds") for t in th.TARGET_SENSITIVITIES]]
OUTPUTS = {"curves": "threshold_analysis_curves.csv", "operating_points": "threshold_analysis_operating_points.csv", "case_level": "threshold_analysis_case_level.csv",
           "paired_differences": "threshold_analysis_paired_differences.csv", "paired_summary": "threshold_analysis_paired_summary.csv", "group_summary": "threshold_analysis_group_summary.csv"}


# %% Inputs: validation rows of the ten runs (train rows are dropped first)
def validation_rows(group: str, pipeline: str, seed: int, manifest: pd.DataFrame) -> pd.DataFrame:
    variant = config.VARIANTS[pipeline]
    raw = pd.read_csv(config.run_paths(variant, seed)["predictions"], dtype=str, keep_default_na=False, na_filter=False)
    if not set(raw["subset"]) <= {"train", "validation"}:
        raise ValueError(f"{group} seed {seed}: unexpected subset values; only train and validation rows are allowed")
    frame = raw.loc[raw["subset"] == "validation"].reset_index(drop=True)
    frame = frame.astype({"label": int, "malignant_score": float, "predicted_label": int, "threshold": float})
    validation = data.subset_frame(manifest, "validation")
    problems = th.validation_frame_problems(frame, EXPECTED)
    if frame["ID"].tolist() != validation["ID"].tolist() or frame["Case"].tolist() != validation["Case"].tolist():
        problems.append("IDs or cases differ from the split")
    if not np.array_equal((frame["malignant_score"].to_numpy() >= 0.5).astype(int), frame["predicted_label"].to_numpy()):
        problems.append("stored 0.5 decisions do not follow score >= 0.5")
    if problems:
        raise ValueError(f"{group} seed {seed}: " + "; ".join(problems))
    return frame[["ID", "Case", "label", "malignant_score"]]


def run_ids_from_table(table: pd.DataFrame) -> dict:
    """Historical run identifiers by (pipeline, seed). Only the stored tables of the earlier studies carry them; a table from busbra.aggregation has none, and then the result tables have no run_id column."""
    if "run_id" not in table.columns:
        return {}
    return {(r.pipeline, int(r.seed)): r.run_id for r in table[table["subset"] == "validation"].itertuples()}


def historical_run_ids(study: str) -> dict:
    path = TABLES / f"{study}_run_metrics.csv"
    return run_ids_from_table(pd.read_csv(path) if path.is_file() else pd.DataFrame())


def sk_counts(labels, scores, threshold):
    tn, fp, fn, tp = confusion_matrix(labels, (scores >= threshold).astype(int), labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


# %% Curves, operating points, case level
def analyse(run_ids: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curves and operating points of the ten runs. run_ids: historical run identifiers by group (default: those of the stored study tables, if any)."""
    run_ids = run_ids if run_ids is not None else {group: historical_run_ids(study) for group, study, _ in GROUPS}
    manifest = data.read_manifest()
    data.check_split(manifest)
    curves, points = [], []
    for group, study, pipeline in GROUPS:
        for seed in SEEDS:
            frame = validation_rows(group, pipeline, seed, manifest)
            ident = {"pipeline_group": group, "pipeline": pipeline, "seed": seed, **({"run_id": run_ids[group][(pipeline, seed)]} if run_ids.get(group) else {})}
            curve, rows = op.run_analysis(frame, ident, th.TARGET_SENSITIVITIES, grid=th.GRID_THRESHOLDS, annotated=True, case_precision=True)
            labels, scores = frame["label"].to_numpy(), frame["malignant_score"].to_numpy()
            if any(sk_counts(labels, scores, t.threshold) != (t.tn, t.fp, t.fn, t.tp) for t in curve.itertuples()):
                raise ValueError(f"{group} seed {seed}: a curve point differs from the independent confusion matrix")
            curves.append(curve)
            points += rows
    return pd.concat(curves, ignore_index=True), pd.DataFrame(points)


# %% Paired and group tables
def paired_tables(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4B minus 3F inside the same seed (never across seeds), image and case level, plus mean / sample SD / sign counts of the five differences."""
    specs = [{"ids": {"comparison": comparison, "comparison_kind": kind}, "a": "3F", "b": "4B", "label": label} for comparison, label, kind in COMPARISONS]
    return op.paired_point_tables(points, specs, SEEDS, "pipeline_group", ("value_3F", "value_4B"), "difference_4B_minus_3F", ["comparison", "level", "metric"], kinds=["reference", "target"])


def group_summary(points: pd.DataFrame) -> pd.DataFrame:
    """Mean / sample SD / min / max of single-run values over the five training runs of one group (no sums, no pooling)."""
    rows = []
    selected = points[points["point_label"].isin(["0.5"] + [f"target_{t}" for t in th.TARGET_SENSITIVITIES]) & points["point_kind"].isin(["reference", "target"])]
    for (group, label), part in selected.groupby(["pipeline_group", "point_label"], sort=False):
        for level, prefix in (("image", ""), ("case", "case_")):
            for metric in op.PAIRED_POINT_METRICS:
                if metric == "threshold" and level == "case":
                    continue
                values = pd.to_numeric(part[metric if metric == "threshold" else f"{prefix}{metric}"], errors="coerce").dropna().to_numpy(dtype=float)
                rows.append({"pipeline_group": group, "point_label": label, "level": level, "metric": metric, "n_runs": int(len(part)), "n_defined": int(values.size),
                             "mean": float(values.mean()) if values.size else None, "sd_ddof1": float(values.std(ddof=1)) if values.size > 1 else None,
                             "min": float(values.min()) if values.size else None, "max": float(values.max()) if values.size else None})
    return pd.DataFrame(rows)


def compute(run_ids: dict | None = None) -> dict[str, pd.DataFrame]:
    curves, points = analyse(run_ids)
    differences, paired_summary = paired_tables(points)
    case_level = points[points["point_kind"].isin(["reference", "target"])][[c for c in ("pipeline_group", "pipeline", "seed", "run_id") if c in points.columns] + ["point_label", "threshold"] + [c for c in points.columns if c.startswith("case_")]].reset_index(drop=True)
    return {"curves": curves, "operating_points": points, "case_level": case_level, "paired_differences": differences, "paired_summary": paired_summary, "group_summary": group_summary(points)}


same = results.same_as_stored


# %% Run
def main(argv: list[str]) -> None:
    tables = compute()
    if "--write" in argv:
        clash = [f for f in OUTPUTS.values() if (TABLES / f).exists()]
        if clash:
            raise FileExistsError(f"already present: {clash}; nothing was changed")
        for key, frame in tables.items():
            frame.to_csv(TABLES / OUTPUTS[key], index=False, lineterminator="\n", na_rep="")
        print(f"wrote {len(tables)} tables")
        return
    for key, frame in tables.items():
        path = TABLES / OUTPUTS[key]
        print(f"{OUTPUTS[key]:46s} {len(frame):6d} rows  " + ("identical to the stored table" if path.is_file() and same(frame, path) else "DIFFERENT or not stored"))
    focus = tables["paired_summary"].query("level == 'image' and metric == 'fp'")
    print("paired 4B minus 3F, image false positives (mean, sample SD, +/-/= over five seeds):")
    for r in focus.itertuples():
        print(f"  {r.comparison}: mean {r.mean_difference:+.1f}, SD {r.sd_ddof1:.1f}, {r.n_positive}/{r.n_negative}/{r.n_zero}")


if __name__ == "__main__":
    main(sys.argv[1:] if "__file__" in globals() else [])
