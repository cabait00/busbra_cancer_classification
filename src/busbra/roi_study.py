"""Validation studies of the ROI branches from stored scores and diagnoses: expert-mask ROI (4E) and automatic ROI (4F-B). Pure pandas / numpy; nothing is trained or inferred.

Groups: the full-image pipeline (4B) and the expert-mask ROI pipeline (4E), each as frozen head and as extended fine-tuning (Adam 1e-4), and the automatic ROI, which feeds automatic crops to the
SAME 4E checkpoints (no classifier is trained on automatic crops). The operating points are the reference threshold 0.5 and the sensitivity targets 0.90 / 0.95 (at least 113 / 119 of the
125 malignant validation images); all curve and point rules are those of busbra.operating. The case score is the maximum of the image scores. Only validation scores are read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from busbra import aggregation, auto_roi as ar, config, data, operating as op
from busbra import evaluation as ev
from busbra import thresholds as th

SEEDS = list(config.SEEDS)
TARGETS = ("0.90", "0.95")
POINT_LABELS = ["0.5"] + [f"target_{t}" for t in TARGETS]
RUN_METRICS = ["roc_auc", "average_precision", "sensitivity", "specificity", "precision", "f1", "accuracy", "tn", "fp", "fn", "tp"]

# ---- expert-mask ROI study: group -> (variant, information); stage -> (full-image group, ROI group, description) in the order of the stored tables
EXPERT_GROUPS = {"4B_frozen": ("mobilenetv2_frozen_160_aug", "full image"), "ROI_frozen": ("mobilenetv2_frozen_160_roi", "expert ROI"),
                 "4B_finetune": ("mobilenetv2_higher_lr_160_aug", "full image"), "ROI_finetune": ("mobilenetv2_higher_lr_160_roi", "expert ROI")}
EXPERT_STAGES = {"finetune": ("4B_finetune", "ROI_finetune", "extended fine-tuning 1e-4"), "frozen": ("4B_frozen", "ROI_frozen", "frozen head")}

# ---- automatic ROI study
AUTO_STAGES = {"frozen": {"full": "4B_frozen", "expert": "4E_frozen", "auto": "4FB_frozen"}, "finetune": {"full": "4B_finetune", "expert": "4E_finetune", "auto": "4FB_finetune"}}
_STAGE_VARIANTS = {"frozen": {"full": "mobilenetv2_frozen_160_aug", "roi": "mobilenetv2_frozen_160_roi"}, "finetune": {"full": "mobilenetv2_higher_lr_160_aug", "roi": "mobilenetv2_higher_lr_160_roi"}}
_INFORMATION = {"full": "full image (4B, stored predictions)", "expert": "expert-mask ROI (4E, stored predictions)", "auto": "automatic ROI (4F-B, new inference with the same 4E checkpoint)"}
AUTO_GROUPS = {names[kind]: {"stage": stage, "kind": kind, "pipeline": _STAGE_VARIANTS[stage]["full" if kind == "full" else "roi"], "information": _INFORMATION[kind]}
               for stage, names in AUTO_STAGES.items() for kind in ("full", "expert", "auto")}
AUTO_PAIRS = [("auto_minus_expert", "expert"), ("auto_minus_full", "full")]  # automatic ROI minus the reference group, within the same seed and stage
HIGHLIGHTED_EXAMPLES = ["bus_1014-r", "bus_1013-l", "bus_1014-l"]  # the examples named by the segmentation inspection; the rest comes from its fixed error-analysis selection

# ---- file names (reports/tables)
EXPERT_OPERATING_FILES = {"threshold_curves": "expert_roi_threshold_curves.csv", "operating_points": "expert_roi_operating_points.csv",
                          "operating_point_differences": "expert_roi_operating_point_differences.csv", "operating_point_summary": "expert_roi_operating_point_summary.csv"}
AUTO_FILES = {"run_metrics": "auto_roi_run_metrics.csv", "summary_validation": "auto_roi_summary_validation.csv", "paired_differences": "auto_roi_paired_differences.csv",
              "paired_summary": "auto_roi_paired_summary.csv", "threshold_curves": "auto_roi_threshold_curves.csv", "operating_points": "auto_roi_operating_points.csv",
              "operating_point_differences": "auto_roi_operating_point_differences.csv", "operating_point_summary": "auto_roi_operating_point_summary.csv", "crop_flags": "auto_roi_crop_flags.csv",
              "crop_categories": "auto_roi_crop_categories.csv", "coverage_summary": "auto_roi_coverage_summary.csv", "box_definitions": "auto_roi_box_definitions.csv",
              "component_selection": "auto_roi_component_selection.csv", "notable_examples": "auto_roi_notable_examples.csv", "concurrence": "auto_roi_concurrence.csv"}
DIAGNOSTICS_FILE = "auto_roi_diagnostics_validation.csv"


def auto_score_path(seed: int, stage: str) -> Path:
    return config.TABLES_DIR / "auto_roi" / f"seed_{seed}" / stage / "scores_auto.csv"


def diagnostics_path() -> Path:
    return config.TABLES_DIR / DIAGNOSTICS_FILE


def segmentation_selection_path() -> Path:
    """The fixed example selection of the segmentation study (a given input of the named examples)."""
    return config.TABLES_DIR / "segmentation" / "unet_seed7" / "preview_selection.csv"


# ---- primary inputs: validation scores -------------------------------------------------------------------------------------------------------------------
def read_score_file(path: Path) -> pd.DataFrame:
    """ID, Case, label, malignant_score of the validation rows of one stored score file (run predictions carry a subset column and also hold the training rows; automatic scores do not)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"stored scores not found: {path}")
    frame = pd.read_csv(path, dtype={"ID": str, "Case": str}, float_precision="round_trip")
    missing = [c for c in ("ID", "Case", "label", "malignant_score") if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if "subset" in frame.columns:
        frame = frame[frame["subset"] == "validation"]
    return frame[["ID", "Case", "label", "malignant_score"]].reset_index(drop=True)


def check_score_sets(sets: dict, validation: pd.DataFrame, seeds=SEEDS) -> None:
    """Every (group, seed) set must describe exactly the validation images in the frozen split order (IDs and cases as unchanged strings, labels), with finite scores in [0, 1].
    Missing or unexpected seeds and groups, duplicated IDs and wrong subsets are reported together."""
    expected = pd.DataFrame({"ID": validation["ID"].astype(str), "Case": validation["Case"].astype(str), "label": data.exact_labels(validation["label"], "validation subset of the split")})
    problems = []
    groups = list(dict.fromkeys(group for group, _ in sets))
    for group in groups:
        found = sorted(seed for g, seed in sets if g == group)
        if found != sorted(seeds):
            problems.append(f"{group}: seeds {found} instead of {sorted(seeds)}")
    for (group, seed), frame in sets.items():
        frame = frame.reset_index(drop=True)
        if frame["ID"].duplicated().any():
            problems.append(f"{group} seed {seed}: duplicated IDs")
        try:
            labels = pd.Series(data.exact_labels(frame["label"], f"{group} seed {seed}"))
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if len(frame) != len(expected) or not frame[["ID", "Case"]].equals(expected[["ID", "Case"]]) or not labels.equals(expected["label"]):
            problems.append(f"{group} seed {seed}: IDs, cases or labels differ from the validation subset of the split (count {len(frame)} instead of {len(expected)})")
        problems += [f"{group} seed {seed}: {p}" for p in th.score_problems(frame["malignant_score"].to_numpy(dtype=float))]
    if problems:
        raise ValueError("score sets are inconsistent: " + "; ".join(problems[:8]) + (" ..." if len(problems) > 8 else ""))


def expert_score_sets(seeds=SEEDS) -> dict:
    """The stored validation scores of the four expert-ROI study groups, seeds outer and groups inner (the order of the stored tables)."""
    return {(group, seed): read_score_file(config.run_paths(config.VARIANTS[variant], seed)["predictions"]) for seed in seeds for group, (variant, _) in EXPERT_GROUPS.items()}


def auto_score_sets(seeds=SEEDS) -> dict:
    """The stored validation scores of the six groups of the automatic-ROI study (stored 4B and 4E runs, new automatic scores), seeds outer and groups inner."""
    sets = {}
    for seed in seeds:
        for stage, names in AUTO_STAGES.items():
            sets[(names["full"], seed)] = read_score_file(config.run_paths(config.VARIANTS[_STAGE_VARIANTS[stage]["full"]], seed)["predictions"])
            sets[(names["expert"], seed)] = read_score_file(config.run_paths(config.VARIANTS[_STAGE_VARIANTS[stage]["roi"]], seed)["predictions"])
            sets[(names["auto"], seed)] = read_score_file(auto_score_path(seed, stage))
    return sets


# ---- expert-mask ROI: threshold curves, operating points, paired differences ---------------------------------------------------------------------------------
def expert_operating_tables(sets: dict) -> dict:
    """Threshold curves, operating points, their paired differences (expert ROI minus full image within the same seed and stage, image and case level) and the summary of the expert-ROI study, keyed like EXPERT_OPERATING_FILES."""
    seeds = sorted({seed for _, seed in sets})  # a partial view uses its own seeds throughout
    curves, points = [], []
    for (group, seed), frame in sets.items():
        variant, information = EXPERT_GROUPS[group]
        curve, rows = op.run_analysis(frame, {"group": group, "pipeline": variant, "information": information, "seed": seed}, TARGETS)
        curves.append(curve)
        points += rows
    points = pd.DataFrame(points)
    specs = [{"ids": {"stage": stage, "point": label}, "a": full, "b": roi_group, "label": label} for stage, (full, roi_group, _) in EXPERT_STAGES.items() for label in POINT_LABELS]
    differences, summary = op.paired_point_tables(points, specs, seeds, "group", ("value_full_image", "value_roi"), "difference_roi_minus_full", ["stage", "point", "level", "metric"])
    return {"threshold_curves": pd.concat(curves, ignore_index=True), "operating_points": points, "operating_point_differences": differences, "operating_point_summary": summary}


FREQUENCY_FLAGS = [("multiple_components", lambda d: d["n_components"] > 1, "more than one connected component (8-connected)"),
                   ("mask_covers_more_than_half_of_the_image", lambda d: d["positive_fraction"] > 0.5, "mask covers more than half of the image"),
                   ("crop_is_at_least_90_percent_of_the_image", lambda d: d["crop_area_fraction"] >= 0.9, "ROI crop covers at least 90 % of the original image area (practically a full image)"),
                   ("tight_box_touches_an_image_border", lambda d: d["touches_any"], "the tight box touches an image border"),
                   ("expansion_cut_by_the_image_border", lambda d: d["expansion_clipped"], "the 20 % expansion is shortened by the image border"),
                   ("crop_enlarged_to_160", lambda d: d["upscaled"], "crop smaller than 160 pixels is enlarged (scale factor above 1)"),
                   ("crop_enlarged_by_a_factor_of_2_or_more", lambda d: d["scale_to_160"] >= 2.0, "enlargement factor at least 2"),
                   ("crop_enlarged_by_a_factor_of_3_or_more", lambda d: d["scale_to_160"] >= 3.0, "enlargement factor at least 3 (strongly enlarged, visibly blurred crops)")]


def geometry_frequencies(geometry: pd.DataFrame) -> pd.DataFrame:
    """Per subset (train, validation) and group (all, benign, malignant): number of flagged images and cases with the denominators (descriptive; the crop rule is never adapted to them)."""
    rows = []
    for subset in ("train", "validation"):
        part = geometry[geometry["subset"] == subset]
        for group, selection in (("all", part), ("benign", part[part["class"] == "benign"]), ("malignant", part[part["class"] == "malignant"])):
            for flag, rule, description in FREQUENCY_FLAGS:
                flagged = rule(selection)
                rows.append({"subset": subset, "group": group, "flag": flag, "description": description, "n_flagged_images": int(flagged.sum()), "n_flagged_cases": int(selection.loc[flagged, "Case"].nunique()),
                             "denominator_images": int(len(selection)), "denominator_cases": int(selection["Case"].nunique()), "share_of_images": float(flagged.sum() / len(selection)) if len(selection) else None})
    return pd.DataFrame(rows)


# ---- automatic ROI: metrics, comparisons and operating points ---------------------------------------------------------------------------------------------------
def run_metrics(sets: dict) -> pd.DataFrame:
    """ROC-AUC, AP and the confusion metrics at the threshold 0.5 for every (group, seed)."""
    rows = []
    for (group, seed), frame in sets.items():
        metrics = ev.binary_metrics(frame["label"].to_numpy(dtype=int), frame["malignant_score"].to_numpy(dtype=float), th.REFERENCE_THRESHOLD)
        info = AUTO_GROUPS[group]
        rows.append({"group": group, "pipeline": group, "model": info["pipeline"], "stage": info["stage"], "information": info["information"], "seed": seed, "subset": "validation",
                     "threshold": th.REFERENCE_THRESHOLD, "n_images": metrics["n_images"], "n_cases": int(frame["Case"].nunique()), "n_malignant_images": metrics["n_positive"],
                     "n_benign_images": metrics["n_negative"], **{k: metrics[k] for k in ("tn", "fp", "fn", "tp", "sensitivity", "specificity", "precision", "f1", "accuracy", "roc_auc", "average_precision")}})
    return pd.DataFrame(rows)


def auto_study_tables(sets: dict) -> dict:
    """The study tables of the automatic ROI from the 30 stored score sets: run metrics, summary, paired run differences and summary, curves, operating points and their paired differences."""
    runs = run_metrics(sets)
    summary = aggregation.summarize(runs, list(AUTO_GROUPS), RUN_METRICS)
    summary.insert(1, "information", [AUTO_GROUPS[p]["information"] for p in summary["pipeline"]])
    pairs = [(f"{pair} ({stage})", names[reference], names["auto"]) for stage, names in AUTO_STAGES.items() for pair, reference in AUTO_PAIRS]
    differences = aggregation.paired_between(runs, runs, pairs, SEEDS, RUN_METRICS)
    curves, points = [], []
    for (group, seed), frame in sets.items():
        info = AUTO_GROUPS[group]
        curve, rows = op.run_analysis(frame, {"group": group, "pipeline": info["pipeline"], "information": info["information"], "stage": info["stage"], "seed": seed}, TARGETS)
        curves.append(curve)
        points += rows
    points = pd.DataFrame(points)
    mismatch = [f"{g} seed {s}" for (g, s) in sets for r in [runs[(runs["group"] == g) & (runs["seed"] == s)].iloc[0]]
                for p in [points[(points["group"] == g) & (points["seed"] == s) & (points["point_label"] == "0.5")].iloc[0]] if (p["tn"], p["fp"], p["fn"], p["tp"]) != tuple(int(r[k]) for k in ("tn", "fp", "fn", "tp"))]
    if mismatch:
        raise ValueError(f"the operating point 0.5 differs from the confusion matrix of the run metrics: {mismatch[:3]}")
    specs = [{"ids": {"pair": pair, "stage": stage, "point": label}, "a": names[reference], "b": names["auto"], "label": label}
             for stage, names in AUTO_STAGES.items() for pair, reference in AUTO_PAIRS for label in POINT_LABELS]
    point_differences, point_summary = op.paired_point_tables(points, specs, SEEDS, "group", ("value_reference", "value_auto"), "difference_auto_minus_reference", ["pair", "stage", "point", "level", "metric"])
    return {"run_metrics": runs, "summary_validation": summary, "paired_differences": differences, "paired_summary": aggregation.summarize_pairs(differences),
            "threshold_curves": pd.concat(curves, ignore_index=True), "operating_points": points, "operating_point_differences": point_differences, "operating_point_summary": point_summary}


# ---- automatic ROI: crop diagnoses, concurrence with classification errors, named examples ----------------------------------------------------------------------
def diagnostic_tables(diag: pd.DataFrame) -> dict:
    """The descriptive tables of the actual automatic crops from the per-image diagnoses (stored 5C/4F-B table; no image, mask or model is read)."""
    return {"crop_flags": ar.crop_flag_table(diag), "crop_categories": ar.category_table(diag), "coverage_summary": ar.coverage_summary(diag), "box_definitions": ar.definition_table(diag),
            "component_selection": ar.component_selection_table(diag)}


def concurrence_table(diag: pd.DataFrame, sets: dict) -> pd.DataFrame:
    """For every category of the actual crop, stage, class and input: the number of images, and per seed the misclassified images at the threshold 0.5 (false negatives for malignant, false
    positives for benign) and the mean score, reported as mean, min and max over the five seeds. Descriptive; no pooled statistic and no rule is derived."""
    rows = []
    labels = diag["label"].to_numpy(dtype=int)
    for stage, names in AUTO_STAGES.items():
        for category in ar.CATEGORIES + ["component_dice_below_0.5", "any_image"]:
            if category == "any_image":
                member = np.ones(len(diag), dtype=bool)
            elif category == "component_dice_below_0.5":
                member = (diag["component_dice"] < 0.5).to_numpy()
            else:
                member = (diag["category"] == category).to_numpy()
            for cls, value in (("malignant", 1), ("benign", 0)):
                selected = member & (labels == value)
                n = int(selected.sum())
                for kind, group in (("full_image_4B", names["full"]), ("expert_roi_4E", names["expert"]), ("automatic_roi_4FB", names["auto"])):
                    errors, mean_scores = [], []
                    for seed in SEEDS:
                        s = sets[(group, seed)]["malignant_score"].to_numpy(dtype=float)[selected]
                        errors.append(int((s < th.REFERENCE_THRESHOLD).sum()) if value == 1 else int((s >= th.REFERENCE_THRESHOLD).sum()))
                        mean_scores.append(float(s.mean()) if n else float("nan"))
                    rows.append({"stage": stage, "category": category, "class": cls, "input": kind, "n_images": n, "n_cases": int(diag.loc[selected, "Case"].nunique()),
                                 "error_type": "false_negatives_at_0.5" if value == 1 else "false_positives_at_0.5", "errors_mean_over_seeds": float(np.mean(errors)), "errors_min": int(min(errors)),
                                 "errors_max": int(max(errors)), "error_share_mean": (float(np.mean(errors)) / n) if n else None, "mean_score_mean_over_seeds": float(np.nanmean(mean_scores)) if n else None,
                                 "errors_by_seed": ";".join(str(e) for e in errors)})
    return pd.DataFrame(rows)


def notable_examples(diag: pd.DataFrame, sets: dict, selection: pd.DataFrame) -> pd.DataFrame:
    """The named validation examples: the three highlighted by the segmentation inspection and all images of its fixed error-analysis selection (`selection` = the stored preview selection of the
    segmentation study), with their crop and the scores of the three inputs. The selection is a given input; nothing is chosen here."""
    errors = selection[selection["block"] == "error_analysis"]
    ids = HIGHLIGHTED_EXAMPLES + [i for i in errors["ID"] if i not in HIGHLIGHTED_EXAMPLES]
    index = {i: k for k, i in enumerate(diag["ID"])}
    missing = [i for i in ids if i not in index]
    if missing:
        raise ValueError(f"examples are not part of the diagnoses: {missing}")
    rank_of = {i: r for r, i in enumerate(errors["ID"], start=1)}
    dice_of = errors.set_index("ID")["dice_original"].to_dict()
    rows = []
    for identifier in ids:
        k = index[identifier]
        d = diag.iloc[k]
        row = {"ID": identifier, "highlighted_in_4FA_inspection": identifier in HIGHLIGHTED_EXAMPLES, "4FA_error_analysis_rank": rank_of.get(identifier),
               "class": d["class"], "Device": d["Device"], "Case": d["Case"], "4FA_dice_all_predicted_pixels": dice_of.get(identifier),
               "predicted_pixels": int(d["predicted_pixels"]), "n_components": int(d["n_components"]), "largest_component_pixels": int(d["largest_size"]), "second_component_pixels": int(d["second_size"]),
               "fallback_full_image": bool(d["fallback_full_image"]), "crop_category": d["category"], "expert_coverage_of_actual_crop": float(d["coverage_actual_crop"]),
               "component_dice": None if pd.isna(d["component_dice"]) else float(d["component_dice"]), "crop_w": int(d["crop_w"]), "crop_h": int(d["crop_h"]),
               "crop_area_fraction": float(d["crop_area_fraction"]), "scale_to_160": float(d["scale_to_160"])}
        for stage in ("frozen", "finetune"):
            for kind in ("full", "expert", "auto"):
                scores = [float(sets[(AUTO_STAGES[stage][kind], seed)]["malignant_score"].iloc[k]) for seed in SEEDS]
                row[f"{stage}_{kind}_mean_score"] = float(np.mean(scores))
                row[f"{stage}_{kind}_seeds_with_score_at_least_0.5"] = int(sum(s >= th.REFERENCE_THRESHOLD for s in scores))
        rows.append(row)
    return pd.DataFrame(rows)


REQUIRED_DIAGNOSTICS = ["ID", "Case", "label", "class", "Device", "fallback_full_image", "category", "predicted_pixels", "n_components", "n_largest_ties", "largest_size", "second_size",
                        "component_dice", "coverage_actual_crop", "crop_w", "crop_h", "crop_area_fraction", "scale_to_160", "scale_ge_2", "scale_ge_3", "upscaled", "touches_any",
                        "expansion_clipped", "best_overlap", "chosen_is_best_overlap", "chosen_zero_but_other_overlaps", "overlap_chosen"] + [definition[1] for definition in ar.DEFINITIONS]
CLASS_OF_LABEL = {0: "benign", 1: "malignant"}


def read_diagnostics(path: Path, validation: pd.DataFrame) -> pd.DataFrame:
    """The per-image diagnoses of the automatic crops (primary input), checked against the validation subset: required columns, the same images in the same order, the same cases, and labels
    and class names exactly as in the split (labels are checked before any conversion)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"stored diagnoses not found: {path}")
    diag = pd.read_csv(path, dtype={"ID": str, "Case": str}, float_precision="round_trip")
    missing = [c for c in REQUIRED_DIAGNOSTICS if c not in diag.columns]
    if missing:
        raise ValueError(f"{path}: the diagnoses lack the columns {missing}")
    if diag["ID"].tolist() != validation["ID"].astype(str).tolist() or diag["Case"].tolist() != validation["Case"].astype(str).tolist():
        raise ValueError("the diagnoses do not list the validation images in the order of the split")
    labels = data.exact_labels(diag["label"], "diagnoses")
    expected = data.exact_labels(validation["label"], "validation subset of the split")
    if not np.array_equal(labels, expected):
        raise ValueError(f"{int((labels != expected).sum())} diagnosis labels differ from the split (first: {diag['ID'][labels != expected].iloc[0]})")
    if diag["class"].tolist() != [CLASS_OF_LABEL[int(v)] for v in expected] or diag["class"].tolist() != validation["Pathology"].astype(str).tolist():
        raise ValueError("the class names of the diagnoses differ from the split")
    return diag


def auto_tables(sets: dict, diag: pd.DataFrame, selection: pd.DataFrame) -> dict:
    """All 15 tables of the automatic-ROI study, keyed like AUTO_FILES."""
    return {**auto_study_tables(sets), **diagnostic_tables(diag), "notable_examples": notable_examples(diag, sets, selection), "concurrence": concurrence_table(diag, sets)}
