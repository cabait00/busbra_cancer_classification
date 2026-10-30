"""Validation operating points of the seven groups and 27 combinations of the final evaluation (CPU, stored tables only).

Nothing here opens an image or a mask, loads a model or imports TensorFlow. The threshold rule is the one of src/busbra/thresholds.py:
  * a unit is positive if score >= threshold (malignant = 1);
  * the needed number of detected positives for a sensitivity target q is ceil(q * n_positive), computed with exact fractions;
  * the threshold is the HIGHEST one that reaches at least that number (the k-th largest positive score); tied scores are classified together;
  * no interpolation and no optimisation of F1, accuracy or Youden; the case score is the maximum of the image scores of the case.
Thresholds are chosen on the validation subset only and are stored as decimal text plus IEEE-754 hex, so they can be read back exactly.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from busbra import config
from busbra import evaluation as ev
from busbra import thresholds as th

SEEDS = tuple(config.SEEDS)
TARGETS = ("0.90", "0.95")
EXPECTED_NEEDED = {"image": {"0.90": 113, "0.95": 119}, "case": {"0.90": 63, "0.95": 66}}
POINTS = (("0.5", "fixed_reference", None), ("target_0.90", "target_sensitivity", "0.90"), ("target_0.95", "target_sensitivity", "0.95"))
LEVELS = ("image", "case")

# group_id, label, seeds, where the validation scores come from
GROUPS = [
    {"group_id": "dummy", "label": "Dummy classifier (class prior)", "seeds": None, "adapter": "baseline", "model": "dummy_prior"},
    {"group_id": "logreg_l2", "label": "L2 logistic regression, 64 x 64 flattened", "seeds": None, "adapter": "baseline", "model": "logreg_l2"},
    {"group_id": "small_cnn", "label": "Small CNN", "seeds": SEEDS, "adapter": "run", "variant": "small_cnn_160"},
    {"group_id": "full_frozen", "label": "MobileNetV2 frozen, augmented full image", "seeds": SEEDS, "adapter": "run", "variant": "mobilenetv2_frozen_160_aug"},
    {"group_id": "full_finetuned", "label": "MobileNetV2 fine-tuned (Adam 1e-4), augmented full image", "seeds": SEEDS, "adapter": "run", "variant": "mobilenetv2_higher_lr_160_aug"},
    {"group_id": "expert_roi_finetuned", "label": "MobileNetV2 fine-tuned, expert-mask ROI", "seeds": SEEDS, "adapter": "run", "variant": "mobilenetv2_higher_lr_160_roi"},
    {"group_id": "auto_roi_finetuned", "label": "U-Net -> automatic ROI -> fine-tuned ROI classifier", "seeds": SEEDS, "adapter": "auto_roi"},
]
GROUP_IDS = [g["group_id"] for g in GROUPS]
N_COMBINATIONS = sum(1 if g["seeds"] is None else len(g["seeds"]) for g in GROUPS)  # 27


def combinations() -> list[dict]:
    """The 27 combinations in the fixed order of GROUPS: combo_id, group, seed (None for the two baselines) and the group definition."""
    entries = []
    for group in GROUPS:
        for seed in group["seeds"] or (None,):
            entries.append({"combo_id": f"{group['group_id']}|{'single' if seed is None else 'seed' + str(seed)}", "group_id": group["group_id"], "seed": seed, "spec": group})
    return entries


def float_text(value) -> str:
    """The shortest decimal text that reads back as exactly the same IEEE-754 double (repr)."""
    return repr(float(value))


def read_validation_scores(entry: dict, validation: pd.DataFrame) -> pd.DataFrame:
    """Validation scores of one combination in manifest order (columns ID, Case, label, malignant_score); raises ValueError on any inconsistency.

    Train rows of the mixed prediction files are ignored; test rows must not occur.
    """
    spec, seed = entry["spec"], entry["seed"]
    if spec["adapter"] == "baseline":
        frame = pd.read_csv(config.TABLES_DIR / "baseline_predictions.csv", dtype={"ID": str, "Case": str}, float_precision="round_trip")
        frame = frame[(frame["model"] == spec["model"]) & (frame["subset"] == "validation")]
    elif spec["adapter"] == "run":
        path = config.run_paths(config.VARIANTS[spec["variant"]], seed)["predictions"]
        frame = pd.read_csv(path, dtype={"ID": str, "Case": str}, float_precision="round_trip")
        frame = frame[(frame["model"] == spec["variant"]) & (frame["seed"] == seed) & (frame["subset"] == "validation")]
    else:  # the frozen automatic-ROI scores of the validation images
        frame = pd.read_csv(config.TABLES_DIR / "auto_roi" / f"seed_{seed}" / "finetune" / "scores_auto.csv", dtype={"ID": str, "Case": str}, float_precision="round_trip")
    ids = validation["ID"].tolist()
    if frame["ID"].duplicated().any() or set(frame["ID"]) != set(ids) or len(frame) != len(ids):
        raise ValueError(f"{entry['combo_id']}: the validation IDs differ from the split")
    frame = frame.set_index("ID").loc[ids].reset_index()
    if (frame["Case"].to_numpy() != validation["Case"].to_numpy()).any() or (frame["label"].to_numpy() != validation["label"].astype(int).to_numpy()).any():
        raise ValueError(f"{entry['combo_id']}: Case or label differ from the split")
    problems = th.score_problems(frame["malignant_score"].to_numpy(dtype=float))
    if problems:
        raise ValueError(f"{entry['combo_id']}: " + "; ".join(problems))
    return frame[["ID", "Case", "label", "malignant_score"]].astype({"label": int, "malignant_score": float})


def units(level: str, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores of the evaluation units: images, or cases with the maximum image score (a case is positive as soon as one image is)."""
    if level == "image":
        return frame["label"].to_numpy(dtype=int), frame["malignant_score"].to_numpy(dtype=float)
    cases = th.aggregate_cases(frame["Case"], frame["label"], frame["malignant_score"])
    return cases["label"].to_numpy(dtype=int), cases["case_score"].to_numpy(dtype=float)


def needed_exact(target: str, n_positive: int) -> int:
    """ceil(q * n) in integer arithmetic (independent of thresholds.needed_true_positives)."""
    q = Fraction(target)
    return -((-q.numerator * n_positive) // q.denominator)


def check_target(labels, scores, target: str, result: dict) -> list:
    """Independent verification of one target threshold: exact needed count, the curve-derived threshold, and the next higher distinct score no longer reaches the target."""
    problems = []
    n_positive = int((labels == 1).sum())
    needed = needed_exact(target, n_positive)
    if needed != result["needed_tp"]:
        problems.append(f"needed true positives {result['needed_tp']} != exact {needed}")
    curve_threshold = th.target_threshold_from_curve(th.operating_curve(labels, scores), needed)
    if curve_threshold != result["threshold"]:
        problems.append(f"curve threshold {curve_threshold!r} != {result['threshold']!r}")
    higher = scores[scores > result["threshold"]]
    if higher.size and th.confusion_counts(labels, scores, float(higher.min()))["tp"] >= needed:
        problems.append("the next higher distinct score still reaches the target: the threshold is not the highest")
    if result["tp"] < needed:
        problems.append("target not reached")
    if float.fromhex(float.hex(result["threshold"])) != result["threshold"] or float(float_text(result["threshold"])) != result["threshold"]:
        problems.append("threshold does not survive a text round trip")
    return problems


def metric_fields(counts: dict) -> dict:
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    n = tn + fp + fn + tp
    return {"n_units": n, "n_positive": tp + fn, "n_negative": tn + fp, "n_predicted_positive": tp + fp, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "sensitivity": ev.rate(tp, tp + fn), "specificity": ev.rate(tn, tn + fp), "precision": ev.rate(tp, tp + fp), "npv": ev.rate(tn, tn + fn),
            "f1": ev.rate(2 * tp, 2 * tp + fp + fn), "accuracy": ev.rate(tp + tn, n), "predicted_positive_share": ev.rate(tp + fp, n)}


def threshold_fields(threshold: float, scores) -> dict:
    return {"threshold": float_text(threshold), "threshold_hex": float(threshold).hex(), "n_units_with_score_equal_to_threshold": int((scores == threshold).sum())}


def compute_combination(entry: dict, frame: pd.DataFrame) -> tuple[list, list, list, list]:
    """Core operating-point rows (6), metric rows (2), historical case-view rows (3) of one combination, plus the problems found by the independent checks."""
    core, metrics, case_view, problems = [], [], [], []
    key = {"combo_id": entry["combo_id"], "group_id": entry["group_id"], "seed": entry["seed"]}
    image_thresholds = {}
    for level in LEVELS:
        labels, scores = units(level, frame)
        both = ev.binary_metrics(labels, scores)
        metrics.append({**key, "level": level, "n_units": int(labels.size), "n_positive": both["n_positive"], "n_negative": both["n_negative"], "roc_auc": both["roc_auc"], "average_precision": both["average_precision"],
                        "n_distinct_scores": int(np.unique(scores).size), "largest_tie_group": int(np.unique(scores, return_counts=True)[1].max())})
        for point, rule, target in POINTS:
            if target is None:
                threshold, needed, result = 0.5, None, None
            else:
                result = th.target_threshold(labels, scores, target)
                threshold, needed = result["threshold"], result["needed_tp"]
                problems += [f"{entry['combo_id']} {level} {point}: {p}" for p in check_target(labels, scores, target, result)]
                if needed != EXPECTED_NEEDED[level][target]:
                    problems.append(f"{entry['combo_id']} {level} {point}: needed {needed}, expected {EXPECTED_NEEDED[level][target]}")
            counts = th.confusion_counts(labels, scores, threshold)
            row = {**key, "level": level, "point": point, "selection_rule": rule if target is None else "highest threshold with at least ceil(q * positives) detected positives",
                   "point_kind": rule, "target_sensitivity": target, "needed_tp": needed, **threshold_fields(threshold, scores), **metric_fields(counts)}
            row["target_met"] = None if target is None else bool(counts["tp"] >= needed)
            row["near_all_positive"] = th.near_all_positive(counts["tp"] + counts["fp"], counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"])
            core.append(row)
            if level == "image":
                image_thresholds[point] = threshold
    case_labels, case_scores = units("case", frame)
    for point, rule, target in POINTS:
        threshold = image_thresholds[point]
        counts = th.confusion_counts(case_labels, case_scores, threshold)
        case_view.append({**key, "point": point, "threshold_origin": "image-level threshold of the same combination (historical comparison view, not a case threshold)",
                          "target_sensitivity": target, **threshold_fields(threshold, case_scores), **metric_fields(counts)})
    return core, metrics, case_view, problems


SUMMARY_METRICS = ["threshold", "n_predicted_positive", "tp", "fn", "fp", "tn", "sensitivity", "specificity", "precision", "npv", "f1", "accuracy", "predicted_positive_share"]


def summarize_operating_points(core: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = core.copy()
    work["threshold"] = work["threshold"].astype(float)
    for group in GROUP_IDS:
        for level in LEVELS:
            for point, _, _ in POINTS:
                part = work[(work["group_id"] == group) & (work["level"] == level) & (work["point"] == point)]
                for metric in SUMMARY_METRICS:
                    rows.append({"group_id": group, "level": level, "point": point, "metric": metric, **ev.stats(part[metric]),
                                 "note": "mean over the available training realisations (seed mean) of the same holdout; SD undefined for a single model"})
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUP_IDS:
        for level in LEVELS:
            part = metrics[(metrics["group_id"] == group) & (metrics["level"] == level)]
            for metric in ("roc_auc", "average_precision"):
                rows.append({"group_id": group, "level": level, "metric": metric, **ev.stats(part[metric]), "note": "seed mean; no pooled AUC over seeds, no ensemble"})
    return pd.DataFrame(rows)
