"""Operating curves, operating points and paired point comparisons of validation score sets (pure pandas / numpy; stored scores only).

One implementation for every study that reports threshold curves and operating points: the sensitivity-oriented threshold analysis (all targets and the threshold grid), the expert-mask
ROI study and the automatic-ROI study (reference threshold 0.5 and the planned targets 0.90 / 0.95). The rules are those of busbra.thresholds: a unit is positive if score >= threshold,
equal scores form one step, a target threshold is the highest threshold that reaches the needed number of true positives, the case score is the maximum of the image scores and the
thresholds are always the image-derived ones. Rates with a zero denominator are undefined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from busbra import evaluation as ev
from busbra import thresholds as th

FLAT_KEYS = ("n_images", "n_positive", "n_negative", "n_predicted_positive", "predicted_positive_share", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "precision")
PAIRED_POINT_METRICS = ["threshold", "n_predicted_positive", "tp", "fn", "fp", "tn", "sensitivity", "specificity", "precision"]


def flat(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}{k}": metrics[k] for k in FLAT_KEYS}


def run_analysis(frame: pd.DataFrame, ident: dict, targets, grid=(), annotated: bool = False, case_precision: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Complete image curve with case counts and the operating points of one score set.

    frame: validation rows with Case, label, malignant_score. ident: identifier columns put in front of every row. Points in this order: the reference threshold 0.5, the `grid` thresholds
    (a sequence of floats), then one point per sensitivity target (strings such as "0.90"). annotated adds point_kind (reference / grid / target) and target_met. Raises ValueError if counts
    are not monotone along the curve, a point differs from the independent confusion-matrix routine or a target threshold differs from the one derived from the curve."""
    labels, scores = frame["label"].to_numpy(dtype=int), frame["malignant_score"].to_numpy(dtype=float)
    case_frame = th.aggregate_cases(frame["Case"], labels, scores)
    case_labels, case_scores = case_frame["label"].to_numpy(), case_frame["case_score"].to_numpy()
    curve = th.operating_curve(labels, scores)
    case_counts = pd.DataFrame([th.confusion_counts(case_labels, case_scores, t) for t in curve["threshold"]])
    for column in ("tn", "fp", "fn", "tp", "n_predicted_positive"):
        curve[f"case_{column}"] = case_counts[column].to_numpy()
    curve["case_sensitivity"] = [th.rate(int(a), int(a + b)) for a, b in zip(case_counts["tp"], case_counts["fn"])]
    curve["case_specificity"] = [th.rate(int(a), int(a + b)) for a, b in zip(case_counts["tn"], case_counts["fp"])]
    if case_precision:
        curve["case_precision"] = [th.rate(int(a), int(a + b)) for a, b in zip(case_counts["tp"], case_counts["fp"])]
    problems = []
    if not (curve["tp"].is_monotonic_increasing and curve["fp"].is_monotonic_increasing and case_counts["tp"].is_monotonic_increasing and case_counts["fp"].is_monotonic_increasing):
        problems.append("counts are not monotone along the curve")
    curve = pd.concat([pd.DataFrame([ident] * len(curve)), curve], axis=1)

    def point(kind: str, label: str, threshold: float, extra: dict) -> dict:
        image = th.metrics_from_counts(th.confusion_counts(labels, scores, threshold))
        case = th.metrics_from_counts(th.confusion_counts(case_labels, case_scores, threshold))
        reference = ev.binary_metrics(labels, scores, threshold)  # independent confusion-matrix routine
        if (image["tn"], image["fp"], image["fn"], image["tp"]) != (reference["tn"], reference["fp"], reference["fn"], reference["tp"]):
            problems.append(f"confusion at {label} differs from the independent routine")
        head = {"point_kind": kind} if annotated else {}
        return {**ident, **head, "point_label": label, "threshold": threshold, **extra, **flat("", image), "near_all_positive": th.near_all_positive(image["n_predicted_positive"], image["n_images"]),
                **flat("case_", case)}

    no_target = {"target_sensitivity": None, "needed_tp": None, **({"target_met": None} if annotated else {})}
    points = [point("reference", "0.5", th.REFERENCE_THRESHOLD, no_target)]
    points += [point("grid", f"{g:.1f}", g, no_target) for g in grid]
    for target in targets:
        result = th.target_threshold(labels, scores, target)
        if th.target_threshold_from_curve(curve, result["needed_tp"]) != result["threshold"]:
            problems.append(f"target {target} threshold differs from the curve")
        extra = {"target_sensitivity": float(target), "needed_tp": result["needed_tp"], **({"target_met": bool(result["tp"] >= result["needed_tp"])} if annotated else {})}
        points.append(point("target", f"target_{target}", result["threshold"], extra))
    if problems:
        raise ValueError("; ".join(problems))
    return curve, points


def paired_point_tables(points: pd.DataFrame, specs: list[dict], seeds, group_column: str, value_names: tuple, difference_name: str, summary_keys: list, kinds=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired differences (b minus a) at operating points within the same seed, image and case level, and their summary over the seeds.

    specs: ordered list of {"ids": identifier columns of the comparison, "a": group, "b": group, "label": point label}. kinds restricts the rows to those point kinds (annotated tables).
    summary_keys: the identifier columns (from ids, plus level and metric) that define a summary row. Mean and sample SD (ddof=1) and the sign counts come from th.paired_difference_summary."""
    def row_of(group, seed, label):
        mask = (points[group_column] == group) & (points["seed"] == seed) & (points["point_label"] == label)
        if kinds is not None:
            mask &= points["point_kind"].isin(kinds)
        selected = points[mask]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one point for {group}, seed {seed}, {label}; found {len(selected)}")
        return selected.iloc[0]

    rows = []
    for spec in specs:
        for seed in seeds:
            a, b = row_of(spec["a"], seed, spec["label"]), row_of(spec["b"], seed, spec["label"])
            for level, prefix in (("image", ""), ("case", "case_")):
                for metric in PAIRED_POINT_METRICS:
                    if metric == "threshold" and level == "case":
                        continue
                    column = metric if metric == "threshold" else f"{prefix}{metric}"
                    va, vb = a[column], b[column]
                    rows.append({**spec["ids"], "level": level, "seed": seed, "metric": metric, value_names[0]: va, value_names[1]: vb,
                                 difference_name: (float(vb) - float(va)) if pd.notna(va) and pd.notna(vb) else np.nan})
    differences = pd.DataFrame(rows)
    summary = []
    for key, part in differences.groupby(summary_keys, sort=False):
        summary.append({**dict(zip(summary_keys, key)), **th.paired_difference_summary(part[difference_name]),
                        "differences_by_seed": ";".join("" if pd.isna(v) else f"{v:.10g}" for v in part.sort_values("seed")[difference_name]), "seeds": ";".join(str(s) for s in seeds)})
    return differences, pd.DataFrame(summary)
