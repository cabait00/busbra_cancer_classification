"""Operating-point helpers for the threshold analysis (step 4C): CPU only, NumPy and pandas, no TensorFlow.

Conventions (identical to busbra.evaluation):
- the score is the model's malignant score; it is NOT claimed to be a calibrated probability;
- hard predictions use the inclusive rule  score >= threshold  (malignant = 1);
- a rate with a zero denominator is undefined (None / NaN), never 0;
- equal scores form one step of the curve: they can only be classified together.

Everything here works on arrays that were already filtered to the validation subset; the functions never read files.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd

from busbra.evaluation import rate  # noqa: F401  (single implementation; re-exported for the callers of thresholds.rate)

TARGET_SENSITIVITIES = ("0.80", "0.90", "0.95")  # strings: converted to exact fractions, never through floating point
GRID_THRESHOLDS = tuple(float(f"0.{k}") for k in range(1, 10))  # 0.1 ... 0.9, the literal decimal values
REFERENCE_THRESHOLD = 0.5
NEAR_ALL_POSITIVE_SHARE = "0.90"  # an operating point that classifies at least this share of all images as positive is flagged openly


def score_problems(scores) -> list[str]:
    """Reasons why an array cannot be used as malignant scores (empty list = usable)."""
    scores = np.asarray(scores)
    problems = []
    if scores.ndim != 1 or scores.size == 0:
        problems.append("scores are not a non-empty 1-D array")
        return problems
    if not np.issubdtype(scores.dtype, np.number):
        problems.append("scores are not numeric")
        return problems
    if not np.isfinite(scores).all():
        problems.append(f"{int((~np.isfinite(scores)).sum())} score(s) are not finite")
    elif scores.min() < 0.0 or scores.max() > 1.0:
        problems.append(f"scores leave [0, 1] (min {scores.min()!r}, max {scores.max()!r})")
    return problems


def confusion_counts(labels, scores, threshold: float) -> dict:
    """tn, fp, fn, tp and the number of positive predictions for the rule score >= threshold (threshold may be +inf: nothing is positive)."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be 1-D arrays of the same length")
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("labels must contain only 0 and 1")
    if score_problems(scores):
        raise ValueError("; ".join(score_problems(scores)))
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "n_predicted_positive": tp + fp}


def metrics_from_counts(counts: dict) -> dict:
    """Counts plus sensitivity, specificity, precision and the positive-prediction share; undefined denominators give None."""
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    n = tn + fp + fn + tp
    return {**counts, "n_images": n, "n_positive": tp + fn, "n_negative": tn + fp, "sensitivity": rate(tp, tp + fn), "specificity": rate(tn, tn + fp),
            "precision": rate(tp, tp + fp), "predicted_positive_share": rate(tp + fp, n)}


def near_all_positive(n_predicted_positive: int, n: int) -> bool:
    """True if at least the pre-registered share (0.90) of all items is classified positive: the target is then reached almost trivially."""
    return n_predicted_positive >= math.ceil(Fraction(NEAR_ALL_POSITIVE_SHARE) * n)


def operating_curve(labels, scores) -> pd.DataFrame:
    """The complete piecewise operating curve over the observed distinct scores.

    One row per distinct score value t (rule score >= t) in descending order of t, preceded by the 'all negative' point (t = +inf).
    The row of the smallest observed score classifies everything as positive ('all positive'). Equal scores are one step. The result does not
    depend on the order of the input rows (stable sort on the negated score only fixes the order of tied items, never the counts)."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels and scores must be non-empty 1-D arrays of the same length")
    if score_problems(scores):
        raise ValueError("; ".join(score_problems(scores)))
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("labels must contain only 0 and 1")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    cumulative_fp = np.cumsum(sorted_labels == 0)
    last_of_group = np.concatenate([sorted_scores[1:] != sorted_scores[:-1], [True]])
    thresholds = np.concatenate([[np.inf], sorted_scores[last_of_group]])
    tp = np.concatenate([[0], cumulative_tp[last_of_group]]).astype(int)
    fp = np.concatenate([[0], cumulative_fp[last_of_group]]).astype(int)
    n_positive, n_negative = int((labels == 1).sum()), int((labels == 0).sum())
    fn, tn = n_positive - tp, n_negative - fp
    n = labels.size
    point_type = np.full(thresholds.size, "score_boundary", dtype=object)
    point_type[0] = "all_negative"
    point_type[tp + fp == n] = "all_positive"
    return pd.DataFrame({"point_index": np.arange(thresholds.size), "point_type": point_type, "threshold": thresholds, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                         "n_predicted_positive": tp + fp, "sensitivity": [rate(int(a), n_positive) for a in tp], "specificity": [rate(int(a), n_negative) for a in tn],
                         "precision": [rate(int(a), int(a + b)) for a, b in zip(tp, fp)]})


def needed_true_positives(target: str, n_positive: int) -> int:
    """Smallest number of detected positives whose sensitivity is at least `target` (exact fraction arithmetic, e.g. 0.95 of 125 -> 119)."""
    if n_positive <= 0:
        raise ValueError("no positive items: a sensitivity target is undefined")
    needed = math.ceil(Fraction(target) * n_positive)
    return max(needed, 1)


def target_threshold(labels, scores, target: str) -> dict:
    """Largest threshold t whose rule score >= t reaches at least the target sensitivity on these items.

    With a fixed ranking this is the k-th largest score among the positive items (k = needed true positives): any larger threshold detects fewer than k
    positives, t itself detects at least k (more if scores are tied). The achieved counts are returned, not assumed."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    n_positive = int((labels == 1).sum())
    needed = needed_true_positives(target, n_positive)
    positive_scores = np.sort(scores[labels == 1])[::-1]
    threshold = float(positive_scores[needed - 1])
    counts = confusion_counts(labels, scores, threshold)
    if counts["tp"] < needed:  # cannot happen by construction; guards against a wrong implementation
        raise AssertionError("target threshold does not reach the needed true positives")
    return {"target": target, "needed_tp": needed, "threshold": threshold, **counts}


def target_threshold_from_curve(curve: pd.DataFrame, needed_tp: int) -> float:
    """The same threshold, derived independently from the operating curve (largest finite threshold with tp >= needed)."""
    reached = curve[(curve["tp"] >= needed_tp) & np.isfinite(curve["threshold"])]
    if reached.empty:
        raise ValueError("no point of the curve reaches the needed true positives")
    return float(reached["threshold"].max())


def aggregate_cases(cases, labels, scores) -> pd.DataFrame:
    """Case level: label (must be identical for all images of the case), number of images, case score = MAXIMUM of the image scores.

    A case is positive at a threshold as soon as one of its images is positive. Rows are sorted by the case identifier (string order)."""
    frame = pd.DataFrame({"Case": pd.Series(cases, dtype="string").to_numpy(), "label": np.asarray(labels, dtype=int), "score": np.asarray(scores, dtype=np.float64)})
    per_case_labels = frame.groupby("Case", sort=True)["label"].nunique()
    if (per_case_labels != 1).any():
        raise ValueError(f"{int((per_case_labels != 1).sum())} case(s) have more than one label: {sorted(per_case_labels[per_case_labels != 1].index)[:5]}")
    grouped = frame.groupby("Case", sort=True)
    out = pd.DataFrame({"label": grouped["label"].first().astype(int), "n_images": grouped.size().astype(int), "case_score": grouped["score"].max().astype(np.float64)})
    return out.reset_index()


def paired_difference_summary(differences: pd.Series, tolerance: float = 1e-12) -> dict:
    """Mean and sample SD (ddof=1) of paired differences plus sign counts; NaN differences (undefined values) are excluded and counted."""
    values = pd.to_numeric(differences, errors="coerce").to_numpy(dtype=float)
    defined = values[np.isfinite(values)]
    n = int(defined.size)
    return {"n_seeds": int(values.size), "n_defined": n, "mean_difference": float(defined.mean()) if n else None, "sd_ddof1": float(defined.std(ddof=1)) if n > 1 else None,
            "min": float(defined.min()) if n else None, "max": float(defined.max()) if n else None, "n_positive": int((defined > tolerance).sum()),
            "n_negative": int((defined < -tolerance).sum()), "n_zero": int((np.abs(defined) <= tolerance).sum())}


def validation_frame_problems(frame: pd.DataFrame, expected: dict) -> list[str]:
    """Contract of one run's validation rows: unique IDs, labels 0/1, one label per case, finite scores in [0, 1], expected counts.

    `frame` needs the columns ID, Case, label (int), malignant_score (float). `expected` gives images, cases, malignant_images, malignant_cases."""
    problems = []
    if frame["ID"].duplicated().any():
        problems.append(f"{int(frame['ID'].duplicated().sum())} duplicated ID(s)")
    if frame["ID"].isna().any() or (frame["ID"].astype(str).str.len() == 0).any():
        problems.append("empty ID(s)")
    if not set(frame["label"].unique()) <= {0, 1}:
        problems.append(f"labels outside {{0, 1}}: {sorted(set(frame['label'].unique()) - {0, 1})}")
    problems += score_problems(frame["malignant_score"].to_numpy(dtype=float))
    per_case = frame.groupby("Case")["label"].nunique()
    if (per_case != 1).any():
        problems.append(f"{int((per_case != 1).sum())} case(s) with mixed labels")
    observed = {"images": int(len(frame)), "cases": int(frame["Case"].nunique()), "malignant_images": int((frame["label"] == 1).sum()),
                "malignant_cases": int(frame.loc[frame["label"] == 1, "Case"].nunique())}
    for key, value in expected.items():
        if observed[key] != value:
            problems.append(f"{key}: observed {observed[key]}, expected {value}")
    return problems
