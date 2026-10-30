"""Binary-classification evaluation for BUS-BRA (positive class = malignant = 1).

Conventions shared by all models of the project:
- The score is the model's score for class 1 (malignant). It is NOT claimed to be a
  calibrated clinical probability.
- Hard predictions use the fixed rule score >= threshold (default 0.5).
- ROC-AUC and average precision use the continuous scores, never the hard labels.
- A rate with a zero denominator is undefined: None here (empty in CSV, null in JSON).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

POSITIVE_LABEL = 1
CLASS_ORDER = [0, 1]  # fixed order of the confusion matrix: benign, malignant


def positive_class_column(classes) -> int:
    """Column of predict_proba that belongs to the malignant class (label 1).

    The column is looked up in classes_ instead of assuming "the last column".
    """
    classes = list(np.asarray(classes).tolist())
    if sorted(classes) != CLASS_ORDER:
        raise ValueError(f"unexpected class labels {classes}; expected exactly {CLASS_ORDER}")
    return classes.index(POSITIVE_LABEL)


def malignant_scores(model, features: np.ndarray) -> np.ndarray:
    """Malignant score (float64, shape (n,)) of a fitted scikit-learn classifier."""
    probabilities = model.predict_proba(features)
    column = positive_class_column(model.classes_)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(model.classes_):
        raise ValueError(f"predict_proba shape {probabilities.shape} does not match classes_")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("predict_proba rows do not sum to 1")
    scores = probabilities[:, column].astype(np.float64)
    if not np.isfinite(scores).all() or scores.min() < 0.0 or scores.max() > 1.0:
        raise ValueError("malignant scores are not finite values in [0, 1]")
    return scores


def rate(numerator: int, denominator: int) -> float | None:
    """numerator / denominator, or None (undefined) when the denominator is zero."""
    return float(numerator) / float(denominator) if denominator else None


def binary_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    """All project metrics for one model on one set of images.

    Confusion-matrix counts are returned as tn, fp, fn, tp for the class order [0, 1].
    F1 = 2TP / (2TP + FP + FN) is defined whenever there is at least one positive
    image or one positive prediction; an undefined precision does not make it undefined.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape or y_true.ndim != 1:
        raise ValueError("y_true and scores must be 1-D arrays of the same length")
    if not set(np.unique(y_true)) <= set(CLASS_ORDER):
        raise ValueError("y_true must contain only the labels 0 and 1")
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).ravel())
    n_positive, n_negative = tp + fn, tn + fp
    both_classes = n_positive > 0 and n_negative > 0
    return {
        "n_images": int(len(y_true)), "n_positive": n_positive, "n_negative": n_negative,
        "n_predicted_positive": tp + fp,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "sensitivity": rate(tp, n_positive),
        "specificity": rate(tn, n_negative),
        "precision": rate(tp, tp + fp),
        "f1": rate(2 * tp, 2 * tp + fp + fn),
        "accuracy": rate(tp + tn, len(y_true)),
        # ROC and PR need both classes in the evaluated set.
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(y_true, scores)) if both_classes else None,
    }


def sigmoid_scores(outputs, expected_length: int) -> np.ndarray:
    """Validate the output of a single-unit sigmoid model and return it as malignant scores.

    A one-unit sigmoid output is directly the score for class 1; there is no classes_ and
    no argmax. Checked: shape (n, 1) or (n,), the expected number of rows (the order is
    that of the images fed in), finite values in [0, 1]. Returns float64 of shape (n,).
    """
    outputs = np.asarray(outputs)
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        outputs = outputs[:, 0]
    if outputs.ndim != 1 or len(outputs) != expected_length:
        raise ValueError(f"unexpected output shape {outputs.shape}; expected ({expected_length},) or "
                         f"({expected_length}, 1)")
    scores = outputs.astype(np.float64)
    if not np.isfinite(scores).all() or scores.min() < 0.0 or scores.max() > 1.0:
        raise ValueError("sigmoid scores are not finite values in [0, 1]")
    return scores


def stats(values, count_name: str = "n_realisations") -> dict:
    """Number, number of defined values, mean, sample standard deviation (ddof=1), minimum and maximum; undefined values are counted, never replaced by zero."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    defined = numeric[np.isfinite(numeric)]
    return {count_name: int(numeric.size), "n_defined": int(defined.size), "mean": float(defined.mean()) if defined.size else None,
            "sd_ddof1": float(defined.std(ddof=1)) if defined.size > 1 else None, "min": float(defined.min()) if defined.size else None, "max": float(defined.max()) if defined.size else None}
