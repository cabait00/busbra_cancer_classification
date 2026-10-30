# %% [02] Baselines: DummyClassifier (class prior) and L2 logistic regression on 64 x 64 pixels (CPU, scikit-learn only)
#
# Both models are fitted on TRAIN only and reported on train and validation. Image preparation: grayscale, whole image fitted into a
# 64 x 64 square with black padding, float32 / 255 (src/busbra/image_processing.py). The score is the model's score for class 1
# (malignant), not a calibrated clinical probability; the fixed decision rule is score >= 0.5. Nothing is tuned.
# Logistic regression: C=1, L2 (l1_ratio=0), lbfgs, no class weights, unscaled pixels. If - and only if - the iteration limit is
# reached, exactly one repeat with max_iter=5000 is done; the choice does not depend on any validation result.
#
# Actions (the default only reads stored results):
#   .venv/bin/python analysis/02_baselines.py            show the stored metrics
#   .venv/bin/python analysis/02_baselines.py --fit      fit and write the results; stops if baseline_metrics.csv already exists

# %% Imports
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from busbra import config, data, image_processing as ip  # noqa: E402
from busbra.evaluation import binary_metrics, malignant_scores  # noqa: E402

TARGET_SIZE = 64
LOGREG_PARAMS = dict(C=1.0, l1_ratio=0.0, solver="lbfgs", fit_intercept=True, class_weight=None, tol=1e-4, max_iter=2000)
FALLBACK_MAX_ITER = 5000
METRICS_PATH = config.TABLES_DIR / "baseline_metrics.csv"
PREDICTIONS_PATH = config.TABLES_DIR / "baseline_predictions.csv"


def fit_logreg(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")  # warnings stay visible
        model = LogisticRegression(**LOGREG_PARAMS).fit(features, labels)
    if any(w.category.__name__ == "ConvergenceWarning" for w in caught):
        print(f"iteration limit reached with max_iter={LOGREG_PARAMS['max_iter']}: one repeat with max_iter={FALLBACK_MAX_ITER}")
        model = LogisticRegression(**{**LOGREG_PARAMS, "max_iter": FALLBACK_MAX_ITER}).fit(features, labels)
    return model


# %% Show or fit
def show() -> None:
    if not METRICS_PATH.is_file():
        print("no stored baseline results")
        return
    table = pd.read_csv(METRICS_PATH)
    print(table[["model", "subset", "n_images", "roc_auc", "average_precision", "sensitivity", "specificity", "accuracy"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def fit() -> None:
    for path in (METRICS_PATH, PREDICTIONS_PATH, *config.BASELINE_MODELS.values()):
        if path.exists():
            raise FileExistsError(f"{path} already exists; nothing was changed")
    manifest = data.read_manifest()
    data.check_split(manifest)
    frames = {s: data.subset_frame(manifest, s) for s in ("train", "validation")}
    features, labels = {}, {}
    for subset, frame in frames.items():
        images = data.load_full_images(frame, TARGET_SIZE)[..., 0]
        features[subset], labels[subset] = ip.flatten_for_sklearn(images), frame["label"].astype(int).to_numpy()
    dummy = DummyClassifier(strategy="prior").fit(features["train"], labels["train"])
    logreg = fit_logreg(features["train"], labels["train"])
    models = {"dummy_prior": dummy, "logreg_l2": logreg}
    rows, predictions = [], []
    for name, model in models.items():
        for subset, frame in frames.items():
            scores = malignant_scores(model, features[subset])
            m = binary_metrics(labels[subset], scores, config.THRESHOLD)
            malignant = frame["label"] == "1"
            rows.append({"model": name, "subset": subset, "positive_class": "malignant (label 1)", "threshold": config.THRESHOLD, "decision_rule": "score >= threshold",
                         "n_images": m["n_images"], "n_cases": frame["Case"].nunique(), "n_malignant_cases": frame.loc[malignant, "Case"].nunique(),
                         "n_benign_cases": frame.loc[~malignant, "Case"].nunique(), "n_malignant_images": m["n_positive"], "n_benign_images": m["n_negative"],
                         "n_predicted_malignant": m["n_predicted_positive"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"], "sensitivity": m["sensitivity"],
                         "specificity": m["specificity"], "precision": m["precision"], "f1": m["f1"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
                         "average_precision": m["average_precision"]})
            predictions.append(pd.DataFrame({"model": name, "subset": subset, "ID": frame["ID"], "Case": frame["Case"], "label": labels[subset], "malignant_score": scores,
                                             "predicted_label": (scores >= config.THRESHOLD).astype(int), "threshold": config.THRESHOLD}))
    config.BASELINE_MODELS["dummy"].parent.mkdir(parents=True, exist_ok=True)
    contract = {"preprocessing": ip.describe_preprocessing(TARGET_SIZE), "labels": {"0": "benign", "1": "malignant"}, "score": "score for class 1 (malignant), not a calibrated probability",
                "threshold": config.THRESHOLD, "decision_rule": "score >= threshold", "trained_on": "train subset only", "scikit_learn_version": sklearn.__version__}
    for key, name in (("dummy", "dummy_prior"), ("logreg_l2", "logreg_l2")):
        joblib.dump({"name": name, "model": models[name], "parameters": models[name].get_params(), **contract}, config.BASELINE_MODELS[key])
    pd.DataFrame(rows).to_csv(METRICS_PATH, index=False, lineterminator="\n")
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTIONS_PATH, index=False, lineterminator="\n")
    show()


if __name__ == "__main__":
    fit() if "--fit" in (sys.argv[1:] if "__file__" in globals() else []) else show()
