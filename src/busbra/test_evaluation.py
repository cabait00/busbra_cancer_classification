"""CPU helpers for the final evaluation of the frozen test subset: frozen thresholds, metric rows, paired cluster bootstrap and segmentation diagnostics.

Nothing in this module opens an image or a mask, loads a model or imports TensorFlow. It works on tables and arrays that the script passes in.
Fixed rules:

* the seven groups, the 27 combinations and the seeds are those of busbra.final_evaluation; nothing is replaced by a more favourable historical run;
* the thresholds are the frozen validation values (image and case level, exact IEEE-754 hex text); no threshold is chosen or averaged with a test label, and none is chosen inside the bootstrap;
* a unit is positive if score >= threshold (malignant = 1); the case score is the maximum of the image scores; undefined rates stay empty;
* the three main comparisons use the groups full_finetuned (4B, extended fine-tuning 1e-4), expert_roi_finetuned (4E) and auto_roi_finetuned (4F-B): the same fine-tuning stage and learning rate;
* the bootstrap draws cases (clusters) with replacement within the case label (stratified), keeps repeated cases as repeated units, uses the same draws for every group and seed, computes every metric per seed
  first and averages the paired differences over the seeds afterwards.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from busbra import auto_roi as ar
from busbra import evaluation as ev
from busbra import final_evaluation as fe
from busbra import roi
from busbra import segmentation as seg
from busbra import thresholds as th

# ---- the registered design ----------------------------------------------------------------------------------------------
SEEDS = list(fe.SEEDS)
GROUP_IDS = list(fe.GROUP_IDS)
N_COMBINATIONS = fe.N_COMBINATIONS  # 27
CLASSIFIER_BATCH_SIZE = 16
UNET_BATCH_SIZE = seg.BATCH_SIZE  # 8
MAIN_GROUPS = {"full": "full_finetuned", "expert": "expert_roi_finetuned", "auto": "auto_roi_finetuned"}
COMPARISONS = [("expert_minus_full", "full", "expert"), ("auto_minus_expert", "expert", "auto"), ("auto_minus_full", "full", "auto")]  # (id, reference, other): difference = other minus reference
BOOTSTRAP = {"replicates": 2000, "rng_seed": 2026, "generator": "numpy.random.default_rng(2026) (PCG64), one generator for all replicates in the order of the draws below",
             "strata": "case label: n_malignant_cases draws from the malignant cases, then n_benign_cases draws from the benign cases, each with replacement (numpy Generator.integers), cases sorted by the string order of the Case value",
             "unit": "case (cluster); all images of a drawn case are used together; a case drawn several times counts several times (image level: its images are repeated; case level: the case is a repeated unit)",
             "quantile": "numpy.percentile(..., method='linear') at 2.5 and 97.5 (two-sided 95 % percentile interval)",
             "same_draws": "the same draws for every group, every seed and both evaluation levels",
             "order_of_averaging": "metric per group and seed on the resampled test set first; the paired difference (other minus reference) per seed; then the mean over the five seeds; the interval is taken over the 2,000 seed-means",
             "fixed_thresholds": "the frozen 5A image threshold (image level) and case threshold (case level) of every combination for the targets 0.90 and 0.95; no threshold is re-chosen inside a replicate"}
TARGETS = fe.TARGETS  # ("0.90", "0.95")
BOOTSTRAP_METRICS = [("auc_image", "image", "auc", None), ("auc_case", "case", "auc", None)] + [
    (f"{kind}_{level}_{target}", level, kind, target) for level in ("image", "case") for target in TARGETS for kind in ("sensitivity", "specificity")]
BOOTSTRAP_METRIC_NAMES = [m[0] for m in BOOTSTRAP_METRICS]
THRESHOLD_POINT_OF = {"0.90": "target_0.90", "0.95": "target_0.95"}
TIE_TOLERANCE = 1e-12


class TestEvaluationError(Exception):
    """An input of the final evaluation violates its contract; the message names the problem. Nothing is repaired silently."""


# ---- frozen thresholds of 5A ---------------------------------------------------------------------------------------------------
def load_frozen_thresholds(path: Path, combo_ids: list) -> tuple[dict, list]:
    """(combo_id, level, point) -> {value, text, hex, needed_tp, target}. The value is read from the exact hex text; text and hex must agree and survive a round trip."""
    table = pd.read_csv(path, dtype={"threshold": str, "threshold_hex": str, "target_sensitivity": str, "point": str, "level": str}, keep_default_na=False)
    problems, out = [], {}
    expected_keys = {(c, level, point) for c in combo_ids for level in fe.LEVELS for point, _, _ in fe.POINTS}
    for row in table.itertuples():
        key = (row.combo_id, row.level, row.point)
        if key in out:
            problems.append(f"duplicated threshold row {key}")
            continue
        value = float.fromhex(row.threshold_hex)
        if not math.isfinite(value) or float(row.threshold) != value or float.hex(value) != row.threshold_hex or repr(value) != row.threshold:
            problems.append(f"{key}: threshold text {row.threshold!r} and hex {row.threshold_hex!r} do not round-trip to the same double")
        out[key] = {"value": value, "text": row.threshold, "hex": row.threshold_hex, "needed_tp": int(float(row.needed_tp)) if str(row.needed_tp).strip() not in ("", "nan") else None,
                    "target": row.target_sensitivity if row.target_sensitivity not in ("", "nan") else None}
    if set(out) != expected_keys:
        problems.append(f"threshold keys differ from the 27 x 2 x 3 expected ones (missing {len(expected_keys - set(out))}, unexpected {len(set(out) - expected_keys)})")
    for key, t in out.items():
        if key[2] == "0.5" and t["value"] != 0.5:
            problems.append(f"{key}: the reference threshold is not 0.5")
    return out, problems


# ---- score frames -----------------------------------------------------------------------------------------------------------------
def score_frame_problems(frame: pd.DataFrame, meta: pd.DataFrame, what: str) -> list:
    """One combination's raw test scores against the frozen metadata: same IDs, Case values and labels in the manifest order, finite scores in [0, 1], one label per case."""
    problems = []
    if list(frame["ID"]) != list(meta["ID"]):
        return [f"{what}: the IDs differ from the manifest order of the test subset"]
    if list(frame["Case"].astype(str)) != list(meta["Case"].astype(str)):
        problems.append(f"{what}: Case values differ from the manifest")
    if list(frame["label"].astype(int)) != list(meta["label"].astype(int)):
        problems.append(f"{what}: labels differ from the manifest")
    problems += [f"{what}: {p}" for p in th.score_problems(frame["malignant_score"].to_numpy(dtype=float))]
    return problems


# ---- metric rows with FROZEN thresholds ---------------------------------------------------------------------------------------
def combination_rows(entry: dict, frame: pd.DataFrame, thresholds: dict) -> tuple[list, list, list]:
    """Metric rows (2: image, case), core operating-point rows (6) and the historical case view under the image thresholds (3) of one combination on the test subset.

    The thresholds are looked up, never computed: no label of the test subset selects anything. `target_sensitivity_reached_on_test` is a plain description (tp >= ceil(q * positives of the test unit set)),
    not a correction: a shortfall is reported."""
    key = {"combo_id": entry["combo_id"], "group_id": entry["group_id"], "seed": entry["seed"]}
    core, metrics, case_view = [], [], []
    for level in fe.LEVELS:
        labels, scores = fe.units(level, frame)
        both = ev.binary_metrics(labels, scores)
        metrics.append({**key, "level": level, "n_units": int(labels.size), "n_positive": both["n_positive"], "n_negative": both["n_negative"], "roc_auc": both["roc_auc"], "average_precision": both["average_precision"],
                        "n_distinct_scores": int(np.unique(scores).size), "largest_tie_group": int(np.unique(scores, return_counts=True)[1].max())})
        for point, rule, target in fe.POINTS:
            t = thresholds[(entry["combo_id"], level, point)]
            counts = th.confusion_counts(labels, scores, t["value"])
            row = {**key, "level": level, "point": point, "point_kind": rule, "target_sensitivity": target, "validation_needed_tp": t["needed_tp"], "threshold_origin": "frozen 5A validation threshold",
                   **fe.threshold_fields(t["value"], scores), **fe.metric_fields(counts)}
            if target is None:
                row["test_needed_tp_for_target"], row["target_sensitivity_reached_on_test"] = None, None
            else:
                needed = fe.needed_exact(target, both["n_positive"])
                row["test_needed_tp_for_target"], row["target_sensitivity_reached_on_test"] = needed, bool(counts["tp"] >= needed)
            row["near_all_positive"] = th.near_all_positive(counts["tp"] + counts["fp"], counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"])
            core.append(row)
    case_labels, case_scores = fe.units("case", frame)
    for point, rule, target in fe.POINTS:
        t = thresholds[(entry["combo_id"], "image", point)]
        counts = th.confusion_counts(case_labels, case_scores, t["value"])
        case_view.append({**key, "point": point, "threshold_origin": "frozen 5A IMAGE-level threshold of the same combination (historical comparison view, not a case threshold)", "target_sensitivity": target,
                          **fe.threshold_fields(t["value"], case_scores), **fe.metric_fields(counts)})
    return core, metrics, case_view


def independent_confusion_check(core: pd.DataFrame, frames: dict, thresholds: dict) -> list:
    """Every core row's confusion recomputed with the sklearn confusion matrix of busbra.evaluation (an independent routine) at the frozen threshold."""
    problems = []
    for row in core.itertuples():
        frame = frames[row.combo_id]
        labels, scores = fe.units(row.level, frame)
        reference = ev.binary_metrics(labels, scores, thresholds[(row.combo_id, row.level, row.point)]["value"])
        if (reference["tn"], reference["fp"], reference["fn"], reference["tp"]) != (row.tn, row.fp, row.fn, row.tp):
            problems.append(f"{row.combo_id} {row.level} {row.point}: the confusion differs from the independent routine")
    return problems


def validation_reference_table(core: pd.DataFrame, validation_core: pd.DataFrame) -> pd.DataFrame:
    """Validation (frozen in 5A) next to test at the same point, per combination, level and point. Description only: the validation numbers were used for development and are optimistic."""
    keys = ["combo_id", "group_id", "seed", "level", "point"]
    left = validation_core[keys + ["threshold", "tp", "fp", "fn", "tn", "sensitivity", "specificity", "precision", "n_predicted_positive", "n_positive", "n_negative"]].rename(columns=lambda c: c if c in keys else f"validation_{c}")
    right = core[keys + ["tp", "fp", "fn", "tn", "sensitivity", "specificity", "precision", "n_predicted_positive", "n_positive", "n_negative"]].rename(columns=lambda c: c if c in keys else f"test_{c}")
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one")
    merged["sensitivity_test_minus_validation"] = merged["test_sensitivity"].astype(float) - merged["validation_sensitivity"].astype(float)
    merged["specificity_test_minus_validation"] = merged["test_specificity"].astype(float) - merged["validation_specificity"].astype(float)
    return merged


def summarize_test(core: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fe.summarize_operating_points(core), fe.summarize_metrics(metrics)


# ---- paired comparisons by seed (observed on the test set, no resampling) ------------------------------------------------------------
PAIRED_POINT_METRICS = ["n_predicted_positive", "tp", "fn", "fp", "tn", "sensitivity", "specificity", "precision", "npv", "f1", "accuracy"]


def paired_by_seed(core: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The three main comparisons per seed (other minus reference): AUC and AP (image and case level) and the metrics at 0.5 and at the frozen target points; summary with mean, sample SD and sign counts."""
    rows = []
    for comparison, reference, other in COMPARISONS:
        gref, goth = MAIN_GROUPS[reference], MAIN_GROUPS[other]
        for seed in SEEDS:
            for level in fe.LEVELS:
                a = metrics[(metrics["group_id"] == gref) & (metrics["seed"] == seed) & (metrics["level"] == level)].iloc[0]
                b = metrics[(metrics["group_id"] == goth) & (metrics["seed"] == seed) & (metrics["level"] == level)].iloc[0]
                for metric in ("roc_auc", "average_precision"):
                    rows.append({"comparison": comparison, "level": level, "point": "", "metric": metric, "seed": seed, "value_reference": float(a[metric]), "value_other": float(b[metric]), "difference_other_minus_reference": float(b[metric]) - float(a[metric])})
                for point, _, _ in fe.POINTS:
                    pa = core[(core["group_id"] == gref) & (core["seed"] == seed) & (core["level"] == level) & (core["point"] == point)].iloc[0]
                    pb = core[(core["group_id"] == goth) & (core["seed"] == seed) & (core["level"] == level) & (core["point"] == point)].iloc[0]
                    for metric in PAIRED_POINT_METRICS:
                        va, vb = pa[metric], pb[metric]
                        rows.append({"comparison": comparison, "level": level, "point": point, "metric": metric, "seed": seed, "value_reference": va, "value_other": vb,
                                     "difference_other_minus_reference": (float(vb) - float(va)) if pd.notna(va) and pd.notna(vb) else np.nan})
    differences = pd.DataFrame(rows)
    summary = []
    for (comparison, level, point, metric), part in differences.groupby(["comparison", "level", "point", "metric"], sort=False):
        part = part.sort_values("seed")
        summary.append({"comparison": comparison, "level": level, "point": point, "metric": metric, **th.paired_difference_summary(part["difference_other_minus_reference"], TIE_TOLERANCE),
                        "differences_by_seed": ";".join("" if pd.isna(v) else repr(float(v)) for v in part["difference_other_minus_reference"]), "seeds": ";".join(str(s) for s in SEEDS)})
    return differences, pd.DataFrame(summary)


# ---- the paired stratified cluster bootstrap --------------------------------------------------------------------------------------
def case_table(frame: pd.DataFrame) -> dict:
    """Case structure of a test frame (ID, Case, label): case IDs in string order, labels per case, the image rows of every case as a padded matrix (-1 = padding)."""
    cases = sorted(frame["Case"].astype(str).unique())
    position = {c: i for i, c in enumerate(cases)}
    case_of = np.array([position[c] for c in frame["Case"].astype(str)], dtype=np.int64)
    labels = frame["label"].to_numpy(dtype=int)
    n_cases = len(cases)
    width = int(np.bincount(case_of, minlength=n_cases).max())
    matrix = -np.ones((n_cases, width), dtype=np.int64)
    fill = np.zeros(n_cases, dtype=np.int64)
    for image_index, c in enumerate(case_of):
        matrix[c, fill[c]] = image_index
        fill[c] += 1
    case_labels = np.zeros(n_cases, dtype=int)
    for c in range(n_cases):
        case_images = matrix[c][matrix[c] >= 0]
        if len(set(labels[case_images])) != 1:
            raise TestEvaluationError(f"case {cases[c]} has several labels")
        case_labels[c] = labels[case_images[0]]
    return {"cases": cases, "case_of_image": case_of, "image_matrix": matrix, "case_labels": case_labels, "n_malignant_cases": int((case_labels == 1).sum()), "n_benign_cases": int((case_labels == 0).sum())}


def bootstrap_draws(case_labels: np.ndarray, replicates: int = BOOTSTRAP["replicates"], rng_seed: int = BOOTSTRAP["rng_seed"]) -> np.ndarray:
    """(replicates, n_cases) indices into the case list: per replicate first n_malignant draws from the malignant cases, then n_benign draws from the benign cases (both with replacement)."""
    rng = np.random.default_rng(rng_seed)
    malignant, benign = np.flatnonzero(case_labels == 1), np.flatnonzero(case_labels == 0)
    draws = np.empty((replicates, len(case_labels)), dtype=np.int64)
    for r in range(replicates):
        draws[r, :len(malignant)] = malignant[rng.integers(0, len(malignant), len(malignant))]
        draws[r, len(malignant):] = benign[rng.integers(0, len(benign), len(benign))]
    return draws


def combo_arrays(frame: pd.DataFrame, structure: dict) -> dict:
    """Per-combination arrays for the bootstrap: image labels / scores and the case labels / maximum scores in the order of structure['cases']."""
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["malignant_score"].to_numpy(dtype=np.float64)
    matrix = structure["image_matrix"]
    case_scores = np.array([scores[row[row >= 0]].max() for row in matrix], dtype=np.float64)
    return {"labels": labels, "scores": scores, "case_labels": structure["case_labels"], "case_scores": case_scores, "matrix": matrix}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _rates(labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity and specificity at several thresholds (score >= threshold); an undefined rate is NaN."""
    predicted = scores[:, None] >= thresholds[None, :]
    positives, negatives = labels == 1, labels == 0
    tp = (predicted & positives[:, None]).sum(axis=0)
    tn = (~predicted & negatives[:, None]).sum(axis=0)
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    return (tp / n_pos if n_pos else np.full(len(thresholds), np.nan)), (tn / n_neg if n_neg else np.full(len(thresholds), np.nan))


def metric_vector(arrays: dict, thresholds: dict, draw: np.ndarray) -> np.ndarray:
    """The BOOTSTRAP_METRICS of one combination on the cases listed in `draw` (indices into the case list, repetitions kept). thresholds: {('image'|'case', '0.90'|'0.95'): float}."""
    members = arrays["matrix"][draw].ravel()
    members = members[members >= 0]
    y_img, s_img = arrays["labels"][members], arrays["scores"][members]
    y_case, s_case = arrays["case_labels"][draw], arrays["case_scores"][draw]
    out = np.empty(len(BOOTSTRAP_METRICS), dtype=np.float64)
    out[0], out[1] = _auc(y_img, s_img), _auc(y_case, s_case)
    index = 2
    for level, (y, s) in (("image", (y_img, s_img)), ("case", (y_case, s_case))):
        taus = np.array([thresholds[(level, t)] for t in TARGETS], dtype=np.float64)
        sens, spec = _rates(y, s, taus)
        for k in range(len(TARGETS)):
            out[index], out[index + 1] = sens[k], spec[k]
            index += 2
    return out


def bootstrap_values(frames: dict, thresholds: dict, draws: np.ndarray, structure: dict) -> tuple[np.ndarray, np.ndarray]:
    """values[group, seed, metric, replicate] and the observed values[group, seed, metric] (every case once) for the three main groups and the five seeds.

    frames: (group_id, seed) -> test frame; thresholds: (combo_id, level, point) -> {value, ...} as returned by load_frozen_thresholds."""
    groups = [MAIN_GROUPS[k] for k in ("full", "expert", "auto")]
    values = np.full((len(groups), len(SEEDS), len(BOOTSTRAP_METRICS), len(draws)), np.nan)
    observed = np.full((len(groups), len(SEEDS), len(BOOTSTRAP_METRICS)), np.nan)
    identity = np.arange(len(structure["cases"]))
    for gi, group in enumerate(groups):
        for si, seed in enumerate(SEEDS):
            frame = frames[(group, seed)]
            arrays = combo_arrays(frame, structure)
            combo = f"{group}|seed{seed}"
            tau = {(level, t): thresholds[(combo, level, THRESHOLD_POINT_OF[t])]["value"] for level in fe.LEVELS for t in TARGETS}
            observed[gi, si] = metric_vector(arrays, tau, identity)
            for r in range(len(draws)):
                values[gi, si, :, r] = metric_vector(arrays, tau, draws[r])
    return values, observed


def comparison_summary(values: np.ndarray, observed: np.ndarray, replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimate (mean over the five seeds of the observed paired differences), the two-sided 95 % percentile interval of the seed-mean differences over the replicates, and the seed spread as a separate quantity."""
    order = {"full": 0, "expert": 1, "auto": 2}
    rows, replicate_rows = [], []
    for comparison, reference, other in COMPARISONS:
        gi_ref, gi_oth = order[reference], order[other]
        for mi, metric in enumerate(BOOTSTRAP_METRIC_NAMES):
            by_seed = observed[gi_oth, :, mi] - observed[gi_ref, :, mi]
            replicate_means = (values[gi_oth, :, mi, :] - values[gi_ref, :, mi, :]).mean(axis=0)
            finite = replicate_means[np.isfinite(replicate_means)]
            low, high = (float(np.percentile(finite, 2.5, method="linear")), float(np.percentile(finite, 97.5, method="linear"))) if finite.size else (None, None)
            rows.append({"comparison": comparison, "reference_group": MAIN_GROUPS[reference], "other_group": MAIN_GROUPS[other], "metric": metric, "point_estimate_seed_mean_difference": float(np.nanmean(by_seed)) if np.isfinite(by_seed).any() else None,
                         "ci95_low": low, "ci95_high": high, "interval_excludes_zero": (bool(low > 0 or high < 0) if low is not None else None), "n_replicates": int(replicates), "n_replicates_defined": int(finite.size),
                         "replicate_mean": float(finite.mean()) if finite.size else None, "replicate_sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
                         "share_replicates_positive": float((finite > 0).mean()) if finite.size else None, "share_replicates_negative": float((finite < 0).mean()) if finite.size else None,
                         "seed_sd_ddof1": float(np.nanstd(by_seed, ddof=1)) if np.isfinite(by_seed).sum() > 1 else None, "differences_by_seed": ";".join(repr(float(v)) for v in by_seed),
                         "note": "interval = sampling uncertainty of the 376 test images / 212 cases for FIXED models and thresholds; the seed spread is a different quantity; no new training, no new threshold selection, no external clinic"})
            for r, v in enumerate(replicate_means):
                replicate_rows.append((r, comparison, metric, float(v)))
    return pd.DataFrame(rows), pd.DataFrame(replicate_rows, columns=["replicate", "comparison", "metric", "seed_mean_difference_other_minus_reference"])


def observed_from_tables(core: pd.DataFrame, metrics: pd.DataFrame) -> np.ndarray:
    """The observed bootstrap metrics read from the 54 metric rows and the 162 core rows (independent of the resampling code); shape (group, seed, metric)."""
    groups = [MAIN_GROUPS[k] for k in ("full", "expert", "auto")]
    out = np.full((len(groups), len(SEEDS), len(BOOTSTRAP_METRICS)), np.nan)
    for gi, group in enumerate(groups):
        for si, seed in enumerate(SEEDS):
            for mi, (name, level, kind, target) in enumerate(BOOTSTRAP_METRICS):
                if kind == "auc":
                    out[gi, si, mi] = float(metrics[(metrics["group_id"] == group) & (metrics["seed"] == seed) & (metrics["level"] == level)].iloc[0]["roc_auc"])
                else:
                    row = core[(core["group_id"] == group) & (core["seed"] == seed) & (core["level"] == level) & (core["point"] == THRESHOLD_POINT_OF[target])].iloc[0]
                    value = row[kind]
                    out[gi, si, mi] = float(value) if pd.notna(value) else np.nan
    return out


# ---- ROI adapters ---------------------------------------------------------------------------------------------------------------------
def expert_geometry_for_inputs(expert_geometry: pd.DataFrame) -> pd.DataFrame:
    """The 4E-style expert geometry table in the column names that busbra.auto_roi.inputs_from_geometry expects (decoded size and expanded box); no mask is involved here."""
    frame = expert_geometry[["ID", "Case", "label", "image_width", "image_height", "exp_x0", "exp_y0", "exp_x1", "exp_y1"]].rename(columns={"image_width": "width", "image_height": "height"})
    return frame.reset_index(drop=True)


# ---- segmentation diagnostics of the test subset (after the automatic ROIs are frozen) --------------------------------------------------------
def segmentation_image_row(row: dict, record: dict, expert: np.ndarray, prediction: np.ndarray) -> dict:
    """Quality of the RAW prediction (all predicted pixels, 4F-A definition) on the decoded pixel grid of one test image; the crop diagnosis (largest component, 4F-B definition) comes from busbra.auto_roi."""
    expert, prediction = np.asarray(expert, dtype=bool), np.asarray(prediction, dtype=bool)
    height, width = prediction.shape
    metrics = seg.dice_iou(prediction, expert)
    sizes = roi.components(prediction)
    area = width * height
    expert_box = seg.box_of(expert)
    out = {"ID": row["ID"], "expert_pixels_original": metrics["truth_pixels"], "pred_pixels_original": metrics["pred_pixels"], "dice_original": metrics["dice"], "iou_original": metrics["iou"],
           "pred_empty": metrics["pred_pixels"] == 0, "pred_components_raw": len(sizes), "pred_fraction_original": metrics["pred_pixels"] / area, "pred_over_half_of_image": metrics["pred_pixels"] / area > 0.5}
    out.update(seg.box_deviation(seg.box_of(prediction), expert_box, expert))
    out.update({"expert_x0": expert_box[0], "expert_y0": expert_box[1], "expert_x1": expert_box[2], "expert_y1": expert_box[3]})
    return out


def segmentation_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the raw-prediction Dice and IoU on the original grid (4F-A definition), empty predictions, Dice below 0.5, predictions with several components; overall, by class and by device with denominators."""
    rows = []
    for group_type, group, frame in ar._groups(per_image):
        n = len(frame)
        base = {"group_type": group_type, "group": group, "n_images": int(n), "n_cases": int(frame["Case"].nunique())}
        for metric in ("dice_original", "iou_original"):
            rows.append({**base, "metric": metric, **seg.distribution(frame[metric]), "note": "small groups are not interpreted" if n < seg.SMALL_GROUP_NOTE_BELOW else ""})
        for name, flagged in (("empty_prediction", frame["pred_empty"]), ("dice_below_0.5", frame["dice_original"] < 0.5), ("several_components_in_the_raw_prediction", frame["pred_components_raw"] > 1),
                              ("prediction_over_half_of_the_image", frame["pred_over_half_of_image"])):
            flagged = flagged.astype(bool)
            rows.append({**base, "metric": f"count_{name}", "n_defined": int(n), "mean": float(flagged.mean()) if n else None, "sd_ddof1": None, "min": None, "p05": None, "q1": None, "median": None, "q3": None, "p95": None, "max": None,
                         "n_flagged": int(flagged.sum()), "n_flagged_cases": int(frame.loc[flagged.to_numpy(), "Case"].nunique()), "note": "share of the images of the group; n_flagged images"})
    return pd.DataFrame(rows)


def dice_histogram(per_image: pd.DataFrame) -> pd.DataFrame:
    values = per_image["dice_original"].to_numpy(dtype=float)
    edges = seg.DICE_BINS
    counts, _ = np.histogram(values, bins=edges)  # the last bin includes 1.0
    return pd.DataFrame({"bin_low": edges[:-1], "bin_high": edges[1:], "n_images": counts.astype(int), "n_images_total": int(len(values))})


