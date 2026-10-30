"""Helpers of step 5E (FGSM main evaluation and device breakdown on the frozen test subset): attack bookkeeping, decision changes, device groups, subset metrics, seed summaries.

NumPy / pandas only (no TensorFlow, no image access). The FGSM step itself is the reviewed helper of step 5B (busbra.model_diagnostics.fgsm_step); this module never perturbs anything. All thresholds are the
frozen 5A thresholds that are passed in (exact doubles); no threshold is searched, averaged or calibrated here, neither per device nor on attacked scores.

Conventions: a frame has the columns ID, Case, label, malignant_score (one row per image); the positive class is malignant = 1; decision rule score >= threshold; the case score is the MAXIMUM of the image scores of the
case; an undefined rate (zero denominator) or an undefined AUC / AP (one class only) is None, never 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from busbra import evaluation as ev
from busbra import final_evaluation as fe
from busbra import thresholds as th

EPSILONS_255 = (0, 1, 2, 4)
ATTACK_CONDITIONS = {"full": "full_finetuned", "auto": "auto_roi_finetuned"}
MIXED = "mixed devices"
ALL = "all"
CORE_METRICS = ["n_predicted_positive", "tp", "fn", "fp", "tn", "sensitivity", "specificity", "precision", "npv", "f1", "accuracy", "predicted_positive_share"]


class RobustnessError(Exception):
    """A precondition of the robustness / device analysis is violated."""


def attack_combos(seeds) -> list:
    return [{"combo_id": f"{group}|seed{seed}", "group_id": group, "seed": int(seed), "condition": condition} for condition, group in ATTACK_CONDITIONS.items() for seed in seeds]


# ---- decisions and their changes --------------------------------------------------------------------------------------------------------
def _aligned_units(level: str, clean: pd.DataFrame, adv: pd.DataFrame):
    if list(clean["ID"]) != list(adv["ID"]) or list(clean["Case"].astype(str)) != list(adv["Case"].astype(str)) or list(clean["label"]) != list(adv["label"]):
        raise RobustnessError("the clean and the attacked frame do not list the same IDs, cases and labels in the same order")
    labels, clean_scores = fe.units(level, clean)
    labels2, adv_scores = fe.units(level, adv)
    if not np.array_equal(labels, labels2):
        raise RobustnessError("unit labels differ between clean and attacked frame")
    return labels, clean_scores, adv_scores


def decision_changes(clean: pd.DataFrame, adv: pd.DataFrame, level: str, threshold: float) -> dict:
    """Separates the four ways a decision can stay or change relative to the label: correct -> wrong, wrong -> correct, stays correct, stays wrong (plus the raw number of flips and their direction by class)."""
    labels, a, b = _aligned_units(level, clean, adv)
    pa, pb = a >= threshold, b >= threshold
    ok_a, ok_b = pa == (labels == 1), pb == (labels == 1)
    flips = pa != pb
    return {"n_units": int(labels.size), "correct_to_wrong": int((ok_a & ~ok_b).sum()), "wrong_to_correct": int((~ok_a & ok_b).sum()), "stay_correct": int((ok_a & ok_b).sum()), "stay_wrong": int((~ok_a & ~ok_b).sum()),
            "decision_flips": int(flips.sum()), "flips_to_malignant": int((~pa & pb).sum()), "flips_to_benign": int((pa & ~pb).sum()),
            "benign_correct_to_wrong": int((ok_a & ~ok_b & (labels == 0)).sum()), "malignant_correct_to_wrong": int((ok_a & ~ok_b & (labels == 1)).sum()),
            "benign_wrong_to_correct": int((~ok_a & ok_b & (labels == 0)).sum()), "malignant_wrong_to_correct": int((~ok_a & ok_b & (labels == 1)).sum())}


def decision_change_rows(key: dict, epsilon_255: int, clean: pd.DataFrame, adv: pd.DataFrame, thresholds: dict) -> list:
    rows = []
    for level in fe.LEVELS:
        for point, _, _ in fe.POINTS:
            t = thresholds[(key["combo_id"], level, point)]
            rows.append({**key, "epsilon_255": epsilon_255, "level": level, "point": point, "threshold": t["text"], **decision_changes(clean, adv, level, t["value"])})
    return rows


# ---- device groups ----------------------------------------------------------------------------------------------------------------------
def device_groups(rows: list) -> dict:
    """image -> device (every image its own device); case -> its device, or the separate group 'mixed devices' when the images of a case come from several devices. Every case is assigned exactly once."""
    image_device = {r["ID"]: r["Device"] for r in rows}
    if len(image_device) != len(rows) or any(not d for d in image_device.values()):
        raise RobustnessError("image IDs are not unique or a device is missing")
    per_case = {}
    for r in rows:
        per_case.setdefault(str(r["Case"]), set()).add(r["Device"])
    case_group = {case: (next(iter(devices)) if len(devices) == 1 else MIXED) for case, devices in per_case.items()}
    return {"image_device": image_device, "case_group": case_group, "devices": sorted(set(image_device.values())), "mixed_cases": sorted(c for c, g in case_group.items() if g == MIXED)}


def device_table(rows: list, groups: dict) -> pd.DataFrame:
    """Per device: images and cases with benign / malignant denominators (cases assigned once; mixed-device cases form their own group)."""
    frame = pd.DataFrame(rows)[["ID", "Case", "Device", "label"]].astype({"label": int})
    frame["Case"] = frame["Case"].astype(str)
    out = []
    for name in [*groups["devices"], MIXED]:
        images = frame[frame["Device"] == name]
        cases = frame[frame["Case"].map(groups["case_group"]) == name].groupby("Case")["label"].first()
        out.append({"group": name, "n_images": int(len(images)), "n_images_malignant": int((images["label"] == 1).sum()), "n_images_benign": int((images["label"] == 0).sum()),
                    "n_cases": int(len(cases)), "n_cases_malignant": int((cases == 1).sum()), "n_cases_benign": int((cases == 0).sum())})
    out.append({"group": ALL, "n_images": int(len(frame)), "n_images_malignant": int((frame["label"] == 1).sum()), "n_images_benign": int((frame["label"] == 0).sum()), "n_cases": int(frame["Case"].nunique()),
                "n_cases_malignant": int(frame.groupby("Case")["label"].first().eq(1).sum()), "n_cases_benign": int(frame.groupby("Case")["label"].first().eq(0).sum())})
    return pd.DataFrame(out)


def subset_frames(frame: pd.DataFrame, groups: dict) -> dict:
    """(level, group) -> frame of the evaluation unit set: images of one device, or all images of the cases of one case group (cases are never split)."""
    work = frame.copy()
    work["Case"] = work["Case"].astype(str)
    work["_device"] = work["ID"].map(groups["image_device"])
    work["_case_group"] = work["Case"].map(groups["case_group"])
    if work["_device"].isna().any() or work["_case_group"].isna().any():
        raise RobustnessError("an image or case has no device assignment")
    out = {}
    for name in [*groups["devices"], ALL]:
        out[("image", name)] = work if name == ALL else work[work["_device"] == name]
    for name in [*groups["devices"], MIXED, ALL]:
        part = work if name == ALL else work[work["_case_group"] == name]
        if len(part):
            out[("case", name)] = part
    return out


def subset_rows(frame: pd.DataFrame, groups: dict, key: dict, thresholds: dict) -> tuple[list, list]:
    """Metric rows (AUC / AP) and core operating-point rows (0.5, target 0.90, target 0.95: the GLOBAL frozen thresholds of the combination at the matching level) for every device group and level. A group without
    both classes has an undefined AUC / AP (None); a rate with a zero denominator is None."""
    metrics, core = [], []
    for (level, name), part in subset_frames(frame, groups).items():
        labels, scores = fe.units(level, part)
        both = ev.binary_metrics(labels, scores)
        metrics.append({**key, "level": level, "group": name, "n_images": int(len(part)), "n_units": int(labels.size), "n_positive": both["n_positive"], "n_negative": both["n_negative"],
                        "roc_auc": both["roc_auc"], "average_precision": both["average_precision"]})
        for point, rule, target in fe.POINTS:
            t = thresholds[(key["combo_id"], level, point)]
            counts = th.confusion_counts(labels, scores, t["value"])
            core.append({**key, "level": level, "group": name, "point": point, "threshold": t["text"], "n_images": int(len(part)), **fe.metric_fields(counts)})
    return metrics, core


def seed_summary(frame: pd.DataFrame, keys: list, value_columns: list) -> pd.DataFrame:
    """Mean and sample SD (ddof = 1) over the seeds (one realisation per seed) of every value column within the grouping keys; no pooling over seeds."""
    rows = []
    for values, part in frame.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        for column in value_columns:
            rows.append({**dict(zip(keys, values)), "metric": column, **ev.stats(part[column])})
    return pd.DataFrame(rows)


def paired_change(frame: pd.DataFrame, keys: list, value_columns: list, epsilon_column: str = "epsilon_255") -> pd.DataFrame:
    """Per seed (identified inside `keys`) the value at epsilon > 0 minus the value of the SAME pipeline and seed at epsilon 0; undefined if either value is undefined."""
    base = frame[frame[epsilon_column] == 0].set_index(keys)
    rows = []
    for _, r in frame[frame[epsilon_column] > 0].iterrows():
        index = tuple(r[k] for k in keys)
        ref = base.loc[index if len(index) > 1 else index[0]]
        if isinstance(ref, pd.DataFrame):
            raise RobustnessError('the epsilon-0 reference of a paired change is not unique')
        for column in value_columns:
            a, b = ref[column], r[column]
            ok = pd.notna(a) and pd.notna(b)
            rows.append({**{k: r[k] for k in keys}, epsilon_column: int(r[epsilon_column]), "metric": column, "value_clean": a if pd.notna(a) else None, "value_attacked": b if pd.notna(b) else None,
                         "change": float(b) - float(a) if ok else None})
    return pd.DataFrame(rows)


