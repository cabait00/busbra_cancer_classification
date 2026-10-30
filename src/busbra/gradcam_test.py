"""Helpers of step 5D (Grad-CAM and qualitative error analysis on the frozen test subset): deterministic example selection from stored tables, batch blocks, preview grouping and descriptive facts of one map.

Everything here is NumPy / pandas only (no TensorFlow, no image access). The selection uses stored scores, the frozen image threshold and the stored 5C segmentation diagnostics; it never looks at an image, a mask or a heat map.

Selection rule (fixed in the 5D task before the first heat map; the concrete IDs are re-derived from the tables and compared with the list given in the task):
  core     seed 7, automatic-ROI branch; confusion category of the stored 5C score against the frozen validation image threshold of the 0.95 sensitivity target (decision: score >= threshold);
           categories in the order TP, TN, FP, FN; per category ascending ID; at most two images per category; at most one image per case over the WHOLE selection (a case that is already used is skipped);
           a missing second image is not replaced.
  special  up to four localisation failures, one per category in a fixed order, each the first image in ascending ID that is neither a core image nor from a case that is already used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BATCH_SIZE = 16  # the original ordered evaluation batches of the 5C classification
CORE_CATEGORIES = ("TP", "TN", "FP", "FN")
PER_CORE_CATEGORY = 2
SPECIAL_CATEGORIES = ("fallback_full_image", "crop_without_expert_pixel", "chosen_component_misses_but_other_overlaps", "crop_covers_below_90_percent")
COVERAGE_BELOW = 0.9
DIAGNOSTIC_COLUMNS = ["ID", "Case", "label", "fallback_full_image", "category", "expert_pixels", "inside_actual_crop", "coverage_actual_crop", "chosen_zero_but_other_overlaps", "pred_empty", "dice_original"]


class GradcamTestError(Exception):
    """A precondition of the selection or of a map description is violated."""


def outcome(label, decision) -> str:
    """Confusion category of one decision: TP / FN for a malignant image (decision True = predicted malignant), FP / TN for a benign one."""
    if int(label) not in (0, 1):
        raise GradcamTestError(f"label {label!r} is not 0 / 1")
    return ("TP" if decision else "FN") if int(label) == 1 else ("FP" if decision else "TN")


def decisions(scores, threshold: float) -> np.ndarray:
    """The frozen decision rule of the project: predicted malignant iff score >= threshold."""
    return np.asarray(scores, dtype=np.float64) >= float(threshold)


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    frame["ID"], frame["Case"] = frame["ID"].astype(str), frame["Case"].astype(str)
    if frame["ID"].duplicated().any():
        raise GradcamTestError("duplicated image IDs in the selection table")
    return frame.sort_values("ID", kind="stable").reset_index(drop=True)


def select_core(scores: pd.DataFrame, threshold: float) -> tuple[list, list]:
    """Core examples from a table with ID, Case, label, malignant_score (any order) and the frozen image threshold. Returns (selected rows in processing order, log of every processed candidate)."""
    frame = _ordered(scores)
    frame["decision"] = decisions(frame["malignant_score"], threshold)
    frame["outcome"] = [outcome(l, d) for l, d in zip(frame["label"], frame["decision"])]
    selected, log, used_cases, used_by = [], [], set(), {}
    for category in CORE_CATEGORIES:
        candidates = frame[frame["outcome"] == category]
        taken = 0
        for row in candidates.itertuples():
            if taken == PER_CORE_CATEGORY:
                break
            if row.Case in used_cases:
                log.append({"phase": "core", "category": category, "ID": row.ID, "Case": row.Case, "status": "skipped_case_already_used", "n_candidates_of_category": len(candidates),
                            "detail": f"case {row.Case} is already used by {used_by[row.Case]}; at most one image per case over the whole selection"})
                continue
            used_cases.add(row.Case)
            used_by[row.Case] = row.ID
            taken += 1
            selected.append({"role": "core", "category": category, "ID": row.ID, "Case": row.Case, "label": int(row.label), "score_auto": float(row.malignant_score)})
            log.append({"phase": "core", "category": category, "ID": row.ID, "Case": row.Case, "status": "selected", "n_candidates_of_category": len(candidates), "detail": f"{taken}. image of category {category} in ascending ID order"})
        if taken < PER_CORE_CATEGORY:
            log.append({"phase": "core", "category": category, "ID": "", "Case": "", "status": "no_further_example", "n_candidates_of_category": len(candidates),
                        "detail": f"only {taken} of {PER_CORE_CATEGORY} images could be selected ({len(candidates)} image(s) of this category exist; the others were skipped as listed); no replacement by a hand-picked image"})
    return selected, log


def _strict_bool(value, name: str, allow_empty: bool = False):
    text = str(value).strip()
    if text in ("True", "true", "1", "1.0"):
        return True
    if text in ("False", "false", "0", "0.0"):
        return False
    if text in ("", "nan", "NaN", "None") and allow_empty:
        return None
    raise GradcamTestError(f"{name}: value {value!r} is not a boolean")


def special_flags(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """The four special-case flags per image from the stored 5C diagnostics, derived BOTH from the stored category text and from the numbers (expert pixels, pixels inside the crop, coverage); any disagreement is an
    error. chosen_zero_but_other_overlaps is empty (None) for a fallback image (no component exists) and is read strictly."""
    missing = [c for c in DIAGNOSTIC_COLUMNS if c not in diagnostics.columns]
    if missing:
        raise GradcamTestError(f"diagnostics columns missing: {missing}")
    frame = diagnostics[DIAGNOSTIC_COLUMNS].copy()
    frame["ID"], frame["Case"] = frame["ID"].astype(str), frame["Case"].astype(str)
    if frame["ID"].duplicated().any():
        raise GradcamTestError("duplicated image IDs in the diagnostics")
    fallback = np.array([_strict_bool(v, "fallback_full_image") for v in frame["fallback_full_image"]])
    pred_empty = np.array([_strict_bool(v, "pred_empty") for v in frame["pred_empty"]])
    zero_other = np.array([_strict_bool(v, "chosen_zero_but_other_overlaps", allow_empty=True) is True for v in frame["chosen_zero_but_other_overlaps"]])
    if (fallback != pred_empty).any():
        raise GradcamTestError("fallback_full_image and pred_empty disagree")
    for v, fb in zip(frame["chosen_zero_but_other_overlaps"], fallback):
        if (str(v).strip() in ("", "nan", "NaN")) != bool(fb):
            raise GradcamTestError("chosen_zero_but_other_overlaps must be empty exactly for the fallback images")
    expert = frame["expert_pixels"].astype(float).to_numpy()
    inside = frame["inside_actual_crop"].astype(float).to_numpy()
    coverage = frame["coverage_actual_crop"].astype(float).to_numpy()
    category = frame["category"].astype(str).to_numpy()
    if (expert <= 0).any() or (inside < 0).any() or (inside > expert).any():
        raise GradcamTestError("expert pixel counts are not positive or pixels inside the crop exceed them")
    if not np.allclose(coverage, inside / expert, rtol=0, atol=1e-12):
        raise GradcamTestError("coverage_actual_crop is not inside_actual_crop / expert_pixels")
    no_pixel_numeric, below_numeric = (~fallback) & (inside == 0), (~fallback) & (inside > 0) & (coverage < COVERAGE_BELOW)
    no_pixel_text, below_text = category == "covers_no_expert_pixel", category == "covers_more_than_0_to_below_90_percent"
    if (fallback != (category == "fallback_full_image")).any() or (no_pixel_numeric != no_pixel_text).any() or (below_numeric != below_text).any():
        raise GradcamTestError("the stored category text and the numbers of the diagnostics disagree")
    out = frame[["ID", "Case", "label"]].copy()
    out["fallback_full_image"], out["crop_without_expert_pixel"] = fallback, no_pixel_numeric
    out["chosen_component_misses_but_other_overlaps"], out["crop_covers_below_90_percent"] = zero_other, below_numeric
    return out


def select_special(diagnostics: pd.DataFrame, core_selected: list) -> tuple[list, list]:
    """At most one image per special category in the fixed order; candidates in ascending ID; an image that is a core image or whose case is already used (core or earlier special) is skipped. A category without a
    usable candidate is documented as 'no additional example available'; no other category is invented."""
    flags = _ordered(special_flags(diagnostics))
    used_ids = {r["ID"] for r in core_selected}
    used_cases = {r["Case"] for r in core_selected}
    used_by = {r["Case"]: r["ID"] for r in core_selected}
    selected, log = [], []
    for category in SPECIAL_CATEGORIES:
        candidates = flags[flags[category]]
        chosen = False
        for row in candidates.itertuples():
            if row.ID in used_ids:
                log.append({"phase": "special", "category": category, "ID": row.ID, "Case": row.Case, "status": "skipped_already_selected", "n_candidates_of_category": len(candidates), "detail": "the image is already part of the selection"})
            elif row.Case in used_cases:
                log.append({"phase": "special", "category": category, "ID": row.ID, "Case": row.Case, "status": "skipped_case_already_used", "n_candidates_of_category": len(candidates),
                            "detail": f"case {row.Case} is already used by {used_by[row.Case]}"})
            else:
                selected.append({"role": "special", "category": category, "ID": row.ID, "Case": row.Case, "label": int(row.label)})
                used_ids.add(row.ID), used_cases.add(row.Case)
                used_by[row.Case] = row.ID
                log.append({"phase": "special", "category": category, "ID": row.ID, "Case": row.Case, "status": "selected", "n_candidates_of_category": len(candidates), "detail": "first usable image in ascending ID order"})
                chosen = True
                break
        if not chosen:
            log.append({"phase": "special", "category": category, "ID": "", "Case": "", "status": "no_additional_example_available", "n_candidates_of_category": len(candidates),
                        "detail": "no additional example available: the category is empty or all its images are already selected or belong to used cases"})
    return selected, log


def batch_blocks(ordered_ids: list, target_ids: list, block_size: int = BATCH_SIZE) -> dict:
    """The original ordered evaluation batches (consecutive runs of block_size IDs of the ordered test subset; the last one may be shorter) that contain at least one target, with all their images."""
    position = {identifier: i for i, identifier in enumerate(ordered_ids)}
    if len(position) != len(ordered_ids):
        raise GradcamTestError("duplicated IDs in the ordered test subset")
    unknown = [t for t in target_ids if t not in position]
    if unknown:
        raise GradcamTestError(f"targets outside the ordered test subset: {unknown[:3]}")
    blocks = sorted({position[t] // block_size for t in target_ids})
    images = []
    for block in blocks:
        for pos in range(block * block_size, min((block + 1) * block_size, len(ordered_ids))):
            images.append({"ID": ordered_ids[pos], "position": pos, "block": block, "role": "target" if ordered_ids[pos] in set(target_ids) else "context"})
    return {"block_size": block_size, "blocks": blocks, "block_lengths": {str(b): sum(1 for i in images if i["block"] == b) for b in blocks}, "images": images}


def preview_groups(selected: list, rows_per_figure: int = 2) -> list:
    """Figure grouping: one figure per core category (at most two rows each), the special cases in figures of at most two rows in the order of the special categories."""
    groups = []
    for category in CORE_CATEGORIES:
        rows = [r["ID"] for r in selected if r["role"] == "core" and r["category"] == category]
        for start in range(0, len(rows), rows_per_figure):
            part = rows[start:start + rows_per_figure]
            groups.append({"name": f"core_{category}" + ("" if start == 0 else f"_{start // rows_per_figure + 1}"), "kind": "core", "category": category, "ids": part})
    special = [r["ID"] for r in selected if r["role"] == "special"]
    for start in range(0, len(special), rows_per_figure):
        groups.append({"name": f"special_{start // rows_per_figure + 1}", "kind": "special", "category": "special", "ids": special[start:start + rows_per_figure]})
    return groups


def map_facts(coarse_raw: np.ndarray, image_map: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Descriptive technical facts of one map (no localisation metric): raw coarse map (h, w) after ReLU, normalised map in the image frame (NaN = not computed), optional geometric validity mask of the model input
    canvas (True = image / crop pixel, False = artificial padding). The peak position follows the map only. The coarse map is normalised BEFORE the padding is removed and the map is resampled, so the maximum of the
    VISIBLE map (visible_max) can be below 1, for example when the largest coarse value lies in the removed padding; peak_cell_center_in_padding reports that case."""
    coarse = np.asarray(coarse_raw, dtype=np.float32)
    finite = np.isfinite(image_map)
    visible_max = float(np.max(image_map[finite])) if finite.any() else None
    positive = int((coarse > 0).sum())
    if positive == 0:
        return {"positive_coarse_cells": 0, "coarse_cells": int(coarse.size), "peak_cell_row": None, "peak_cell_col": None, "peak_x": None, "peak_y": None, "visible_max": visible_max, "peak_cell_center_in_padding": None}
    row, col = np.unravel_index(int(np.argmax(coarse)), coarse.shape)
    masked = np.where(finite, image_map, -np.inf)
    y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    in_padding = None
    if valid is not None:
        size_y, size_x = valid.shape
        cy, cx = min(size_y - 1, int((row + 0.5) * size_y / coarse.shape[0])), min(size_x - 1, int((col + 0.5) * size_x / coarse.shape[1]))
        in_padding = not bool(valid[cy, cx])
    return {"positive_coarse_cells": positive, "coarse_cells": int(coarse.size), "peak_cell_row": int(row), "peak_cell_col": int(col), "peak_x": int(x), "peak_y": int(y), "visible_max": visible_max,
            "peak_cell_center_in_padding": in_padding}


# ---- stored table formats read by the preview renderer (one definition for the writer, the renderer and the tests) --------------------------------------------
CONDITIONS = ("full", "expert", "auto")
SELECTION_COLUMNS = (["selection_order", "role", "category", "selection_reason", "ID", "Case", "label", "Device", "position_in_test_order", "batch_block", "position_in_block"]
                     + [f"{name}_{c}{suffix}" for c in CONDITIONS for name, suffix in (("score", ""), ("threshold", "_0.95"), ("threshold", "_0.95_hex"), ("decision", "_0.95"), ("outcome", "_0.95"))]
                     + ["auto_fallback_full_image", "auto_crop_category_5c", "expert_pixels_5c", "pixels_inside_actual_crop_5c", "coverage_actual_crop_5c", "chosen_zero_but_other_overlaps_5c", "dice_raw_prediction_5c"])
GEOMETRY_COLUMNS = ["condition", "ID", "role", "category", "box_source", "image_width", "image_height", "box_x0", "box_y0", "box_x1", "box_y1", "crop_width", "crop_height", "crop_area_fraction",
                    "fitted_width", "fitted_height", "pad_x", "pad_y", "pad_right", "pad_bottom", "scale", "auto_fallback_full_image"]
CAM_COLUMNS_USED = ["condition", "ID", "raw_max", "empty_after_relu", "positive_coarse_cells", "coarse_cells", "peak_cell_row", "peak_cell_col", "peak_cell_center_in_padding", "visible_max"]
BOX_SOURCE = {"full": "whole image (4B preparation)", "expert": "frozen expert geometry table (5C, roi-expert-mask-v1)", "auto": "frozen automatic geometry table (5C)"}
SPECIAL_REASON = {
    "fallback_full_image": "special case: empty segmentation: the frozen automatic crop is the whole image (fallback_full_image = True); first usable image in ascending ID order (not a core image, case not yet used)",
    "crop_without_expert_pixel": "special case: the actual automatic crop contains no expert-mask pixel (category covers_no_expert_pixel; not a fallback); first usable image in ascending ID order (not a core image, case not yet used)",
    "chosen_component_misses_but_other_overlaps": "special case: the chosen largest component does not overlap the expert mask although another component does (chosen_zero_but_other_overlaps = True); first usable image in ascending ID order (not a core image, case not yet used)",
    "crop_covers_below_90_percent": "special case: the actual automatic crop contains more than 0 but less than 90 % of the expert-mask pixels (category covers_more_than_0_to_below_90_percent; not a fallback); first usable image in ascending ID order (not a core image, case not yet used)"}


def core_reason(category: str, threshold: float) -> str:
    return (f"core {category}: automatic-ROI branch, seed 7, stored 5C score against the frozen 5A image threshold of the 0.95 target ({threshold!r}); "
            "2 images per category at most, ascending ID, at most one image per case over the whole selection")


def table_problems(selection: pd.DataFrame, geometry: pd.DataFrame, cam: pd.DataFrame, maps, ids) -> list[str]:
    """Missing columns or map keys that the preview renderer reads, for the given image IDs (empty list = the renderer inputs are complete)."""
    problems = [f"selection: missing column {c}" for c in SELECTION_COLUMNS if c not in selection.columns]
    problems += [f"geometry: missing column {c}" for c in GEOMETRY_COLUMNS if c not in geometry.columns]
    problems += [f"cam: missing column {c}" for c in CAM_COLUMNS_USED if c not in cam.columns]
    have_geometry, have_cam = set(zip(geometry.get("condition", []), geometry.get("ID", []))), set(zip(cam.get("condition", []), cam.get("ID", [])))
    for identifier in ids:
        if identifier not in set(selection.get("ID", [])):
            problems.append(f"selection: no row for {identifier}")
        for c in CONDITIONS:
            problems += [f"{name}: no row for {c}/{identifier}" for name, have in (("geometry", have_geometry), ("cam", have_cam)) if (c, identifier) not in have]
            if f"image__{c}__{identifier}" not in maps:
                problems.append(f"maps: no key image__{c}__{identifier}")
    return problems


def case_title_lines(selection_row, special_label: dict) -> list[str]:
    """Title of the original-image panel: ID, case, class, device and the selection category."""
    s = selection_row
    kind = f"{s['category']} of the automatic-ROI branch (0.95 target)" if s["role"] == "core" else "special case: " + special_label[s["category"]]
    return [f"{s['ID']} | case {s['Case']} | {'malignant' if int(s['label']) == 1 else 'benign'}", str(s["Device"]), kind]


def panel_lines(selection_row, cam_row, condition: str, condition_label: str) -> list[str]:
    """Title lines of one Grad-CAM panel: stored score, frozen threshold, decision and the descriptive facts of the map."""
    s, entry = selection_row, cam_row
    fallback = str(s["auto_fallback_full_image"]) == "True"
    empty = str(entry["empty_after_relu"]) == "True"
    lines = [condition_label + (" - FALLBACK: whole image" if condition == "auto" and fallback else ""),
             f"score {float(s[f'score_{condition}']):.3f} | threshold {float(s[f'threshold_{condition}_0.95']):.4f}",
             f"decision: {s[f'decision_{condition}_0.95']} ({s[f'outcome_{condition}_0.95']})",
             (f"raw max {float(entry['raw_max']):.2e} | {int(entry['positive_coarse_cells'])}/{int(entry['coarse_cells'])} cells > 0 | peak r{entry['peak_cell_row']},c{entry['peak_cell_col']}" if not empty
              else "raw max 0 | empty map is a regular finding")]
    if not empty and (str(entry["peak_cell_center_in_padding"]) == "True" or float(entry["visible_max"]) < 0.9):
        lines.append(f"note: the largest coarse value lies in the removed padding or at the edge; visible maximum {float(entry['visible_max']):.2f}")
    return lines
