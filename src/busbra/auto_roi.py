"""Helpers for the automatic ROI (step 4F-B): automatic ROI from the stored 4F-A segmentation, applied to the unchanged 4E classifiers (inference only).

INFORMATION CONDITION. The automatic ROI needs no manual mask at application time: the stored best U-Net checkpoint of step 4F-A predicts the lesion pixels from the image alone. The
4E classifiers were TRAINED on expert-mask ROIs; step 4F-B only changes their INPUT at inference (an automatic crop instead of the expert crop). It is not a classifier trained on automatic
crops. The U-Net itself was trained with expert masks and selected with the validation images, so the validation results of this step are explorative.

Rule (registered in the 4F-A plan, applied here without change; every coordinate is a coordinate of the DECODED original image)
1. The image goes through the unchanged 4F-A preparation (256 x 256, black padding) into the stored U-Net; the probability map is mapped back to the decoded size (busbra.segmentation) and only
   THERE binarised with probability >= 0.5.
2. 8-connected components of the binary mask; only the component with the most pixels is kept. A tie in pixel count goes to the component whose first pixel in row-major order (smallest row,
   then smallest column) comes first.
3. Empty mask: the whole unchanged image is the crop (a logged 'full-image fallback' for the SAME classifier; no switch to the 4B model).
4. Otherwise the tight box of the kept component is expanded by the frozen contract roi-expert-mask-v1 (ceil(edge / 5) per side, clipped at the image border), the UNMASKED original is cropped and
   prepared with the existing resize rule to 160 x 160 x 1 float32 in [0, 1] (busbra.roi).
No function that creates a crop takes an expert mask: the validation expert masks cannot influence a pixel of an automatic crop. They are read afterwards for diagnostics only.

The module imports the standard library, numpy, pandas, scipy, Pillow and the light project helpers; Keras is not imported here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from busbra import config, roi
from busbra import segmentation as seg

UNET_BATCH_SIZE = seg.BATCH_SIZE  # 8, as in the segmentation step: the same kernels for the same checkpoint
CLASSIFIER_BATCH_SIZE = config.BATCH_SIZE  # 16, as in the classification studies
# fine-tuning stage -> variant name of the expert-ROI classifier (the automatic ROI re-uses the SAME checkpoints) and its learning rate
STAGE_VARIANTS = {"frozen": "mobilenetv2_frozen_160_roi", "finetune": "mobilenetv2_higher_lr_160_roi"}


class AutoRoiError(Exception):
    """The input of the automatic ROI generation violates the contract; nothing is repaired silently."""


# ---------------------------------------------------------------------------------------------------------------------
# Components and the automatic crop (no function in this section takes an expert mask)
def label_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """8-connected labelling of a boolean mask: (label image with 0 = background, sizes of the components 1..n in label order)."""
    labelled, n = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labelled.ravel(), minlength=n + 1)[1:] if n else np.zeros(0, dtype=np.int64)
    return labelled, sizes


def first_pixels(labelled: np.ndarray, n: int) -> np.ndarray:
    """For every label 1..n the flat row-major index of its first pixel (smallest row, then smallest column)."""
    flat = labelled.ravel()
    positions = np.flatnonzero(flat)
    labels, first = np.unique(flat[positions], return_index=True)
    out = np.full(n + 1, -1, dtype=np.int64)
    out[labels] = positions[first]
    return out[1:]


def select_component(mask: np.ndarray) -> dict:
    """The component with the most pixels; a tie goes to the component whose first pixel in row-major order comes first. An empty mask has no component (component_mask None)."""
    mask = np.asarray(mask, dtype=bool)
    labelled, sizes = label_components(mask)
    n = int(len(sizes))
    base = {"predicted_pixels": int(mask.sum()), "n_components": n}
    if n == 0:
        return {**base, "chosen_label": None, "chosen_size": 0, "second_size": 0, "n_largest_ties": 0, "first_row": None, "first_col": None, "component_mask": None, "sizes": []}
    largest = int(sizes.max())
    tied = np.flatnonzero(sizes == largest) + 1
    starts = first_pixels(labelled, n)
    chosen = int(tied[int(np.argmin(starts[tied - 1]))])
    ordered = sorted((int(s) for s in sizes), reverse=True)
    row, col = divmod(int(starts[chosen - 1]), mask.shape[1])
    return {**base, "chosen_label": chosen, "chosen_size": largest, "second_size": ordered[1] if n > 1 else 0, "n_largest_ties": int(len(tied)), "first_row": row, "first_col": col,
            "component_mask": labelled == chosen, "sizes": ordered}


GEOMETRY_COLUMNS = ["ID", "Case", "label", "class", "Device", "width", "height", "predicted_pixels", "predicted_fraction", "n_components", "largest_size", "second_size", "n_largest_ties", "first_row", "first_col",
                    "fallback_full_image", "tight_x0", "tight_y0", "tight_x1", "tight_y1", "exp_x0", "exp_y0", "exp_x1", "exp_y1", "crop_w", "crop_h", "crop_area_fraction", "scale_to_160", "upscaled",
                    "scale_ge_2", "scale_ge_3", "touches_any", "touch_sides", "margin_deficit_pixels", "expansion_clipped"]


def derive_roi(prediction: np.ndarray) -> dict:
    """Box of the automatic ROI from a binary prediction on the DECODED pixel grid: (fallback flag, tight box of the kept component or None, expanded box)."""
    prediction = np.asarray(prediction, dtype=bool)
    height, width = prediction.shape
    selection = select_component(prediction)
    if selection["n_components"] == 0:
        return {"selection": selection, "fallback": True, "tight": None, "expanded": (0, 0, width, height)}
    tight = roi.tight_box(selection["component_mask"])
    return {"selection": selection, "fallback": False, "tight": tight, "expanded": roi.expanded_box(tight, width, height)}


def geometry_row(manifest_row: dict, prediction: np.ndarray) -> dict:
    """The per-image record of the automatic ROI (all numbers derive from the prediction and the image size)."""
    derived = derive_roi(prediction)
    selection, tight, expanded = derived["selection"], derived["tight"], derived["expanded"]
    height, width = prediction.shape
    crop_w, crop_h = expanded[2] - expanded[0], expanded[3] - expanded[1]
    scale = roi.resize_geometry(crop_w, crop_h)["scale"]
    if tight is None:
        contact, margin = {"any": False, "n_sides": 0}, {"deficit_pixels": 0}
    else:
        contact, margin = roi.border_contact(tight, width, height), roi.margins(tight, expanded)
    row = {"ID": manifest_row["ID"], "Case": manifest_row["Case"], "label": int(manifest_row["label"]), "class": manifest_row["Pathology"], "Device": manifest_row["Device"], "width": width, "height": height,
           "predicted_pixels": selection["predicted_pixels"], "predicted_fraction": selection["predicted_pixels"] / (width * height), "n_components": selection["n_components"], "largest_size": selection["chosen_size"],
           "second_size": selection["second_size"], "n_largest_ties": selection["n_largest_ties"], "first_row": selection["first_row"], "first_col": selection["first_col"], "fallback_full_image": derived["fallback"],
           "exp_x0": expanded[0], "exp_y0": expanded[1], "exp_x1": expanded[2], "exp_y1": expanded[3], "crop_w": crop_w, "crop_h": crop_h, "crop_area_fraction": crop_w * crop_h / (width * height),
           "scale_to_160": scale, "upscaled": scale > 1, "scale_ge_2": scale >= 2.0, "scale_ge_3": scale >= 3.0, "touches_any": contact["any"], "touch_sides": contact["n_sides"],
           "margin_deficit_pixels": margin["deficit_pixels"], "expansion_clipped": margin["deficit_pixels"] > 0}
    for name, value in zip(("tight_x0", "tight_y0", "tight_x1", "tight_y1"), tight if tight is not None else (None,) * 4):
        row[name] = value
    return row


def generate_rois(rows: list[dict], images: list, probabilities: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, list]:
    """Automatic ROI inputs of the listed images in EXACTLY the given order: (N, 160, 160, 1) float32, the geometry table and the binary predictions (all predicted pixels on the decoded grid).

    `images` are the decoded grayscale images (PIL, mode L), `probabilities` the (N, 256, 256) maps of the stored U-Net. No expert mask enters: there is no parameter for one."""
    if not (len(rows) == len(images) == len(probabilities)):
        raise AutoRoiError("rows, images and probability maps must describe the same images")
    arrays = np.empty((len(rows), roi.TARGET_SIZE, roi.TARGET_SIZE, 1), dtype=np.float32)
    records, predictions = [], []
    for index, (row, image) in enumerate(zip(rows, images)):
        width, height = image.size
        if (int(row["image_width"]), int(row["image_height"])) != (width, height):
            raise AutoRoiError(f"{row['ID']}: decoded size {(width, height)} differs from the audited size {(int(row['image_width']), int(row['image_height']))}")
        prediction = seg.to_original_mask(probabilities[index], seg.fit_geometry(width, height))
        record = geometry_row(row, prediction)
        arrays[index] = roi.prepare_roi_input(image, (record["exp_x0"], record["exp_y0"], record["exp_x1"], record["exp_y1"]))
        records.append(record)
        predictions.append(prediction)
    return arrays, pd.DataFrame(records, columns=GEOMETRY_COLUMNS), predictions


def inputs_from_geometry(images: list, geometry: pd.DataFrame) -> np.ndarray:
    """The ROI inputs from the FROZEN boxes and the images only (no U-Net, no mask): the classification phase uses this path."""
    arrays = np.empty((len(geometry), roi.TARGET_SIZE, roi.TARGET_SIZE, 1), dtype=np.float32)
    for index, (image, row) in enumerate(zip(images, geometry.itertuples())):
        if image.size != (int(row.width), int(row.height)):
            raise AutoRoiError(f"{row.ID}: image size {image.size} differs from the frozen geometry {(int(row.width), int(row.height))}")
        arrays[index] = roi.prepare_roi_input(image, (int(row.exp_x0), int(row.exp_y0), int(row.exp_x1), int(row.exp_y1)))
    return arrays


def pack_masks(ids: list, masks: list) -> dict:
    """Binary predictions as packed bits (one concatenated byte array plus shapes): compact and independent of the zip container."""
    shapes = np.array([m.shape for m in masks], dtype=np.int64)
    packed = [np.packbits(np.asarray(m, dtype=bool).ravel()) for m in masks]
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in packed])]).astype(np.int64)
    return {"ids": np.array(ids), "shapes": shapes, "offsets": offsets, "bits": np.concatenate(packed) if packed else np.zeros(0, dtype=np.uint8)}


def unpack_masks(archive) -> dict:
    out = {}
    for i, identifier in enumerate(archive["ids"]):
        h, w = (int(v) for v in archive["shapes"][i])
        bits = archive["bits"][int(archive["offsets"][i]):int(archive["offsets"][i + 1])]
        out[str(identifier)] = np.unpackbits(bits)[:h * w].reshape(h, w).astype(bool)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Diagnostics against the validation expert masks (read only AFTER the automatic ROIs are frozen; nothing here feeds back into a crop)
CATEGORIES = ["fallback_full_image", "covers_all_expert_pixels", "covers_90_to_below_100_percent", "covers_more_than_0_to_below_90_percent", "covers_no_expert_pixel"]


def coverage_category(fallback: bool, inside: int, total: int) -> str:
    """Category of the ACTUAL crop by integer arithmetic on pixel counts (no floating-point threshold)."""
    if fallback:
        return "fallback_full_image"
    if inside == total:
        return "covers_all_expert_pixels"
    if inside == 0:
        return "covers_no_expert_pixel"
    if inside * 10 >= total * 9:
        return "covers_90_to_below_100_percent"
    return "covers_more_than_0_to_below_90_percent"


def _inside(expert: np.ndarray, box) -> int:
    if box is None:
        return 0
    x0, y0, x1, y1 = box
    return int(np.count_nonzero(expert[y0:y1, x0:x1]))


def diagnose_image(expert: np.ndarray, all_pixels: np.ndarray, record: dict) -> dict:
    """What the frozen crop of one image does to the expert mask. `record` is the frozen geometry row; the kept component is re-derived from the stored predicted pixels and must equal it."""
    expert, all_pixels = np.asarray(expert, dtype=bool), np.asarray(all_pixels, dtype=bool)
    if expert.shape != all_pixels.shape:
        raise AutoRoiError(f"{record['ID']}: expert mask {expert.shape} and stored prediction {all_pixels.shape} differ in size")
    total = int(expert.sum())
    if total == 0:
        raise AutoRoiError(f"{record['ID']}: empty expert mask")
    labelled, sizes = label_components(all_pixels)
    selection = select_component(all_pixels)
    if (selection["n_components"], selection["chosen_size"], selection["first_row"], selection["first_col"]) != (int(record["n_components"]), int(record["largest_size"]), record["first_row"] if pd.notna(record["first_row"]) else None,
                                                                                                                  record["first_col"] if pd.notna(record["first_col"]) else None):
        raise AutoRoiError(f"{record['ID']}: the kept component re-derived from the stored prediction differs from the frozen geometry")
    fallback = bool(record["fallback_full_image"])
    expanded = (int(record["exp_x0"]), int(record["exp_y0"]), int(record["exp_x1"]), int(record["exp_y1"]))
    inside_actual = _inside(expert, expanded)
    row = {"ID": record["ID"], "expert_pixels": total, "inside_actual_crop": inside_actual, "coverage_actual_crop": inside_actual / total, "category": coverage_category(fallback, inside_actual, total)}
    nan = float("nan")
    allpix_box = seg.box_of(all_pixels)
    row["inside_all_pixels_box"] = _inside(expert, allpix_box) if allpix_box is not None else None
    row["coverage_all_pixels_box"] = (row["inside_all_pixels_box"] / total) if allpix_box is not None else nan
    allpix_expanded = roi.expanded_box(allpix_box, expert.shape[1], expert.shape[0]) if allpix_box is not None else None
    row["inside_all_pixels_box_plus_20"] = _inside(expert, allpix_expanded) if allpix_expanded is not None else None
    row["coverage_all_pixels_box_plus_20"] = (row["inside_all_pixels_box_plus_20"] / total) if allpix_expanded is not None else nan
    if fallback:
        row.update(inside_component_tight_box=None, coverage_component_tight_box=nan, component_dice=nan, component_iou=nan, overlap_chosen=None, best_overlap=None, n_components_overlapping=None,
                   chosen_is_best_overlap=None, chosen_zero_but_other_overlaps=None)
        return row
    tight = (int(record["tight_x0"]), int(record["tight_y0"]), int(record["tight_x1"]), int(record["tight_y1"]))
    component = selection["component_mask"]
    metrics = seg.dice_iou(component, expert)
    overlaps = np.bincount(labelled[expert & (labelled > 0)], minlength=len(sizes) + 1)[1:]
    chosen_overlap = int(overlaps[selection["chosen_label"] - 1])
    best = int(overlaps.max()) if len(overlaps) else 0
    row.update(inside_component_tight_box=_inside(expert, tight), coverage_component_tight_box=_inside(expert, tight) / total, component_dice=metrics["dice"], component_iou=metrics["iou"], overlap_chosen=chosen_overlap,
               best_overlap=best, n_components_overlapping=int((overlaps > 0).sum()), chosen_is_best_overlap=chosen_overlap >= best, chosen_zero_but_other_overlaps=bool(chosen_overlap == 0 and best > 0))
    return row


def _groups(frame: pd.DataFrame):
    yield "all", "all", frame
    for name in ("benign", "malignant"):
        yield "class", name, frame[frame["class"] == name]
    for device in sorted(frame["Device"].unique()):
        yield "device", device, frame[frame["Device"] == device]


CROP_FLAGS = [("fallback_full_image", lambda d: d["fallback_full_image"], "all", "the predicted mask is empty: the whole image is the crop (logged fallback)"),
              ("multiple_components", lambda d: d["n_components"] > 1, "all", "more than one 8-connected component in the prediction (only the largest is kept)"),
              ("tie_between_largest_components", lambda d: d["n_largest_ties"] > 1, "all", "several components share the largest pixel count (row-major rule decides)"),
              ("expert_coverage_below_90_percent", lambda d: d["category"].isin(["covers_more_than_0_to_below_90_percent", "covers_no_expert_pixel"]), "non_fallback", "less than 90 % of the expert-mask pixels lie inside the actual crop"),
              ("expert_coverage_below_100_percent", lambda d: d["category"] != "covers_all_expert_pixels", "non_fallback", "not every expert-mask pixel lies inside the actual crop"),
              ("expert_coverage_zero", lambda d: d["category"] == "covers_no_expert_pixel", "non_fallback", "no expert-mask pixel lies inside the actual crop (the crop misses the lesion)"),
              ("component_dice_below_0.5", lambda d: d["component_dice"] < 0.5, "non_fallback", "Dice between the kept component and the expert mask below 0.5"),
              ("crop_enlarged_by_a_factor_of_2_or_more", lambda d: d["scale_ge_2"], "all", "the crop is enlarged by a factor of at least 2 (interpolation, visibly blurred)"),
              ("crop_enlarged_by_a_factor_of_3_or_more", lambda d: d["scale_ge_3"], "all", "the crop is enlarged by a factor of at least 3"),
              ("expansion_cut_by_the_image_border", lambda d: d["expansion_clipped"], "non_fallback", "the 20 % expansion is shortened by the image border"),
              ("tight_box_touches_an_image_border", lambda d: d["touches_any"], "non_fallback", "the tight box of the kept component touches an image border"),
              ("crop_is_at_least_90_percent_of_the_image", lambda d: d["crop_area_fraction"] >= 0.9, "non_fallback", "the crop covers at least 90 % of the image area (practically a full image)")]


def crop_flag_table(diag: pd.DataFrame) -> pd.DataFrame:
    """Counts with denominators (images and cases) by group; the denominator basis of every flag is explicit (all images, or the images without fallback). Zero denominators are undefined (empty)."""
    rows = []
    for group_type, group, frame in _groups(diag):
        for flag, rule, basis, description in CROP_FLAGS:
            base = frame if basis == "all" else frame[~frame["fallback_full_image"]]
            flagged = rule(base).fillna(False).astype(bool) if len(base) else pd.Series([], dtype=bool)
            n_flagged = int(flagged.sum())
            rows.append({"group_type": group_type, "group": group, "flag": flag, "description": description, "denominator_basis": "all images of the group" if basis == "all" else "images without fallback",
                         "n_flagged_images": n_flagged, "n_flagged_cases": int(base.loc[flagged.to_numpy(), "Case"].nunique()) if n_flagged else 0, "denominator_images": int(len(base)),
                         "denominator_cases": int(base["Case"].nunique()), "share_of_images": (n_flagged / len(base)) if len(base) else None})
    return pd.DataFrame(rows)


def category_table(diag: pd.DataFrame) -> pd.DataFrame:
    """The five crop categories (fallback listed on its own) by group, with images and cases."""
    rows = []
    for group_type, group, frame in _groups(diag):
        for category in CATEGORIES:
            part = frame[frame["category"] == category]
            rows.append({"group_type": group_type, "group": group, "category": category, "n_images": int(len(part)), "n_cases": int(part["Case"].nunique()), "denominator_images": int(len(frame)),
                         "denominator_cases": int(frame["Case"].nunique()), "share_of_images": (len(part) / len(frame)) if len(frame) else None})
    return pd.DataFrame(rows)


def coverage_summary(diag: pd.DataFrame) -> pd.DataFrame:
    """Share of the expert mask inside the actual crop. Two rows per group, clearly labelled: WITHOUT the fallback images (the real localisation quality) and WITH the fallback counted as full
    coverage (a whole-image crop trivially contains the lesion; this row must not be read as localisation quality)."""
    rows = []
    for group_type, group, frame in _groups(diag):
        for basis, part in (("without_fallback_images", frame[~frame["fallback_full_image"]]), ("fallback_images_counted_as_full_coverage_NOT_a_localisation_quality", frame)):
            values = part["coverage_actual_crop"].to_numpy(dtype=float)
            rows.append({"group_type": group_type, "group": group, "basis": basis, "n_images": int(len(part)), "n_cases": int(part["Case"].nunique()), **seg.distribution(values),
                         "n_below_100_percent": int((part["category"].isin(CATEGORIES[2:])).sum()), "n_below_90_percent": int((part["category"].isin(CATEGORIES[3:])).sum()), "n_zero": int((part["category"] == CATEGORIES[4]).sum())})
    return pd.DataFrame(rows)


DEFINITIONS = [("all_predicted_pixels_box", "coverage_all_pixels_box", "box around ALL predicted pixels, no expansion (the 4F-A measure)"),
               ("all_predicted_pixels_box_plus_20_percent", "coverage_all_pixels_box_plus_20", "box around ALL predicted pixels expanded by 20 % (not used for any crop)"),
               ("largest_component_tight_box", "coverage_component_tight_box", "tight box of the kept (largest) component, no expansion"),
               ("largest_component_plus_20_percent_actual_crop", "coverage_actual_crop", "the ACTUAL crop of this step: largest component plus 20 % (fallback images excluded)")]


def definition_table(diag: pd.DataFrame) -> pd.DataFrame:
    """The same validation images judged with the four box definitions, on the images WITHOUT fallback (non-empty predictions): counts of coverage below 100 %, below 90 % and zero. The 4F-A figure is the first
    definition; the crop of this step is the last."""
    base = diag[~diag["fallback_full_image"]]
    rows = []
    for name, column, description in DEFINITIONS:
        values = base[column].to_numpy(dtype=float)
        rows.append({"definition": name, "description": description, "n_images_without_fallback": int(len(base)), "n_below_100_percent": int((values < 1.0).sum()), "n_below_90_percent": int((values < 0.9).sum()),
                     "n_zero": int((values == 0.0).sum()), "mean_coverage": float(np.nanmean(values)), "median_coverage": float(np.nanmedian(values))})
    return pd.DataFrame(rows)


def component_selection_table(diag: pd.DataFrame) -> pd.DataFrame:
    """How the 'largest component' rule behaves where it matters (images with more than one component), judged afterwards against the expert mask."""
    multi = diag[(diag["n_components"] > 1)]
    rows = []
    for group_type, group, frame in _groups(multi):
        rows.append({"group_type": group_type, "group": group, "n_images_with_several_components": int(len(frame)), "n_cases": int(frame["Case"].nunique()),
                     "n_largest_is_also_best_overlap_with_expert": int(frame["chosen_is_best_overlap"].fillna(False).astype(bool).sum()),
                     "n_chosen_overlaps_expert_but_another_component_overlaps_more": int(((frame["overlap_chosen"] > 0) & (~frame["chosen_is_best_overlap"].fillna(True).astype(bool))).sum()),
                     "n_chosen_has_no_overlap_but_another_component_does": int(frame["chosen_zero_but_other_overlaps"].fillna(False).astype(bool).sum()),
                     "n_no_component_overlaps_expert": int((frame["best_overlap"] == 0).sum()), "n_ties_for_the_largest": int((frame["n_largest_ties"] > 1).sum())})
    allp = diag
    rows.append({"group_type": "all_images", "group": "images with exactly one component", "n_images_with_several_components": int((allp["n_components"] == 1).sum()), "n_cases": int(allp.loc[allp["n_components"] == 1, "Case"].nunique()),
                 "n_largest_is_also_best_overlap_with_expert": None, "n_chosen_overlaps_expert_but_another_component_overlaps_more": None, "n_chosen_has_no_overlap_but_another_component_does": None,
                 "n_no_component_overlaps_expert": int((allp.loc[allp["n_components"] == 1, "best_overlap"] == 0).sum()), "n_ties_for_the_largest": 0})
    return pd.DataFrame(rows)
