"""Helpers for step 4F-A: a first fixed segmentation of the annotated lesion on full ultrasound images (data geometry, mapping back to the original pixels, metrics, plan).

WHAT IS SEGMENTED. The network predicts the pixels of the annotated lesion, not benign / malignant. The only model input is the supplied ultrasound image; Pathology, Case, Device, BI-RADS,
bounding boxes and masks are never input features (the expert mask is the training TARGET of this step, and the reference of the quality measurement).

Geometry contract (image and mask share ONE geometry; the decoded sizes steer it, never the CSV Width/Height columns)
1. The whole image is fitted into 256 x 256 with the existing rule of busbra.image_processing.resize_with_padding: proportional scaling (Pillow ImageOps.contain, bilinear), centred on black.
   Pillow's contain also ENLARGES an image that is smaller than the target (not the case for BUS-BRA: every image has a side above 256 pixels).
2. The mask is transformed with exactly the same fitted size and padding offsets, but with nearest-neighbour interpolation; its values stay {0, 1}. The padding is background (0).
3. fit_geometry() records, for every image, the decoded size, the fitted size, the padding offsets and the two scale factors, so that every prediction can be traced back to original pixels.
4. Mapping back: crop the fitted region out of the 256 x 256 probability map (the padding is discarded, never counted), resize it bilinearly to the decoded size (float32 map), then apply the
   binary rule probability >= 0.5. Pixel centres follow x_model = pad_x + (x + 0.5) * scale_x - 0.5 (and the inverse).

The module imports the standard library, numpy, pandas, Pillow, scipy and the light project helpers at the top level; Keras lives in busbra.segmentation_model.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from busbra import image_processing as ip
from busbra import roi

# ---- the registered design ------------------------------------------------------------------------------------------
SIZE = 256
SEED = 7
SEED_OFFSETS = {"flip": 1, "shuffle": 3}  # the mapping of the classification studies (flip s+1, shuffle s+3); no dropout and no other random layer exist in this network
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
THRESHOLD = 0.5  # fixed binary rule of this first attempt: probability >= 0.5
DICE_SMOOTH = 1.0
FILTERS = (16, 32, 64, 128)
BOTTLENECK_FILTERS = 256
SCORE_TOLERANCE = 1e-5
RUN_NAME = "unet_256_seed7"
DICE_BINS = [round(0.05 * k, 2) for k in range(21)]  # 0.00 ... 1.00; the last bin includes 1.00
QUANTILES = (("p05", 5), ("q1", 25), ("median", 50), ("q3", 75), ("p95", 95))
SMALL_GROUP_NOTE_BELOW = 10  # groups with fewer images are listed with a note and not interpreted
PREVIEW_ERROR_N = 12


class SegmentationDataError(Exception):
    """A train/validation pair violates the data contract; the message lists the problems. Nothing is repaired silently."""


def run_paths(project_root: Path) -> dict:
    """All files of the run in the own namespace (reports/tables/segmentation, models/segmentation, local previews in data/processed)."""
    tables = Path(project_root) / "reports" / "tables" / "segmentation" / "unet_seed7"
    models = Path(project_root) / "models" / "segmentation" / "unet_seed7"
    processed = Path(project_root) / "data" / "processed"
    return {"table_dir": tables, "model_dir": models, "history": tables / "history.csv", "config": tables / "config.json",
            "summary": tables / "metrics_summary.csv", "groups": tables / "metrics_by_group.csv", "flags": tables / "quality_flags.csv", "box": tables / "box_deviation_summary.csv",
            "histogram": tables / "dice_histogram.csv", "selection": tables / "preview_selection.csv", "per_image": tables / "per_image_train_validation.csv",
            "checkpoint": models / f"{RUN_NAME}_best.keras", "model_summary": models / "model_summary.txt",
            "preview_representative": processed / "segmentation_preview_representative.png", "preview_errors": processed / "segmentation_preview_error_analysis_validation.png"}


# ---------------------------------------------------------------------------------------------------------------------
# Seeds, epoch order and the synchronous flip (pure numpy; used by the Keras batch generator and by the tests)
def seed_plan(seed: int = SEED) -> dict:
    return {"study_seed": seed, "PYTHONHASHSEED": str(seed), "python_numpy_tensorflow_keras": seed,
            "weight_initialisation": f"fresh construction after keras.utils.set_random_seed({seed})",
            "train_flip": seed + SEED_OFFSETS["flip"], "train_shuffle": seed + SEED_OFFSETS["shuffle"],
            "epoch_streams": "numpy default_rng([seed, epoch]) for the shuffle order and for the flip decisions of every epoch (index = epoch, first epoch 0); the flip decision is indexed by IMAGE, not by position",
            "random_layers_in_the_network": "none (no dropout; the augmentation is done on image/mask pairs outside the network)"}


def epoch_order(shuffle_seed: int, epoch: int, n: int) -> np.ndarray:
    return np.random.default_rng([shuffle_seed, epoch]).permutation(n)


def epoch_flips(flip_seed: int, epoch: int, n: int) -> np.ndarray:
    """One boolean per IMAGE for this epoch: True = flip image and mask horizontally (probability one half)."""
    return np.random.default_rng([flip_seed, epoch]).random(n) < 0.5


def flip_pair_batch(images: np.ndarray, masks: np.ndarray, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal flip of the flagged samples; the SAME decision acts on the image and on its mask. Inputs are (B, H, W, 1); copies are returned."""
    if images.shape != masks.shape or len(flags) != len(images):
        raise ValueError("images, masks and flags must describe the same batch")
    out_x, out_y = images.copy(), masks.copy()
    flags = np.asarray(flags, dtype=bool)
    out_x[flags] = images[flags][:, :, ::-1, :]
    out_y[flags] = masks[flags][:, :, ::-1, :]
    return out_x, out_y


# ---------------------------------------------------------------------------------------------------------------------
# Geometry: forward (image and mask), the record per image, and the mapping back to the original pixels
def fit_geometry(width: int, height: int, size: int = SIZE) -> dict:
    """The geometry of resize_with_padding for an image of this DECODED size (Pillow's own computation on a dummy image): fitted size, centring offsets, scale factors."""
    fitted_width, fitted_height = ImageOps.contain(Image.new("L", (width, height)), (size, size), method=ip.INTERPOLATION).size
    if min(fitted_width, fitted_height) < 1:
        raise ValueError(f"image of size {(width, height)} degenerates to {(fitted_width, fitted_height)}")
    return {"width": int(width), "height": int(height), "fitted_width": int(fitted_width), "fitted_height": int(fitted_height), "pad_x": int((size - fitted_width) // 2),
            "pad_y": int((size - fitted_height) // 2), "scale_x": fitted_width / width, "scale_y": fitted_height / height, "size": int(size)}


def prepare_image(image: Image.Image, size: int = SIZE) -> np.ndarray:
    """The existing rule (whole image, bilinear, centred on black) and float scaling: (size, size, 1) float32 in [0, 1]."""
    return ip.to_unit_float32(ip.resize_with_padding(image, size))[..., np.newaxis]


def prepare_mask(mask: np.ndarray, size: int = SIZE) -> np.ndarray:
    """The same fitted size and padding offsets as the image, nearest-neighbour interpolation: (size, size, 1) uint8 with the values {0, 1} (padding = 0)."""
    mask = np.asarray(mask, dtype=bool)
    source = Image.fromarray(mask.astype(np.uint8))
    fitted = ImageOps.contain(source, (size, size), method=Image.Resampling.NEAREST)
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    array = np.asarray(canvas, dtype=np.uint8)
    if not set(np.unique(array).tolist()) <= {0, 1}:
        raise ValueError("the transformed mask is not binary")
    return array[..., np.newaxis]


def image_area(canvas: np.ndarray, geometry: dict) -> np.ndarray:
    """The fitted image region of a 2-D map on the padded canvas (the padding is excluded)."""
    px, py, fw, fh = geometry["pad_x"], geometry["pad_y"], geometry["fitted_width"], geometry["fitted_height"]
    region = np.asarray(canvas)[py:py + fh, px:px + fw]
    if region.shape != (fh, fw):
        raise ValueError(f"canvas {np.asarray(canvas).shape} does not contain the fitted region {(fh, fw)} at {(py, px)}")
    return region


def to_original_probability(probability: np.ndarray, geometry: dict) -> np.ndarray:
    """Probability map of the padded canvas -> (height, width) float32 map on the original pixel grid: crop the fitted region, resize bilinearly to the decoded size."""
    region = np.ascontiguousarray(image_area(np.asarray(probability, dtype=np.float32), geometry))
    width, height = geometry["width"], geometry["height"]
    if region.shape == (height, width):
        return region.copy()
    return np.asarray(Image.fromarray(region).resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.float32)


def to_original_mask(probability: np.ndarray, geometry: dict, threshold: float = THRESHOLD) -> np.ndarray:
    """Binary mask on the original pixel grid: probability map mapped back (no padding), then probability >= threshold."""
    return to_original_probability(probability, geometry) >= threshold


# ---------------------------------------------------------------------------------------------------------------------
# Metrics and per-image analysis on the original pixel grid
def dice_iou(prediction: np.ndarray, truth: np.ndarray) -> dict:
    """Pixel Dice and IoU of two boolean masks. A zero denominator is UNDEFINED (None), never zero (the expert masks are never empty, so it does not occur here)."""
    prediction, truth = np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool)
    tp = int(np.count_nonzero(prediction & truth))
    p, g = int(np.count_nonzero(prediction)), int(np.count_nonzero(truth))
    union = p + g - tp
    return {"dice": (2 * tp / (p + g)) if p + g else None, "iou": (tp / union) if union else None, "tp": tp, "pred_pixels": p, "truth_pixels": g}


def box_of(mask: np.ndarray):
    """(x0, y0, x1, y1) half-open around ALL positive pixels, or None for an empty mask (the box of an empty prediction is undefined)."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def box_iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def box_deviation(predicted_box, expert_box, expert_mask: np.ndarray) -> dict:
    """Location and size of the predicted box relative to the expert box (original pixels, half-open boxes). Undefined (None) for an empty prediction.

    centre offsets are signed (predicted minus expert), in units of the EXPERT box width / height; the centre distance is relative to the expert box diagonal; the ratios are predicted / expert;
    expert_inside = share of the positive expert-mask pixels that lie inside the predicted box."""
    keys = ["pred_x0", "pred_y0", "pred_x1", "pred_y1", "box_iou", "center_dx_rel", "center_dy_rel", "center_distance_rel", "width_ratio", "height_ratio", "area_ratio", "expert_inside_pred_box"]
    if predicted_box is None:
        return {k: None for k in keys}
    pw, ph = predicted_box[2] - predicted_box[0], predicted_box[3] - predicted_box[1]
    ew, eh = expert_box[2] - expert_box[0], expert_box[3] - expert_box[1]
    dx = (predicted_box[0] + predicted_box[2]) / 2 - (expert_box[0] + expert_box[2]) / 2
    dy = (predicted_box[1] + predicted_box[3]) / 2 - (expert_box[1] + expert_box[3]) / 2
    x0, y0, x1, y1 = predicted_box
    inside = float(np.count_nonzero(expert_mask[y0:y1, x0:x1])) / float(np.count_nonzero(expert_mask))
    return {"pred_x0": x0, "pred_y0": y0, "pred_x1": x1, "pred_y1": y1, "box_iou": box_iou(predicted_box, expert_box), "center_dx_rel": dx / ew, "center_dy_rel": dy / eh,
            "center_distance_rel": math.hypot(dx, dy) / math.hypot(ew, eh), "width_ratio": pw / ew, "height_ratio": ph / eh, "area_ratio": (pw * ph) / (ew * eh), "expert_inside_pred_box": inside}


def analyse_prediction(probability: np.ndarray, expert_model: np.ndarray, expert_original: np.ndarray, geometry: dict, threshold: float = THRESHOLD) -> dict:
    """Everything measured for one image: Dice / IoU on the model grid (padding included) and on the ORIGINAL pixel grid after the mapping back, empty prediction, number of 8-connected components,
    predicted share of the image area, round trip of the expert mask (the resolution ceiling of the 256 grid) and the box deviation."""
    probability = np.asarray(probability, dtype=np.float32)
    prediction_model = probability >= threshold
    model = dice_iou(prediction_model, np.asarray(expert_model, dtype=bool))
    prediction_original = to_original_mask(probability, geometry, threshold)
    original = dice_iou(prediction_original, expert_original)
    roundtrip = dice_iou(to_original_mask(np.asarray(expert_model, dtype=np.float32), geometry, threshold), expert_original)
    area = geometry["width"] * geometry["height"]
    sizes = roi.components(prediction_original)
    expert_box = box_of(expert_original)
    row = {"dice_model": model["dice"], "iou_model": model["iou"], "pred_pixels_model": model["pred_pixels"], "expert_pixels_model": model["truth_pixels"],
           "dice_original": original["dice"], "iou_original": original["iou"], "pred_pixels_original": original["pred_pixels"], "pred_fraction_original": original["pred_pixels"] / area,
           "pred_empty": original["pred_pixels"] == 0, "pred_components": len(sizes), "pred_over_half_of_image": original["pred_pixels"] / area > 0.5,
           "roundtrip_dice": roundtrip["dice"], "roundtrip_iou": roundtrip["iou"], "expert_x0": expert_box[0], "expert_y0": expert_box[1], "expert_x1": expert_box[2], "expert_y1": expert_box[3]}
    row.update(box_deviation(box_of(prediction_original), expert_box, expert_original))
    return row


# ---------------------------------------------------------------------------------------------------------------------
# Loading the train / validation pairs (the manifest split decides which files are opened)
GEOMETRY_COLUMNS = ["ID", "Case", "label", "class", "Device", "subset", "width", "height", "fitted_width", "fitted_height", "pad_x", "pad_y", "scale_x", "scale_y",
                    "expert_pixels_original", "expert_fraction_original", "expert_pixels_model", "roundtrip_dice_expert_mask"]


def load_pairs(project_root: Path, rows: list[dict], subset: str) -> dict:
    """Prepare every listed pair in EXACTLY the given order.

    Per pair: image (mode 'L') and mask (PNG mode '1', bool, not empty, same DECODED size as the image and as the audited size of the manifest) are linked through the manifest row; the image and the mask
    go through the common geometry; the transformed mask must stay non-empty. Returns images (N, 256, 256, 1) float32, masks (N, 256, 256, 1) uint8, the original masks (list of bool arrays) and the geometry
    table. Only the paths of the given rows are opened. Any violation is collected and raised as SegmentationDataError; there is no repair and no exclusion."""
    project_root = Path(project_root)
    images = np.empty((len(rows), SIZE, SIZE, 1), dtype=np.float32)
    masks = np.empty((len(rows), SIZE, SIZE, 1), dtype=np.uint8)
    originals, records, problems = [], [], []
    for index, r in enumerate(rows):
        try:
            image = ip.load_grayscale_image(project_root / r["image_relpath"])
            width, height = image.size
            description, array = roi.read_mask_raw(project_root / r["mask_relpath"])
            expert = roi.decode_mask(description, array, expected_size=(width, height))
        except (ip.ImageLoadError, roi.MaskError) as exc:
            problems.append(f"{r['ID']}: {exc}")
            originals.append(None)
            continue
        audited = (int(r["image_width"]), int(r["image_height"]), int(r["mask_width"]), int(r["mask_height"]))
        if audited != (width, height, width, height):
            problems.append(f"{r['ID']}: decoded sizes image {(width, height)} / mask {(description['width'], description['height'])} differ from the audited {audited}")
            originals.append(None)
            continue
        geometry = fit_geometry(width, height)
        images[index] = prepare_image(image)
        masks[index] = prepare_mask(expert)
        if not masks[index].any():
            problems.append(f"{r['ID']}: the mask is empty after the transformation to {SIZE} x {SIZE}")
        originals.append(expert)
        pixels = int(np.count_nonzero(expert))
        records.append({"ID": r["ID"], "Case": r["Case"], "label": int(r["label"]), "class": r["Pathology"], "Device": r["Device"], "subset": subset, "width": width, "height": height,
                        "fitted_width": geometry["fitted_width"], "fitted_height": geometry["fitted_height"], "pad_x": geometry["pad_x"], "pad_y": geometry["pad_y"],
                        "scale_x": geometry["scale_x"], "scale_y": geometry["scale_y"], "expert_pixels_original": pixels, "expert_fraction_original": pixels / (width * height),
                        "expert_pixels_model": int(masks[index].sum()), "roundtrip_dice_expert_mask": dice_iou(to_original_mask(masks[index][..., 0].astype(np.float32), geometry), expert)["dice"]})
    if problems:
        raise SegmentationDataError(f"{len(problems)} of {len(rows)} {subset} pairs violate the contract (no repair, no exclusion): " + " | ".join(problems[:8]) + (" ..." if len(problems) > 8 else ""))
    if not (np.isfinite(images).all() and 0.0 <= float(images.min()) and float(images.max()) <= 1.0):
        raise SegmentationDataError(f"{subset}: images are not finite values in [0, 1]")
    return {"images": images, "masks": masks, "originals": originals, "geometry": pd.DataFrame(records, columns=GEOMETRY_COLUMNS)}


# ---------------------------------------------------------------------------------------------------------------------
# Tables from the per-image analysis (descriptive; images of one case are dependent; groups with few images are not interpreted)
def distribution(values) -> dict:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return {"n_defined": 0, "mean": None, "sd_ddof1": None, "min": None, **{k: None for k, _ in QUANTILES}, "max": None}
    return {"n_defined": int(len(values)), "mean": float(values.mean()), "sd_ddof1": float(values.std(ddof=1)) if len(values) > 1 else None, "min": float(values.min()),
            **{k: float(np.percentile(values, q)) for k, q in QUANTILES}, "max": float(values.max())}


def metric_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    """Per subset and space (model grid with padding / original pixel grid) the distribution of the per-image Dice and IoU, plus the round trip of the expert mask (resolution ceiling)."""
    rows = []
    for subset in ("train", "validation"):
        part = per_image[per_image["subset"] == subset]
        for space, metric, column in (("model_grid_256", "dice", "dice_model"), ("model_grid_256", "iou", "iou_model"), ("original_area", "dice", "dice_original"), ("original_area", "iou", "iou_original"),
                                      ("original_area", "roundtrip_dice_of_the_expert_mask", "roundtrip_dice")):
            rows.append({"subset": subset, "space": space, "metric": metric, "n_images": int(len(part)), "n_cases": int(part["Case"].nunique()), **distribution(part[column])})
    return pd.DataFrame(rows)


def group_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    """Original-area Dice and IoU by class and by device, per subset, with image and case denominators (descriptive; no fairness statement; small groups are marked)."""
    rows = []
    for subset in ("train", "validation"):
        part = per_image[per_image["subset"] == subset]
        groups = [("all", "all", part)] + [("class", c, part[part["class"] == c]) for c in ("benign", "malignant")] + [("device", d, part[part["Device"] == d]) for d in sorted(part["Device"].unique())]
        for group_type, group, selection in groups:
            dice, iou = distribution(selection["dice_original"]), distribution(selection["iou_original"])
            rows.append({"subset": subset, "group_type": group_type, "group": group, "n_images": int(len(selection)), "n_cases": int(selection["Case"].nunique()),
                         "dice_mean": dice["mean"], "dice_median": dice["median"], "dice_min": dice["min"], "dice_max": dice["max"], "iou_mean": iou["mean"], "iou_median": iou["median"],
                         "n_empty_predictions": int(selection["pred_empty"].sum()), "note": f"fewer than {SMALL_GROUP_NOTE_BELOW} images: not interpreted" if len(selection) < SMALL_GROUP_NOTE_BELOW else ""})
    return pd.DataFrame(rows)


FLAGS = [("empty_prediction", lambda d: d["pred_empty"], "all", "no pixel reaches the probability 0.5 (the predicted mask is empty)"),
         ("multiple_components", lambda d: d["pred_components"] > 1, "all", "the predicted mask has more than one 8-connected component"),
         ("prediction_over_half_of_the_image", lambda d: d["pred_over_half_of_image"], "all", "the predicted mask covers more than half of the original image area"),
         ("dice_below_0.5", lambda d: d["dice_original"] < 0.5, "all", "original-area Dice below 0.5"),
         ("dice_below_0.25", lambda d: d["dice_original"] < 0.25, "all", "original-area Dice below 0.25"),
         ("box_without_overlap", lambda d: d["box_iou"] == 0, "non_empty", "the predicted box does not overlap the expert box (box IoU 0)"),
         ("box_area_ratio_below_0.5", lambda d: d["area_ratio"] < 0.5, "non_empty", "predicted box area below half of the expert box area"),
         ("box_area_ratio_above_2", lambda d: d["area_ratio"] > 2.0, "non_empty", "predicted box area above twice the expert box area"),
         ("expert_inside_predicted_box_below_0.9", lambda d: d["expert_inside_pred_box"] < 0.9, "non_empty", "less than 90 % of the expert-mask pixels lie inside the predicted box")]


def flag_table(per_image: pd.DataFrame) -> pd.DataFrame:
    """Counts with denominators (images and cases of the group; the box flags use the images with a non-empty prediction as denominator). Rates with a zero denominator are undefined (empty)."""
    rows = []
    for subset in ("train", "validation"):
        part = per_image[per_image["subset"] == subset]
        groups = [("all", "all", part)] + [("class", c, part[part["class"] == c]) for c in ("benign", "malignant")] + [("device", d, part[part["Device"] == d]) for d in sorted(part["Device"].unique())]
        for group_type, group, selection in groups:
            for flag, rule, applies_to, description in FLAGS:
                base = selection if applies_to == "all" else selection[~selection["pred_empty"]]
                flagged = rule(base) if len(base) else pd.Series([], dtype=bool)
                n_flagged = int(flagged.sum())
                rows.append({"subset": subset, "group_type": group_type, "group": group, "flag": flag, "description": description, "denominator_basis": "all images of the group" if applies_to == "all" else "images with a non-empty predicted mask",
                             "n_flagged_images": n_flagged, "n_flagged_cases": int(base.loc[flagged.index[flagged.to_numpy()], "Case"].nunique()) if n_flagged else 0, "denominator_images": int(len(base)),
                             "denominator_cases": int(base["Case"].nunique()), "share_of_images": (n_flagged / len(base)) if len(base) else None})
    return pd.DataFrame(rows)


BOX_METRICS = ["box_iou", "center_dx_rel", "center_dy_rel", "center_distance_rel", "width_ratio", "height_ratio", "area_ratio", "expert_inside_pred_box"]


def box_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    """Location and size deviation of the predicted box (images with a non-empty prediction only; the denominators are listed)."""
    rows = []
    for subset in ("train", "validation"):
        part = per_image[(per_image["subset"] == subset) & (~per_image["pred_empty"])]
        for metric in BOX_METRICS:
            rows.append({"subset": subset, "metric": metric, "n_images_with_prediction": int(len(part)), "n_images_in_subset": int((per_image["subset"] == subset).sum()), **distribution(part[metric])})
    return pd.DataFrame(rows)


def dice_histogram(per_image: pd.DataFrame) -> pd.DataFrame:
    """Counts of the original-area Dice in bins of width 0.05 (the last bin includes 1.0) per subset and class: the data of the distribution figure."""
    rows = []
    edges = np.array(DICE_BINS)
    for subset in ("train", "validation"):
        for group in ("all", "benign", "malignant"):
            part = per_image[(per_image["subset"] == subset) & ((per_image["class"] == group) | (group == "all"))]
            counts, _ = np.histogram(part["dice_original"].to_numpy(dtype=float), bins=edges)
            for left, right, count in zip(edges[:-1], edges[1:], counts):
                rows.append({"subset": subset, "group": group, "bin_left": float(left), "bin_right": float(right), "n_images": int(count), "n_images_in_group": int(len(part))})
    return pd.DataFrame(rows)


def select_representative(table: pd.DataFrame) -> list[tuple[str, str]]:
    """`table` of ONE subset with ID, class, Device and expert_fraction_original."""
    selected = []
    for device in sorted(table["Device"].unique()):
        for name in ("benign", "malignant"):
            group = table[(table["Device"] == device) & (table["class"] == name)]
            if group.empty:
                continue
            median = float(np.median(group["expert_fraction_original"].to_numpy()))
            pick = group.assign(_distance=(group["expert_fraction_original"] - median).abs()).sort_values(["_distance", "ID"]).iloc[0]["ID"]
            selected.append((pick, f"median expert-mask share of {device} / {name}"))
    return selected


def select_error_analysis(per_image: pd.DataFrame, n: int = PREVIEW_ERROR_N) -> list[tuple[str, str]]:
    part = per_image[per_image["subset"] == "validation"].sort_values(["dice_original", "ID"], ascending=[True, True]).head(n)
    return [(row.ID, f"error analysis: validation Dice rank {rank} from below") for rank, row in enumerate(part.itertuples(), start=1)]


# ---------------------------------------------------------------------------------------------------------------------
# The immutable study plan
EXPECTED_PARAMETERS = {"total": 1_945_233, "trainable": 1_942_289, "non_trainable": 2_944}  # recomputed by the tests with a dry build of the network


