"""Expert-mask ROI helpers (step 4D): mask decoding, box geometry, crop and preparation of one ROI input.

Information condition (read this before using the functions)
-------------------------------------------------------------
The ROI is derived from the lesion mask drawn by an expert. It is therefore a different information condition than the full-image branch: applying it later
requires a manual mask. A classification result obtained with these crops must not be reported as a pure image-to-score pipeline. The mask annotates the lesion;
it is neither a cancer label (benign lesions have masks too) nor a marker of histologically healthy background. Only the crop of the UNMASKED original image
enters a later model: no mask channel, no blackening outside the lesion, no contrast stretching, no OCR, no marker removal, no BBOX or BI-RADS metadata.

Geometry contract (all coordinates are original pixel coordinates, half-open: [x0, x1) x [y0, y1), w = x1 - x0, h = y1 - y0)
1. tight box = axis-parallel box around ALL positive mask pixels (all connected components included). An empty mask is an error, never replaced by a box.
2. expansion on every side by 20 % of the box width (left/right) or height (top/bottom), computed in exact rational arithmetic:
   left = floor(x0 - w/5), right = ceil(x1 + w/5), top = floor(y0 - h/5), bottom = ceil(y1 + h/5). Because w/5 is added after x0 resp. x1 (integers) this equals a margin of
   ceil(w/5) pixels on both sides (e.g. w = 7: 1.4 -> 2 pixels, w = 5: exactly 1 pixel, w = 1: 1 pixel); the expansion is therefore at least 20 % on every side.
3. clip to [0, W] x [0, H]. A side that touches the image border is not extended beyond it (no mirroring, no padding with invented pixels); the clipping is recorded.
4. crop the unmasked original image to the expanded box, then prepare it with the existing aspect-ratio preserving rule (busbra.image_processing.resize_with_padding:
   whole crop fitted into the square, bilinear, centred on black) to 160 x 160 x 1 float32 in [0, 1]. Pillow's ImageOps.contain also ENLARGES a crop that is smaller than the
   target; the scale factor is reported per image (a factor above 1 means interpolation, not new information).
The 20 % are a fixed experimental choice: not optimised on validation data and not a millimetre value.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from busbra import image_processing as ip

MARGIN = Fraction(1, 5)  # 20 % of the box edge on each side
CONNECTIVITY = 8  # a component is an 8-connected set of positive pixels
TARGET_SIZE = 160
ACCEPTED_MASK_ENCODING = {"png_mode": "1", "array_dtype": "bool", "values": [False, True]}
DIRECTION_RULE = "True is the lesion iff the median True share over the training masks is below one half (the typical mask covers a small part of the image); masks above one half are reported as findings, not repaired"


class MaskError(Exception):
    """A mask (or a mask/image pair) violates the contract; the message names the file and the reason. Nothing is repaired silently."""


# ---------------------------------------------------------------------------------------------------------------------
# Raw encoding (inspected BEFORE any binarisation) and the one accepted conversion
def read_mask_raw(path: Path) -> tuple[dict, np.ndarray]:
    """Open one mask PNG read-only and describe its raw encoding: PNG mode, array dtype, distinct values (with counts) and size. Returns (description, array)."""
    path = Path(path)
    try:
        with Image.open(path) as image:
            image.load()
            description = {"file": path.name, "format": image.format, "png_mode": image.mode, "width": image.size[0], "height": image.size[1]}
            array = np.asarray(image)
    except FileNotFoundError as exc:
        raise MaskError(f"{path.name}: file not found") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise MaskError(f"{path.name}: mask cannot be read ({type(exc).__name__}: {exc})") from exc
    values, counts = np.unique(array, return_counts=True)
    description.update({"array_dtype": str(array.dtype), "values": [v.item() for v in values], "value_counts": {str(v.item()): int(c) for v, c in zip(values, counts)}})
    return description, array


def decode_mask(description: dict, array: np.ndarray, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    """The one accepted conversion: PNG mode '1' / bool with the values {False, True}; True = annotated lesion (1), False = background (0).

    The direction (True = lesion) is justified by the training files (DIRECTION_RULE, evaluated by the analysis script over all training masks: the median True share is far below one half).
    Any other encoding, intermediate value, wrong size or empty mask is an error: there is no threshold and no repair. A mask that covers more than half of the image is NOT an error here;
    the analysis script reports such masks as findings for the visual inspection."""
    name = description["file"]
    encoding = {"png_mode": description["png_mode"], "array_dtype": description["array_dtype"]}
    if encoding != {k: ACCEPTED_MASK_ENCODING[k] for k in encoding} or not set(description["values"]) <= set(ACCEPTED_MASK_ENCODING["values"]):
        raise MaskError(f"{name}: unexpected mask encoding (PNG mode {description['png_mode']!r}, dtype {description['array_dtype']}, values {description['values']}); "
                        f"accepted is exactly {ACCEPTED_MASK_ENCODING}. No silent threshold or conversion is applied.")
    if expected_size is not None and (description["width"], description["height"]) != tuple(expected_size):
        raise MaskError(f"{name}: decoded mask size {(description['width'], description['height'])} differs from the expected size {tuple(expected_size)}")
    mask = np.asarray(array, dtype=bool)
    if mask.ndim != 2:
        raise MaskError(f"{name}: mask array is not two-dimensional (shape {mask.shape})")
    if not mask.any():
        raise MaskError(f"{name}: empty mask (no positive pixel); the ROI is undefined and no substitute box is used")
    return mask


# ---------------------------------------------------------------------------------------------------------------------
# Geometry
def positive_pixels(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def components(mask: np.ndarray, connectivity: int = CONNECTIVITY) -> list[int]:
    """Sizes (pixels) of the connected components of the positive pixels, largest first. 8-connected by default (4 is available for comparison only)."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    structure = np.ones((3, 3), dtype=int) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    labelled, n = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labelled.ravel())[1:] if n else np.array([], dtype=int)
    return sorted((int(s) for s in sizes), reverse=True)


def tight_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), half-open, around ALL positive pixels. An empty mask raises MaskError."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise MaskError("empty mask: no tight box")
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def expanded_box(box: tuple[int, int, int, int], width: int, height: int, margin: Fraction = MARGIN) -> tuple[int, int, int, int]:
    """Order: expand in exact rational arithmetic (floor left/top, ceil right/bottom), then clip to [0, width] x [0, height]."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    left = math.floor(Fraction(x0) - margin * w)
    right = math.ceil(Fraction(x1) + margin * w)
    top = math.floor(Fraction(y0) - margin * h)
    bottom = math.ceil(Fraction(y1) + margin * h)
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def border_contact(box: tuple[int, int, int, int], width: int, height: int) -> dict:
    """Which image borders the box touches (left/top: coordinate 0; right/bottom: the full width/height)."""
    x0, y0, x1, y1 = box
    sides = {"left": x0 == 0, "right": x1 == width, "top": y0 == 0, "bottom": y1 == height}
    return {**sides, "n_sides": int(sum(sides.values())), "any": any(sides.values())}


def margins(tight: tuple[int, int, int, int], expanded: tuple[int, int, int, int]) -> dict:
    """Actual margins in pixels (left, right, top, bottom) and the number of pixels that the image border cut off compared with the unclipped rule."""
    (x0, y0, x1, y1), (e0, f0, e1, f1) = tight, expanded
    wanted = {"x": math.ceil(MARGIN * (x1 - x0)), "y": math.ceil(MARGIN * (y1 - y0))}
    actual = {"left": x0 - e0, "right": e1 - x1, "top": y0 - f0, "bottom": f1 - y1}
    deficit = (wanted["x"] - actual["left"]) + (wanted["x"] - actual["right"]) + (wanted["y"] - actual["top"]) + (wanted["y"] - actual["bottom"])
    return {**actual, "wanted_x": wanted["x"], "wanted_y": wanted["y"], "deficit_pixels": int(deficit)}


def geometry_problems(mask: np.ndarray, tight: tuple[int, int, int, int], expanded: tuple[int, int, int, int], width: int, height: int) -> list[str]:
    """Independent checks of one pair's boxes (empty list = fine): inclusion, minimality, containment, exact margins, shortening only by image borders."""
    problems = []
    x0, y0, x1, y1 = tight
    e0, f0, e1, f1 = expanded
    if mask.shape != (height, width):
        problems.append(f"mask shape {mask.shape} differs from (height, width) {(height, width)}")
        return problems
    if mask[y0:y1, x0:x1].sum() != mask.sum():
        problems.append("positive mask pixels lie outside the tight box")
    if not (mask[y0:y1, x0].any() and mask[y0:y1, x1 - 1].any() and mask[y0, x0:x1].any() and mask[y1 - 1, x0:x1].any()):
        problems.append("tight box is not minimal (a side has no positive pixel)")
    if not (0 <= e0 <= x0 and 0 <= f0 <= y0 and x1 <= e1 <= width and y1 <= f1 <= height):
        problems.append("expanded box does not contain the tight box or leaves the image")
    need_x, need_y = math.ceil(MARGIN * (x1 - x0)), math.ceil(MARGIN * (y1 - y0))
    for name, actual, need, room in (("left", x0 - e0, need_x, x0), ("right", e1 - x1, need_x, width - x1), ("top", y0 - f0, need_y, y0), ("bottom", f1 - y1, need_y, height - y1)):
        if actual != min(need, room):
            problems.append(f"{name} margin {actual} differs from min(required {need}, available {room}): shortened by something other than the image border")
        if actual < (MARGIN * (x1 - x0 if name in ('left', 'right') else y1 - y0)) and actual != room:
            problems.append(f"{name} margin {actual} is below 20 % without touching the image border")
    crop_inside = mask[f0:f1, e0:e1].sum()
    if crop_inside != mask.sum():
        problems.append("positive mask pixels are lost by the crop")
    return problems


def crop_original(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """The UNMASKED original inside the box (a plain crop; nothing outside the lesion is blackened)."""
    return image.crop(box)


def resize_geometry(crop_width: int, crop_height: int, size: int = TARGET_SIZE) -> dict:
    """Geometry of the existing resize rule for a crop of this size: fitted size, scale factor, centring offsets (the same function Pillow uses, on a dummy image)."""
    fitted = ImageOps.contain(Image.new("L", (crop_width, crop_height)), (size, size), method=ip.INTERPOLATION).size
    return {"fitted_width": fitted[0], "fitted_height": fitted[1], "scale": min(size / crop_width, size / crop_height), "pad_x": (size - fitted[0]) // 2, "pad_y": (size - fitted[1]) // 2}


def prepare_roi_input(image: Image.Image, box: tuple[int, int, int, int], size: int = TARGET_SIZE) -> np.ndarray:
    """Crop the unmasked original to the box, then the existing resize/padding rule and float scaling: array (size, size, 1), float32, [0, 1]."""
    resized = ip.resize_with_padding(crop_original(image, box), size)
    return ip.to_unit_float32(resized)[..., np.newaxis]


def contour(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of the mask (positive pixels with a non-positive 8-neighbour; the image border counts as outside). Only for the local preview."""
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


