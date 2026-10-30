"""Data access: the fixed split manifest, one central split check and the model inputs (full image or expert-mask ROI).

`data/processed/manifest_split.csv` is the only authoritative assignment of images to train / validation / test. It is read here
and never rewritten. Importing the module has no side effects. Images are only opened for the subsets that are asked for
explicitly; the training code requests train and validation, never test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from busbra import config
from busbra import image_processing as ip
from busbra import roi

SIZE = 160  # input side of the expert-mask ROI arrays
SUBSETS = ("train", "validation", "test")


def read_manifest(path: Path = config.MANIFEST_PATH) -> pd.DataFrame:
    """The split manifest as strings (IDs keep leading zeros, row order is the file order)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}. Run analysis/01_prepare_data.py (needs the extracted BUS-BRA files).")
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def check_split(manifest: pd.DataFrame) -> None:
    """Central consistency check of the split; raises ValueError listing every violation."""
    problems = []
    if not manifest["ID"].is_unique:
        problems.append("image IDs are not unique")
    if not (manifest["label"] == manifest["Pathology"].map({"benign": "0", "malignant": "1"})).all():
        problems.append("label does not follow Pathology (benign = 0, malignant = 1)")
    if set(manifest["split"]) != set(SUBSETS):
        problems.append(f"split names {sorted(set(manifest['split']))}")
    if (manifest.groupby("Case")["label"].nunique() != 1).any():
        problems.append("a case has more than one label")
    cases = {s: set(manifest.loc[manifest["split"] == s, "Case"]) for s in SUBSETS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if cases[a] & cases[b]:
            problems.append(f"{len(cases[a] & cases[b])} cases occur in both {a} and {b}")
    for subset in SUBSETS:
        part = manifest[manifest["split"] == subset]
        if len(part) != config.EXPECTED_IMAGES[subset] or part["Case"].nunique() != config.EXPECTED_CASES[subset]:
            problems.append(f"{subset}: {len(part)} images / {part['Case'].nunique()} cases, expected {config.EXPECTED_IMAGES[subset]} / {config.EXPECTED_CASES[subset]}")
    if (manifest["image_mask_size_match"] != "True").any():
        problems.append("an image and its mask differ in decoded size")
    if problems:
        raise ValueError("split check failed: " + "; ".join(problems))


def exact_labels(values, what: str) -> np.ndarray:
    """Class labels as integers, checked BEFORE the conversion: every value defined, numeric and exactly 0 or 1 (a truncating cast would turn 0.25 into 0). Raises ValueError otherwise."""
    series = pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce")
    bad = numeric.isna() | ~numeric.isin([0, 1])
    if bad.any():
        raise ValueError(f"{what}: labels must be defined and exactly 0 or 1; found {series[bad].unique()[:5].tolist()}")
    return numeric.to_numpy().astype(int)


def subset_frame(manifest: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Rows of one subset in manifest order."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    return manifest[manifest["split"] == subset].reset_index(drop=True)


def load_full_images(frame: pd.DataFrame, size: int, project_root: Path = config.PROJECT_ROOT) -> np.ndarray:
    """(n, size, size, 1) float32 in [0, 1]; the decoded sizes of the manifest (not the partly wrong CSV sizes) are checked."""
    expected = list(zip(frame["image_width"].astype(int), frame["image_height"].astype(int)))
    images = ip.preprocess_image_set(project_root, frame["image_relpath"].tolist(), size, expected)
    return images[..., np.newaxis]


class RoiDataError(Exception):
    """The ROI input of a pair violates the contract; the message lists the problems. Nothing is repaired silently."""


GEOMETRY_COLUMNS = ["ID", "Case", "label", "class", "Device", "subset", "image_width", "image_height", "mask_width", "mask_height", "positive_pixels", "positive_fraction", "n_components",
                    "x0", "y0", "x1", "y1", "box_w", "box_h", "touches_left", "touches_right", "touches_top", "touches_bottom", "touch_sides", "touches_any",
                    "exp_x0", "exp_y0", "exp_x1", "exp_y1", "crop_w", "crop_h", "crop_area_fraction", "margin_deficit_pixels", "expansion_clipped", "scale_to_160", "upscaled"]


def measure_pair(project_root: Path, r: dict) -> tuple:
    """Open one image-mask pair of a manifest row and measure it (the one place where the ROI geometry of a pair is derived).

    Returns (image, expanded box, record): the decoded image, the 20 % expanded box (roi-expert-mask-v1) and every measurement of the pair (mask size and encoding, pixel counts, components, tight and
    expanded box, border contact, margins, resize geometry). The DECODED size steers every geometry; the CSV Width/Height columns never do. Raises RoiDataError for any violation of the contract."""
    project_root = Path(project_root)
    try:
        image = ip.load_grayscale_image(project_root / r["image_relpath"])
        width, height = image.size
        description, array = roi.read_mask_raw(project_root / r["mask_relpath"])
        mask = roi.decode_mask(description, array, expected_size=(width, height))
    except (ip.ImageLoadError, roi.MaskError) as exc:
        raise RoiDataError(f"{r['ID']}: {exc}") from exc
    audited = (int(r["image_width"]), int(r["image_height"]), int(r["mask_width"]), int(r["mask_height"]))
    if audited != (width, height, width, height):
        raise RoiDataError(f"{r['ID']}: decoded sizes image {(width, height)} / mask {(description['width'], description['height'])} differ from the audited {audited}")
    tight = roi.tight_box(mask)
    expanded = roi.expanded_box(tight, width, height)
    geometry = roi.geometry_problems(mask, tight, expanded, width, height)
    if geometry:
        raise RoiDataError(f"{r['ID']}: " + "; ".join(geometry))
    sizes = roi.components(mask)
    contact = roi.border_contact(tight, width, height)
    margin = roi.margins(tight, expanded)
    crop_w, crop_h = expanded[2] - expanded[0], expanded[3] - expanded[1]
    resize = roi.resize_geometry(crop_w, crop_h)
    pixels = roi.positive_pixels(mask)
    record = {
        "ID": r["ID"], "Case": r["Case"], "label": int(r["label"]), "class": r["Pathology"], "Device": r["Device"], "image_width": width, "image_height": height,
        "mask_width": description["width"], "mask_height": description["height"], "size_match": True, "csv_size_mismatch_flag": r.get("csv_size_mismatch") == "True",
        "positive_pixels": pixels, "positive_fraction": pixels / (width * height), "n_components": len(sizes), "largest_component_pixels": sizes[0], "largest_component_share": sizes[0] / pixels,
        "x0": tight[0], "y0": tight[1], "x1": tight[2], "y1": tight[3], "box_w": tight[2] - tight[0], "box_h": tight[3] - tight[1],
        "box_area_fraction": (tight[2] - tight[0]) * (tight[3] - tight[1]) / (width * height),
        "touches_left": contact["left"], "touches_right": contact["right"], "touches_top": contact["top"], "touches_bottom": contact["bottom"], "touch_sides": contact["n_sides"], "touches_any": contact["any"],
        "exp_x0": expanded[0], "exp_y0": expanded[1], "exp_x1": expanded[2], "exp_y1": expanded[3], "crop_w": crop_w, "crop_h": crop_h, "crop_area_fraction": crop_w * crop_h / (width * height),
        "margin_left": margin["left"], "margin_right": margin["right"], "margin_top": margin["top"], "margin_bottom": margin["bottom"], "margin_deficit_pixels": margin["deficit_pixels"],
        "expansion_clipped": margin["deficit_pixels"] > 0, "scale_to_160": resize["scale"], "upscaled": resize["scale"] > 1, "fitted_w": resize["fitted_width"], "fitted_h": resize["fitted_height"],
        "pad_x": resize["pad_x"], "pad_y": resize["pad_y"]}
    return image, expanded, record


def measure_pairs(project_root: Path, rows: list[dict], subset: str) -> list[tuple]:
    """measure_pair for every row in the given order; all violations are collected and raised together (no repair, no exclusion)."""
    measured, problems = [], []
    for r in rows:
        try:
            measured.append(measure_pair(project_root, r))
        except RoiDataError as exc:
            problems.append(str(exc))
    if problems:
        raise RoiDataError(f"{len(problems)} of {len(rows)} {subset} pairs violate the contract (no repair, no exclusion): " + " | ".join(problems[:8]) + (" ..." if len(problems) > 8 else ""))
    return measured


def load_roi_arrays(project_root: Path, rows: list[dict], subset: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Prepare the ROI input of every listed pair in EXACTLY the given order: (N, 160, 160, 1) float32 in [0, 1] plus the geometry table.

    Per pair: the image (mode 'L') and the mask (PNG mode '1', bool, not empty, same decoded size as the image) are linked through the manifest row (see measure_pair); the UNMASKED original is cropped
    to the expanded box and prepared with the existing resize rule. Only the paths of the given rows are opened. Any violation is raised as RoiDataError; there is no repair and no exclusion."""
    measured = measure_pairs(project_root, rows, subset)
    arrays = np.empty((len(rows), SIZE, SIZE, 1), dtype=np.float32)
    for index, (image, expanded, _) in enumerate(measured):
        arrays[index] = roi.prepare_roi_input(image, expanded)
    check_roi_arrays(arrays, subset)
    return arrays, geometry_table(measured, subset)


def geometry_table(measured: list[tuple], subset: str) -> pd.DataFrame:
    """The geometry table (data.GEOMETRY_COLUMNS) of measure_pairs results."""
    return pd.DataFrame([{**record, "subset": subset} for _, _, record in measured], columns=GEOMETRY_COLUMNS)


def check_roi_arrays(arrays: np.ndarray, what: str) -> None:
    if not (np.isfinite(arrays).all() and 0.0 <= float(arrays.min()) and float(arrays.max()) <= 1.0):
        raise RoiDataError(f"{what}: ROI arrays are not finite values in [0, 1]")


# ---- the expert geometry of train and validation: measured once from the masks, then read by every consumer ---------------------------------------------------
def expert_geometry_path() -> Path:
    return config.TABLES_DIR / "expert_roi_geometry_train_validation.csv"


def expert_frequencies_path() -> Path:
    return config.TABLES_DIR / "expert_roi_geometry_frequencies.csv"


EXPERT_SUBSETS = ("train", "validation")  # the test masks are never part of this table
PREPARE_COMMAND = ".venv/bin/python analysis/03_classification.py --preset expert_roi --prepare"


def measure_expert_geometry(project_root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """Open the image-mask pairs of the training and validation subset once and measure them (measure_pairs); one row per pair in manifest order, training rows first. Needs the extracted files."""
    return pd.concat([geometry_table(measure_pairs(project_root, subset_frame(manifest, s).to_dict("records"), s), s) for s in EXPERT_SUBSETS], ignore_index=True)


def validate_expert_geometry(geometry: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Check an expert geometry table against the manifest and return it with typed columns. Every violation is reported together; nothing is repaired, joined or defaulted.

    Required: all data.GEOMETRY_COLUMNS; the train and validation pairs of the manifest, exactly once each, in manifest order (training first), with the same ID, Case, label, class, device and
    decoded image size; integer boxes inside the image; crop sizes equal to the box; the stored resize scale equal to the existing resize rule."""
    missing = [c for c in GEOMETRY_COLUMNS if c not in geometry.columns]
    if missing:
        raise RoiDataError(f"the expert geometry lacks the columns {missing}")
    frame = geometry[GEOMETRY_COLUMNS].copy()
    frame["ID"], frame["Case"], frame["class"], frame["Device"], frame["subset"] = (frame[c].astype(str) for c in ("ID", "Case", "class", "Device", "subset"))
    problems = []
    try:
        frame["label"] = exact_labels(frame["label"], "geometry")
    except ValueError as exc:
        raise RoiDataError(f"expert geometry is not usable: {exc}") from exc
    if frame["ID"].duplicated().any():
        problems.append(f"duplicated IDs {sorted(frame.loc[frame['ID'].duplicated(), 'ID'])[:3]}")
    expected = pd.concat([subset_frame(manifest, s) for s in EXPERT_SUBSETS], ignore_index=True)
    if len(frame) != len(expected):
        problems.append(f"{len(frame)} rows instead of {len(expected)} (train and validation pairs of the manifest)")
    else:
        same = {"ID": expected["ID"], "Case": expected["Case"], "label": pd.Series(exact_labels(expected["label"], "manifest")), "class": expected["Pathology"], "Device": expected["Device"], "subset": expected["split"],
                "image_width": expected["image_width"].astype(int), "image_height": expected["image_height"].astype(int), "mask_width": expected["mask_width"].astype(int),
                "mask_height": expected["mask_height"].astype(int)}
        for column, values in same.items():
            if not frame[column].astype(values.dtype).reset_index(drop=True).equals(values.reset_index(drop=True)):
                problems.append(f"column {column} differs from the manifest (IDs, order, subsets, cases, labels or sizes)")
    for column in ("x0", "y0", "x1", "y1", "exp_x0", "exp_y0", "exp_x1", "exp_y1", "crop_w", "crop_h", "image_width", "image_height"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values != values.round()).any():
            problems.append(f"column {column} is not integer")
        else:
            frame[column] = values.astype(int)
    if not problems:
        inside = (frame["exp_x0"] >= 0) & (frame["exp_y0"] >= 0) & (frame["exp_x1"] > frame["exp_x0"]) & (frame["exp_y1"] > frame["exp_y0"]) & (frame["exp_x1"] <= frame["image_width"]) & (frame["exp_y1"] <= frame["image_height"])
        contains = (frame["exp_x0"] <= frame["x0"]) & (frame["exp_y0"] <= frame["y0"]) & (frame["exp_x1"] >= frame["x1"]) & (frame["exp_y1"] >= frame["y1"]) & (frame["x1"] > frame["x0"]) & (frame["y1"] > frame["y0"])
        if not inside.all():
            problems.append(f"{int((~inside).sum())} expanded boxes are empty or lie outside the image")
        if not contains.all():
            problems.append(f"{int((~contains).sum())} tight boxes are empty or not inside the expanded box")
        if not ((frame["crop_w"] == frame["exp_x1"] - frame["exp_x0"]) & (frame["crop_h"] == frame["exp_y1"] - frame["exp_y0"])).all():
            problems.append("crop sizes differ from the expanded boxes")
        if contains.all() and inside.all():  # the stored expanded box must be the one the existing rule (20 % per side, rounded up, clipped at the border) gives for the tight box and the image size
            expected_boxes = [roi.expanded_box((int(r.x0), int(r.y0), int(r.x1), int(r.y1)), int(r.image_width), int(r.image_height)) for r in frame.itertuples()]
            stored_boxes = list(zip(frame["exp_x0"], frame["exp_y0"], frame["exp_x1"], frame["exp_y1"]))
            wrong = [i for i, (a, b) in enumerate(zip(expected_boxes, stored_boxes)) if a != tuple(int(v) for v in b)]
            if wrong:
                problems.append(f"{len(wrong)} expanded boxes differ from the 20 % expansion of the tight box (first: {frame['ID'].iloc[wrong[0]]})")
        if inside.all():
            scale = [roi.resize_geometry(int(w), int(h))["scale"] for w, h in zip(frame["crop_w"], frame["crop_h"])]
            if not np.allclose(frame["scale_to_160"].astype(float).to_numpy(), scale, rtol=0, atol=1e-12):
                problems.append("the stored resize scales differ from the existing resize rule")
    if problems:
        raise RoiDataError("expert geometry is not usable: " + "; ".join(problems))
    return frame


def read_expert_geometry(manifest: pd.DataFrame | None = None, path: Path | None = None) -> pd.DataFrame:
    """The validated expert geometry of the training and validation pairs (see validate_expert_geometry). A missing file is an error that names the preparation command: nothing is measured here."""
    path = Path(expert_geometry_path() if path is None else path)
    if not path.is_file():
        raise FileNotFoundError(f"the expert geometry {path} does not exist. Prepare it once with: {PREPARE_COMMAND}")
    manifest = read_manifest() if manifest is None else manifest
    return validate_expert_geometry(pd.read_csv(path, dtype={"ID": str, "Case": str, "class": str, "Device": str, "subset": str}, float_precision="round_trip"), manifest)


def load_roi_inputs(project_root: Path, frame: pd.DataFrame, geometry: pd.DataFrame) -> tuple[list, np.ndarray, list]:
    """Images, ROI arrays (N, 160, 160, 1) float32 in [0, 1] and expanded boxes of the pairs of `frame` (a manifest subset in manifest order) from the stored expert geometry and the original images.

    No mask is opened. The decoded image size must equal the manifest and the geometry. The unmasked original is cropped to the stored box and prepared with the existing resize rule, exactly
    as load_roi_arrays does after measuring the mask."""
    project_root = Path(project_root)
    table = geometry.set_index("ID")
    missing = [i for i in frame["ID"] if i not in table.index]
    if missing:
        raise RoiDataError(f"no stored geometry for {missing[:3]}")
    images, boxes = [], []
    arrays = np.empty((len(frame), SIZE, SIZE, 1), dtype=np.float32)
    for index, r in enumerate(frame.to_dict("records")):
        g = table.loc[r["ID"]]
        if str(g["Case"]) != r["Case"] or int(g["label"]) != int(r["label"]):
            raise RoiDataError(f"{r['ID']}: case or label differ from the stored geometry")
        image = ip.load_grayscale_image(project_root / r["image_relpath"], (int(g["image_width"]), int(g["image_height"])))
        box = (int(g["exp_x0"]), int(g["exp_y0"]), int(g["exp_x1"]), int(g["exp_y1"]))
        arrays[index] = roi.prepare_roi_input(image, box)
        images.append(image)
        boxes.append(box)
    check_roi_arrays(arrays, "stored-geometry ROI inputs")
    return images, arrays, boxes


def load_classification_inputs(data: str, size: int, subsets=("train", "validation"), project_root: Path = config.PROJECT_ROOT):
    """Inputs and labels for the requested subsets: {subset: (frame, images, labels)}. data is 'full' or 'roi'; the ROI inputs come from the stored expert geometry (read_expert_geometry), never from a new mask measurement."""
    manifest = read_manifest()
    check_split(manifest)
    geometry = read_expert_geometry(manifest) if data == "roi" else None  # before any image is opened
    result = {}
    for subset in subsets:
        frame = subset_frame(manifest, subset)
        if data == "full":
            images = load_full_images(frame, size, project_root)
        elif data == "roi":
            _, images, _ = load_roi_inputs(project_root, frame, geometry)
        else:
            raise ValueError(f"unknown data type {data!r}")
        result[subset] = (frame, images, frame["label"].astype(int).to_numpy())
    return result


# ---- preparation of the split manifest from the extracted BUS-BRA files ---------------------------------------------------
MANIFEST_COLUMNS = ["ID", "Case", "Pathology", "label", "outer_kfold", "split", "Device", "Histology", "BIRADS", "Width", "Height", "Side", "BBOX", "image_relpath", "mask_relpath",
                    "image_width", "image_height", "mask_width", "mask_height", "mask_foreground_pixels", "mask_is_empty", "image_mask_size_match", "csv_width_mismatch",
                    "csv_height_mismatch", "csv_size_mismatch"]


def read_metadata(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """bus_data.csv joined with the outer fold of 5-fold-cv.csv (all strings, IDs unchanged), plus label and split.

    The inner valid_* columns are not used: they put cases into both training and validation. The split follows the outer folds only
    (fold 1 test, 2 validation, 3-5 train); it is a fixed holdout, not a cross-validation.
    """
    raw_dir = Path(raw_dir)
    bus = pd.read_csv(raw_dir / "bus_data.csv", dtype=str, keep_default_na=False, na_filter=False)
    folds = pd.read_csv(raw_dir / "5-fold-cv.csv", dtype=str, keep_default_na=False, na_filter=False)
    if not (bus["ID"].is_unique and folds["ID"].is_unique and set(bus["ID"]) == set(folds["ID"])):
        raise ValueError("bus_data.csv and 5-fold-cv.csv do not list the same unique image IDs")
    merged = bus.merge(folds[["ID", "Pathology", "kFold"]], on="ID", how="left", suffixes=("", "_fold"))
    if (merged["Pathology"] != merged["Pathology_fold"]).any():
        raise ValueError("Pathology differs between bus_data.csv and 5-fold-cv.csv")
    merged["label"] = merged["Pathology"].map({"benign": "0", "malignant": "1"})
    if merged["label"].isna().any():
        raise ValueError("unexpected Pathology value")
    merged["outer_kfold"] = merged["kFold"]
    merged["split"] = merged["kFold"].astype(int).map(config.SPLIT_OF_OUTER_FOLD)
    return merged.drop(columns=["Pathology_fold", "kFold"]).sort_values("ID").reset_index(drop=True)


def build_manifest(raw_dir: Path = config.RAW_DIR, project_root: Path = config.PROJECT_ROOT) -> pd.DataFrame:
    """Manifest with the decoded image and mask sizes. Opens every image and mask; the CSV sizes never steer anything (four CSV heights are wrong)."""
    from PIL import Image
    raw_dir, project_root = Path(raw_dir), Path(project_root)
    meta = read_metadata(raw_dir)
    rows = []
    for record in meta.to_dict("records"):
        image_path = raw_dir / "Images" / f"{record['ID']}.png"
        mask_path = raw_dir / "Masks" / f"mask_{record['ID'][len('bus_'):]}.png"
        with Image.open(image_path) as image:
            image.load()
            image_mode, image_size = image.mode, image.size
        with Image.open(mask_path) as mask:
            mask.load()
            mask_size, foreground = mask.size, int((np.asarray(mask.convert("L")) > 0).sum())
        if image_mode != "L":
            raise ValueError(f"{image_path.name}: image mode {image_mode!r}, expected 'L'")
        width_bad, height_bad = int(record["Width"]) != image_size[0], int(record["Height"]) != image_size[1]
        rows.append({**record, "image_relpath": image_path.relative_to(project_root).as_posix(), "mask_relpath": mask_path.relative_to(project_root).as_posix(),
                     "image_width": image_size[0], "image_height": image_size[1], "mask_width": mask_size[0], "mask_height": mask_size[1], "mask_foreground_pixels": foreground,
                     "mask_is_empty": foreground == 0, "image_mask_size_match": image_size == mask_size, "csv_width_mismatch": width_bad, "csv_height_mismatch": height_bad,
                     "csv_size_mismatch": width_bad or height_bad})
    return pd.DataFrame(rows)[MANIFEST_COLUMNS].astype(str)


def data_findings(raw_dir: Path = config.RAW_DIR, manifest: pd.DataFrame | None = None) -> dict:
    """The documented data findings, recomputed from the metadata (images are not opened).

    csv_size_mismatches  IDs whose CSV width/height differ from the decoded size (needs a manifest with decoded sizes)
    birads_conflicts     cases whose images carry different BI-RADS values
    inner_overlap        per inner valid_k column (1 = training, 0 = validation, NaN = outer test fold), cases in both groups (reason not to use them)
    """
    raw_dir = Path(raw_dir)
    meta = read_metadata(raw_dir)
    findings = {"birads_conflicts": sorted(meta.groupby("Case")["BIRADS"].nunique().loc[lambda s: s > 1].index.tolist(), key=int)}
    folds = pd.read_csv(raw_dir / "5-fold-cv.csv", dtype=str, keep_default_na=False, na_filter=False).merge(meta[["ID", "Case"]], on="ID")
    findings["inner_overlap"] = {}
    for column in [c for c in folds.columns if c.startswith("valid_")]:  # 1 = inner training, 0 = inner validation, NaN = outer test fold
        training, validation = set(folds.loc[folds[column] == "1", "Case"]), set(folds.loc[folds[column] == "0", "Case"])
        findings["inner_overlap"][column] = len(training & validation)
    if manifest is not None and "csv_size_mismatch" in manifest:
        findings["csv_size_mismatches"] = manifest.loc[manifest["csv_size_mismatch"] == "True", "ID"].tolist()
    return findings
