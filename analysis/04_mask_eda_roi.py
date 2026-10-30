# %% [04] Expert masks of the TRAINING pairs (EDA) and the technical definition of the expert-mask ROI crop
#
# Every image of BUS-BRA comes with an expert lesion mask. Before a classifier is trained on a crop around the lesion, the training masks are inspected
# (encoding, foreground direction, size, connected components, contact with the image border) and the crop rule is fixed in the ROI contract
# `roi-expert-mask-v1` (src/busbra/roi.py, reports/tables/roi_contract.json): tight box around ALL positive mask pixels, expanded by ceil(edge / 5) per side,
# clipped at the image border; the UNMASKED original is cropped and resized with the existing rule (aspect ratio kept, black padding) to 160 x 160.
#
# The mask is used only to define the crop box. It is not a model input (no mask channel, no blackening outside the lesion), not a cancer label (benign
# lesions have masks as well) and does not mark healthy tissue. The crop needs a manual mask at application time, so results of an expert-ROI model are not a
# pure image-to-score pipeline; the full-image branch stays valid. Decoded pixel sizes steer all geometry; the four wrong CSV heights are only flagged.
# Only the 1,114 training images and masks are opened. Nothing is trained and TensorFlow is not imported. The validation and test masks are not read here.
#
# Actions (the default only reads the stored summary):
#   .venv/bin/python analysis/04_mask_eda_roi.py            show the stored flag counts
#   .venv/bin/python analysis/04_mask_eda_roi.py --run      decode the training masks, write the five tables; stops if one exists
# The statistics figure is drawn by plots/plot_mask_eda.py from mask_eda_train_pairs.csv.

# %% Imports
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from busbra import config, data, roi  # noqa: E402

TABLES = {name: config.TABLES_DIR / f"mask_eda_{name}.csv" for name in ("encoding", "train_pairs", "summary", "flag_counts", "extremes")}
SUMMARY_METRICS = ["positive_fraction", "positive_pixels", "n_components", "box_w", "box_h", "box_area_fraction", "crop_w", "crop_h", "crop_area_fraction", "scale_to_160"]
FLAGS = [("multiple_components", lambda d: d["n_components"] > 1), ("mask_covers_more_than_half_of_the_image", lambda d: d["positive_fraction"] > 0.5),
         ("tight_box_touches_any_border", lambda d: d["touches_any"]), ("tight_box_touches_left", lambda d: d["touches_left"]), ("tight_box_touches_right", lambda d: d["touches_right"]),
         ("tight_box_touches_top", lambda d: d["touches_top"]), ("tight_box_touches_bottom", lambda d: d["touches_bottom"]), ("expansion_cut_by_image_border", lambda d: d["expansion_clipped"]),
         ("crop_enlarged_to_160", lambda d: d["upscaled"]), ("known_csv_size_discrepancy", lambda d: d["csv_size_mismatch_flag"])]


# %% Raw encoding of all training masks before any binarisation
def inspect_encodings(rows: list[dict]) -> pd.DataFrame:
    accepted = (roi.ACCEPTED_MASK_ENCODING["png_mode"], roi.ACCEPTED_MASK_ENCODING["array_dtype"], "|".join(str(v) for v in roi.ACCEPTED_MASK_ENCODING["values"]))
    groups: dict[tuple, list[float]] = {}
    findings = []
    for r in rows:
        description, array = roi.read_mask_raw(config.PROJECT_ROOT / r["mask_relpath"])
        key = (description["png_mode"], description["array_dtype"], "|".join(str(v) for v in description["values"]))
        n_true = description["value_counts"].get("True", 0)
        groups.setdefault(key, []).append(n_true / array.size)
        if key != accepted:
            findings.append(f"{r['ID']}: encoding {key}")
        if n_true == 0:
            findings.append(f"{r['ID']}: empty mask")
    fractions = [f for v in groups.values() for f in v]
    if findings or list(groups) != [accepted] or float(np.median(fractions)) >= 0.5:
        raise ValueError(f"unexpected mask encoding or foreground direction (True must be the minority = the lesion): {findings[:10]}")
    return pd.DataFrame([{"png_mode": k[0], "array_dtype": k[1], "distinct_values": k[2], "n_masks": len(v), "true_share_min": min(v), "true_share_median": float(np.median(v)), "true_share_max": max(v)}
                         for k, v in sorted(groups.items())])


# %% Per-pair measurements and the geometry gate (the same measurement as the ROI loader: busbra.data.measure_pair)
def analyse_pairs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([record for _, _, record in data.measure_pairs(config.PROJECT_ROOT, rows, "train")])


# %% Compact tables
def groups_of(pairs: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    out = [("all", "all training pairs", pairs)]
    out += [("class", name, part) for name, part in pairs.groupby("class", sort=True)]
    out += [("device", name, part) for name, part in pairs.groupby("Device", sort=True)]
    return out


def build_tables(pairs: pd.DataFrame, encoding: pd.DataFrame) -> dict:
    summary_rows, flag_rows = [], []
    for kind, name, part in groups_of(pairs):
        for metric in SUMMARY_METRICS:
            v = part[metric].to_numpy(dtype=float)
            q = np.percentile(v, [5, 25, 50, 75, 95])
            summary_rows.append({"group_type": kind, "group": name, "n_images": int(len(part)), "n_cases": int(part["Case"].nunique()), "metric": metric, "min": float(v.min()), "p05": float(q[0]),
                                 "q1": float(q[1]), "median": float(q[2]), "q3": float(q[3]), "p95": float(q[4]), "max": float(v.max()), "mean": float(v.mean())})
        for flag, rule in FLAGS:
            flagged = rule(part)
            flag_rows.append({"group_type": kind, "group": name, "flag": flag, "n_flagged_images": int(flagged.sum()), "n_flagged_cases": int(part.loc[flagged, "Case"].nunique()),
                              "denominator_images": int(len(part)), "denominator_cases": int(part["Case"].nunique()), "share_of_images": float(flagged.sum() / len(part))})
    extremes = []
    for category, column, ascending, condition in (("smallest_positive_pixels", "positive_pixels", True, None), ("largest_positive_fraction", "positive_fraction", False, None),
                                                    ("most_components", "n_components", False, lambda d: d["n_components"] > 1), ("most_border_sides", "touch_sides", False, lambda d: d["touch_sides"] > 0),
                                                    ("largest_crop_cut_by_border", "margin_deficit_pixels", False, lambda d: d["margin_deficit_pixels"] > 0), ("largest_enlargement", "scale_to_160", False, None),
                                                    ("smallest_enlargement_or_largest_reduction", "scale_to_160", True, None)):
        part = pairs if condition is None else pairs[condition(pairs)]
        for rank, row in enumerate(part.sort_values([column, "ID"], ascending=[ascending, True]).head(10).to_dict("records"), start=1):
            extremes.append({"category": category, "rank": rank, **{k: row[k] for k in ("ID", "Case", "class", "Device", "positive_pixels", "positive_fraction", "n_components", "touch_sides",
                                                                                          "margin_deficit_pixels", "crop_area_fraction", "scale_to_160")}})
    return {"encoding": encoding, "train_pairs": pairs, "summary": pd.DataFrame(summary_rows), "flag_counts": pd.DataFrame(flag_rows), "extremes": pd.DataFrame(extremes)}


# %% Run
def show() -> None:
    path = TABLES["flag_counts"]
    if not path.is_file():
        print("no stored mask EDA")
        return
    flags = pd.read_csv(path)
    print(flags[flags["group_type"] == "all"][["flag", "n_flagged_images", "denominator_images", "share_of_images"]].to_string(index=False))


def run() -> None:
    clash = [p.name for p in TABLES.values() if p.exists()]
    if clash:
        raise FileExistsError(f"already present: {clash}; nothing was changed")
    manifest = data.read_manifest()
    data.check_split(manifest)
    rows = data.subset_frame(manifest, "train").to_dict("records")  # the split filter comes first: only training files are opened
    tables = build_tables(analyse_pairs(rows), inspect_encodings(rows))
    for name, table in tables.items():
        table.to_csv(TABLES[name], index=False, lineterminator="\n")
    print(f"wrote {len(tables)} tables for {len(rows)} training pairs")


if __name__ == "__main__":
    run() if "--run" in (sys.argv[1:] if "__file__" in globals() else []) else show()
