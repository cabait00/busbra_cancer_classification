# %% [10] Grad-CAM and qualitative error analysis on rule-selected frozen test images (seed 7)
#
# Examples are chosen by a deterministic rule from the STORED test scores of the automatic-ROI branch (seed 7) against its frozen 95 %-sensitivity image threshold:
# up to two images per category TP / TN / FP / FN in ascending ID order, at most one image per case, plus up to four localisation failures of the automatic ROI.
# Grad-CAM (target layer Conv_1 of the MobileNetV2 backbone, malignant logit before the sigmoid) is computed for the full-image model (4B), the expert-ROI model (4E) and
# the automatic ROI (same 4E classifier, frozen automatic boxes), always inside the original 16-image evaluation batches of the test run. The padding is removed and the map is
# mapped back to the original image frame. Maps show where the gradient-weighted activation is, not why; no cause and no clinical correctness is claimed.
# Nothing is trained and no threshold, model or rule is changed; the visual inspection of the overlays is done by the project owner.
#
# Actions (the default derives the selection from the stored tables and compares it with the stored selection; nothing is written):
#   .venv/bin/python analysis/10_interpretation.py
#   env -u LD_LIBRARY_PATH .venv/bin/python analysis/10_interpretation.py --run
#       explicit GPU action: computes the maps and writes gradcam_test_selection.csv, gradcam_test_geometry.csv, gradcam_test_score_fidelity.csv, gradcam_test_cam.csv and the local
#       map file data/processed/gradcam_test_maps.npz (image-derived, stays local); stops if one of them exists.

# %% Imports
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from busbra import auto_roi as ar, config, data, final_evaluation as fe, gradcam_test as gt, image_processing as ip, inference  # noqa: E402
from busbra import model_diagnostics as md, results, test_evaluation as te  # noqa: E402
from busbra.evaluation import sigmoid_scores  # noqa: E402

TABLES = config.TABLES_DIR
TEST_DIR = TABLES / "test_evaluation"
MAPS_PATH = config.PROJECT_ROOT / "data" / "processed" / "gradcam_test_maps.npz"
OUTPUTS = {name: TABLES / f"gradcam_test_{name}.csv" for name in ("selection", "geometry", "score_fidelity", "cam")}
SEED = 7
THRESHOLD_POINT = "target_0.95"
SCORE_TOLERANCE = 1e-5
CONDITIONS = {"full": {"combo": "full_finetuned|seed7", "variant": "mobilenetv2_higher_lr_160_aug"},
              "expert": {"combo": "expert_roi_finetuned|seed7", "variant": "mobilenetv2_higher_lr_160_roi"},
              "auto": {"combo": "auto_roi_finetuned|seed7", "variant": "mobilenetv2_higher_lr_160_roi"}}  # the automatic ROI uses the SAME classifier as the expert ROI


# %% Selection and geometry from stored tables (CPU)
def derive_selection() -> dict:
    """The selection table (stored format), the geometry table of the three input conditions and the evaluation batches, all from the stored score, threshold, diagnostics and geometry tables."""
    manifest = data.read_manifest()
    data.check_split(manifest)
    test = data.subset_frame(manifest, "test")
    entries = [e["combo_id"] for e in fe.combinations()]
    thresholds, problems = te.load_frozen_thresholds(TABLES / "final_operating_points.csv", entries)
    if problems:
        raise ValueError("; ".join(problems[:3]))
    scores = pd.read_csv(TABLES / "test_evaluation_scores.csv", dtype={"ID": str, "Case": str}, float_precision="round_trip")
    by_condition = {c: scores[scores["combo_id"] == spec["combo"]].set_index("ID")["malignant_score"] for c, spec in CONDITIONS.items()}
    meta = pd.DataFrame({"ID": test["ID"], "Case": test["Case"], "label": test["label"].astype(int)})
    threshold_of = {c: thresholds[(spec["combo"], "image", THRESHOLD_POINT)]["value"] for c, spec in CONDITIONS.items()}
    core, core_log = gt.select_core(meta.assign(malignant_score=by_condition["auto"].loc[meta["ID"]].to_numpy()), threshold_of["auto"])
    diagnostics = pd.read_csv(TEST_DIR / "auto_roi_diagnostics_test.csv", dtype=str, keep_default_na=False)
    special, special_log = gt.select_special(diagnostics, core)
    selected = core + special
    blocks = gt.batch_blocks(meta["ID"].tolist(), [r["ID"] for r in selected])
    position = {i["ID"]: i for i in blocks["images"]}
    diag = diagnostics.set_index("ID")
    info = test.set_index("ID")
    rows = []
    for order, r in enumerate(selected, start=1):
        identifier, d = r["ID"], diag.loc[r["ID"]]
        reason = gt.core_reason(r["category"], threshold_of["auto"]) if r["role"] == "core" else gt.SPECIAL_REASON[r["category"]]
        row = {"selection_order": order, "role": r["role"], "category": r["category"], "selection_reason": reason, "ID": identifier, "Case": info.loc[identifier, "Case"], "label": int(r["label"]),
               "Device": info.loc[identifier, "Device"], "position_in_test_order": position[identifier]["position"], "batch_block": position[identifier]["block"],
               "position_in_block": position[identifier]["position"] - position[identifier]["block"] * gt.BATCH_SIZE}
        for condition in CONDITIONS:
            t, score = threshold_of[condition], float(by_condition[condition].loc[identifier])
            row.update({f"score_{condition}": score, f"threshold_{condition}_0.95": repr(t), f"threshold_{condition}_0.95_hex": float(t).hex(), f"decision_{condition}_0.95": "malignant" if score >= t else "benign",
                        f"outcome_{condition}_0.95": gt.outcome(r["label"], score >= t)})
        row.update({"auto_fallback_full_image": d["fallback_full_image"], "auto_crop_category_5c": d["category"], "expert_pixels_5c": d["expert_pixels"], "pixels_inside_actual_crop_5c": d["inside_actual_crop"],
                    "coverage_actual_crop_5c": d["coverage_actual_crop"], "chosen_zero_but_other_overlaps_5c": d["chosen_zero_but_other_overlaps"], "dice_raw_prediction_5c": d["dice_original"]})
        rows.append(row)
    selection = pd.DataFrame(rows)[gt.SELECTION_COLUMNS]
    return {"selected": selected, "rows": selection, "geometry": derive_geometry(selection, test), "blocks": blocks, "test": test, "scores": by_condition, "log": core_log + special_log}


def derive_geometry(selection: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Box and resize geometry of the selected images for the full-image, expert-ROI and automatic-ROI input (boxes from the stored geometry tables, nothing is measured on an image)."""
    stored = {"expert": pd.read_csv(TEST_DIR / "expert_roi_geometry_test.csv", dtype={"ID": str}).set_index("ID"), "auto": pd.read_csv(TEST_DIR / "auto_roi_geometry_test.csv", dtype={"ID": str}).set_index("ID")}
    size = test.set_index("ID")[["image_width", "image_height"]].astype(int)
    rows = []
    for s in selection.itertuples():
        width, height = int(size.loc[s.ID, "image_width"]), int(size.loc[s.ID, "image_height"])
        for condition in CONDITIONS:
            box = (0, 0, width, height) if condition == "full" else tuple(int(stored[condition].loc[s.ID, k]) for k in ("exp_x0", "exp_y0", "exp_x1", "exp_y1"))
            crop_w, crop_h = box[2] - box[0], box[3] - box[1]
            g = md.padding_geometry(crop_w, crop_h)
            rows.append({"condition": condition, "ID": s.ID, "role": s.role, "category": s.category, "box_source": gt.BOX_SOURCE[condition], "image_width": width, "image_height": height,
                         "box_x0": box[0], "box_y0": box[1], "box_x1": box[2], "box_y1": box[3], "crop_width": crop_w, "crop_height": crop_h, "crop_area_fraction": crop_w * crop_h / (width * height),
                         "fitted_width": g["fitted_width"], "fitted_height": g["fitted_height"], "pad_x": g["pad_x"], "pad_y": g["pad_y"], "pad_right": g["pad_right"], "pad_bottom": g["pad_bottom"],
                         "scale": g["scale"], "auto_fallback_full_image": s.auto_fallback_full_image if condition == "auto" else ""})
    return pd.DataFrame(rows)[gt.GEOMETRY_COLUMNS]


def show() -> None:
    selection = derive_selection()
    print(f"{len(selection['selected'])} examples in {len(selection['blocks']['blocks'])} evaluation batches:")
    print(selection["rows"][["selection_order", "role", "category", "ID"]].to_string(index=False))
    for key, name in (("rows", "selection"), ("geometry", "geometry")):
        stored = OUTPUTS[name]
        if stored.is_file():
            print(f"{name} table identical to the stored table:", results.same_as_stored(selection[key], stored))
        else:
            print(f"no stored {name} table to compare with")


# %% Grad-CAM (explicit GPU action)
def run() -> None:
    clash = [p.name for p in (*OUTPUTS.values(), MAPS_PATH) if p.exists()]
    if clash:
        raise FileExistsError(f"already present: {clash}; nothing was changed")
    selection = derive_selection()
    test, blocks, selected = selection["test"], selection["blocks"], selection["selected"]
    rows_by_id = {r["ID"]: r for r in test.to_dict("records")}
    order = [i["ID"] for i in blocks["images"]]
    block_of = {i["ID"]: i["block"] for i in blocks["images"]}
    target_ids = {r["ID"] for r in selected}
    images = {i: ip.load_grayscale_image(config.PROJECT_ROOT / rows_by_id[i]["image_relpath"], (int(rows_by_id[i]["image_width"]), int(rows_by_id[i]["image_height"]))) for i in order}
    auto_frame = pd.read_csv(TEST_DIR / "auto_roi_geometry_test.csv", dtype={"ID": str, "Case": str}, float_precision="round_trip").set_index("ID", drop=False).loc[order].reset_index(drop=True)
    expert_frame = te.expert_geometry_for_inputs(pd.read_csv(TEST_DIR / "expert_roi_geometry_test.csv", dtype={"ID": str, "Case": str}, float_precision="round_trip").set_index("ID", drop=False).loc[order].reset_index(drop=True))
    listing = [images[i] for i in order]
    arrays = {"full": inference.full_array(listing, md.SIZE), "expert": ar.inputs_from_geometry(listing, expert_frame), "auto": ar.inputs_from_geometry(listing, auto_frame)}
    boxes = {"full": {i: (0, 0, images[i].size[0], images[i].size[1]) for i in order}, "expert": {r.ID: (int(r.exp_x0), int(r.exp_y0), int(r.exp_x1), int(r.exp_y1)) for r in expert_frame.itertuples()},
             "auto": {r.ID: (int(r.exp_x0), int(r.exp_y0), int(r.exp_x1), int(r.exp_y1)) for r in auto_frame.itertuples()}}
    valid = {c: np.stack([md.validity_mask(boxes[c][i][2] - boxes[c][i][0], boxes[c][i][3] - boxes[c][i][1]) for i in order]) for c in CONDITIONS}
    from busbra import models, training  # TensorFlow is imported only here, after all CPU preparation
    import keras
    training.configure_gpu()
    fidelity, cam_rows, maps = [], [], {}
    for variant_name in dict.fromkeys(spec["variant"] for spec in CONDITIONS.values()):
        variant = config.VARIANTS[variant_name]
        models.reset_random_sources(SEED)
        model = models.compile_model(keras.models.load_model(config.run_paths(variant, SEED)["checkpoint"], compile=False), variant.lr)
        graph = md.DiagnosticGraph(model)
        for condition in [c for c, spec in CONDITIONS.items() if spec["variant"] == variant_name]:
            for block in blocks["blocks"]:
                index = [k for k, i in enumerate(order) if block_of[i] == block]
                ids = [order[k] for k in index]
                x = arrays[condition][index]
                reference = selection["scores"][condition].loc[ids].to_numpy(dtype=np.float64)
                predicted = sigmoid_scores(model.predict(training.make_eval_dataset(x, np.zeros(len(x), dtype=np.float32), gt.BATCH_SIZE), verbose=0), len(x))
                cam_inputs = graph.gradcam_inputs(x)
                worst = max(float(np.abs(predicted - reference).max()), float(np.abs(md.sigmoid(cam_inputs["logit"]) - reference).max()))
                if worst > SCORE_TOLERANCE:
                    raise ValueError(f"{condition} block {block}: the input does not reproduce the stored scores (max difference {worst:.2e}); no map is made")
                for k, identifier in enumerate(ids):
                    fidelity.append({"condition": condition, "batch_block": block, "ID": identifier, "role": "target" if identifier in target_ids else "context", "stored_score": reference[k], "predict_score": predicted[k],
                                     "cam_graph_score": float(md.sigmoid(cam_inputs["logit"][k]))})
                cams = md.grad_cam_maps(cam_inputs["activation"], cam_inputs["gradient"])
                for r in selected:
                    if block_of[r["ID"]] != block:
                        continue
                    identifier, k = r["ID"], ids.index(r["ID"])
                    normalized, raw_max, empty = md.normalize_cam(cams["cam"][k])
                    box, (width, height) = boxes[condition][identifier], images[identifier].size
                    mapped = md.map_cam_to_image(normalized, box, width, height)
                    facts = gt.map_facts(cams["cam"][k], mapped["image"], valid[condition][order.index(identifier)])
                    maps[(condition, identifier)] = {"image": mapped["image"], "coarse_raw": cams["cam"][k].copy()}
                    cam_rows.append({"condition": condition, "ID": identifier, "role": r["role"], "category": r["category"], "label": r["label"], "stored_score": reference[k], "raw_max": raw_max, "empty_after_relu": empty,
                                     "gradient_all_zero": cams["gradient_all_zero"][k], **facts, "box_x0": box[0], "box_y0": box[1], "box_x1": box[2], "box_y1": box[3], "image_width": width, "image_height": height,
                                     "batch_block": block})
        del graph, model
        keras.backend.clear_session()
    selection["rows"].to_csv(OUTPUTS["selection"], index=False, lineterminator="\n")
    selection["geometry"].to_csv(OUTPUTS["geometry"], index=False, lineterminator="\n")
    pd.DataFrame(fidelity).to_csv(OUTPUTS["score_fidelity"], index=False, lineterminator="\n")
    pd.DataFrame(cam_rows).to_csv(OUTPUTS["cam"], index=False, lineterminator="\n")
    arrays_out = {}
    for (condition, identifier), entry in maps.items():
        arrays_out[f"image__{condition}__{identifier}"], arrays_out[f"coarse__{condition}__{identifier}"] = entry["image"], entry["coarse_raw"]
    np.savez_compressed(MAPS_PATH, **arrays_out)
    print(f"{len(cam_rows)} Grad-CAM maps computed; the maps stay local")


if __name__ == "__main__":
    run() if "--run" in (sys.argv[1:] if "__file__" in globals() else []) else show()
