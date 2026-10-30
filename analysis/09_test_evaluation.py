# %% [09] Final test evaluation of the frozen models: metrics, comparisons and cluster bootstrap (CPU, stored test scores only)
#
# The test subset was evaluated once with the models, thresholds and rules frozen on the validation subset (07). This script does not run any
# model: it recomputes the test tables from the stored raw test scores (reports/tables/test_evaluation_scores.csv, 27 combinations x 376 images)
# and the frozen thresholds (reports/tables/final_operating_points.csv). Nothing is tuned or selected with a test label.
#
#   * metrics at the frozen thresholds (0.5 and the validation thresholds for 90 % / 95 % sensitivity), image level and case level (case score = maximum);
#   * the three main comparisons per seed (expert ROI - full image, automatic ROI - expert ROI, automatic ROI - full image), fine-tuned stage, Adam 1e-4;
#   * paired stratified cluster bootstrap over cases: 2,000 replicates, RNG seed 2026, the same draws for all groups and seeds;
#   * U-Net quality (Dice, IoU) and automatic-ROI diagnostics from the stored per-image table.
#
# Actions (the default recomputes everything in memory and compares it with the stored tables; nothing is written):
#   .venv/bin/python analysis/09_test_evaluation.py
#   .venv/bin/python analysis/09_test_evaluation.py --write    write the test tables (and, after --infer, the long score table the other analyses read); stops without writing if a table exists
#   env -u LD_LIBRARY_PATH .venv/bin/python analysis/09_test_evaluation.py --infer
#       explicit GPU action, meant to be run ONCE: model inference on the 376 test images (automatic ROIs, expert geometry, raw scores of the 27 combinations,
#       U-Net diagnostics). It stops if any of its outputs (including the central score table) already exist and never changes a model or a threshold. The order is fixed: the automatic ROIs are
#       derived from the test IMAGES only, before any test mask is read.

# %% Imports
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from busbra import results, auto_roi as ar, config, data, final_evaluation as fe, inference, roi, test_evaluation as te  # noqa: E402

TABLES = config.TABLES_DIR
TEST_DIR = TABLES / "test_evaluation"
SCORES_DIR = TEST_DIR / "scores"
AUTO_GEOMETRY_PATH = TEST_DIR / "auto_roi_geometry_test.csv"
EXPERT_GEOMETRY_PATH = TEST_DIR / "expert_roi_geometry_test.csv"
DIAGNOSTICS_PATH = TEST_DIR / "auto_roi_diagnostics_test.csv"
MASKS_PATH = config.PROJECT_ROOT / "data" / "processed" / "test_unet_predicted_masks.npz"  # pixel predictions of test images: local only
CENTRAL_SCORES = TABLES / "test_evaluation_scores.csv"  # long table of the 27 combinations; written by --write from the per-combination files of --infer


def score_path(combo: str) -> Path:
    return SCORES_DIR / f"{combo.replace('|', '_')}.csv"


def stored_scores() -> pd.DataFrame:
    """Raw test scores of all combinations in one long table: the stored table, or (after --infer) the per-combination files."""
    path = CENTRAL_SCORES
    if path.is_file():
        return pd.read_csv(path, dtype={"ID": str, "Case": str}, float_precision="round_trip")
    parts = []
    for entry in fe.combinations():
        frame = pd.read_csv(score_path(entry["combo_id"]), dtype={"ID": str, "Case": str}, float_precision="round_trip")
        parts.append(frame.assign(combo_id=entry["combo_id"], group_id=entry["group_id"], seed=entry["seed"]))
    return pd.concat(parts, ignore_index=True)[["combo_id", "group_id", "seed", "ID", "Case", "label", "malignant_score"]].astype({"seed": "Int64"})


# %% Computation
def compute() -> dict:
    manifest = data.read_manifest()
    data.check_split(manifest)
    test = data.subset_frame(manifest, "test")
    meta = pd.DataFrame({"ID": test["ID"], "Case": test["Case"], "label": test["label"].astype(int)})
    entries = {e["combo_id"]: e for e in fe.combinations()}
    thresholds, problems = te.load_frozen_thresholds(TABLES / "final_operating_points.csv", list(entries))
    long_scores = stored_scores()
    frames = {combo: part[["ID", "Case", "label", "malignant_score"]].reset_index(drop=True) for combo, part in long_scores.groupby("combo_id", sort=False)}
    for combo in entries:
        problems += te.score_frame_problems(frames[combo], meta, combo)
    if problems or len(frames) != te.N_COMBINATIONS:
        raise ValueError("inputs inconsistent: " + "; ".join(problems[:5]))
    core_rows, metric_rows, case_rows = [], [], []
    for combo, entry in entries.items():
        c, m, v = te.combination_rows(entry, frames[combo], thresholds)
        core_rows, metric_rows, case_rows = core_rows + c, metric_rows + m, case_rows + v
    core, metrics, case_view = pd.DataFrame(core_rows), pd.DataFrame(metric_rows), pd.DataFrame(case_rows)
    for frame in (core, metrics, case_view):
        frame["seed"] = frame["seed"].astype("Int64")
    for column in ("validation_needed_tp", "test_needed_tp_for_target"):
        core[column] = core[column].astype("Int64")
    if te.independent_confusion_check(core, frames, thresholds):
        raise ValueError("a confusion matrix differs from the independent routine")
    validation_core = pd.read_csv(TABLES / "final_operating_points.csv", dtype={"seed": "Int64"}, float_precision="round_trip")
    reference = te.validation_reference_table(core, validation_core)
    op_summary, metrics_summary = te.summarize_test(core, metrics)
    differences, difference_summary = te.paired_by_seed(core, metrics)
    structure = te.case_table(meta)
    draws = te.bootstrap_draws(structure["case_labels"])
    boot_frames = {(g, s): frames[f"{g}|seed{s}"] for g in te.MAIN_GROUPS.values() for s in te.SEEDS}
    started = time.perf_counter()
    values, observed = te.bootstrap_values(boot_frames, thresholds, draws, structure)
    if not np.allclose(observed, te.observed_from_tables(core, metrics), atol=1e-12, equal_nan=True):
        raise ValueError("the bootstrap code at the identity draw differs from the metric tables")
    bootstrap_summary, replicates = te.comparison_summary(values, observed, len(draws))
    print(f"bootstrap: {len(draws)} replicates in {time.perf_counter() - started:.0f} s")
    diag = pd.read_csv(TEST_DIR / "auto_roi_diagnostics_test.csv", dtype={"ID": str, "Case": str}, float_precision="round_trip")
    tables = {"metrics": metrics, "metrics_summary": metrics_summary, "operating_points": core, "operating_point_summary": op_summary, "case_view_image_thresholds": case_view,
              "validation_vs_test": reference, "comparisons_by_seed": differences, "comparisons_summary": difference_summary, "bootstrap_summary": bootstrap_summary,
              "bootstrap_replicates": replicates, "segmentation_summary": te.segmentation_summary(diag), "segmentation_dice_histogram": te.dice_histogram(diag),
              "segmentation_crop_flags": ar.crop_flag_table(diag), "segmentation_crop_categories": ar.category_table(diag), "segmentation_coverage": ar.coverage_summary(diag),
              "segmentation_box_definitions": ar.definition_table(diag), "segmentation_component_selection": ar.component_selection_table(diag)}
    return {"tables": tables, "draws": draws}


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", na_rep="")


same = results.same_as_stored


# %% Model inference on the test images (explicit GPU action, run once)
def infer() -> None:
    outputs = [AUTO_GEOMETRY_PATH, EXPERT_GEOMETRY_PATH, DIAGNOSTICS_PATH, MASKS_PATH, CENTRAL_SCORES] + [score_path(e["combo_id"]) for e in fe.combinations()]
    clash = [config.project_relative(p) for p in outputs if p.exists()]
    if clash:
        raise FileExistsError(f"already present: {clash[:3]} ...; the test subset is evaluated once; nothing was changed")
    manifest = data.read_manifest()
    data.check_split(manifest)
    test = data.subset_frame(manifest, "test")
    rows, ids = test.to_dict("records"), test["ID"].tolist()
    images = inference.load_images(rows)
    from busbra import training
    training.configure_gpu()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    # A. automatic ROIs from the test IMAGES only (no mask is opened), frozen before anything else is derived
    _, auto_geometry, predictions = ar.generate_rois(rows, images, inference.unet_probabilities(images))
    auto_geometry.to_csv(AUTO_GEOMETRY_PATH, index=False, lineterminator="\n", na_rep="")
    np.savez(MASKS_PATH, **ar.pack_masks(ids, predictions))
    # B. expert geometry from the test masks (after A)
    _, expert_geometry = data.load_roi_arrays(config.PROJECT_ROOT, rows, "test")
    expert_geometry.to_csv(EXPERT_GEOMETRY_PATH, index=False, lineterminator="\n", na_rep="")
    # C. raw scores of the 27 combinations; the crops are rebuilt from the frozen boxes and the images only
    auto_input = ar.inputs_from_geometry(images, auto_geometry)
    expert_input = ar.inputs_from_geometry(images, te.expert_geometry_for_inputs(expert_geometry))
    full_input = inference.full_array(images, 160)
    meta = {"ID": test["ID"], "Case": test["Case"], "label": test["label"].astype(int)}
    for entry in fe.combinations():
        spec, seed = entry["spec"], entry["seed"]
        if spec["adapter"] == "baseline":
            key = {"dummy_prior": "dummy", "logreg_l2": "logreg_l2"}[spec["model"]]
            scores = inference.baseline_scores(config.BASELINE_MODELS[key], images)
        else:
            variant = config.VARIANTS[spec.get("variant", "mobilenetv2_higher_lr_160_roi")]
            array = auto_input if spec["adapter"] == "auto_roi" else (expert_input if variant.data == "roi" else full_input)
            scores = inference.classifier_scores(config.run_paths(variant, seed)["checkpoint"], seed, array, variant.lr)
        pd.DataFrame({**meta, "malignant_score": scores}).to_csv(score_path(entry["combo_id"]), index=False, lineterminator="\n")
        print(f"{entry['combo_id']}: {len(scores)} scores")
    # D. diagnostics of the automatic segmentation against the test masks (no effect on any crop)
    crop_rows, image_rows = [], []
    for row, record, prediction in zip(rows, auto_geometry.to_dict("records"), predictions):
        description, array = roi.read_mask_raw(config.PROJECT_ROOT / row["mask_relpath"])
        expert = roi.decode_mask(description, array, expected_size=(int(record["width"]), int(record["height"])))
        crop_rows.append(ar.diagnose_image(expert, prediction, record))
        image_rows.append(te.segmentation_image_row(row, record, expert, prediction))
    diagnostics = auto_geometry.merge(pd.DataFrame(crop_rows), on="ID", how="left", validate="one_to_one").merge(pd.DataFrame(image_rows), on="ID", how="left", validate="one_to_one")
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False, lineterminator="\n", na_rep="")
    print("inference finished; run the default action to compute metrics, comparisons and the bootstrap")


# %% Run
def main(argv: list[str]) -> None:
    if "--infer" in argv:
        infer()
        return
    result = compute()
    tables = result["tables"]
    if "--write" in argv:
        paths = {name: TABLES / f"test_evaluation_{name}.csv" for name in tables}
        clash = [p.name for p in paths.values() if p.exists()]
        if clash:
            raise FileExistsError(f"already present: {clash}; nothing was changed")
        for name, frame in tables.items():
            write_csv(frame, paths[name])
        if not CENTRAL_SCORES.exists():  # a new result set: the other analyses read the long table of raw scores
            stored_scores().to_csv(CENTRAL_SCORES, index=False, lineterminator="\n")
        print(f"wrote {len(tables)} tables")
        return
    for name, frame in tables.items():
        path = TABLES / f"test_evaluation_{name}.csv"
        print(f"test_evaluation_{name + '.csv':44s} {len(frame):5d} rows  " + ("identical to the stored table" if path.is_file() and same(frame, path) else "DIFFERENT or not stored"))
    reference = TEST_DIR / "bootstrap_arrays.npz"
    if reference.is_file():  # optional historical reference of the draws of the stored result set; the draws themselves are generated from the case labels and the RNG seed
        stored_draws = np.load(reference)["draws"]
        print("bootstrap draws identical to the stored reference draws:", bool(np.array_equal(stored_draws.astype(np.int64), result["draws"])))
    else:
        print("no stored reference draws; no equality with a stored draw file is claimed (the draws are generated deterministically)")

if __name__ == "__main__":
    main(sys.argv[1:] if "__file__" in globals() else [])
