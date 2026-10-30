# %% [06] Automatic ROI: stored U-Net -> largest component -> 20 % box -> the SAME expert-ROI classifiers (validation images, inference only)
#
# The automatic ROI needs no manual mask at application time. The classifiers were trained on expert-mask ROIs (03_classification.py --preset expert_roi);
# this step only changes their INPUT at inference (an automatic crop instead of the expert crop). No second classifier is trained.
# Rule (src/busbra/auto_roi.py): the image goes through the 256 x 256 preparation into the stored U-Net, the probability map is mapped back to the decoded size
# and binarised there (>= 0.5); only the largest 8-connected component is kept (ties: first pixel in row-major order); an empty mask means the whole image is
# the crop (logged as full-image fallback); otherwise the tight box of that component is expanded by the frozen 20 % rule and the UNMASKED original is cropped.
# No function that creates a crop takes an expert mask. The U-Net was trained with expert masks and selected on the validation images: the results are explorative.
#
# Actions (exactly one; the default only reads stored results and starts nothing):
#   .venv/bin/python analysis/06_auto_roi.py                                show the stored crop statistics; derive the 15 study tables in memory from the stored scores and
#                                                                           diagnoses and compare them with the stored tables (nothing is written, nothing is inferred)
#   env -u LD_LIBRARY_PATH .venv/bin/python analysis/06_auto_roi.py --run   U-Net inference, crops, classifier scores (5 seeds x frozen / fine-tuned), diagnostics (GPU)
#   .venv/bin/python analysis/06_auto_roi.py --aggregate                    CPU: write the 15 study and diagnosis tables (run metrics, summaries, paired differences, threshold curves,
#                                                                           operating points, crop flags and categories, coverage, box definitions, component selection, concurrence with
#                                                                           classification errors, named examples) from the stored scores and diagnoses; run it after --run
# --run and --aggregate stop before any work if one of their output files already exists. --aggregate reads the stored validation scores of the full-image and expert-ROI runs, the new
# automatic scores, the per-image diagnoses and the fixed example selection of the segmentation study; no image, mask or model is opened. The test subset is handled by
# 09_test_evaluation.py with the same functions.

# %% Imports
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from busbra import auto_roi as ar, config, data, inference, results, roi, roi_study  # noqa: E402



def geometry_path() -> Path:
    return config.TABLES_DIR / "auto_roi_geometry_validation.csv"


def masks_path() -> Path:
    return config.TABLES_DIR / "auto_roi" / "predicted_masks_validation.npz"  # pixel predictions of validation images: local only


def table_paths() -> dict:
    return {key: config.TABLES_DIR / file for key, file in roi_study.AUTO_FILES.items()}


# %% Show: stored crop statistics and the in-memory derivation of the study tables
def derive_tables() -> dict:
    """The 15 study and diagnosis tables from the PRIMARY stored inputs only (scores, per-image diagnoses, fixed example selection); never from an older output table. Raises FileNotFoundError
    for a missing input and ValueError for inconsistent inputs."""
    manifest = data.read_manifest()
    data.check_split(manifest)
    validation = data.subset_frame(manifest, "validation")
    sets = roi_study.auto_score_sets()
    roi_study.check_score_sets(sets, validation)
    diagnostics = roi_study.read_diagnostics(roi_study.diagnostics_path(), validation)
    selection = pd.read_csv(roi_study.segmentation_selection_path(), dtype={"ID": str, "Case": str}, float_precision="round_trip")
    return roi_study.auto_tables(sets, diagnostics, selection)


def show() -> None:
    if not geometry_path().is_file():
        print("no stored automatic ROIs")
        return
    geometry = pd.read_csv(geometry_path())
    print(f"{len(geometry)} validation images: {int(geometry['fallback_full_image'].sum())} full-image fallbacks, {int((geometry['n_components'] > 1).sum())} with several components, "
          f"{int((~geometry['fallback_full_image']).sum())} crops from a component")
    try:
        tables = derive_tables()
    except FileNotFoundError as exc:
        print(f"the study tables were not derived: {exc}")
        return
    for key, path in table_paths().items():
        stored = path.is_file()
        print(f"{path.name:44s} {len(tables[key]):6d} rows  " + (("identical to the stored table" if results.same_as_stored(tables[key], path) else "DIFFERENT from the stored table") if stored
                                                                  else "derived; no stored table, no equality claimed"))


# %% Aggregate (CPU)
def aggregate() -> None:
    targets = table_paths()
    clash = [path.name for path in targets.values() if path.exists()]
    if clash:
        raise FileExistsError(f"already present: {clash}; nothing was changed")
    tables = derive_tables()
    for key, path in targets.items():
        tables[key].to_csv(path, index=False, lineterminator="\n", na_rep="")
    print(f"wrote {len(targets)} tables to {config.TABLES_DIR}")


# %% Run
def run() -> None:
    outputs = [geometry_path(), masks_path(), roi_study.diagnostics_path()] + [roi_study.auto_score_path(s, st) for s in config.SEEDS for st in ar.STAGE_VARIANTS]
    clash = [config.project_relative(p) for p in outputs if p.exists()]
    if clash:
        raise FileExistsError(f"already present: {clash[:3]} ...; nothing was changed")
    manifest = data.read_manifest()
    data.check_split(manifest)
    validation = data.subset_frame(manifest, "validation")
    rows = validation.to_dict("records")
    labels = validation["label"].astype(int).to_numpy()
    images = inference.load_images(rows)
    from busbra import training
    training.configure_gpu()
    # phase 1: crops from the images only; the expert masks are read afterwards, for diagnostics
    arrays, geometry, predictions = ar.generate_rois(rows, images, inference.unet_probabilities(images))
    packed = ar.pack_masks(list(validation["ID"]), predictions)
    geometry.to_csv(geometry_path(), index=False, lineterminator="\n", na_rep="")
    masks_path().parent.mkdir(parents=True, exist_ok=True)
    np.savez(masks_path(), **packed)
    # phase 2: the same classifier checkpoints as in the expert-ROI study, applied to the automatic crops
    for seed in config.SEEDS:
        for stage, name in ar.STAGE_VARIANTS.items():
            variant = config.VARIANTS[name]
            scores = inference.classifier_scores(config.run_paths(variant, seed)["checkpoint"], seed, arrays, variant.lr)
            path = roi_study.auto_score_path(seed, stage)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"ID": validation["ID"], "Case": validation["Case"], "label": labels, "malignant_score": scores}).to_csv(path, index=False, lineterminator="\n")
            print(f"seed {seed} {stage}: {len(scores)} scores written")
    # phase 3: diagnostics against the expert masks (no effect on any crop)
    diagnostics = []
    for row, record, prediction in zip(rows, geometry.to_dict("records"), predictions):
        description, array = roi.read_mask_raw(config.PROJECT_ROOT / row["mask_relpath"])
        expert = roi.decode_mask(description, array, expected_size=(int(record["width"]), int(record["height"])))
        diagnostics.append(ar.diagnose_image(expert, prediction, record))
    geometry.merge(pd.DataFrame(diagnostics), on="ID", how="left", validate="one_to_one").to_csv(roi_study.diagnostics_path(), index=False, lineterminator="\n", na_rep="")
    show()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true", help="U-Net inference, crops, classifier scores and diagnoses (GPU)")
    action.add_argument("--aggregate", action="store_true", help="write the 15 study and diagnosis tables from the stored scores and diagnoses (CPU)")
    args = parser.parse_args(argv)
    if args.run:
        run()
    elif args.aggregate:
        aggregate()
    else:
        show()


if __name__ == "__main__":
    main(sys.argv[1:] if "__file__" in globals() else [])
