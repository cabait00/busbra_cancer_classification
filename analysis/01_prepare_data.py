# %% [01] Data preparation: BUS-BRA metadata, fixed split and the documented data findings
#
# Source: Zenodo record 8231412 (CC BY 4.0), original BUSBRA.zip extracted to data/raw/BUSBRA/ (1,875 images and masks, 1,064 cases;
# Gomez-Flores et al., Medical Physics 51, 2024, doi:10.1002/mp.16812; the bundled LICENSE.txt requires citing the paper).
# The split is a fixed holdout from the OUTER folds of 5-fold-cv.csv (fold 1 test, 2 validation, 3-5 train); the inner valid_* columns
# are not used and no cross-validation was run. data/processed/manifest_split.csv is the only authoritative assignment of images to
# subsets. It is read by every later step and is never rewritten by this script.
#
# Actions (the default reads only metadata and writes nothing):
#   .venv/bin/python analysis/01_prepare_data.py            check the stored manifest against the raw metadata and print the data findings
#   .venv/bin/python analysis/01_prepare_data.py --build    decode all images and masks and write the manifest; stops if a manifest already exists
#
# Findings that stay visible (not corrected silently): four CSV image heights differ from the decoded sizes (the decoded sizes are used);
# case 856 has two different BI-RADS values; every inner valid_* column puts cases into both inner training and inner validation.
# Images contain embedded text and device markers that a model may use as shortcuts; they are neither removed nor read by OCR.

# %% Imports
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from busbra import config, data  # noqa: E402


# %% Check
def check() -> None:
    manifest = data.read_manifest()
    data.check_split(manifest)
    metadata = data.read_metadata()
    columns = ["ID", "Case", "Pathology", "label", "outer_kfold", "split", "Device", "Histology", "BIRADS", "Width", "Height", "Side", "BBOX"]
    if not manifest[columns].equals(metadata[columns]):
        raise ValueError("the stored manifest differs from bus_data.csv / 5-fold-cv.csv in the metadata columns")
    print(f"manifest_split.csv agrees with the raw metadata: {len(manifest)} images, {manifest['Case'].nunique()} cases")
    for subset in data.SUBSETS:
        part = manifest[manifest["split"] == subset]
        print(f"  {subset:10s} {len(part):5d} images {part['Case'].nunique():4d} cases  malignant {int((part['label'] == '1').sum()):4d} benign {int((part['label'] == '0').sum()):4d}")
    findings = data.data_findings(manifest=manifest)
    print("CSV size mismatches (decoded sizes are used):", findings["csv_size_mismatches"])
    print("cases with different BI-RADS values:", findings["birads_conflicts"])
    print("inner valid_* columns, cases in both training and validation:", findings["inner_overlap"])
    train = manifest[manifest["split"] == "train"]
    print("\ntraining subset, images per device and class")
    print(pd.crosstab(train["Device"], train["Pathology"]).to_string())


# %% Build
def build() -> None:
    if config.MANIFEST_PATH.exists():
        raise FileExistsError(f"{config.MANIFEST_PATH} already exists and is the frozen split; nothing was changed")
    manifest = data.build_manifest()
    data.check_split(manifest)
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.MANIFEST_PATH, index=False, lineterminator="\n")
    print(f"wrote {config.MANIFEST_PATH.relative_to(config.PROJECT_ROOT)} ({len(manifest)} rows)")


if __name__ == "__main__":
    build() if "--build" in (sys.argv[1:] if "__file__" in globals() else []) else check()
