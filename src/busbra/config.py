"""Paths, fixed design constants and the classification variants of the BUS-BRA project.

Everything that differs between the neural-network experiments is listed in VARIANTS. Importing this module has no side
effects (no file access, no TensorFlow).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest_split.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "BUSBRA"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
IMAGENET_WEIGHTS_FILE = (Path(os.environ.get("KERAS_HOME", str(Path.home() / ".keras"))).expanduser() / "models"
                         / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5")

# ---- fixed design -------------------------------------------------------------------------------------------------------
# Fixed holdout from the outer folds of 5-fold-cv.csv (no cross-validation): folds 3-5 train, 2 validation, 1 test.
SPLIT_OF_OUTER_FOLD = {1: "test", 2: "validation", 3: "train", 4: "train", 5: "train"}
EXPECTED_IMAGES = {"train": 1114, "validation": 385, "test": 376}
EXPECTED_CASES = {"train": 639, "validation": 213, "test": 212}
SEEDS = [7, 21, 42, 123, 2026]
SEED_OFFSETS = {"flip": 1, "dropout": 2, "shuffle": 3, "rotation": 4, "zoom": 5}
HISTORICAL_SEED = 42  # the single runs of steps 3A-3E use this one seed for flip and shuffle; their Dropout is not seeded
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 5
THRESHOLD = 0.5
BACKBONE_NAME = "mobilenetv2_backbone"
HEAD_LAYER = "output"
EXPECTED_BATCHNORM_LAYERS = 52
TOTAL_PARAMETERS = 2_259_265
ROTATION_DEGREES = 10.0
ROTATION_FACTOR = ROTATION_DEGREES / 360.0  # Keras takes a fraction of a full turn
ZOOM_RANGE = (-0.10, 0.10)

# Backbone layers that are released in fine-tuning (everything else, including all BatchNorm layers, stays fixed).
_BLOCK_14_16 = [f"block_{b}_{part}" for b in (14, 15, 16) for part in ("expand", "depthwise", "project")]
RELEASED_LAYERS = {
    "limited": ["block_16_expand", "block_16_depthwise", "block_16_project", "Conv_1"],
    "extended": _BLOCK_14_16 + ["Conv_1"],
}
TRAINABLE_PARAMETERS = {"limited": 880_321, "extended": 1_512_001, "frozen": 1_281, "cnn": 100_993}


@dataclass(frozen=True)
class Variant:
    """One neural classification experiment.

    model        'cnn' or 'mobilenet'
    size         input side length in pixels
    augmentation 'flip' or 'flip_rotation_zoom' (random layers are active in training only)
    data         'full' image or expert-mask 'roi' crop
    start        'random', 'imagenet' (local weight file), 'imagenet_via_160' (250 px graph that receives the 160 px
                 backbone values) or 'checkpoint' (best checkpoint of `source`, same seed)
    release      None, 'limited' or 'extended' (released backbone layers in fine-tuning)
    seed_mode    'study' (explicit offsets of SEED_OFFSETS) or 'historical' (steps 3A-3E, see HISTORICAL_SEED)
    carry        'weights' (network weights are copied into a freshly built graph) or 'object' (the loaded model is
                 continued, historical rule of steps 3C-3E)
    graph_weights  'none' or 'imagenet': with which weights the fresh graph is BUILT before the trained network weights are copied in
                 (all backbone and head values are overwritten; the build order of the random initialisers is kept as in the original studies)
    """
    name: str
    namespace: str
    model: str
    size: int
    augmentation: str
    data: str
    start: str
    lr: float
    max_epochs: int
    release: str | None = None
    source: str | None = None
    seed_mode: str = "study"
    carry: str = "weights"
    graph_weights: str = "none"
    graph_name: str = "mobilenetv2_frozen"


def _v(**kw) -> Variant:
    return Variant(**kw)


VARIANTS: dict[str, Variant] = {v.name: v for v in [
    # steps 3A-3E: single runs with seed 42 (historical seed rules)
    _v(name="small_cnn", namespace="single_runs", model="cnn", size=160, augmentation="flip", data="full", start="random", lr=1e-3, max_epochs=30, seed_mode="historical", graph_name="small_cnn"),
    _v(name="mobilenetv2_frozen", namespace="single_runs", model="mobilenet", size=160, augmentation="flip", data="full", start="imagenet", lr=1e-3, max_epochs=30, seed_mode="historical"),
    _v(name="mobilenetv2_finetuned", namespace="single_runs", model="mobilenet", size=160, augmentation="flip", data="full", start="checkpoint", source="mobilenetv2_frozen", lr=1e-5,
       max_epochs=20, release="limited", seed_mode="historical", carry="object"),
    _v(name="mobilenetv2_extended", namespace="single_runs", model="mobilenet", size=160, augmentation="flip", data="full", start="checkpoint", source="mobilenetv2_frozen", lr=1e-5,
       max_epochs=20, release="extended", seed_mode="historical", carry="object"),
    _v(name="mobilenetv2_higher_lr", namespace="single_runs", model="mobilenet", size=160, augmentation="flip", data="full", start="checkpoint", source="mobilenetv2_frozen", lr=1e-4,
       max_epochs=20, release="extended", seed_mode="historical", carry="object"),
    # step 3F: five pipelines over five seeds
    _v(name="small_cnn_160", namespace="multiseed", model="cnn", size=160, augmentation="flip", data="full", start="random", lr=1e-3, max_epochs=30, graph_name="small_cnn"),
    _v(name="mobilenetv2_frozen_160", namespace="multiseed", model="mobilenet", size=160, augmentation="flip", data="full", start="imagenet", lr=1e-3, max_epochs=30),
    _v(name="mobilenetv2_finetuned_160", namespace="multiseed", model="mobilenet", size=160, augmentation="flip", data="full", start="checkpoint", source="mobilenetv2_frozen_160", lr=1e-5,
       max_epochs=20, release="limited"),
    _v(name="mobilenetv2_extended_160", namespace="multiseed", model="mobilenet", size=160, augmentation="flip", data="full", start="checkpoint", source="mobilenetv2_frozen_160", lr=1e-5,
       max_epochs=20, release="extended"),
    _v(name="mobilenetv2_higher_lr_160", namespace="multiseed", model="mobilenet", size=160, augmentation="flip", data="full", start="checkpoint", source="mobilenetv2_frozen_160", lr=1e-4,
       max_epochs=20, release="extended"),
    # step 4A: 250 px input
    _v(name="mobilenetv2_frozen_250", namespace="resolution250", model="mobilenet", size=250, augmentation="flip", data="full", start="imagenet_via_160", lr=1e-3, max_epochs=30,
       graph_name="mobilenetv2_250"),
    _v(name="mobilenetv2_higher_lr_250", namespace="resolution250", model="mobilenet", size=250, augmentation="flip", data="full", start="checkpoint", source="mobilenetv2_frozen_250",
       lr=1e-4, max_epochs=20, release="extended", graph_name="mobilenetv2_250"),
    # step 4B: flip + rotation + zoom
    _v(name="mobilenetv2_frozen_160_aug", namespace="augmentation160", model="mobilenet", size=160, augmentation="flip_rotation_zoom", data="full", start="imagenet", lr=1e-3,
       max_epochs=30, graph_name="mobilenetv2_160_aug"),
    _v(name="mobilenetv2_higher_lr_160_aug", namespace="augmentation160", model="mobilenet", size=160, augmentation="flip_rotation_zoom", data="full", start="checkpoint",
       source="mobilenetv2_frozen_160_aug", lr=1e-4, max_epochs=20, release="extended", graph_name="mobilenetv2_160_aug", graph_weights="imagenet"),
    # step 4E: expert-mask ROI crop (same augmentation and settings as 4B)
    _v(name="mobilenetv2_frozen_160_roi", namespace="expert_roi", model="mobilenet", size=160, augmentation="flip_rotation_zoom", data="roi", start="imagenet", lr=1e-3,
       max_epochs=30, graph_name="mobilenetv2_160_aug"),
    _v(name="mobilenetv2_higher_lr_160_roi", namespace="expert_roi", model="mobilenet", size=160, augmentation="flip_rotation_zoom", data="roi", start="checkpoint",
       source="mobilenetv2_frozen_160_roi", lr=1e-4, max_epochs=20, release="extended", graph_name="mobilenetv2_160_aug", graph_weights="imagenet"),
]}

# Validation loss and ROC-AUC of the loaded frozen checkpoint (epoch 20) before the first fine-tuning update of step 3C: the start reference of the learning-curve figure. They come from
# the exact evaluation of that checkpoint (float64); the ROC-AUC equals the stored validation metric of the frozen run, the stored epoch loss differs from the loss only in the 8th digit (float32 logging).
FROZEN_START_VALIDATION = {"loss": 0.47328917620064515, "roc_auc": 0.8159692307692308}

# Command-line presets: which variants (in order) belong to one study.
PRESETS = {
    "single_runs": ["small_cnn", "mobilenetv2_frozen", "mobilenetv2_finetuned", "mobilenetv2_extended", "mobilenetv2_higher_lr"],
    "multiseed": [n for n, v in VARIANTS.items() if v.namespace == "multiseed"],
    "resolution250": [n for n, v in VARIANTS.items() if v.namespace == "resolution250"],
    "augmentation160": [n for n, v in VARIANTS.items() if v.namespace == "augmentation160"],
    "expert_roi": [n for n, v in VARIANTS.items() if v.namespace == "expert_roi"],
}
PRESET_SEEDS = {"single_runs": [HISTORICAL_SEED]}
PRESET_STEPS = {"single_runs": "3A-3E", "multiseed": "3F", "resolution250": "4A", "augmentation160": "4B", "expert_roi": "4E"}
# Reference groups of the later studies: preset -> [(preset of the stored runs, variants)]; their stored runs are read, never trained again.
_FROZEN_HIGHER_160 = ["mobilenetv2_frozen_160", "mobilenetv2_higher_lr_160"]
PRESET_REFERENCES = {"multiseed": [], "resolution250": [("multiseed", _FROZEN_HIGHER_160)], "augmentation160": [("multiseed", _FROZEN_HIGHER_160), ("resolution250", PRESETS["resolution250"])],
                     "expert_roi": [("augmentation160", PRESETS["augmentation160"])]}
# The multi-seed pipelines and the early single runs of steps 3A-3E share their configuration; the single run is the seed-42 value stored in an earlier flat layout.
SINGLE_RUN_OF = {"small_cnn_160": "small_cnn", "mobilenetv2_frozen_160": "mobilenetv2_frozen", "mobilenetv2_finetuned_160": "mobilenetv2_finetuned", "mobilenetv2_extended_160": "mobilenetv2_extended",
                 "mobilenetv2_higher_lr_160": "mobilenetv2_higher_lr"}


def seeds_for(preset: str) -> list[int]:
    return PRESET_SEEDS.get(preset, SEEDS)


# The single runs of steps 3A-3E (seed 42) were stored in a flat layout: reports/tables/<prefix>_{history,metrics,predictions}.csv and models/<directory>/...
_SINGLE_RUN_FILES = {"small_cnn": ("small_cnn", "small_cnn"), "mobilenetv2_frozen": ("mobilenetv2_frozen", "mobilenetv2_frozen"), "mobilenetv2_finetuned": ("mobilenetv2_finetuned", "mobilenetv2_finetuned"),
                     "mobilenetv2_extended": ("mobilenetv2_extended", "mobilenetv2_extended"), "mobilenetv2_higher_lr": ("mobilenetv2_higher_lr", "mobilenetv2_higher_lr")}


def run_paths(variant: Variant, seed: int) -> dict[str, Path]:
    """Files of one run. Every seed and variant has its own directory, nothing is shared between runs (except the historical flat layout of the single runs)."""
    if variant.namespace == "single_runs":
        prefix, directory = _SINGLE_RUN_FILES[variant.name]
        models = MODELS_DIR / directory
        return {"table_dir": TABLES_DIR, "model_dir": models, "history": TABLES_DIR / f"{prefix}_history.csv", "metrics": TABLES_DIR / f"{prefix}_metrics.csv", "predictions": TABLES_DIR / f"{prefix}_predictions.csv",
                "config": models / f"{variant.name}_config.json", "checkpoint": models / f"{variant.name}_best.keras", "summary": models / "model_summary.txt"}
    tables = TABLES_DIR / variant.namespace / f"seed_{seed}" / variant.name
    models = MODELS_DIR / variant.namespace / f"seed_{seed}" / variant.name
    return {"table_dir": tables, "model_dir": models, "history": tables / "history.csv", "metrics": tables / "metrics.csv", "predictions": tables / "predictions.csv",
            "config": tables / "config.json", "checkpoint": models / f"{variant.name}_seed{seed}_best.keras", "summary": models / "model_summary.txt"}


def project_relative(path: Path) -> str:
    """Path relative to the project folder for messages (the full path for anything outside it)."""
    try:
        return Path(path).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_outputs(variant: Variant, seed: int) -> list[Path]:
    """Every file that training one variant for one seed writes (a run is only started when none of them exists)."""
    paths = run_paths(variant, seed)
    return [paths[k] for k in ("history", "metrics", "predictions", "config", "checkpoint", "summary")]


BASELINE_MODELS = {"dummy": MODELS_DIR / "baselines" / "dummy_prior.joblib", "logreg_l2": MODELS_DIR / "baselines" / "logreg_l2_64px.joblib"}
UNET_CHECKPOINT = MODELS_DIR / "segmentation" / "unet_seed7" / "unet_256_seed7_best.keras"
