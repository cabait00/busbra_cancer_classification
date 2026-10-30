# %% [05] Segmentation of the annotated lesion: compact U-Net from scratch (train and validation only)
#
# The network predicts the pixels of the annotated lesion, not benign / malignant. The only model input is the ultrasound image; the expert mask is the
# training TARGET and the reference of the quality measurement. Image and mask share one geometry (whole image fitted into 256 x 256, black padding,
# masks with nearest-neighbour interpolation); predictions are mapped back to the decoded original pixels (probability >= 0.5). The validation masks
# steer early stopping and the checkpoint choice, so the validation Dice is explorative and probably somewhat optimistic. Test data stay untouched.
# Design: src/busbra/segmentation.py (geometry, metrics), src/busbra/segmentation_model.py (U-Net, loss BCE + soft Dice, batch stream with synchronous flip).
# Seed 7 (flip seed 8, shuffle seed 10), Adam 1e-3, batch 8, at most 40 epochs, early stopping on val_loss with patience 7, float32, jit_compile=False.
#
# Actions (the default only reads the stored summary):
#   .venv/bin/python analysis/05_segmentation.py            show the stored validation quality
#   PYTHONHASHSEED=7 env -u LD_LIBRARY_PATH .venv/bin/python analysis/05_segmentation.py --train    train on the GPU and write the results; stops before any work if one of its output files exists
# Local contact-sheet previews of train / validation images are not produced here; the visual inspection of segmentations is done by the project owner.

# %% Imports
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from busbra import config, data, segmentation as seg  # noqa: E402

PATHS = seg.run_paths(config.PROJECT_ROOT)
OUTPUT_KEYS = ("history", "config", "summary", "groups", "flags", "box", "histogram", "selection", "per_image", "checkpoint", "model_summary")  # every file --train writes


# %% Show
def show() -> None:
    if not PATHS["summary"].is_file():
        print("no stored segmentation results")
        return
    table = pd.read_csv(PATHS["summary"])
    print(table[(table["space"] == "original_area") & (table["metric"] == "dice")].to_string(index=False))


# %% Train
def train() -> None:
    clash = [config.project_relative(PATHS[k]) for k in OUTPUT_KEYS if PATHS[k].exists()]
    if clash:
        raise FileExistsError(f"{len(clash)} segmentation output files already exist, for example {clash[0]}; nothing was changed")
    if os.environ.get("PYTHONHASHSEED") != str(seg.SEED):
        raise RuntimeError(f"the U-Net run is defined with PYTHONHASHSEED={seg.SEED} in the environment of the Python process (it cannot be set from inside); start it as "
                           f"PYTHONHASHSEED={seg.SEED} env -u LD_LIBRARY_PATH .venv/bin/python analysis/05_segmentation.py --train")
    manifest = data.read_manifest()
    data.check_split(manifest)
    frames = {s: data.subset_frame(manifest, s) for s in ("train", "validation")}
    loaded = {s: seg.load_pairs(config.PROJECT_ROOT, frame.to_dict("records"), s) for s, frame in frames.items()}
    x_train, y_train, x_val, y_val = (loaded["train"]["images"], loaded["train"]["masks"], loaded["validation"]["images"], loaded["validation"]["masks"])
    from busbra import models, segmentation_model as sm, training  # TensorFlow is imported here, only for training
    import keras
    import tensorflow as tf
    gpus = training.configure_gpu()
    models.reset_random_sources(seg.SEED)
    with tf.device("/GPU:0"):
        model = sm.compile_model(sm.build_unet(), seg.LEARNING_RATE)
    parameters = sm.parameter_report(model)
    if not parameters["ok"]:
        raise ValueError(f"unexpected parameter counts {parameters}")
    PATHS["table_dir"].mkdir(parents=True, exist_ok=True)
    PATHS["model_dir"].mkdir(parents=True, exist_ok=True)
    callbacks, early_stopping, timer = sm.build_callbacks(PATHS["checkpoint"])
    batch = seg.BATCH_SIZE
    train_set = sm.make_train_dataset(x_train, y_train, batch, seg.SEED + seg.SEED_OFFSETS["shuffle"], seg.SEED + seg.SEED_OFFSETS["flip"], seg.MAX_EPOCHS)
    val_set = sm.make_eval_dataset(x_val, y_val, batch)
    started = time.perf_counter()
    fit = model.fit(train_set, steps_per_epoch=sm.steps_per_epoch(len(x_train), batch), epochs=seg.MAX_EPOCHS, validation_data=val_set, callbacks=callbacks, shuffle=False, verbose=2)
    seconds = round(time.perf_counter() - started, 1)
    history = pd.DataFrame(fit.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    history["epoch_seconds"] = timer.seconds[:len(history)]
    best_epoch = int(history["val_loss"].to_numpy().argmin()) + 1
    history["is_best_epoch"] = history["epoch"] == best_epoch
    history.to_csv(PATHS["history"], index=False, lineterminator="\n")
    stop_reason = "early_stopping" if early_stopping.stopped_epoch > 0 else ("epoch_limit_reached" if len(history) == seg.MAX_EPOCHS else "other")
    # the reloaded model is compiled again (jit_compile=False) before predicting
    best = sm.compile_model(keras.models.load_model(PATHS["checkpoint"], compile=False), seg.LEARNING_RATE)
    probabilities = {"train": sm.predict_probabilities(best, x_train, batch), "validation": sm.predict_probabilities(best, x_val, batch)}
    masks = {"train": y_train, "validation": y_val}
    per_image = []
    for subset in frames:
        rows = []
        for i, base in enumerate(loaded[subset]["geometry"].to_dict("records")):
            analysis = seg.analyse_prediction(probabilities[subset][i], masks[subset][i, ..., 0], loaded[subset]["originals"][i], seg.fit_geometry(base["width"], base["height"]))
            rows.append({**base, **analysis})
        per_image.append(pd.DataFrame(rows))
    per_image = pd.concat(per_image, ignore_index=True)
    selection = [(i, "train", r) for i, r in seg.select_representative(per_image[per_image["subset"] == "train"])]
    selection += [(i, "validation", r) for i, r in seg.select_representative(per_image[per_image["subset"] == "validation"])]
    selection += [(i, "validation", r) for i, r in seg.select_error_analysis(per_image)]
    selection = pd.DataFrame([{"block": "error_analysis" if r.startswith("error analysis") else "representative", "subset": s, "ID": i, "reason": r} for i, s, r in selection]).merge(
        per_image[["subset", "ID", "Case", "class", "Device", "expert_fraction_original", "dice_original", "iou_original", "pred_empty", "pred_components"]], on=["subset", "ID"], how="left")
    tables = {"summary": seg.metric_summary(per_image), "groups": seg.group_summary(per_image), "flags": seg.flag_table(per_image), "box": seg.box_summary(per_image),
              "histogram": seg.dice_histogram(per_image), "selection": selection}
    for key, table in tables.items():
        table.to_csv(PATHS[key], index=False, lineterminator="\n", na_rep="")
    per_image.to_csv(PATHS["per_image"], index=False, lineterminator="\n", na_rep="")
    lines: list[str] = []
    best.summary(print_fn=lambda line, **_: lines.append(line))
    PATHS["model_summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    settings = {"seed": seg.SEED, "random_sources": seg.seed_plan(seg.SEED), "optimizer": {"name": "Adam", "learning_rate": seg.LEARNING_RATE}, "batch_size": batch, "max_epochs": seg.MAX_EPOCHS,
                "early_stopping": {"monitor": "val_loss", "patience": seg.EARLY_STOPPING_PATIENCE, "restore_best_weights": True}, "threshold": seg.THRESHOLD, "parameters": parameters,
                "result": {"best_epoch": best_epoch, "epochs_run": len(history), "stop_reason": stop_reason, "best_val_loss": float(history["val_loss"].min()), "training_seconds": seconds},
                "versions": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__, "gpus": gpus},
                "note": "the validation expert masks steer early stopping and the checkpoint; the validation quality is explorative"}
    PATHS["config"].write_text(json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8")
    val = per_image[per_image["subset"] == "validation"]
    print(f"epochs {len(history)}, best epoch {best_epoch}; validation original-area Dice mean {val['dice_original'].mean():.4f}, median {val['dice_original'].median():.4f}")


if __name__ == "__main__":
    train() if "--train" in (sys.argv[1:] if "__file__" in globals() else []) else show()
