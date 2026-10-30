"""Shared classification training: one function trains any variant of `config.VARIANTS` for one seed and writes its results.

Flow (identical for the small CNN and all MobileNetV2 variants): refuse to overwrite existing results -> load train and validation
inputs -> build the start model (fresh session and seed) -> fit with the fixed callbacks -> reload the best checkpoint -> predict ->
write history, metrics, predictions, config and model summary. Test images are never loaded here. Nothing runs at import time.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from busbra import config, data, models
from busbra.config import Variant
from busbra.evaluation import binary_metrics, sigmoid_scores

os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # the backend of all project models; chosen before Keras is first imported
MONITOR = "val_loss"


def configure_gpu() -> list[str]:
    """Memory growth before the GPU is initialised; a missing GPU is an error (there is no CPU fallback)."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow sees no GPU. Training needs the GPU; there is no CPU fallback.")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        if not all(tf.config.experimental.get_memory_growth(g) is True for g in gpus):
            raise RuntimeError("Memory growth can no longer be set because the GPU runtime is already initialised; restart the Python process.") from exc
    tf.config.set_soft_device_placement(False)  # an op pinned to the GPU must not move to the CPU
    return [g.name for g in gpus]


def make_train_dataset(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle_seed: int):
    """Shuffled batches, reshuffled every epoch with a fixed seed; the last smaller batch is kept."""
    import tensorflow as tf
    dataset = tf.data.Dataset.from_tensor_slices((images, labels.astype("float32").reshape(-1, 1)))
    dataset = dataset.shuffle(len(images), seed=shuffle_seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def make_eval_dataset(images: np.ndarray, labels: np.ndarray, batch_size: int):
    """Fixed order (the order of the arrays), no shuffling, nothing dropped."""
    import tensorflow as tf
    dataset = tf.data.Dataset.from_tensor_slices((images, labels.astype("float32").reshape(-1, 1)))
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


_TIMER_CLASS = None


def epoch_timer():
    """Callback that records the wall-clock seconds of every epoch (training plus validation); Keras is imported on first use."""
    global _TIMER_CLASS
    if _TIMER_CLASS is None:
        import keras

        class EpochTimer(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.seconds: list[float] = []
                self._started = 0.0

            def on_epoch_begin(self, epoch, logs=None):
                self._started = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                self.seconds.append(round(time.perf_counter() - self._started, 3))

        _TIMER_CLASS = EpochTimer
    return _TIMER_CLASS()


def build_callbacks(checkpoint_path, patience: int = config.EARLY_STOPPING_PATIENCE):
    """Checkpointing and early stopping both follow val_loss (min). With min_delta=0 the first best epoch stays the best epoch on a tie."""
    import keras
    checkpoint = keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor=MONITOR, mode="min", save_best_only=True, save_weights_only=False, verbose=0)
    early_stopping = keras.callbacks.EarlyStopping(monitor=MONITOR, mode="min", patience=patience, min_delta=0, restore_best_weights=True, verbose=1)
    timer = epoch_timer()
    return [checkpoint, early_stopping, keras.callbacks.TerminateOnNaN(), timer], early_stopping, timer


def shuffle_seed(variant: Variant, seed: int) -> int:
    return config.HISTORICAL_SEED if variant.seed_mode == "historical" else seed + config.SEED_OFFSETS["shuffle"]


def train_variant(variant: Variant, seed: int, batch_size: int = config.BATCH_SIZE) -> dict:
    """Train one variant for one seed. Raises FileExistsError if results of this run already exist (nothing is overwritten or deleted).

    Study variants require PYTHONHASHSEED=<seed> in the process environment, as the worker processes of the earlier studies had."""
    if variant.seed_mode == "study" and os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(f"study runs need PYTHONHASHSEED={seed} in the environment of the Python process (it cannot be set from inside); start the process as "
                           f"PYTHONHASHSEED={seed} .venv/bin/python ... --seeds {seed}")
    paths = config.run_paths(variant, seed)
    existing = [str(path) for path in config.run_outputs(variant, seed) if path.exists()]
    if existing:
        raise FileExistsError(f"results of {variant.name} / seed {seed} already exist: {existing}. Nothing was changed.")
    inputs = data.load_classification_inputs(variant.data, variant.size)
    gpus = configure_gpu()
    import keras
    import tensorflow as tf

    def checkpoint_of(name: str):
        return config.run_paths(config.VARIANTS[name], seed)["checkpoint"]

    model = models.build_start_model(variant, seed, checkpoint_of)
    paths["table_dir"].mkdir(parents=True, exist_ok=True)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    (train_frame, train_x, train_y), (val_frame, val_x, val_y) = inputs["train"], inputs["validation"]
    callbacks, early_stopping, timer = build_callbacks(paths["checkpoint"])
    train_set = make_train_dataset(train_x, train_y, batch_size, shuffle_seed(variant, seed))
    val_set = make_eval_dataset(val_x, val_y, batch_size)
    started = time.perf_counter()
    fit = model.fit(train_set, validation_data=val_set, epochs=variant.max_epochs, callbacks=callbacks, verbose=2)
    training_seconds = round(time.perf_counter() - started, 1)

    history = pd.DataFrame(fit.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    history["epoch_seconds"] = timer.seconds[:len(history)]
    best_epoch = int(history["val_loss"].to_numpy().argmin()) + 1  # first minimum wins on a tie
    history["is_best_epoch"] = history["epoch"] == best_epoch
    stop_reason = "early_stopping" if early_stopping.stopped_epoch > 0 else ("epoch_limit_reached" if len(history) == variant.max_epochs else "other")
    history.to_csv(paths["history"], index=False, lineterminator="\n")

    # The reloaded model must be compiled again (jit_compile=False) before predicting: a loaded model otherwise uses other numerics.
    loaded = models.compile_model(keras.models.load_model(paths["checkpoint"], compile=False), variant.lr)
    metric_rows, prediction_frames = [], []
    for subset, (frame, images, labels) in inputs.items():
        scores = sigmoid_scores(loaded.predict(make_eval_dataset(images, labels, batch_size), verbose=0), len(frame))
        m = binary_metrics(labels, scores, config.THRESHOLD)
        malignant = frame["label"] == "1"
        metric_rows.append({"model": variant.name, "seed": seed, "subset": subset, "positive_class": "malignant (label 1)", "threshold": config.THRESHOLD, "decision_rule": "score >= threshold",
                            "n_images": m["n_images"], "n_cases": frame["Case"].nunique(), "n_malignant_cases": frame.loc[malignant, "Case"].nunique(),
                            "n_benign_cases": frame.loc[~malignant, "Case"].nunique(), "n_malignant_images": m["n_positive"], "n_benign_images": m["n_negative"],
                            "n_predicted_malignant": m["n_predicted_positive"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"], "sensitivity": m["sensitivity"],
                            "specificity": m["specificity"], "precision": m["precision"], "f1": m["f1"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
                            "average_precision": m["average_precision"], "best_epoch": best_epoch})
        prediction_frames.append(pd.DataFrame({"model": variant.name, "seed": seed, "subset": subset, "ID": frame["ID"], "Case": frame["Case"], "label": labels,
                                               "malignant_score": scores, "predicted_label": (scores >= config.THRESHOLD).astype(int), "threshold": config.THRESHOLD}))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(paths["metrics"], index=False, lineterminator="\n")
    pd.concat(prediction_frames, ignore_index=True).to_csv(paths["predictions"], index=False, lineterminator="\n")

    lines: list[str] = []
    loaded.summary(print_fn=lambda line, **_: lines.append(line))
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    settings = {"variant": asdict(variant), "seed": seed, "random_sources": _random_sources(variant, seed), "batch_size": batch_size,
                "optimizer": {"name": "Adam", "learning_rate": variant.lr, "state": "new, nothing inherited"}, "loss": "BinaryCrossentropy(from_logits=False)",
                "released_layers": config.RELEASED_LAYERS.get(variant.release) if variant.release else None, "class_weights": None,
                "callbacks": {"early_stopping": {"monitor": MONITOR, "mode": "min", "patience": config.EARLY_STOPPING_PATIENCE, "min_delta": 0, "restore_best_weights": True},
                              "model_checkpoint": {"monitor": MONITOR, "mode": "min", "save_best_only": True}, "terminate_on_nan": True},
                "numerics": {"dtype": "float32", "jit_compile": False, "mixed_precision": False},
                "result": {"best_epoch": best_epoch, "epochs_run": len(history), "stop_reason": stop_reason, "best_val_loss": float(history["val_loss"].min()), "training_seconds": training_seconds},
                "versions": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__, "gpus": gpus}}
    paths["config"].write_text(json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8")
    validation = metrics[metrics["subset"] == "validation"].iloc[0]
    print(f"{variant.name} seed {seed}: epochs {len(history)}, best epoch {best_epoch}, stop {stop_reason}, validation ROC-AUC {validation['roc_auc']:.4f}")
    return settings["result"]


def _random_sources(variant: Variant, seed: int) -> dict:
    if variant.seed_mode == "historical":
        return {"global": config.HISTORICAL_SEED, "RandomFlip": config.HISTORICAL_SEED, "Dropout": "unseeded", "train_shuffle": config.HISTORICAL_SEED}
    sources = {"global": seed, "RandomFlip": seed + config.SEED_OFFSETS["flip"], "Dropout": seed + config.SEED_OFFSETS["dropout"], "train_shuffle": seed + config.SEED_OFFSETS["shuffle"]}
    if variant.augmentation == "flip_rotation_zoom":
        sources.update(RandomRotation=seed + config.SEED_OFFSETS["rotation"], RandomZoom=seed + config.SEED_OFFSETS["zoom"])
    return sources
