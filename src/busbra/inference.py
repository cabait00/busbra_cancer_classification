"""Inference helpers: images in manifest order, U-Net probabilities, classifier scores and baseline scores (no training, no weight update).

TensorFlow / Keras are imported inside the functions that need them; the caller configures the GPU first (busbra.training.configure_gpu).
Scores are the models' malignant scores in [0, 1]; a one-unit sigmoid output is directly the score (no argmax).
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from busbra import config
from busbra import evaluation as ev
from busbra import image_processing as ip
from busbra import models, segmentation as seg


def load_images(rows: list[dict], project_root: Path = config.PROJECT_ROOT) -> list:
    """Decoded 8-bit grayscale images in the given order; the decoded size must equal the audited decoded size (the CSV Width / Height columns never steer anything)."""
    return [ip.load_grayscale_image(Path(project_root) / r["image_relpath"], (int(r["image_width"]), int(r["image_height"]))) for r in rows]


def full_array(images: list, size: int) -> np.ndarray:
    """The whole image, aspect ratio kept, bilinear, centred black padding, float32 / 255: (N, size, size, 1); identical to image_processing.preprocess_image_set on the same images."""
    return np.stack([ip.to_unit_float32(ip.resize_with_padding(image, size)) for image in images])[..., np.newaxis]


def unet_probabilities(images: list, checkpoint: Path = config.UNET_CHECKPOINT) -> np.ndarray:
    """(N, 256, 256) probability maps of the stored U-Net (batch size 8, the same kernels as in the segmentation step)."""
    import keras
    from busbra import segmentation_model as sm
    unet = sm.compile_model(keras.models.load_model(checkpoint, compile=False), seg.LEARNING_RATE)
    x256 = np.stack([seg.prepare_image(image) for image in images])
    return sm.predict_probabilities(unet, x256, seg.BATCH_SIZE)


def classifier_scores(checkpoint: Path, seed: int, arrays: np.ndarray, lr: float, batch_size: int = config.BATCH_SIZE) -> np.ndarray:
    """One classifier checkpoint on one prepared input array: ordered batches (no shuffling, nothing dropped), compiled with jit_compile=False, session cleared afterwards."""
    import keras
    from busbra import training
    models.reset_random_sources(seed)
    model = models.compile_model(keras.models.load_model(checkpoint, compile=False), lr)
    dataset = training.make_eval_dataset(arrays, np.zeros(len(arrays), dtype=np.float32), batch_size)
    scores = ev.sigmoid_scores(model.predict(dataset, verbose=0), len(arrays))
    del model
    keras.backend.clear_session()
    gc.collect()
    return scores


def baseline_scores(model_file: Path, images: list) -> np.ndarray:
    """Malignant scores of a fitted scikit-learn baseline (joblib bundle) on the 64 x 64 flattened images."""
    import joblib
    x = ip.flatten_for_sklearn(np.stack([ip.to_unit_float32(ip.resize_with_padding(image, 64)) for image in images]))
    return ev.malignant_scores(joblib.load(model_file)["model"], x)
