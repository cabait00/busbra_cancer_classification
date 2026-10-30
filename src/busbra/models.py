"""Model construction for the classification experiments: small CNN and MobileNetV2 pipeline, start weights, released layers, compile.

One builder per architecture. Which experiment uses which seeds, augmentation, start weights and released layers is decided by
`config.Variant`. Keras is imported inside the functions, so importing this module neither loads TensorFlow nor touches the GPU.

Seed rules (see config.Variant.seed_mode):
  study       RandomFlip s+1, Dropout s+2, RandomRotation s+4, RandomZoom s+5 (training shuffle s+3 lives in training.py)
  historical  RandomFlip 42, Dropout unseeded (global generator after set_random_seed(42)); steps 3A-3E
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np

from busbra import config
from busbra.config import Variant

os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # the backend of all project models; chosen before Keras is first imported


def reset_random_sources(seed: int) -> None:
    """Clear the Keras session (name counters, global generators) and seed Python, NumPy and TensorFlow."""
    import keras
    keras.backend.clear_session()
    gc.collect()
    keras.utils.set_random_seed(seed)


def _seed(variant: Variant, seed: int, source: str) -> int | None:
    if variant.seed_mode == "historical":
        return None if source == "dropout" else config.HISTORICAL_SEED
    return seed + config.SEED_OFFSETS[source]


def _augmentation_layers(variant: Variant, seed: int) -> list:
    from keras import layers
    stack = [layers.RandomFlip("horizontal", seed=_seed(variant, seed, "flip"), name="augment_flip")]
    if variant.augmentation == "flip_rotation_zoom":
        stack += [layers.RandomRotation(factor=config.ROTATION_FACTOR, fill_mode="constant", fill_value=0.0, interpolation="bilinear",
                                        seed=_seed(variant, seed, "rotation"), name="augment_rotation"),
                  layers.RandomZoom(height_factor=config.ZOOM_RANGE, width_factor=None, fill_mode="constant", fill_value=0.0, interpolation="bilinear",
                                    seed=_seed(variant, seed, "zoom"), name="augment_zoom")]
    return stack


def build_small_cnn(variant: Variant, seed: int):
    """3-block CNN with global average pooling (no large Flatten-Dense head). The sigmoid output is the malignant score."""
    import keras
    from keras import layers
    inputs = keras.Input(shape=(variant.size, variant.size, 1), dtype="float32", name="image")
    x = inputs
    for layer in _augmentation_layers(variant, seed):
        x = layer(x)
    for index, filters in enumerate((32, 64, 128), start=1):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"conv{index}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{index}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.3, seed=_seed(variant, seed, "dropout"), name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name=config.HEAD_LAYER)(x)
    return keras.Model(inputs, outputs, name=variant.graph_name)


def build_mobilenet(variant: Variant, seed: int, weights: str | Path | None):
    """MobileNetV2 (alpha 1.0, no top) on the grey image copied to three channels and scaled to [-1, 1]; the backbone is called with training=False.

    `weights` is the local ImageNet weight file (never a download) or None for a graph whose weights are copied in afterwards.
    """
    import keras
    from keras import layers
    size = variant.size
    inputs = keras.Input(shape=(size, size, 1), dtype="float32", name="image")
    x = inputs
    for layer in _augmentation_layers(variant, seed):
        x = layer(x)
    x = layers.Concatenate(axis=-1, name="gray_to_rgb")([x, x, x])
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="scale_to_minus1_1")(x)
    backbone = keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=1.0, include_top=False, weights=str(weights) if weights else None, pooling=None,
                                              name=config.BACKBONE_NAME)
    backbone.trainable = False  # every backbone weight, including BatchNorm statistics, stays fixed
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, seed=_seed(variant, seed, "dropout"), name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name=config.HEAD_LAYER)(x)
    return keras.Model(inputs, outputs, name=variant.graph_name)


def transfer_network_weights(source, target) -> None:
    """Copy the trained backbone and head weights into a freshly built graph of the same architecture.

    Generator states and optimizer state are NOT copied: the target keeps its own fresh ones.
    """
    target.get_layer(config.BACKBONE_NAME).set_weights(source.get_layer(config.BACKBONE_NAME).get_weights())
    target.get_layer(config.HEAD_LAYER).set_weights(source.get_layer(config.HEAD_LAYER).get_weights())


def load_pretrained_backbone_160(weights_file: Path):
    """The 160 px no-top backbone with the values of the local ImageNet file; the values are copied into a 250 px backbone (same layers, input-size independent weights)."""
    import keras
    return keras.applications.MobileNetV2(input_shape=(160, 160, 3), alpha=1.0, include_top=False, weights=str(weights_file), pooling=None, name="pretrained_backbone_160")


def release_layers(model, release: str) -> None:
    """Release exactly the named backbone convolutions and the head; everything else, including all 52 BatchNorm layers, stays fixed.

    Order matters: first make the model and the nested backbone trainable at all, THEN set every backbone layer individually; a later
    blanket assignment would undo the selection. Raises ValueError if the structure or the parameter counts differ from the design.
    """
    from keras import layers
    names = config.RELEASED_LAYERS[release]
    backbone = model.get_layer(config.BACKBONE_NAME)
    by_name = {layer.name: layer for layer in backbone.layers}
    problems = [n for n in names if n not in by_name]
    n_bn = sum(isinstance(layer, layers.BatchNormalization) for layer in backbone.layers)
    if problems or n_bn != config.EXPECTED_BATCHNORM_LAYERS:
        raise ValueError(f"unexpected backbone structure: missing {problems}, {n_bn} BatchNorm layers")
    model.trainable = True
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = layer.name in names
    model.get_layer(config.HEAD_LAYER).trainable = True
    check_parameters(model, release)


def check_parameters(model, kind: str) -> dict:
    """Trainable / total parameter counts against the design; BatchNorm weights must not be trainable. Returns the counts."""
    count = lambda ws: int(sum(int(np.prod(w.shape)) for w in ws))  # noqa: E731
    report = {"total": int(model.count_params()), "trainable": count(model.trainable_weights), "n_trainable_tensors": len(model.trainable_weights)}
    expected = config.TRAINABLE_PARAMETERS[kind]
    expected_total = report["total"] if kind == "cnn" else config.TOTAL_PARAMETERS
    if report["trainable"] != expected or report["total"] != expected_total:
        raise ValueError(f"parameter counts {report} differ from the design (trainable {expected}, total {expected_total})")
    return report


def compile_model(model, lr: float):
    """New Adam(lr) without inherited state, binary cross-entropy on the sigmoid output, jit_compile=False.

    The two running metrics are curve estimates only. jit_compile=False must also be set again after loading a saved model: a loaded
    model otherwise reports jit_compile=True and predicts with slightly different numerics.
    """
    import keras
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy"), keras.metrics.AUC(name="auc")], jit_compile=False)
    return model


def build_start_model(variant: Variant, seed: int, checkpoint_of=None):
    """The compiled model at the start of training, following the variant's start rule.

    checkpoint_of(source_variant_name) -> Path of the best checkpoint of the source variant for the same seed (fine-tuning only).
    The session is reset and seeded immediately before the graph is built.
    """
    import keras
    import tensorflow as tf
    reset_random_sources(config.HISTORICAL_SEED if variant.seed_mode == "historical" else seed)
    kind = "cnn" if variant.model == "cnn" else (variant.release or "frozen")
    with tf.device("/GPU:0"):
        if variant.model == "cnn":
            model = build_small_cnn(variant, seed)
        elif variant.start == "imagenet":
            model = build_mobilenet(variant, seed, _require(config.IMAGENET_WEIGHTS_FILE))
        elif variant.start == "imagenet_via_160":
            model = build_mobilenet(variant, seed, None)  # random backbone, replaced completely below
            source = load_pretrained_backbone_160(_require(config.IMAGENET_WEIGHTS_FILE))
            model.get_layer(config.BACKBONE_NAME).set_weights(source.get_weights())
        elif variant.start == "checkpoint":
            source_path = checkpoint_of(variant.source)
            if variant.carry == "object":  # historical rule: the loaded model itself is continued
                model = keras.models.load_model(source_path, compile=False)
            else:
                model = build_mobilenet(variant, seed, _require(config.IMAGENET_WEIGHTS_FILE) if variant.graph_weights == "imagenet" else None)
                loaded = keras.models.load_model(source_path, compile=False)
                transfer_network_weights(loaded, model)
                del loaded
            release_layers(model, variant.release)
        else:
            raise ValueError(f"unsupported start rule {variant.start!r}")
        if variant.start != "checkpoint":
            check_parameters(model, kind)
        compile_model(model, variant.lr)
    return model


def _require(path: Path) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(f"local ImageNet weight file not found: {path} (there is no download path)")
    return path
