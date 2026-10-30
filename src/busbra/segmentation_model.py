"""Keras parts of step 4F-A: the compact U-Net, the loss (BCE + soft Dice), the curve metrics, the batch generator with the synchronous flip, callbacks and the compile step.

Importing this module imports TensorFlow/Keras but does not touch the GPU: memory growth must be configured by the caller (busbra.training.configure_gpu) before the first GPU operation.

Architecture (docs/segmentation_4fa.md): input 256 x 256 x 1 -> four encoder blocks (16, 32, 64, 128 filters, each Conv-BN-ReLU twice, followed by 2 x 2 max pooling) -> bottleneck block (256) ->
four decoder levels (transposed convolution 2 x 2 stride 2, concatenation with the skip of the same level, Conv-BN-ReLU twice) -> 1 x 1 convolution with sigmoid. The expected parameter counts are
registered in the plan (busbra.segmentation.EXPECTED_PARAMETERS) and compared with the built network.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # chosen before Keras is first imported

import keras  # noqa: E402
import numpy as np
import tensorflow as tf
from keras import layers, ops

from busbra import segmentation as seg
from busbra import training


def conv_block(x, filters: int, name: str):
    """Conv2D 3x3 same (he_normal, no bias) -> BatchNormalization -> ReLU, twice."""
    for index in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal", name=f"{name}_conv{index}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{index}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{index}")(x)
    return x


def build_unet() -> keras.Model:
    """The compact U-Net of the plan. Call busbra.models.reset_random_sources(seed) first so that the initial weights follow the study seed."""
    inputs = keras.Input(shape=(seg.SIZE, seg.SIZE, 1), dtype="float32", name="image")
    x, skips = inputs, []
    for level, filters in enumerate(seg.FILTERS, start=1):
        x = conv_block(x, filters, f"enc{level}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{level}")(x)
    x = conv_block(x, seg.BOTTLENECK_FILTERS, "bottleneck")
    for level in range(len(seg.FILTERS), 0, -1):
        filters = seg.FILTERS[level - 1]
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", kernel_initializer="he_normal", name=f"up{level}")(x)
        x = layers.Concatenate(axis=-1, name=f"skip{level}")([x, skips[level - 1]])
        x = conv_block(x, filters, f"dec{level}")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(x)
    return keras.Model(inputs, outputs, name="unet_256")


# ---- loss and curve metrics ------------------------------------------------------------------------------------------
def _soft_dice(y_true, y_pred):
    axes = (1, 2, 3)
    intersection = ops.sum(y_true * y_pred, axis=axes)
    return (2.0 * intersection + seg.DICE_SMOOTH) / (ops.sum(y_true, axis=axes) + ops.sum(y_pred, axis=axes) + seg.DICE_SMOOTH)


@keras.saving.register_keras_serializable(package="busbra_seg")
def bce_dice_loss(y_true, y_pred):
    """Per image: BCE (mean over all pixels) + (1 - soft Dice), equal weights; Keras averages over the images of the batch."""
    bce = ops.mean(keras.losses.binary_crossentropy(y_true, y_pred), axis=(1, 2))
    return bce + (1.0 - _soft_dice(y_true, y_pred))


@keras.saving.register_keras_serializable(package="busbra_seg")
def dice_soft(y_true, y_pred):
    return _soft_dice(y_true, y_pred)


def _hard(y_pred):
    return ops.cast(y_pred >= seg.THRESHOLD, "float32")


@keras.saving.register_keras_serializable(package="busbra_seg")
def dice_hard_0_5(y_true, y_pred):
    """Per image Dice of the binary prediction (probability >= 0.5) on the model grid; an image without any positive pixel in both masks would be undefined (does not occur: expert masks are not empty)."""
    axes = (1, 2, 3)
    h = _hard(y_pred)
    return 2.0 * ops.sum(y_true * h, axis=axes) / (ops.sum(y_true, axis=axes) + ops.sum(h, axis=axes))


@keras.saving.register_keras_serializable(package="busbra_seg")
def iou_hard_0_5(y_true, y_pred):
    axes = (1, 2, 3)
    h = _hard(y_pred)
    intersection = ops.sum(y_true * h, axis=axes)
    return intersection / (ops.sum(y_true, axis=axes) + ops.sum(h, axis=axes) - intersection)


def compile_model(model: keras.Model, learning_rate: float = seg.LEARNING_RATE) -> keras.Model:
    """New Adam(learning_rate), the registered loss, curve metrics, jit_compile=False (an uncompiled model would use Keras' automatic XLA and differ by rounding)."""
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=bce_dice_loss, metrics=[dice_soft, dice_hard_0_5, iou_hard_0_5], jit_compile=False)
    return model


def parameter_report(model: keras.Model) -> dict:
    total = int(model.count_params())
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    report = {"total": total, "trainable": trainable, "non_trainable": total - trainable}
    report["n_trainable_weights"] = len(model.trainable_weights)
    report["n_batchnorm_layers"] = sum(isinstance(l, layers.BatchNormalization) for l in model.layers)
    report["n_conv_layers"] = sum(isinstance(l, layers.Conv2D) for l in model.layers)
    report["ok"] = {"total": report["total"], "trainable": report["trainable"], "non_trainable": report["non_trainable"]} == seg.EXPECTED_PARAMETERS
    return report


# ---- data: the synchronous flip on image / mask pairs (outside the network) ------------------------------------------
def train_batch_stream(images: np.ndarray, masks: np.ndarray, batch_size: int, shuffle_seed: int, flip_seed: int, max_epochs: int):
    """Generator over the training batches of ALL epochs in order: epoch e uses the numpy streams seeded with (seed, e) for the image order and for one flip decision per IMAGE; the same decision acts on the
    image and on its mask. The last, smaller batch of every epoch is kept. The stream does not depend on anything that happens during training (no hook, no thread timing), so it is exactly the planned one."""
    n = len(images)
    for epoch in range(max_epochs):
        order, flips = seg.epoch_order(shuffle_seed, epoch, n), seg.epoch_flips(flip_seed, epoch, n)
        for start in range(0, n, batch_size):
            selection = order[start:start + batch_size]
            yield seg.flip_pair_batch(images[selection], masks[selection].astype(np.float32), flips[selection])


def steps_per_epoch(n: int, batch_size: int) -> int:
    return int(np.ceil(n / batch_size))


def make_train_dataset(images: np.ndarray, masks: np.ndarray, batch_size: int, shuffle_seed: int, flip_seed: int, max_epochs: int) -> tf.data.Dataset:
    """One continuous dataset over all epochs; model.fit consumes it with steps_per_epoch = ceil(n / batch_size) (Keras keeps the iterator across epochs when steps_per_epoch is given)."""
    signature = (tf.TensorSpec((None, seg.SIZE, seg.SIZE, 1), tf.float32), tf.TensorSpec((None, seg.SIZE, seg.SIZE, 1), tf.float32))
    stream = tf.data.Dataset.from_generator(lambda: train_batch_stream(images, masks, batch_size, shuffle_seed, flip_seed, max_epochs), output_signature=signature)
    return stream.prefetch(2)


def make_eval_dataset(images: np.ndarray, masks: np.ndarray, batch_size: int) -> tf.data.Dataset:
    """Fixed order, no shuffling, no augmentation, nothing dropped."""
    return tf.data.Dataset.from_tensor_slices((images, masks.astype(np.float32))).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def build_callbacks(checkpoint_path):
    """Checkpoint and early stopping both follow val_loss (min). With min_delta=0 an improvement must be strictly smaller, so on a tie the first best epoch stays the best epoch."""
    checkpoint = keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, verbose=0)
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=seg.EARLY_STOPPING_PATIENCE, min_delta=0, restore_best_weights=True, verbose=1)
    timer = training.epoch_timer()
    return [checkpoint, early_stopping, keras.callbacks.TerminateOnNaN(), timer], early_stopping, timer


def predict_probabilities(model: keras.Model, images: np.ndarray, batch_size: int) -> np.ndarray:
    """(N, 256, 256) float32 probabilities in the order of the images (no shuffling, no augmentation)."""
    outputs = model.predict(tf.data.Dataset.from_tensor_slices(images).batch(batch_size), verbose=0)
    outputs = np.asarray(outputs, dtype=np.float32)
    if outputs.shape != (len(images), seg.SIZE, seg.SIZE, 1) or not np.isfinite(outputs).all() or outputs.min() < 0.0 or outputs.max() > 1.0:
        raise ValueError(f"unexpected prediction output {outputs.shape} (finite values in [0, 1] expected)")
    return outputs[..., 0]
