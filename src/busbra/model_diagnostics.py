"""Grad-CAM and FGSM for the single-output sigmoid MobileNetV2 classifiers of the project (step 5B, technical preparation).

Design decisions (project specific; the method follows the course notebooks 12_01 Grad-CAM and 19 / 20_01 FGSM, the adaptations are listed here):
  * The output is ONE sigmoid unit: there is no class vector and no argmax. The explained quantity is the MALIGNANT LOGIT before the sigmoid. It is made accessible by a separate diagnostic
    graph that re-uses the layer OBJECTS of the loaded model (same weights, nothing is replaced in place, nothing is saved): logit = pooled_features @ kernel + bias with the kernel and bias of the
    existing output layer. sigmoid(logit) must reproduce the stored score; a logit recomputed from rounded scores is never used.
  * The target layer sits inside the nested MobileNetV2 backbone. Activation and logit are produced in ONE connected computation (a functional extractor over the existing backbone graph,
    called inside the same GradientTape as the head); the gradient is taken with respect to that very tensor. A missing layer or a gradient of None is an error.
  * Everything runs with training=False (augmentation off, dropout off, BatchNorm moving statistics): no weight and no state changes.
  * FGSM: x_adv = clip(x + epsilon * sign(grad_x BCE(y_true, model(x))), 0, 1) on the EXTERNAL float32 input in [0, 1], one step from the clean input, true labels, untargeted. The loss is computed from
    the logit (sigmoid_cross_entropy_with_logits, numerically stable); it equals BinaryCrossentropy(from_logits=False) on the sigmoid output up to the clipping that Keras applies to probabilities.
    Artificial resize padding is kept constant with a GEOMETRIC validity mask (never the lesion mask).
The geometry part (padding removal, mapping back, validity masks) uses only NumPy and Pillow; TensorFlow / Keras are imported inside the functions that need them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from busbra import roi

SIZE = 160
BACKBONE_NAME = "mobilenetv2_backbone"
TARGET_LAYER = "Conv_1"
PRE_LAYERS = ("augment_flip", "augment_rotation", "augment_zoom", "gray_to_rgb", "scale_to_minus1_1")
HEAD_LAYERS = ("gap", "dropout")
OUTPUT_LAYER = "output"
EPSILONS_255 = (0, 1, 2, 4)  # epsilon = k / 255 in the external input scale [0, 1]
CHANGE_TOLERANCE = 1e-7  # float32 rounding of x + epsilon * sign(g) near 1.0


class DiagnosticError(Exception):
    """A diagnostic precondition is violated (missing layer, unconnected or non-finite gradient, ...)."""


# ---- geometry: resize padding, mapping back to the image, validity masks (NumPy / Pillow only) ---------------------
def padding_geometry(crop_width: int, crop_height: int, size: int = SIZE) -> dict:
    """Geometry of the existing resize rule (ImageOps.contain, bilinear, centred black padding) for a crop of this size: fitted width / height and the offsets (odd remainder lies right / bottom)."""
    g = roi.resize_geometry(crop_width, crop_height, size)
    return {"fitted_width": g["fitted_width"], "fitted_height": g["fitted_height"], "pad_x": g["pad_x"], "pad_y": g["pad_y"], "scale": g["scale"],
            "pad_right": size - g["pad_x"] - g["fitted_width"], "pad_bottom": size - g["pad_y"] - g["fitted_height"]}


def validity_mask(crop_width: int, crop_height: int, size: int = SIZE) -> np.ndarray:
    """(size, size) bool: True for pixels that come from the image / crop, False for the artificial padding. Follows the image geometry only; real black pixels inside the image are True."""
    g = padding_geometry(crop_width, crop_height, size)
    mask = np.zeros((size, size), dtype=bool)
    mask[g["pad_y"]:g["pad_y"] + g["fitted_height"], g["pad_x"]:g["pad_x"] + g["fitted_width"]] = True
    return mask


def _resize_float(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize of a float32 2-D array with Pillow (mode 'F'), result (height, width) float32."""
    return np.asarray(Image.fromarray(np.ascontiguousarray(array, dtype=np.float32), mode="F").resize((int(width), int(height)), Image.Resampling.BILINEAR), dtype=np.float32)


def cam_to_canvas(cam: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Coarse activation map (h, w) -> the size x size model input canvas (bilinear, half-pixel centres): the canvas still contains the padding."""
    return _resize_float(cam, size, size)


def canvas_to_crop(canvas: np.ndarray, crop_width: int, crop_height: int, size: int = SIZE) -> np.ndarray:
    """Remove EXACTLY the resize padding from the canvas map, then map the remaining fitted region onto the real crop size (crop_height, crop_width)."""
    g = padding_geometry(crop_width, crop_height, size)
    region = canvas[g["pad_y"]:g["pad_y"] + g["fitted_height"], g["pad_x"]:g["pad_x"] + g["fitted_width"]]
    return _resize_float(region, crop_width, crop_height)


def embed_in_image(crop_map: np.ndarray, box: tuple, width: int, height: int) -> np.ndarray:
    """Insert the crop map into the frozen box of the original image; everything outside the box is NaN = NOT COMPUTED (it is not 'unimportant = 0')."""
    x0, y0, x1, y1 = box
    if crop_map.shape != (y1 - y0, x1 - x0):
        raise DiagnosticError(f"crop map {crop_map.shape} does not fit the box {(x1 - x0, y1 - y0)} (width, height)")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise DiagnosticError(f"box {box} lies outside the image {(width, height)}")
    out = np.full((height, width), np.nan, dtype=np.float32)
    out[y0:y1, x0:x1] = crop_map
    return out


def map_cam_to_image(cam: np.ndarray, box: tuple, width: int, height: int, size: int = SIZE) -> dict:
    """Coarse map -> 160 canvas -> padding removed -> crop size -> original image frame (NaN outside the box)."""
    canvas = cam_to_canvas(cam, size)
    crop = canvas_to_crop(canvas, box[2] - box[0], box[3] - box[1], size)
    return {"canvas": canvas, "crop": crop, "image": embed_in_image(crop, box, width, height)}


def normalize_cam(cam: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """ReLU-ed class activation map -> own positive maximum = 1. Returns (map, raw maximum, empty). An empty map (no positive value) stays all zero: no division by zero, no NaN."""
    cam = np.asarray(cam, dtype=np.float32)
    if not np.isfinite(cam).all() or cam.min() < 0.0:
        raise DiagnosticError("class activation map is not finite or not non-negative (apply ReLU first)")
    maximum = float(cam.max())
    if maximum <= 0.0:
        return np.zeros_like(cam), maximum, True
    return (cam / np.float32(maximum)).astype(np.float32), maximum, False


# ---- FGSM on arrays (NumPy only) ------------------------------------------------------------------------------------
def epsilon_value(k255: int) -> np.float32:
    return np.float32(k255) / np.float32(255.0)


def fgsm_step(x: np.ndarray, grad: np.ndarray, epsilon, valid: np.ndarray) -> np.ndarray:
    """One FGSM step from the CLEAN input: clip(x + epsilon * sign(grad), 0, 1), applied only where `valid` is True (artificial padding stays exactly as it is). All float32, no quantisation.

    x, grad: (N, H, W, 1); valid: (N, H, W) bool (geometric validity mask). For epsilon 0 the result is identical to x."""
    if not np.isfinite(grad).all():
        raise DiagnosticError("the input gradient is not finite")
    step = np.float32(epsilon) * np.sign(grad).astype(np.float32)
    step = np.where(valid[..., np.newaxis], step, np.float32(0.0)).astype(np.float32)
    return np.clip(x + step, np.float32(0.0), np.float32(1.0)).astype(np.float32)


def perturbation_report(x: np.ndarray, x_adv: np.ndarray, valid: np.ndarray, epsilon) -> dict:
    """Facts about one perturbed row: actual maximum change, L2 norm, number of changed pixels, padding unchanged, value range, epsilon bound with the registered float32 rounding tolerance."""
    delta = x_adv.astype(np.float64) - x.astype(np.float64)
    padding = ~valid[..., np.newaxis]
    return {"linf": float(np.abs(delta).max()), "l2": float(np.sqrt((delta ** 2).sum())), "n_changed": int((delta != 0).sum()), "n_valid": int(valid.sum()),
            "padding_unchanged": bool(np.array_equal(x_adv[np.broadcast_to(padding, x.shape)], x[np.broadcast_to(padding, x.shape)])),
            "min": float(x_adv.min()), "max": float(x_adv.max()), "range_ok": bool(x_adv.min() >= 0.0 and x_adv.max() <= 1.0),
            "within_epsilon": bool(np.abs(delta).max() <= float(epsilon) + CHANGE_TOLERANCE), "identical_to_input": bool(np.array_equal(x, x_adv))}


# ---- the diagnostic graph (TensorFlow / Keras inside) ---------------------------------------------------------------
class DiagnosticGraph:
    """Weight-identical diagnostic view of a loaded classifier: logit, target-layer activation, Grad-CAM inputs and input gradients. The loaded model itself is never modified."""

    def __init__(self, model, backbone_name: str = BACKBONE_NAME, layer_name: str = TARGET_LAYER, pre_layers=PRE_LAYERS, head_layers=HEAD_LAYERS, output_layer: str = OUTPUT_LAYER):
        import keras
        import tensorflow as tf
        self._tf, self._keras, self.model = tf, keras, model
        try:
            self.backbone = model.get_layer(backbone_name)
        except ValueError as exc:
            raise DiagnosticError(f"backbone layer {backbone_name!r} not found in the model") from exc
        if not isinstance(self.backbone, keras.Model):
            raise DiagnosticError(f"{backbone_name!r} is not a nested model")
        try:
            self.target = self.backbone.get_layer(layer_name)
        except ValueError as exc:
            raise DiagnosticError(f"target layer {layer_name!r} not found inside {backbone_name!r}") from exc
        expected_order = ["image", *pre_layers, backbone_name, *head_layers, output_layer]
        if [l.name for l in model.layers] != expected_order:
            raise DiagnosticError(f"unexpected layer order {[l.name for l in model.layers]}; expected {expected_order}")
        self.pre, self.head = [model.get_layer(n) for n in pre_layers], [model.get_layer(n) for n in head_layers]
        self.dense = model.get_layer(output_layer)
        config = self.dense.get_config()
        if config.get("units") != 1 or config.get("activation") != "sigmoid":
            raise DiagnosticError(f"output layer is not a single sigmoid unit: {config.get('units')} units, activation {config.get('activation')!r}")
        self.layer_path = f"{backbone_name}/{layer_name}"
        self.extractor = keras.Model(self.backbone.inputs, [self.target.output, self.backbone.output], name="diagnostic_feature_extractor")  # one connected graph: activation and features
        self.activation_shape = tuple(int(v) for v in self.target.output.shape[1:])
        self._cam_fn = tf.function(self._cam, jit_compile=False)
        self._grad_fn = tf.function(self._input_gradient, jit_compile=False)
        self._logit_fn = tf.function(lambda x: self._forward(x)[1], jit_compile=False)

    def _forward(self, x):
        tf = self._tf
        h = x
        for layer in self.pre:
            h = layer([h, h, h]) if type(layer).__name__ == "Concatenate" else layer(h, training=False)
        activation, features = self.extractor(h, training=False)
        z = features
        for layer in self.head:
            z = layer(z, training=False)
        logit = tf.matmul(z, self.dense.kernel) + self.dense.bias  # exactly the pre-activation of the existing output layer
        return activation, logit

    def _cam(self, x):
        tf = self._tf
        with tf.GradientTape() as tape:
            tape.watch(x)
            activation, logit = self._forward(x)
            total = tf.reduce_sum(logit)  # no coupling between images: inference mode, no batch statistics
        grads = tape.gradient(total, activation)
        return activation, logit, grads

    def _input_gradient(self, x, y, row_weight):
        tf = self._tf
        with tf.GradientTape() as tape:
            tape.watch(x)
            _, logit = self._forward(x)
            rows = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)  # BCE from the logit, stable
            total = tf.reduce_sum(rows * row_weight)
        return logit, rows, tape.gradient(total, x)

    # --- public numpy interface
    def logits(self, x: np.ndarray) -> np.ndarray:
        with self._tf.device("/GPU:0"):
            return self._logit_fn(self._tf.constant(x)).numpy().astype(np.float64)[:, 0]

    def gradcam_inputs(self, x: np.ndarray) -> dict:
        """Activation (N, h, w, K), logit (N,), gradient of the logit w.r.t. the activation (N, h, w, K) and the devices the tensors live on."""
        tf = self._tf
        with tf.device("/GPU:0"):
            activation, logit, grads = self._cam_fn(tf.constant(x))
        if grads is None:
            raise DiagnosticError("the gradient w.r.t. the target activation is None: the activation is not connected to the logit")
        a, g = activation.numpy(), grads.numpy()
        if not (np.isfinite(a).all() and np.isfinite(g).all() and np.isfinite(logit.numpy()).all()):
            raise DiagnosticError("non-finite activation, logit or gradient")
        return {"activation": a, "gradient": g, "logit": logit.numpy().astype(np.float64)[:, 0], "devices": {"activation": activation.device, "gradient": grads.device, "logit": logit.device}}

    def input_gradient(self, x: np.ndarray, y: np.ndarray, row_weight: np.ndarray) -> dict:
        tf = self._tf
        with tf.device("/GPU:0"):
            logit, rows, grad = self._grad_fn(tf.constant(x), tf.constant(y.reshape(-1, 1).astype(np.float32)), tf.constant(row_weight.reshape(-1, 1).astype(np.float32)))
        if grad is None:
            raise DiagnosticError("the input gradient is None")
        g = grad.numpy()
        if not np.isfinite(g).all():
            raise DiagnosticError("non-finite input gradient")
        # tape_logit / tape_bce come from the forward pass INSIDE the gradient graph. Under TF32 that pass is not bit-comparable with the inference path (different graph optimisation: the inference graph
        # folds BatchNorm into the convolutions); reported scores and losses therefore always come from logits() (inference path), and only the SIGN of this gradient is used.
        return {"tape_logit": logit.numpy().astype(np.float64)[:, 0], "tape_bce": rows.numpy().astype(np.float64)[:, 0], "gradient": g, "devices": {"logit": logit.device, "gradient": grad.device}}


def bce_from_logit(logit, y) -> np.ndarray:
    """Binary cross-entropy of a sigmoid unit from its LOGIT (stable form max(z,0) - z*y + log(1 + exp(-|z|))), float64. Equals BinaryCrossentropy(from_logits=False) on sigmoid(z) up to the clipping of probabilities."""
    z, y = np.asarray(logit, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))


def sigmoid(logit) -> np.ndarray:
    z = np.asarray(logit, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def grad_cam_maps(activation: np.ndarray, gradient: np.ndarray) -> dict:
    """Grad-CAM from the activation and the gradient of the logit: per-image channel weights = mean of the gradient over the spatial axes, weighted sum of the channels, then ReLU (N, h, w).

    Also reports, per image, the facts that must not be hidden: all-zero gradient (finite zero gradient) and an empty map after ReLU."""
    if activation.shape != gradient.shape or activation.ndim != 4:
        raise DiagnosticError(f"activation {activation.shape} and gradient {gradient.shape} do not match")
    weights = gradient.mean(axis=(1, 2))  # (N, K): spatial average per image; no mixing between images
    cam = np.maximum(np.einsum("nhwk,nk->nhw", activation.astype(np.float64), weights.astype(np.float64)), 0.0).astype(np.float32)
    return {"cam": cam, "weights": weights, "gradient_all_zero": [bool((np.abs(gradient[i]) == 0).all()) for i in range(len(gradient))],
            "gradient_abs_max": [float(np.abs(gradient[i]).max()) for i in range(len(gradient))], "n_nonzero_channel_weights": [int((weights[i] != 0).sum()) for i in range(len(weights))]}
