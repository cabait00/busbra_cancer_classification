"""Exploratory add-on experiment (adversarial continuation, validation only): a short continuation of the expert-ROI classifiers (4E, extended fine-tuning 1e-4) with FGSM examples, and the evaluation on clean and attacked validation images.

Question: how does a short continuation of the expert-ROI training with FGSM examples change the classification of unchanged and deliberately perturbed images?
Two arms per seed continue from the SAME original 4E checkpoint: `clean_continuation` ([x, x]) and `fgsm_continuation` ([x, x_adv]); five further epochs, a fresh Adam, identical step counts.

Method notes (project specific; the idea follows the adversarial-training section of course notebook 19_02, Goodfellow et al. 2015 for FGSM, Madry et al. 2018 for adversarial training as a robust-optimisation view; Wong et al. 2020
document the limits of simple single-step FGSM training):
  * the attack for a training batch is generated from the CURRENT model in inference mode (augmentation and dropout off, BatchNorm fixed), one step from the unchanged input, true labels, BCE, epsilon on the external [0, 1] input,
    changes only on the geometrically valid image area (the artificial resize padding stays exactly as it is);
  * the perturbed input is cut off from the training graph (stop_gradient): no derivative flows through the attack generation, there is exactly one optimiser step per batch;
  * the mixed batch [x, x_adv] with duplicated true labels is processed in training mode (the existing flip / rotation / zoom and dropout act; the backbone is still called with training=False, so BatchNorm is not updated);
  * the attack is generated BEFORE the random geometric augmentation; the two halves then receive separately drawn augmentations: this combination is what is studied, not an attack optimised for all random transformations.
TensorFlow / Keras are imported inside the functions that need them; the helpers for batching, tables and statistics use only NumPy and pandas.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from busbra import evaluation as ev
from busbra import model_diagnostics as md
from busbra import config
from busbra import thresholds as th

SEEDS = list(config.SEEDS)
ARMS = ("clean_continuation", "fgsm_continuation")
STATES = ("original", *ARMS)
CONDITIONS = ("expert", "auto")
EPOCHS = 5
LEARNING_RATE = 1e-4
TRAIN_EPSILON_255 = 2
BATCH_ORIGINAL = 8  # original images per logical batch: 16 examples per optimiser step
EVAL_EPSILONS_255 = (0, 1, 2, 4)
INFERENCE_BATCH = 16
SCORE_TOLERANCE = 1e-5
SHUFFLE_OFFSET = config.SEED_OFFSETS["shuffle"]
LEVELS = ("image", "case")
METRICS = ["roc_auc", "average_precision", "bce", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "precision", "f1", "accuracy"]
TIE_TOLERANCE = 1e-12


# ---- batching: the same order for both arms --------------------------------------------------------------------------
def epoch_batches(n_images: int, seed: int, epoch: int, batch: int = BATCH_ORIGINAL) -> list:
    """Index batches of one epoch: a permutation that depends ONLY on (seed, epoch) - never on the arm - cut into batches of `batch` original images; the last partial batch is kept as it is (no padding, no dropping)."""
    permutation = np.random.default_rng([int(seed), SHUFFLE_OFFSET, int(epoch)]).permutation(n_images)
    return [permutation[i:i + batch] for i in range(0, n_images, batch)]


def epsilon_value(k255: int) -> np.float32:
    return md.epsilon_value(k255)


# ---- the training step (TensorFlow inside) ---------------------------------------------------------------------------
class TrainingError(Exception):
    """A training step produced a non-finite loss or gradient, or violated a contract."""


def make_train_step(model, graph, optimizer, arm: str, epsilon, return_batch: bool = False):
    """tf.function for ONE optimiser step on a logical batch of original images.

    arm 'fgsm_continuation': (1) the BCE input gradient of the current model in inference mode (diagnostic graph: no augmentation, no dropout, BatchNorm fixed); (2) x_adv = clip(x + epsilon * sign(g), 0, 1) on the valid area only
    (padding unchanged), cut off with stop_gradient; (3) the batch [x, x_adv] with the duplicated labels goes through model(training=True); (4) ONE Adam step on the mean BCE of the 2b examples.
    arm 'clean_continuation': the same, with [x, x] and no attack. `return_batch` exposes the mixed batch for tests."""
    import keras
    import tensorflow as tf
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
    epsilon = float(epsilon)

    def step(x, y, valid):
        b = tf.shape(x)[0]
        if arm == "fgsm_continuation":
            with tf.GradientTape() as attack_tape:
                attack_tape.watch(x)
                _, logit = graph._forward(x)  # inference mode: augmentation off, dropout off, BatchNorm fixed
                attack_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit))
            attack_gradient = attack_tape.gradient(attack_loss, x)
            x_adv = tf.stop_gradient(tf.clip_by_value(x + tf.cast(epsilon, x.dtype) * tf.sign(attack_gradient) * valid, 0.0, 1.0))
            second = x_adv
            attack_linf = tf.reduce_max(tf.abs(x_adv - x))
            attack_zero_rows = tf.reduce_sum(tf.cast(tf.reduce_all(tf.equal(tf.reshape(attack_gradient, (b, -1)), 0.0), axis=1), tf.int32))
            attack_finite = tf.reduce_all(tf.math.is_finite(attack_gradient))
        else:
            second = x
            attack_linf = tf.constant(0.0)
            attack_zero_rows = tf.constant(0, tf.int32)
            attack_finite = tf.constant(True)
        mixed = tf.concat([x, second], axis=0)
        labels = tf.concat([y, y], axis=0)
        with tf.GradientTape() as tape:
            probability = model(mixed, training=True)  # augmentation and dropout act here; the backbone is called with training=False inside the graph
            loss = loss_function(labels, probability)  # mean over the 2b examples
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        if any(g is None for g in gradients):
            raise TrainingError("a trainable variable received no gradient: it is not connected to the loss")
        optimizer.apply_gradients(zip(gradients, variables))
        clipped = tf.clip_by_value(probability, 1e-7, 1.0 - 1e-7)  # the clipping of Keras BinaryCrossentropy
        per_example = -(labels * tf.math.log(clipped) + (1.0 - labels) * tf.math.log(1.0 - clipped))
        stats = {"loss": loss, "loss_first_half": tf.reduce_mean(per_example[:b]), "loss_second_half": tf.reduce_mean(per_example[b:]), "attack_linf": attack_linf, "attack_zero_gradient_rows": attack_zero_rows,
                 "attack_gradient_finite": attack_finite, "gradient_global_norm": tf.linalg.global_norm(gradients), "gradients_finite": tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])}
        if return_batch:
            stats.update({"mixed_batch": mixed, "mixed_labels": labels})
        return stats

    return tf.function(step, jit_compile=False)


# ---- evaluation of one state ------------------------------------------------------------------------------------------
def predict_scores(model, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Scores of the compiled predict function in blocks of 16 in the given order (the same call as in the earlier steps)."""
    from busbra import training
    return ev.sigmoid_scores(model.predict(training.make_eval_dataset(x, labels.astype(int), INFERENCE_BATCH), verbose=0), len(x))


def blockwise_logits(graph, x: np.ndarray) -> np.ndarray:
    return np.concatenate([graph.logits(x[i:i + INFERENCE_BATCH]) for i in range(0, len(x), INFERENCE_BATCH)])


def blockwise_gradient(graph, x: np.ndarray, labels: np.ndarray) -> dict:
    """Input gradient of the BCE for every image at the CLEAN input (blocks of 16): only its sign is used. The forward logit inside the gradient graph is returned as a diagnostic."""
    gradients, tape_logits, devices = [], [], {}
    for i in range(0, len(x), INFERENCE_BATCH):
        part = graph.input_gradient(x[i:i + INFERENCE_BATCH], labels[i:i + INFERENCE_BATCH].astype(np.float32), np.ones(len(x[i:i + INFERENCE_BATCH]), dtype=np.float32))
        gradients.append(part["gradient"])
        tape_logits.append(part["tape_logit"])
        devices = part["devices"]
    return {"gradient": np.concatenate(gradients), "tape_logit": np.concatenate(tape_logits), "devices": devices}


def metric_rows(labels: np.ndarray, cases: np.ndarray, scores: np.ndarray, logits: np.ndarray) -> list:
    """Image-level and case-level metrics at the reference threshold 0.5 (case score = maximum of the image scores; the existing rule). The BCE is defined on image level only (from the logit, stable form)."""
    image = ev.binary_metrics(labels, scores)
    image["bce"] = float(np.mean(md.bce_from_logit(logits, labels)))
    case_frame = th.aggregate_cases(pd.Series(cases), labels, scores)
    case = ev.binary_metrics(case_frame["label"].to_numpy(dtype=int), case_frame["case_score"].to_numpy(dtype=float))
    case["bce"] = None
    return [{"level": "image", **image}, {"level": "case", **case}]


def evaluate_state(model, graph, inputs: dict, labels: np.ndarray, ids: list, cases: np.ndarray, epsilons=EVAL_EPSILONS_255) -> dict:
    """Evaluate one classifier state on every condition and every epsilon. The attack is generated FROM THIS MODEL at the clean input of each condition (gradient sign only), one step per epsilon from the clean input.

    inputs: condition -> {"x": (N,160,160,1) float32, "valid": (N,160,160) bool}. Scores and BCE come from the inference path; the forward logit of the gradient graph is stored only as a diagnostic."""
    rows, score_rows, checks = [], [], []
    for condition, data in inputs.items():
        x, valid = data["x"], data["valid"]
        grad = blockwise_gradient(graph, x, labels)
        clean_logits = blockwise_logits(graph, x)
        for k255 in epsilons:
            eps = epsilon_value(k255)
            x_adv = md.fgsm_step(x, grad["gradient"], eps, valid)
            scores = predict_scores(model, x_adv, labels)
            logits = blockwise_logits(graph, x_adv)
            delta = x_adv.astype(np.float64) - x.astype(np.float64)
            padding = ~np.broadcast_to(valid[..., np.newaxis], x.shape)
            check = {"condition": condition, "epsilon_255": k255, "max_perturbation": float(np.abs(delta).max()), "within_epsilon": bool(np.abs(delta).max() <= float(eps) + md.CHANGE_TOLERANCE),
                     "padding_unchanged": bool(np.array_equal(x_adv[padding], x[padding])), "range_ok": bool(x_adv.min() >= 0.0 and x_adv.max() <= 1.0), "identical_to_input": bool(np.array_equal(x_adv, x)),
                     "gradient_finite": bool(np.isfinite(grad["gradient"]).all()), "images_with_all_zero_gradient": int(sum((grad["gradient"][i] == 0).all() for i in range(len(x)))),
                     "scores_finite": bool(np.isfinite(scores).all() and scores.min() >= 0.0 and scores.max() <= 1.0), "max_abs_tape_forward_minus_inference_logit": float(np.abs(grad["tape_logit"] - clean_logits).max()),
                     "sigmoid_logit_vs_predict_max_abs": float(np.abs(md.sigmoid(logits) - scores).max())}
            checks.append(check)
            for row in metric_rows(labels, cases, scores, logits):
                rows.append({"condition": condition, "epsilon_255": k255, **row})
            for i, identifier in enumerate(ids):
                score_rows.append({"condition": condition, "epsilon_255": k255, "ID": identifier, "label": int(labels[i]), "score": float(scores[i]), "logit": float(logits[i]), "max_abs_change": float(np.abs(delta[i]).max())})
    return {"rows": rows, "scores": score_rows, "checks": checks, "gradient_devices": grad["devices"]}


def evaluation_problems(checks: list, epsilons=EVAL_EPSILONS_255) -> list:
    """Contract of the attack in every evaluation block: finite gradients, range, padding, epsilon bound, epsilon 0 identical to the input, finite scores, both score paths agree."""
    problems = []
    for c in checks:
        tag = f"{c['condition']} eps {c['epsilon_255']}/255"
        if not c["gradient_finite"]:
            problems.append(f"{tag}: non-finite input gradient")
        if not (c["within_epsilon"] and c["padding_unchanged"] and c["range_ok"] and c["scores_finite"]):
            problems.append(f"{tag}: perturbation contract violated (epsilon bound {c['within_epsilon']}, padding {c['padding_unchanged']}, range {c['range_ok']}, finite scores {c['scores_finite']})")
        if c["epsilon_255"] == 0 and not c["identical_to_input"]:
            problems.append(f"{tag}: epsilon 0 changed the input")
        if c["sigmoid_logit_vs_predict_max_abs"] > SCORE_TOLERANCE:
            problems.append(f"{tag}: the logit path and the predict path differ by {c['sigmoid_logit_vs_predict_max_abs']:.2e}")
    return problems


def summarize_states(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state, condition, k255, level), part in runs.groupby(["state", "condition", "epsilon_255", "level"], sort=False):
        for metric in METRICS:
            rows.append({"state": state, "condition": condition, "epsilon_255": int(k255), "level": level, "metric": metric, **ev.stats(part[metric], count_name="n"),
                         "note": "mean and sample SD over the five seeds (training variation on the same holdout); no pooled AUC, no ensemble"})
    return pd.DataFrame(rows)


def attack_effect(runs: pd.DataFrame) -> pd.DataFrame:
    """Per seed, state, condition and level: the value at epsilon k minus the value at epsilon 0 of the SAME state (the effect of the attack relative to the clean input)."""
    rows = []
    for (seed, state, condition, level), part in runs.groupby(["seed", "state", "condition", "level"], sort=False):
        clean = part[part["epsilon_255"] == 0].iloc[0]
        for _, row in part[part["epsilon_255"] > 0].iterrows():
            for metric in METRICS:
                a, b = clean[metric], row[metric]
                rows.append({"seed": int(seed), "state": state, "condition": condition, "level": level, "epsilon_255": int(row["epsilon_255"]), "metric": metric, "clean_value": a, "attacked_value": b,
                             "change": (float(b) - float(a)) if pd.notna(a) and pd.notna(b) else None})
    return pd.DataFrame(rows)


PAIRS = [("fgsm_continuation", "clean_continuation", "fgsm_continuation minus clean_continuation"), ("clean_continuation", "original", "clean_continuation minus original"),
         ("fgsm_continuation", "original", "fgsm_continuation minus original")]


def paired_differences(runs: pd.DataFrame) -> pd.DataFrame:
    """Within the same seed, condition, epsilon and level: state B minus state A for every metric (single values per seed)."""
    rows = []
    for (condition, k255, level), part in runs.groupby(["condition", "epsilon_255", "level"], sort=False):
        for b, a, label in PAIRS:
            pb, pa = part[part["state"] == b].set_index("seed"), part[part["state"] == a].set_index("seed")
            if sorted(pb.index) != sorted(pa.index) or sorted(pb.index) != sorted(SEEDS):
                raise ValueError(f"{label}: the seeds do not match one to one")
            for metric in METRICS:
                for seed in sorted(pb.index):
                    x, y = pa.loc[seed, metric], pb.loc[seed, metric]
                    rows.append({"pair": label, "state_a": a, "state_b": b, "condition": condition, "epsilon_255": int(k255), "level": level, "metric": metric, "seed": int(seed), "value_a": x, "value_b": y,
                                 "difference_b_minus_a": (float(y) - float(x)) if pd.notna(x) and pd.notna(y) else None})
    return pd.DataFrame(rows)


def summarize_pairs(differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, part in differences.groupby(["pair", "condition", "epsilon_255", "level", "metric"], sort=False):
        d = pd.to_numeric(part.sort_values("seed")["difference_b_minus_a"], errors="coerce").to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        rows.append({"pair": key[0], "condition": key[1], "epsilon_255": int(key[2]), "level": key[3], "metric": key[4], "n_seeds": int(len(part)), "n_defined": int(d.size),
                     "mean_difference": float(d.mean()) if d.size else None, "sd_ddof1": float(d.std(ddof=1)) if d.size > 1 else None, "min": float(d.min()) if d.size else None, "max": float(d.max()) if d.size else None,
                     "n_positive": int((d > TIE_TOLERANCE).sum()), "n_negative": int((d < -TIE_TOLERANCE).sum()), "n_equal": int((np.abs(d) <= TIE_TOLERANCE).sum()),
                     "differences_by_seed": ";".join(f"{v:.10f}" for v in d)})
    return pd.DataFrame(rows)


def summarize_effect(effect: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample SD of the attack effect (value at epsilon k minus value at epsilon 0 of the same state) over the five seeds, with the signs."""
    rows = []
    for key, part in effect.groupby(["state", "condition", "epsilon_255", "level", "metric"], sort=False):
        d = pd.to_numeric(part.sort_values("seed")["change"], errors="coerce").to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        rows.append({"state": key[0], "condition": key[1], "epsilon_255": int(key[2]), "level": key[3], "metric": key[4], "n_seeds": int(len(part)), "n_defined": int(d.size), "mean_change": float(d.mean()) if d.size else None,
                     "sd_ddof1": float(d.std(ddof=1)) if d.size > 1 else None, "min": float(d.min()) if d.size else None, "max": float(d.max()) if d.size else None, "n_positive": int((d > TIE_TOLERANCE).sum()),
                     "n_negative": int((d < -TIE_TOLERANCE).sum()), "n_equal": int((np.abs(d) <= TIE_TOLERANCE).sum()), "changes_by_seed": ";".join(f"{v:.10f}" for v in d)})
    return pd.DataFrame(rows)
