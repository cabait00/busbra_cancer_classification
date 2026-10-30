# %% [08] Exploratory add-on: short FGSM adversarial-training continuation of the expert-ROI classifiers (train and validation only)
#
# Question: how does a short continuation of the expert-ROI training with FGSM examples change the classification of unchanged and deliberately perturbed images?
# The five fine-tuned expert-ROI classifiers (one per seed) are the starting points. Per seed two NEW continuations of five epochs each start from the same original checkpoint:
#   clean_continuation  further training with unchanged training images (the existing geometric augmentation), batch [x, x]
#   fgsm_continuation   50 % unchanged and 50 % freshly generated FGSM images, batch [x, x_adv] (epsilon 2/255, one step from the current model, true labels, BCE)
# The original checkpoint is a third reference state and is never saved again. All states are evaluated on the expert ROIs (primary) and on the frozen automatic ROIs
# (transfer check, the SAME weights), clean and under FGSM epsilon 0, 1, 2, 4 /255 (the attack is generated against the evaluated model). Every run is reported; a gain is not
# a success criterion. Fixed: exactly five epochs, no early stopping, fresh Adam 1e-4, original batches of 8 (16 examples per step), same order for both arms.
# This is an exploratory validation experiment, not part of the frozen test scope (27 combinations, frozen thresholds); a gain under FGSM is no evidence of robustness
# against other attacks, devices or shifts. No test image, mask or prediction is touched, and no mask is opened (the expert boxes come from the stored geometry table, data.read_expert_geometry).
#
# Actions (the default recomputes the aggregate tables from the stored per-run evaluations and compares them with the stored tables; nothing is written):
#   .venv/bin/python analysis/08_adversarial_addon.py
#   env -u LD_LIBRARY_PATH .venv/bin/python analysis/08_adversarial_addon.py --train
#       explicit GPU action: five reference evaluations and ten continuations; stops before any work if one of its outputs (evaluation, scores, history, checkpoint) already exists
#   .venv/bin/python analysis/08_adversarial_addon.py --write   write the aggregate tables from the stored evaluations; stops without writing if one exists

# %% Imports
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from busbra import results, adversarial_roi as adv, auto_roi as ar, config, data, model_diagnostics as md  # noqa: E402

TABLES = config.TABLES_DIR
VARIANT = config.VARIANTS["mobilenetv2_higher_lr_160_roi"]  # the starting classifiers (and the graph that is continued)
AUTO_GEOMETRY = TABLES / "auto_roi_geometry_validation.csv"
AGGREGATES = {"run_metrics": "adversarial_roi_run_metrics.csv", "summary": "adversarial_roi_summary.csv", "attack_effect": "adversarial_roi_attack_effect.csv",
              "attack_effect_summary": "adversarial_roi_attack_effect_summary.csv", "paired_differences": "adversarial_roi_paired_differences.csv", "paired_summary": "adversarial_roi_paired_summary.csv"}


def state_paths(seed: int, state: str) -> dict:
    tables = TABLES / "adversarial_roi" / f"seed_{seed}" / state
    models = config.MODELS_DIR / "adversarial_roi" / f"seed_{seed}" / state
    return {"dir": tables, "model_dir": models, "history": tables / "history.csv", "config": tables / "config.json", "evaluation": tables / "evaluation.csv", "scores": tables / "scores.csv",
            "checkpoint": models / f"adversarial_roi_{state}_seed{seed}_epoch{adv.EPOCHS}.keras"}


def train_outputs() -> list[Path]:
    """Every file --train writes: evaluation and scores of each state, history and checkpoint of the two continuations."""
    outputs = []
    for seed in adv.SEEDS:
        for state in adv.STATES:
            paths = state_paths(seed, state)
            outputs += [paths["evaluation"], paths["scores"]] + ([] if state == "original" else [paths["history"], paths["checkpoint"]])
    return outputs


# %% Aggregation from the stored per-run evaluations (CPU)
def aggregate() -> dict[str, pd.DataFrame]:
    order = [(seed, "original") for seed in adv.SEEDS] + [(seed, arm) for seed in adv.SEEDS for arm in adv.ARMS]  # the five originals first, then both arms per seed
    runs = pd.concat([pd.read_csv(state_paths(seed, state)["evaluation"]) for seed, state in order], ignore_index=True)
    effect = adv.attack_effect(runs)
    differences = adv.paired_differences(runs)
    return {"run_metrics": runs, "summary": adv.summarize_states(runs), "attack_effect": effect, "attack_effect_summary": adv.summarize_effect(effect), "paired_differences": differences,
            "paired_summary": adv.summarize_pairs(differences)}


same = results.same_as_stored


def show() -> None:
    for name, frame in aggregate().items():
        path = TABLES / AGGREGATES[name]
        print(f"{AGGREGATES[name]:48s} {len(frame):5d} rows  " + ("identical to the stored table" if path.is_file() and same(frame, path) else "DIFFERENT or not stored"))


# %% Inputs: expert boxes (train, validation) from the stored geometry and frozen automatic boxes (validation); images only, no mask
def build_inputs(frame: pd.DataFrame, geometry: pd.DataFrame) -> dict:
    """ROI inputs of a manifest subset from the stored expert geometry (busbra.data.load_roi_inputs) plus the geometric validity masks of the resize padding."""
    images, x, boxes = data.load_roi_inputs(config.PROJECT_ROOT, frame, geometry)
    valid = np.stack([md.validity_mask(box[2] - box[0], box[3] - box[1]) for box in boxes])
    return {"images": images, "x": x, "valid": valid, "ids": frame["ID"].tolist(), "cases": frame["Case"].to_numpy(), "labels": frame["label"].astype(int).to_numpy()}


# %% Training and evaluation (explicit GPU action)
def train() -> None:
    clash = [str(path) for path in train_outputs() if path.exists()]
    if clash:
        raise FileExistsError(f"{len(clash)} output files already present, for example {clash[0]}; nothing was changed")
    manifest = data.read_manifest()
    data.check_split(manifest)
    import keras
    import tensorflow as tf
    from busbra import models, training
    geometry = data.read_expert_geometry(manifest)  # the stored geometry (prepared once with 03_classification.py --preset expert_roi --prepare); stops with that command if it is missing
    train_data = build_inputs(data.subset_frame(manifest, "train"), geometry)
    validation = build_inputs(data.subset_frame(manifest, "validation"), geometry)
    auto_geometry = pd.read_csv(AUTO_GEOMETRY, dtype={"ID": str, "Case": str}, float_precision="round_trip").set_index("ID").loc[validation["ids"]].reset_index()
    inputs = {"expert": {"x": validation["x"], "valid": validation["valid"]},
              "auto": {"x": ar.inputs_from_geometry(validation["images"], auto_geometry), "valid": np.stack([md.validity_mask(int(r.crop_w), int(r.crop_h)) for r in auto_geometry.itertuples()])}}
    training.configure_gpu()

    def evaluate(model, seed: int, state: str, paths: dict) -> None:
        graph = md.DiagnosticGraph(model)
        result = adv.evaluate_state(model, graph, inputs, validation["labels"], validation["ids"], validation["cases"])
        problems = adv.evaluation_problems(result["checks"])
        if problems:
            raise ValueError("; ".join(problems[:3]))
        rows = pd.DataFrame([{"state": state, **r} for r in result["rows"]])
        rows.insert(0, "seed", seed)
        scores = pd.DataFrame([{"state": state, **r} for r in result["scores"]])
        scores.insert(3, "Case", [dict(zip(validation["ids"], validation["cases"]))[i] for i in scores["ID"]])
        scores.insert(0, "seed", seed)
        rows.to_csv(paths["evaluation"], index=False, lineterminator="\n", na_rep="")
        scores.to_csv(paths["scores"], index=False, lineterminator="\n", na_rep="")

    for seed in adv.SEEDS:
        start = config.run_paths(VARIANT, seed)["checkpoint"]
        reference = state_paths(seed, "original")
        reference["dir"].mkdir(parents=True, exist_ok=True)
        models.reset_random_sources(seed)
        evaluate(models.compile_model(keras.models.load_model(start, compile=False), adv.LEARNING_RATE), seed, "original", reference)
        for arm in adv.ARMS:
            paths = state_paths(seed, arm)
            paths["dir"].mkdir(parents=True, exist_ok=True)
            paths["model_dir"].mkdir(parents=True, exist_ok=True)
            models.reset_random_sources(seed)
            source = keras.models.load_model(start, compile=False)
            model = models.build_mobilenet(VARIANT, seed, None)  # fresh graph with fresh generators; only the network weights are copied
            models.transfer_network_weights(source, model)
            del source
            models.release_layers(model, "extended")
            models.compile_model(model, adv.LEARNING_RATE)  # fresh Adam
            graph = md.DiagnosticGraph(model)
            step = adv.make_train_step(model, graph, model.optimizer, arm, adv.epsilon_value(adv.TRAIN_EPSILON_255))
            t = train_data
            history = []
            with tf.device("/GPU:0"):
                for epoch in range(adv.EPOCHS):
                    acc = {"loss": [], "first": [], "second": [], "linf": [], "zero": 0, "norm": [], "examples": 0}
                    batches = adv.epoch_batches(len(t["ids"]), seed, epoch)
                    for index in batches:
                        out = step(tf.constant(t["x"][index]), tf.constant(t["labels"][index].astype(np.float32).reshape(-1, 1)), tf.constant(t["valid"][index][..., np.newaxis].astype(np.float32)))
                        loss = float(out["loss"])
                        if not (math.isfinite(loss) and bool(out["gradients_finite"]) and bool(out["attack_gradient_finite"])):
                            raise ValueError(f"seed {seed} {arm} epoch {epoch + 1}: non-finite loss or gradient")
                        acc["loss"].append(loss * 2 * len(index))
                        acc["first"].append(float(out["loss_first_half"]) * len(index))
                        acc["second"].append(float(out["loss_second_half"]) * len(index))
                        acc["linf"].append(float(out["attack_linf"]))
                        acc["zero"] += int(out["attack_zero_gradient_rows"])
                        acc["norm"].append(float(out["gradient_global_norm"]))
                        acc["examples"] += 2 * len(index)
                    n_orig = acc["examples"] // 2
                    history.append({"epoch": epoch + 1, "steps": len(batches), "examples": acc["examples"], "mean_loss": sum(acc["loss"]) / acc["examples"], "mean_loss_first_half": sum(acc["first"]) / n_orig,
                                    "mean_loss_second_half": sum(acc["second"]) / n_orig, "mean_attack_linf_per_step": float(np.mean(acc["linf"])), "attack_zero_gradient_rows": acc["zero"],
                                    "max_gradient_global_norm": float(np.max(acc["norm"]))})
            pd.DataFrame(history).to_csv(paths["history"], index=False, lineterminator="\n")
            model.save(paths["checkpoint"])
            del model, graph, step
            keras.backend.clear_session()
            models.reset_random_sources(seed)
            evaluate(models.compile_model(keras.models.load_model(paths["checkpoint"], compile=False), adv.LEARNING_RATE), seed, arm, paths)  # the saved state is evaluated after reloading
            print(f"seed {seed} {arm}: trained and evaluated")
    print("finished; the default action writes nothing and compares the aggregates")


def write() -> None:
    """Write the aggregate tables from the stored per-run evaluations; nothing is written if one of them exists."""
    tables = aggregate()
    targets = {name: TABLES / AGGREGATES[name] for name in tables}
    clash = [p.name for p in targets.values() if p.exists()]
    if clash:
        raise FileExistsError(f"already present: {clash}; nothing was changed")
    for name, frame in tables.items():
        frame.to_csv(targets[name], index=False, lineterminator="\n", na_rep="")
    print(f"wrote {len(tables)} tables")


if __name__ == "__main__":
    argv = sys.argv[1:] if "__file__" in globals() else []
    train() if "--train" in argv else (write() if "--write" in argv else show())
