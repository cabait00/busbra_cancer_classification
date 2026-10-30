# %% [11] FGSM stress test and device breakdown on the frozen test subset
#
# FGSM: one step on the classifier input only, x_adv = clip(x + eps * sign(grad_x BCE(true label)), 0, 1), applied only inside the valid image area (padding stays 0);
# eps = 0, 1/255, 2/255, 4/255 in the model input scale [0, 1]; full-image fine-tuned model (4B) and automatic-ROI pipeline, five seeds each. The gradient runs through
# the classifier only; the U-Net, the crop rule and the box are not attacked (the boxes are the frozen ones). Thresholds are the frozen validation thresholds (07).
# Device breakdown: descriptive counts and metrics at the GLOBAL thresholds (no device-specific thresholds, no fairness statement; the dataset has four devices from
# one institution; cases with images from several devices form their own group); undefined rates stay undefined.
#
# Actions (the default recomputes the tables from the stored attack scores in memory and compares them with the stored tables; nothing is written):
#   .venv/bin/python analysis/11_robustness_devices.py
#   .venv/bin/python analysis/11_robustness_devices.py --write    write the tables that do not exist yet; stops if one exists
#   env -u LD_LIBRARY_PATH .venv/bin/python analysis/11_robustness_devices.py --attack
#       explicit GPU action: runs the ten attacks (2 conditions x 5 seeds) on the test images and writes reports/tables/robustness_test_scores.csv; stops if it exists

# %% Imports
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from busbra import results, auto_roi as ar, config, data, final_evaluation as fe, inference, robustness_test as rt, test_evaluation as te  # noqa: E402
from busbra import evaluation as ev  # noqa: E402

TABLES = config.TABLES_DIR
SCORES_PATH = TABLES / "robustness_test_scores.csv"
SEEDS = list(config.SEEDS)
COMBOS = rt.attack_combos(SEEDS)
BATCH = config.BATCH_SIZE  # the original ordered classifier batches (376 = 23 x 16 + 8)
MODEL_OF = {"full": "mobilenetv2_higher_lr_160_aug", "auto": "mobilenetv2_higher_lr_160_roi"}  # the attacked classifiers (the automatic ROI uses the expert-ROI checkpoints)


# %% Attack (explicit GPU action)
def attack() -> None:
    if SCORES_PATH.exists():
        raise FileExistsError(f"{SCORES_PATH} already exists; nothing was changed")
    from busbra import model_diagnostics as md, models, training
    manifest = data.read_manifest()
    data.check_split(manifest)
    test = data.subset_frame(manifest, "test")
    rows, ids = test.to_dict("records"), test["ID"].tolist()
    labels = test["label"].astype(int).to_numpy().astype(np.float32)
    images = inference.load_images(rows)
    auto_geometry = pd.read_csv(TABLES / "test_evaluation" / "auto_roi_geometry_test.csv", dtype={"ID": str, "Case": str}, float_precision="round_trip")
    if auto_geometry["ID"].tolist() != ids:
        raise ValueError("the frozen automatic geometry does not list the test IDs in the manifest order")
    inputs = {"full": (inference.full_array(images, md.SIZE), [(0, 0, im.size[0], im.size[1]) for im in images]),
              "auto": (ar.inputs_from_geometry(images, auto_geometry), [(int(g.exp_x0), int(g.exp_y0), int(g.exp_x1), int(g.exp_y1)) for g in auto_geometry.itertuples()])}
    import keras
    training.configure_gpu()
    n = len(rows)
    slices = [slice(s, min(s + BATCH, n)) for s in range(0, n, BATCH)]
    out_rows = []
    for combo in COMBOS:
        x, boxes = inputs[combo["condition"]]
        valid = np.stack([md.validity_mask(b[2] - b[0], b[3] - b[1]) for b in boxes])
        variant = config.VARIANTS[MODEL_OF[combo["condition"]]]
        models.reset_random_sources(combo["seed"])
        model = models.compile_model(keras.models.load_model(config.run_paths(variant, combo["seed"])["checkpoint"], compile=False), 1e-3)
        graph = md.DiagnosticGraph(model)

        def predict(array):
            from busbra.evaluation import sigmoid_scores
            return sigmoid_scores(model.predict(training.make_eval_dataset(array, np.zeros(len(array), dtype=np.float32), BATCH), verbose=0), len(array))

        def logits_of(array):
            return np.concatenate([graph.logits(array[s]) for s in slices])

        clean_logits = logits_of(x)
        gradient, tape_logit = np.empty_like(x), np.empty(n, dtype=np.float64)
        for s in slices:
            result = graph.input_gradient(x[s], labels[s], np.ones(s.stop - s.start, dtype=np.float32))
            gradient[s], tape_logit[s] = result["gradient"], result["tape_logit"]
        zero_gradient = np.array([bool((gradient[i] == 0).all()) for i in range(n)])
        for k255 in rt.EPSILONS_255:
            eps = md.epsilon_value(k255)
            x_adv = md.fgsm_step(x, gradient, eps, valid)
            scores, logits = predict(x_adv), logits_of(x_adv)
            loss = md.bce_from_logit(logits, labels)
            reports = [md.perturbation_report(x[i], x_adv[i], valid[i], eps) for i in range(n)]
            for i, r in enumerate(rows):
                out_rows.append({"combo_id": combo["combo_id"], "group_id": combo["group_id"], "seed": combo["seed"], "condition": combo["condition"], "epsilon_255": k255, "ID": r["ID"], "Case": r["Case"],
                                 "label": int(r["label"]), "malignant_score": float(scores[i]), "logit": float(logits[i]), "bce": float(loss[i]), "gradient_all_zero": bool(zero_gradient[i]),
                                 "perturbation_linf": reports[i]["linf"], "pixels_changed": reports[i]["n_changed"], "tape_forward_minus_inference_logit": float(tape_logit[i] - clean_logits[i])})
        del model, graph
        keras.backend.clear_session()
        print(f"{combo['combo_id']} ({combo['condition']}): attack scores computed")
    pd.DataFrame(out_rows).to_csv(SCORES_PATH, index=False, lineterminator="\n", na_rep="")


# %% Computation from stored scores
def compute() -> dict[str, pd.DataFrame]:
    manifest = data.read_manifest()
    data.check_split(manifest)
    test_rows = data.subset_frame(manifest, "test").to_dict("records")
    meta = pd.DataFrame(test_rows)[["ID", "Case", "label"]].astype({"label": int})
    entries = {e["combo_id"]: e for e in fe.combinations()}
    thresholds, problems = te.load_frozen_thresholds(TABLES / "final_operating_points.csv", list(entries))
    if problems:
        raise ValueError("; ".join(problems[:3]))
    groups = rt.device_groups(test_rows)
    scores = pd.read_csv(SCORES_PATH, dtype={"ID": str, "Case": str, "seed": "Int64"}, float_precision="round_trip")
    clean = pd.read_csv(TABLES / "test_evaluation_scores.csv", dtype={"ID": str, "Case": str, "seed": "Int64"}, float_precision="round_trip")
    metrics_rows, core_rows, case_rows, decision_rows, frame_of = [], [], [], [], {}
    for combo in COMBOS:
        key = {"combo_id": combo["combo_id"], "group_id": combo["group_id"], "seed": combo["seed"]}
        for k in rt.EPSILONS_255:
            part = scores[(scores["combo_id"] == combo["combo_id"]) & (scores["epsilon_255"] == k)]
            frame = part[["ID", "Case", "label", "malignant_score"]].reset_index(drop=True)
            if list(frame["ID"]) != list(meta["ID"]):
                raise ValueError(f"{combo['combo_id']} eps {k}: IDs differ from the test order")
            frame_of[(combo["combo_id"], k)] = frame
            core, metrics, case_view = te.combination_rows({**key}, frame, thresholds)
            metrics_rows += [{"condition": combo["condition"], "epsilon_255": k, **r} for r in metrics]
            core_rows += [{"condition": combo["condition"], "epsilon_255": k, **r} for r in core]
            case_rows += [{"condition": combo["condition"], "epsilon_255": k, **r} for r in case_view]
            if k > 0:
                decision_rows += [{"condition": combo["condition"], **r} for r in rt.decision_change_rows(key, k, frame_of[(combo["combo_id"], 0)], frame, thresholds)]
    metrics, core, case_view, decisions = (pd.DataFrame(r) for r in (metrics_rows, core_rows, case_rows, decision_rows))
    summary_metrics = rt.seed_summary(metrics, ["condition", "group_id", "epsilon_255", "level"], ["roc_auc", "average_precision"])
    core_work = core.copy()
    core_work["threshold"] = core_work["threshold"].astype(float)
    summary_ops = rt.seed_summary(core_work, ["condition", "group_id", "epsilon_255", "level", "point"], ["threshold", *rt.CORE_METRICS])
    key_m, key_c = ["combo_id", "condition", "group_id", "seed", "level"], ["combo_id", "condition", "group_id", "seed", "level", "point"]
    changes = pd.concat([rt.paired_change(metrics, key_m, ["roc_auc", "average_precision"]).assign(point=""), rt.paired_change(core, key_c, rt.CORE_METRICS)], ignore_index=True)
    summary_changes = []
    for values, part in changes.assign(point=changes["point"].fillna("")).groupby(["condition", "group_id", "epsilon_255", "level", "point", "metric"], sort=True):
        change = pd.to_numeric(part["change"], errors="coerce")
        summary_changes.append({**dict(zip(["condition", "group_id", "epsilon_255", "level", "point", "metric"], values)), **ev.stats(part["change"]), "n_negative": int((change < 0).sum()),
                                "n_positive": int((change > 0).sum()), "n_zero": int((change == 0).sum())})
    decision_summary = rt.seed_summary(decisions, ["condition", "group_id", "epsilon_255", "level", "point"],
                                       ["correct_to_wrong", "wrong_to_correct", "stay_correct", "stay_wrong", "decision_flips", "flips_to_malignant", "flips_to_benign",
                                        "benign_correct_to_wrong", "malignant_correct_to_wrong", "benign_wrong_to_correct", "malignant_wrong_to_correct"])
    # device breakdown
    clean_metrics, clean_core = [], []
    for combo_id, part in clean.groupby("combo_id", sort=False):
        key = {"combo_id": combo_id, "group_id": entries[combo_id]["group_id"], "seed": entries[combo_id]["seed"]}
        m, c = rt.subset_rows(part[["ID", "Case", "label", "malignant_score"]].reset_index(drop=True), groups, key, thresholds)
        clean_metrics += m
        clean_core += c
    clean_metrics, clean_core = pd.DataFrame(clean_metrics), pd.DataFrame(clean_core)
    att_metrics, att_core = [], []
    for combo in COMBOS:
        key = {"combo_id": combo["combo_id"], "group_id": combo["group_id"], "seed": combo["seed"]}
        for k in rt.EPSILONS_255:
            m, c = rt.subset_rows(frame_of[(combo["combo_id"], k)], groups, key, thresholds)
            att_metrics += [{"condition": combo["condition"], "epsilon_255": k, **r} for r in m]
            att_core += [{"condition": combo["condition"], "epsilon_255": k, **r} for r in c]
    att_metrics, att_core = pd.DataFrame(att_metrics), pd.DataFrame(att_core)
    point_columns = ["tp", "fn", "fp", "tn", "sensitivity", "specificity", "precision", "npv", "f1", "accuracy", "predicted_positive_share"]  # the threshold is not summarised here
    clean_core_f, att_core_f = clean_core.copy(), att_core.copy()
    clean_core_f["threshold"], att_core_f["threshold"] = clean_core_f["threshold"].astype(float), att_core_f["threshold"].astype(float)
    clean_summary = pd.concat([rt.seed_summary(clean_metrics, ["group_id", "group", "level"], ["roc_auc", "average_precision"]).assign(point=""),
                               rt.seed_summary(clean_core_f, ["group_id", "group", "level", "point"], point_columns)], ignore_index=True)
    attacked_summary = pd.concat([rt.seed_summary(att_metrics, ["condition", "epsilon_255", "group", "level"], ["roc_auc", "average_precision"]).assign(point=""),
                                  rt.seed_summary(att_core_f, ["condition", "epsilon_255", "group", "level", "point"], point_columns)], ignore_index=True)
    dev_changes = pd.concat([rt.paired_change(att_metrics, ["combo_id", "condition", "seed", "group", "level"], ["roc_auc", "average_precision"]).assign(point=""),
                             rt.paired_change(att_core, ["combo_id", "condition", "seed", "group", "level", "point"], ["fp", "fn", "sensitivity", "specificity"])], ignore_index=True)
    dev_changes["point"] = dev_changes["point"].fillna("")
    attacked_changes = []
    for values, part in dev_changes.groupby(["condition", "epsilon_255", "group", "level", "point", "metric"], sort=True):
        change = pd.to_numeric(part["change"], errors="coerce")
        attacked_changes.append({**dict(zip(["condition", "epsilon_255", "group", "level", "point", "metric"], values)), **ev.stats(part["change"]), "n_negative": int((change < 0).sum()),
                                 "n_positive": int((change > 0).sum()), "n_zero": int((change == 0).sum())})
    return {"robustness_test_metrics": metrics, "robustness_test_operating_points": core, "robustness_test_case_view_image_thresholds": case_view,
            "robustness_test_summary_metrics": summary_metrics, "robustness_test_summary_operating_points": summary_ops, "robustness_test_changes": changes,
            "robustness_test_changes_summary": pd.DataFrame(summary_changes), "robustness_test_decision_changes": decisions, "robustness_test_decision_changes_summary": decision_summary,
            "device_test_assignment": rt.device_table(test_rows, groups), "device_test_clean_metrics": clean_metrics, "device_test_clean_operating_points": clean_core,
            "device_test_clean_summary": clean_summary, "device_test_attacked_metrics": att_metrics, "device_test_attacked_operating_points": att_core,
            "device_test_attacked_summary": attacked_summary, "device_test_attacked_changes_summary": pd.DataFrame(attacked_changes)}


same = results.same_as_stored


# %% Run
def main(argv: list[str]) -> None:
    if "--attack" in argv:
        attack()
        return
    tables = compute()
    if "--write" in argv:
        clash = [n for n in tables if (TABLES / f"{n}.csv").exists()]
        if clash:
            raise FileExistsError(f"already present: {clash[:3]} ...; nothing was changed")
        for name, frame in tables.items():
            frame.to_csv(TABLES / f"{name}.csv", index=False, lineterminator="\n", na_rep="")
        print(f"wrote {len(tables)} tables")
        return
    for name, frame in tables.items():
        path = TABLES / f"{name}.csv"
        print(f"{name + '.csv':58s} {len(frame):5d} rows  " + ("identical to the stored table" if path.is_file() and same(frame, path) else "DIFFERENT or not stored"))


if __name__ == "__main__":
    main(sys.argv[1:] if "__file__" in globals() else [])
