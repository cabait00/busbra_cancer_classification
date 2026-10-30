"""Tests of the connections between the producers and the consumers of the result tables, the explicit-action guards and the process launcher (CPU only).

Run from the project folder:  .venv/bin/python -m unittest discover -s tests
New tables are only built in memory or in a temporary folder; stored results are read, never written. Nothing here starts a training, loads a model or opens a dataset image: the
geometry test uses small synthetic PNG files in a temporary folder.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "plots"))

from busbra import aggregation, config, data, gradcam_test as gt, launcher, model_diagnostics as md, results, test_evaluation as te, training  # noqa: E402

TABLES = config.TABLES_DIR


def load_script(name: str):
    """Import an analysis script by file name without running it (every script starts its action under __main__ only)."""
    spec = importlib.util.spec_from_file_location("script_" + name, ROOT / "analysis" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal(path: Path, name: str):
    """A module-level literal of a script (read from the source text; the script is not executed)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class GradcamFormatTests(unittest.TestCase):
    """The new selection and geometry have the stored format, and the renderer inputs are complete."""

    @classmethod
    def setUpClass(cls):
        cls.script = load_script("10_interpretation")
        cls.selection = cls.script.derive_selection()

    def synthetic_cam(self):
        rng = np.random.default_rng(0)
        rows, maps = [], {}
        for r in self.selection["rows"].itertuples():
            for condition in gt.CONDITIONS:
                coarse = rng.random((5, 5)).astype(np.float32)
                image_map = np.full((8, 8), np.nan, dtype=np.float32)
                image_map[2:6, 2:6] = 0.5
                facts = gt.map_facts(coarse, image_map, md.validity_mask(40, 30))
                rows.append({"condition": condition, "ID": r.ID, "raw_max": float(coarse.max()), "empty_after_relu": False, **facts})
                maps[f"image__{condition}__{r.ID}"] = image_map
        return pd.DataFrame(rows), maps

    def test_selection_and_geometry_equal_the_stored_tables(self):
        self.assertEqual(list(self.selection["rows"].columns), gt.SELECTION_COLUMNS)
        self.assertEqual(list(self.selection["geometry"].columns), gt.GEOMETRY_COLUMNS)
        self.assertEqual(len(self.selection["rows"]), 11)
        self.assertEqual(len(self.selection["geometry"]), 33)
        self.assertTrue(results.same_as_stored(self.selection["rows"], self.script.OUTPUTS["selection"]))
        self.assertTrue(results.same_as_stored(self.selection["geometry"], self.script.OUTPUTS["geometry"]))

    def test_renderer_inputs_are_complete_for_new_tables(self):
        cam, maps = self.synthetic_cam()
        ids = list(self.selection["rows"]["ID"])
        self.assertEqual(gt.table_problems(self.selection["rows"], self.selection["geometry"], cam, maps, ids), [])
        selection = self.selection["rows"].set_index("ID", drop=False)
        cam_by = cam.set_index(["condition", "ID"], drop=False)
        for identifier in ids:
            self.assertEqual(len(gt.case_title_lines(selection.loc[identifier], {c: c for c in gt.SPECIAL_REASON})), 3)
            for condition in gt.CONDITIONS:
                lines = gt.panel_lines(selection.loc[identifier], cam_by.loc[(condition, identifier)], condition, condition)
                self.assertTrue(4 <= len(lines) <= 5)

    def test_missing_columns_and_map_keys_are_reported(self):
        cam, maps = self.synthetic_cam()
        ids = list(self.selection["rows"]["ID"])
        problems = gt.table_problems(self.selection["rows"].drop(columns=["Device", "auto_fallback_full_image"]), self.selection["geometry"], cam, maps, ids)
        self.assertEqual(len(problems), 2)
        maps.pop(f"image__auto__{ids[0]}")
        self.assertTrue(any("image__auto" in p for p in gt.table_problems(self.selection["rows"], self.selection["geometry"], cam, maps, ids)))

    def test_stored_grad_cam_tables_match_the_renderer(self):
        maps_file = config.PROJECT_ROOT / "data" / "processed" / "gradcam_test_maps.npz"  # local derived array, not part of the public repository
        if not maps_file.is_file():
            self.skipTest("optional local reference data/processed/gradcam_test_maps.npz is not present")
        stored = {n: pd.read_csv(TABLES / f"gradcam_test_{n}.csv", keep_default_na=False) for n in ("selection", "geometry", "cam")}
        with np.load(maps_file, allow_pickle=False) as archive:
            keys = set(archive.files)
        self.assertEqual(gt.table_problems(stored["selection"], stored["geometry"], stored["cam"], keys, stored["selection"]["ID"]), [])


class AggregationAndThresholdTests(unittest.TestCase):
    """New shared tables do not carry run identifiers, and the threshold analysis does not need them."""

    @classmethod
    def setUpClass(cls):
        cls.script = load_script("12_threshold_analysis")

    def test_aggregated_table_has_no_run_id_and_is_accepted(self):
        runs = aggregation.collect_runs(["mobilenetv2_higher_lr_160"], config.SEEDS)
        self.assertNotIn("run_id", runs.columns)
        self.assertEqual(self.script.run_ids_from_table(runs), {})
        stored = pd.read_csv(TABLES / "multiseed_run_metrics.csv")
        self.assertEqual(len(self.script.run_ids_from_table(stored)), 25)

    def test_threshold_tables_without_run_ids_equal_the_stored_tables_except_the_identifier(self):
        tables = self.script.compute({})
        for key, frame in tables.items():
            self.assertNotIn("run_id", frame.columns)
            self.assertTrue(results.same_as_stored(frame, TABLES / self.script.OUTPUTS[key], ignore=("run_id",)), key)


class ReferenceGroupTests(unittest.TestCase):
    """The study summaries hold the reference groups the plots read."""

    EXPECTED_GROUPS = {"multiseed": 7, "resolution250": 4, "augmentation160": 6, "expert_roi": 4}

    @classmethod
    def setUpClass(cls):
        baselines = pd.read_csv(TABLES / "baseline_metrics.csv")
        cls.summary = {p: aggregation.study_summary(p, config.SEEDS, baselines) for p in cls.EXPECTED_GROUPS}

    def test_groups_columns_and_values_equal_the_stored_summaries(self):
        for preset, expected in self.EXPECTED_GROUPS.items():
            new, stored = self.summary[preset], pd.read_csv(TABLES / f"{preset}_summary_validation.csv")
            self.assertEqual(new["pipeline"].nunique(), expected, preset)
            self.assertEqual(list(new.columns), list(stored.columns), preset)
            for column in stored.columns:
                if column == "source":
                    continue  # descriptive wording only
                a, b = new[column], stored[column]
                if pd.api.types.is_numeric_dtype(b):
                    np.testing.assert_allclose(pd.to_numeric(a).to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=0, atol=1e-12, equal_nan=True, err_msg=f"{preset} {column}")
                else:
                    self.assertEqual(a.fillna("").astype(str).tolist(), b.fillna("").astype(str).tolist(), f"{preset} {column}")

    def test_every_group_and_pair_the_plots_use_is_present(self):
        for preset in ("resolution250", "augmentation160", "expert_roi"):
            path = ROOT / "plots" / f"plot_{preset}_results.py"
            groups, pairs = literal(path, "GROUPS"), literal(path, "PAIRS")
            summary = self.summary[preset]
            paired = aggregation.paired_between(*(2 * [pd.concat([aggregation.collect_runs(config.PRESETS[ref], config.SEEDS) for ref in dict.fromkeys(
                ["multiseed", "resolution250", "augmentation160", "expert_roi"])], ignore_index=True)]), [(label, a, b) for label, a, b in self.cross_pairs(preset)], config.SEEDS, ("roc_auc",))
            for name, _ in groups:
                row = summary[(summary["pipeline"] == name) & (summary["metric"] == "roc_auc")]
                self.assertEqual(len(row), 1, f"{preset}: {name}")
                self.assertTrue(np.isfinite(row["sd_ddof1"].iloc[0]))
            for pair, _ in pairs:
                self.assertEqual(len(paired[paired["pair"] == pair]), 5, f"{preset}: {pair}")

    @staticmethod
    def cross_pairs(preset):
        script = load_script("03_classification")
        return script.CROSS[preset][2]

    def test_plot_values_of_the_study_figure_are_reconstructed_from_stored_tables(self):
        import study_figures
        for preset in ("resolution250", "augmentation160", "expert_roi"):
            path = ROOT / "plots" / f"plot_{preset}_results.py"
            groups, pairs = literal(path, "GROUPS"), literal(path, "PAIRS")
            paired = pd.read_csv(TABLES / f"{preset}_paired_differences.csv")
            _, values = study_figures.auc_values(paired, pairs)
            for name, _ in groups:
                self.assertEqual(sorted(s for (n, s) in values if n == name), sorted(config.SEEDS), f"{preset}: {name}")


class SingleRunTests(unittest.TestCase):
    def test_historical_metrics_are_mapped_to_the_common_format_once(self):
        variant = config.VARIANTS["small_cnn"]
        table = aggregation.read_run_metrics(variant, config.HISTORICAL_SEED)
        self.assertEqual(set(table["seed"]), {42})
        with self.assertRaises(ValueError):
            aggregation.read_run_metrics(variant, 7)
        runs = aggregation.collect_runs(config.PRESETS["single_runs"], [42])
        self.assertEqual(sorted(runs["pipeline"].unique()), sorted(config.PRESETS["single_runs"]))

    def test_unsupported_actions_are_argument_errors_and_the_read_view_shows_the_metrics(self):
        script = load_script("03_classification")
        for argv in (["--preset", "single_runs", "--aggregate"], ["--preset", "single_runs", "--seeds", "7"], ["--preset", "multiseed", "--train", "--variants", "nonsense"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                script.main(argv)
            self.assertEqual(caught.exception.code, 2, argv)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            script.main(["--preset", "single_runs"])
        self.assertIn("small_cnn", out.getvalue())
        self.assertIn("roc_auc", out.getvalue())

    def test_study_tables_equal_the_stored_tables(self):
        script = load_script("03_classification")
        for preset in ("multiseed", "resolution250", "augmentation160", "expert_roi"):
            lines = script.compare_with_stored(preset, script.study_tables(preset, config.SEEDS))
            self.assertEqual(len(lines), 8 if preset == "expert_roi" else 4, preset)
            self.assertTrue(all(line.endswith("identical") for line in lines), lines)


class BootstrapTests(unittest.TestCase):
    def test_draws_are_the_documented_generator_output(self):
        draws = te.bootstrap_draws(np.array([1] * 5 + [0] * 9), replicates=3)
        expected = [[4, 0, 0, 3, 1, 9, 5, 8, 10, 8, 12, 12, 11, 13], [3, 0, 4, 3, 0, 7, 6, 13, 11, 13, 7, 10, 10, 11], [0, 2, 3, 4, 3, 9, 9, 8, 6, 7, 6, 7, 12, 9]]
        self.assertEqual(draws.tolist(), expected)  # numpy.random.default_rng(2026), malignant cases first, then benign
        self.assertEqual((te.BOOTSTRAP["replicates"], te.BOOTSTRAP["rng_seed"]), (2000, 2026))

    def test_the_stored_reference_is_optional(self):
        script = load_script("09_test_evaluation")
        with tempfile.TemporaryDirectory() as tmp:
            fake = {"tables": {}, "draws": np.zeros((1, 1), dtype=np.int64)}
            out = io.StringIO()
            with mock.patch.object(script, "compute", return_value=fake), mock.patch.object(script, "TEST_DIR", Path(tmp)), contextlib.redirect_stdout(out):
                script.main([])
            self.assertIn("no stored reference draws", out.getvalue())
            self.assertNotIn("identical to the stored reference draws", out.getvalue())


class LauncherTests(unittest.TestCase):
    def test_plan_order_and_validation(self):
        plan = launcher.plan_runs("multiseed", [7, 21], ["mobilenetv2_frozen_160", "small_cnn_160"])
        self.assertEqual(plan, [("small_cnn_160", 7), ("mobilenetv2_frozen_160", 7), ("small_cnn_160", 21), ("mobilenetv2_frozen_160", 21)])
        self.assertEqual(launcher.plan_runs("single_runs"), [(n, 42) for n in config.PRESETS["single_runs"]])
        for bad in (("single_runs", [7], None), ("multiseed", [99], None), ("multiseed", [7, 7], None), ("multiseed", None, ["nonsense"])):
            with self.assertRaises(ValueError):
                launcher.plan_runs(*bad)

    def test_child_environment(self):
        base = {"LD_LIBRARY_PATH": "/foreign", "PYTHONHASHSEED": "1", "OTHER": "x"}
        study = launcher.child_environment(config.VARIANTS["small_cnn_160"], 123, base)
        self.assertEqual((study["PYTHONHASHSEED"], study["OTHER"], "LD_LIBRARY_PATH" in study), ("123", "x", False))
        historical = launcher.child_environment(config.VARIANTS["small_cnn"], 42, base)
        self.assertNotIn("PYTHONHASHSEED", historical)
        self.assertEqual(base["PYTHONHASHSEED"], "1")  # the caller's mapping is not changed

    def test_commands_environments_and_the_stop_after_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = lambda variant, seed: [Path(tmp) / f"{variant.name}_{seed}.out"]  # noqa: E731
            calls = []

            def starter(command, env):
                calls.append((command, env))
                return mock.Mock(returncode=1 if len(calls) == 2 else 0)

            with self.assertRaises(RuntimeError):
                launcher.run_plan("expert_roi", Path("analysis/03_classification.py"), [7, 21], None, starter=starter, base_env={"LD_LIBRARY_PATH": "/x"}, outputs=outputs, geometry_reader=lambda: None)
            self.assertEqual(len(calls), 2)  # the third run was not started
            first, second = calls
            self.assertEqual(first[0][3:], ["expert_roi", "--train-one", "mobilenetv2_frozen_160_roi", "--seeds", "7"])
            self.assertEqual(second[0][5], "mobilenetv2_higher_lr_160_roi")
            self.assertTrue(all(env["PYTHONHASHSEED"] == "7" and "LD_LIBRARY_PATH" not in env for _, env in calls))

    def test_existing_outputs_block_the_whole_plan_before_any_process_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "history.csv"
            existing.write_text("x")
            started = []
            outputs = lambda variant, seed: [existing] if (variant.name, seed) == ("mobilenetv2_higher_lr_160_roi", 21) else [Path(tmp) / "absent.out"]  # noqa: E731
            with self.assertRaises(FileExistsError):
                launcher.run_plan("expert_roi", Path("x.py"), [7, 21], None, starter=lambda *a, **k: started.append(a), outputs=outputs, geometry_reader=lambda: None)
            self.assertEqual(started, [])

    def test_a_fine_tuning_run_needs_its_source_checkpoint_or_its_source_in_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = {"mobilenetv2_frozen_160_roi": Path(tmp) / "nothing.keras"}
            real = config.run_paths
            patched = lambda variant, seed: {**real(variant, seed), "checkpoint": missing.get(variant.name, real(variant, seed)["checkpoint"])}  # noqa: E731
            absent = lambda variant, seed: [Path(tmp) / "absent.out"]  # noqa: E731
            with mock.patch.object(config, "run_paths", patched):
                with self.assertRaises(FileExistsError):
                    launcher.check_plan([("mobilenetv2_higher_lr_160_roi", 7)], absent)
                launcher.check_plan([("mobilenetv2_frozen_160_roi", 7), ("mobilenetv2_higher_lr_160_roi", 7)], absent)

    def test_the_trainer_refuses_a_wrong_hash_seed_and_existing_results_before_any_data_access(self):
        variant = config.VARIANTS["small_cnn_160"]
        with mock.patch.object(data, "load_classification_inputs", side_effect=AssertionError("data were accessed")):
            with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "1"}), self.assertRaises(RuntimeError):
                training.train_variant(variant, 7)
            with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "7"}), self.assertRaises(FileExistsError):
                training.train_variant(variant, 7)  # the stored results of this run exist


class OutputGuardTests(unittest.TestCase):
    """Every explicit scientific action checks all of its own outputs before any data or model is touched, and one existing file is enough to stop it."""

    class Reached(Exception):
        pass

    def assert_guard(self, module, action, patches: dict, existing: Path, data_hooks: list):
        """With the output paths redirected into a temporary folder: one existing file raises FileExistsError; with none existing the action proceeds to the (replaced) data access."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"keras": mock.Mock(), "tensorflow": mock.Mock()}))  # a real import of the libraries must never happen in a test
            for target, value in patches.items():
                stack.enter_context(mock.patch.object(module, target, value))
            for owner, name in data_hooks:
                stack.enter_context(mock.patch.object(owner, name, side_effect=self.Reached))
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_text("x")
            with self.assertRaises(FileExistsError):
                action()
            existing.unlink()
            with self.assertRaises(self.Reached):
                action()

    def test_segmentation(self):
        script = load_script("05_segmentation")
        with tempfile.TemporaryDirectory() as tmp:
            paths = {k: Path(tmp) / f"{k}.out" for k in script.PATHS}
            for key in script.OUTPUT_KEYS:  # each single output blocks the action
                paths[key].write_text("x")
                with mock.patch.object(script, "PATHS", paths), mock.patch.object(script.data, "read_manifest", side_effect=self.Reached), self.assertRaises(FileExistsError):
                    script.train()
                paths[key].unlink()
            with mock.patch.object(script, "PATHS", paths), mock.patch.object(script.data, "read_manifest", side_effect=self.Reached):
                with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "7"}), self.assertRaises(self.Reached):
                    script.train()
                with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "1"}), self.assertRaises(RuntimeError):  # wrong hash seed: refused before any data access
                    script.train()

    def test_automatic_roi(self):
        script = load_script("06_auto_roi")
        with tempfile.TemporaryDirectory() as tmp:
            masks = Path(tmp) / "auto_roi" / "predicted_masks_validation.npz"
            with mock.patch.object(config, "TABLES_DIR", Path(tmp)):
                self.assert_guard(script, script.run, {}, masks, [(script.data, "read_manifest")])
                with mock.patch.object(script.data, "read_manifest", side_effect=self.Reached):  # --aggregate: every one of the 15 outputs blocks it before any input is read
                    for key, path in script.table_paths().items():
                        path.write_text("x")
                        with self.assertRaises(FileExistsError, msg=key):
                            script.aggregate()
                        path.unlink()
                    with self.assertRaises(self.Reached):
                        script.aggregate()
    def test_adversarial_addon(self):
        script = load_script("08_adversarial_addon")
        with tempfile.TemporaryDirectory() as tmp:
            outputs = [Path(tmp) / f"o{i}" for i in range(4)]
            self.assert_guard(script, script.train, {"train_outputs": lambda: outputs}, outputs[3], [(script.data, "read_manifest")])
        paths = script.train_outputs()
        self.assertEqual(len(paths), 5 * (2 + 2 * 4))  # per seed: original (evaluation, scores) and two continuations (evaluation, scores, history, checkpoint)

    def test_test_inference(self):
        script = load_script("09_test_evaluation")
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            self.assert_guard(script, script.infer, {"AUTO_GEOMETRY_PATH": t / "a.csv", "EXPERT_GEOMETRY_PATH": t / "e.csv", "DIAGNOSTICS_PATH": t / "d.csv", "MASKS_PATH": t / "m.npz",
                                                     "SCORES_DIR": t / "scores", "CENTRAL_SCORES": t / "central.csv"}, t / "central.csv", [(script.data, "read_manifest")])

    def test_grad_cam(self):
        script = load_script("10_interpretation")
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            outputs = {name: t / f"{name}.csv" for name in script.OUTPUTS}
            self.assert_guard(script, script.run, {"OUTPUTS": outputs, "MAPS_PATH": t / "maps.npz"}, outputs["geometry"], [(script, "derive_selection")])

    def test_fgsm(self):
        script = load_script("11_robustness_devices")
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_guard(script, script.attack, {"SCORES_PATH": Path(tmp) / "s.csv"}, Path(tmp) / "s.csv", [(script.data, "read_manifest")])

    def test_baselines(self):
        script = load_script("02_baselines")
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            models = {"dummy": t / "dummy.joblib", "logreg_l2": t / "logreg.joblib"}
            with mock.patch.object(script.config, "BASELINE_MODELS", models):
                self.assert_guard(script, script.fit, {"METRICS_PATH": t / "m.csv", "PREDICTIONS_PATH": t / "p.csv"}, models["logreg_l2"], [(script.data, "read_manifest")])


class RoiGeometryTests(unittest.TestCase):
    """The ROI loader and the mask EDA use one measurement (synthetic images and masks in a temporary folder)."""

    def make_row(self, root: Path, name: str, box):
        (root / "img").mkdir(exist_ok=True)
        (root / "msk").mkdir(exist_ok=True)
        image = Image.fromarray((np.arange(60 * 80, dtype=np.uint32).reshape(60, 80) % 251).astype(np.uint8), mode="L")
        mask = np.zeros((60, 80), dtype=bool)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
        image.save(root / "img" / f"{name}.png")
        Image.fromarray(mask).save(root / "msk" / f"{name}.png")
        return {"ID": name, "Case": "1", "label": "1", "Pathology": "malignant", "Device": "synthetic", "image_relpath": f"img/{name}.png", "mask_relpath": f"msk/{name}.png",
                "image_width": 80, "image_height": 60, "mask_width": 80, "mask_height": 60, "csv_size_mismatch": "False"}

    def test_loader_and_eda_agree_and_follow_the_rounding_rule(self):
        eda = load_script("04_mask_eda_roi")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [self.make_row(root, "a", (10, 12, 30, 32)), self.make_row(root, "b", (0, 0, 7, 5))]
            arrays, geometry = data.load_roi_arrays(root, rows, "train")
            with mock.patch.object(config, "PROJECT_ROOT", root):
                pairs = eda.analyse_pairs(rows)
        self.assertEqual(arrays.shape, (2, 160, 160, 1))
        for column in data.GEOMETRY_COLUMNS:
            if column != "subset":
                self.assertEqual(geometry[column].tolist(), pairs[column].tolist(), column)
        first = geometry.iloc[0]  # tight box 10..30 x 12..32 (20 x 20): 20 % rounded up per side = 4 pixels
        self.assertEqual((first.exp_x0, first.exp_y0, first.exp_x1, first.exp_y1), (6, 8, 34, 36))
        second = geometry.iloc[1]  # the expansion is clipped at the image border
        self.assertEqual((second.exp_x0, second.exp_y0), (0, 0))
        self.assertTrue(second.expansion_clipped)

    def test_a_violation_is_raised_without_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.make_row(root, "a", (10, 12, 30, 32))
            row["image_width"] = 81
            with self.assertRaises(data.RoiDataError):
                data.load_roi_arrays(root, [row], "train")


class FineTuningPlotInputsTests(unittest.TestCase):
    def test_start_values_and_curve_facts_equal_the_documented_ones(self):
        history = pd.read_csv(TABLES / "mobilenetv2_finetuned_history.csv")
        self.assertEqual((len(history), int(history.loc[history["val_loss"].idxmin(), "epoch"])), (12, 7))
        frozen_metrics = pd.read_csv(TABLES / "mobilenetv2_frozen_metrics.csv", float_precision="round_trip")
        self.assertEqual(frozen_metrics[frozen_metrics["subset"] == "validation"]["roc_auc"].iloc[0], config.FROZEN_START_VALIDATION["roc_auc"])
        frozen_history = pd.read_csv(TABLES / "mobilenetv2_frozen_history.csv")
        self.assertAlmostEqual(float(frozen_history.loc[frozen_history["is_best_epoch"], "val_loss"].iloc[0]), config.FROZEN_START_VALIDATION["loss"], places=7)


class ImportTests(unittest.TestCase):
    def test_modules_import_without_tensorflow(self):
        self.assertNotIn("tensorflow", sys.modules)
        self.assertNotIn("keras", sys.modules)

    def test_the_u_net_module_is_checked_statically_only(self):
        tree = ast.parse((ROOT / "src" / "busbra" / "segmentation_model.py").read_text(encoding="utf-8"))
        imported = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        self.assertTrue({"keras", "tensorflow"} <= imported)  # the only module with a top-level Keras import; scripts import it inside the explicit training action
        for script in sorted((ROOT / "analysis").glob("*.py")) + sorted((ROOT / "plots").glob("*.py")):
            top = ast.parse(script.read_text(encoding="utf-8")).body
            names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(names & {"keras", "tensorflow"}, script.name)
            self.assertFalse(any(a.name == "segmentation_model" for n in top if isinstance(n, ast.ImportFrom) for a in n.names), script.name)


if __name__ == "__main__":
    unittest.main()
