"""Tests of the ROI connections: expert threshold curves and operating points, the complete automatic-ROI study, and the expert geometry that is measured once and read by the consumers.

CPU only. Primary stored inputs (scores, diagnoses, example selection) are copied into a temporary folder that replaces config.TABLES_DIR; every table is derived and written THERE and compared
with the unchanged stored tables of the project. Geometry tests use small synthetic PNG files. No model, no dataset image and no TensorFlow/Keras is touched.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import shutil
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

from busbra import auto_roi as ar, config, data, launcher, operating as op, results, roi_study as rs  # noqa: E402

STORED = config.TABLES_DIR


def load_script(name: str):
    spec = importlib.util.spec_from_file_location("roi_script_" + name, ROOT / "analysis" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_primary_inputs(target: Path, with_auto: bool) -> None:
    """Primary inputs only: the metrics and predictions of the 4B and 4E runs, and (automatic ROI) the new scores, the per-image diagnoses and the fixed example selection."""
    for preset in ("augmentation160", "expert_roi"):
        for variant in config.PRESETS[preset]:
            for seed in config.SEEDS:
                source = config.run_paths(config.VARIANTS[variant], seed)
                for key in ("metrics", "predictions"):
                    destination = target / source[key].relative_to(STORED)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source[key], destination)
    if with_auto:
        for seed in config.SEEDS:
            for stage in ("frozen", "finetune"):
                destination = target / rs.auto_score_path(seed, stage).relative_to(STORED)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(rs.auto_score_path(seed, stage), destination)
        for source in (rs.diagnostics_path(), rs.segmentation_selection_path()):
            destination = target / source.relative_to(STORED)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def read_literal(path: Path, name: str):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


FLOAT_TOLERANCE = 1e-15  # absolute; the observed differences are single last-digit steps (at most 1.2e-16), see the comparison log


def table_difference(new_text: str, stored_text: str, shared_columns_only: bool = False) -> tuple[int, float]:
    """(number of cells that differ in text, largest absolute difference of those cells). Every differing cell must be a float written with a decimal point on both sides; texts, integers,
    booleans, empty cells and infinities must be equal as written."""
    new, stored = pd.read_csv(io.StringIO(new_text), dtype=str, keep_default_na=False), pd.read_csv(io.StringIO(stored_text), dtype=str, keep_default_na=False)
    if shared_columns_only:  # the stored run metrics of the earlier studies hold more technical columns than the common format
        columns = [c for c in new.columns if c in stored.columns]
        new, stored = new[columns], stored[columns]
    assert list(new.columns) == list(stored.columns) and len(new) == len(stored)
    cells, largest = 0, 0.0
    for column in stored.columns:
        for a, b in zip(stored[column], new[column]):
            if a == b:
                continue
            assert "." in a and "." in b and "inf" not in a + b and "nan" not in a + b, f"{column}: {a!r} vs {b!r} is not a float rounding difference"
            cells += 1
            largest = max(largest, abs(float(a) - float(b)))
    return cells, largest


class SharedOperatingRuleTests(unittest.TestCase):
    def test_target_threshold_with_ties_and_undefined_rates(self):
        frame = pd.DataFrame({"Case": ["a", "a", "b", "c", "c", "d"], "label": [1, 1, 0, 1, 1, 0], "malignant_score": [0.9, 0.4, 0.4, 0.4, 0.2, 0.1]})
        curve, points = op.run_analysis(frame, {"group": "g", "seed": 1}, ("0.90", "0.95"))
        by = {p["point_label"]: p for p in points}
        self.assertEqual(list(by), ["0.5", "target_0.90", "target_0.95"])
        self.assertEqual(by["target_0.90"]["needed_tp"], 4)  # 0.90 of 4 positives = 3.6 -> 4: the tie at 0.4 is taken together with the benign image of the same score
        self.assertEqual(by["target_0.90"]["threshold"], 0.2)
        self.assertEqual((by["0.5"]["tp"], by["0.5"]["fp"]), (1, 0))
        self.assertEqual(curve["threshold"].iloc[0], np.inf)  # the "all negative" point
        self.assertEqual(by["0.5"]["case_tp"], 1)  # case a is positive through its maximum score 0.9

    def test_paired_summary_counts_signs_and_keeps_undefined_values(self):
        values = {("a", 1): 0, ("a", 2): 0, ("b", 1): 1, ("b", 2): 2}  # false positives; the difference b - a is +1 and +2
        points = pd.DataFrame([{"group": g, "seed": s, "point_label": "0.5", "threshold": 0.5, "n_predicted_positive": 1, "tp": 1, "fn": 0, "fp": v, "tn": 0, "sensitivity": 1.0, "specificity": None,
                                "precision": None, "case_n_predicted_positive": 1, "case_tp": 1, "case_fn": 0, "case_fp": v, "case_tn": 0, "case_sensitivity": 1.0, "case_specificity": None, "case_precision": None}
                               for (g, s), v in values.items()])
        specs = [{"ids": {"point": "0.5"}, "a": "a", "b": "b", "label": "0.5"}]
        differences, summary = op.paired_point_tables(points, specs, [1, 2], "group", ("value_a", "value_b"), "difference_b_minus_a", ["point", "level", "metric"])
        fp = summary[(summary["level"] == "image") & (summary["metric"] == "fp")].iloc[0]
        self.assertEqual((fp["n_positive"], fp["n_negative"], fp["n_zero"]), (2, 0, 0))
        specificity = summary[(summary["level"] == "image") & (summary["metric"] == "specificity")].iloc[0]
        self.assertEqual(int(specificity["n_defined"]), 0)  # undefined stays undefined
        self.assertTrue(pd.isna(specificity["mean_difference"]))
        self.assertNotIn(("case", "threshold"), set(zip(differences["level"], differences["metric"])))


class ExpertOperatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script("03_classification")

    def test_the_eight_tables_are_written_in_a_temporary_folder_and_equal_the_stored_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            copy_primary_inputs(tmp, with_auto=False)
            with mock.patch.object(config, "TABLES_DIR", tmp):
                self.assertEqual(set(self.script.table_keys("expert_roi")), set(self.script.study_tables("expert_roi", config.SEEDS)))
                self.script.write_tables("expert_roi", config.SEEDS)
                written = {p.name: p.read_bytes() for p in tmp.glob("expert_roi_*.csv")}
                self.assertEqual(len(written), 8)
                # the second call is blocked and leaves every byte as it was
                with self.assertRaises(FileExistsError):
                    self.script.write_tables("expert_roi", config.SEEDS)
                self.assertEqual({p.name: p.read_bytes() for p in tmp.glob("expert_roi_*.csv")}, written)
            for name in ("expert_roi_threshold_curves.csv", "expert_roi_operating_points.csv", "expert_roi_operating_point_differences.csv", "expert_roi_operating_point_summary.csv",
                         "expert_roi_run_metrics.csv", "expert_roi_summary_validation.csv", "expert_roi_paired_differences.csv", "expert_roi_paired_summary.csv"):
                cells, largest = table_difference((tmp / name).read_text(), (STORED / name).read_text(), shared_columns_only=name.endswith("run_metrics.csv"))
                self.assertLessEqual(largest, FLOAT_TOLERANCE, name)  # the stored threshold tables were written from scores read with the fast float parser: last-digit steps only
                if not name.endswith(("threshold_curves.csv", "operating_points.csv", "operating_point_differences.csv", "operating_point_summary.csv")):
                    self.assertEqual(cells, 0, name)  # the four study tables of the earlier aggregation are byte-identical

    def test_a_later_existing_output_blocks_before_any_input_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "expert_roi_operating_point_summary.csv").write_text("x")
            with mock.patch.object(config, "TABLES_DIR", tmp), mock.patch.object(rs, "expert_score_sets", side_effect=AssertionError("an input was read")):
                with self.assertRaises(FileExistsError):
                    self.script.write_tables("expert_roi", config.SEEDS)
            self.assertEqual([p.name for p in tmp.iterdir()], ["expert_roi_operating_point_summary.csv"])

    def test_the_tables_hold_what_the_plot_reads(self):
        sets = rs.expert_score_sets()
        tables = rs.expert_operating_tables(sets)
        curves, points, differences, summary = (tables[k] for k in rs.EXPERT_OPERATING_FILES)
        self.assertTrue({"group", "seed", "tn", "fp", "fn", "tp", "sensitivity"} <= set(curves.columns))  # the columns the sensitivity-versus-false-positives panels read
        self.assertTrue({"group", "seed", "point_label", "fp", "sensitivity"} <= set(points.columns))
        self.assertEqual(curves.groupby(["group", "seed"]).ngroups, 20)
        n_negative = int(curves["tn"].iloc[0] + curves["fp"].iloc[0])
        self.assertEqual(n_negative, 260)  # the panels print the validation counts of 260 benign and 125 malignant images
        self.assertEqual(int(curves["tp"].iloc[0] + curves["fn"].iloc[0]), 125)
        self.assertEqual(sorted(curves["group"].unique()), sorted(rs.EXPERT_GROUPS))
        self.assertEqual(set(points["group"]), {"4B_frozen", "ROI_frozen", "4B_finetune", "ROI_finetune"})
        self.assertEqual(sorted(points["seed"].unique()), sorted(config.SEEDS))
        self.assertEqual(set(points["point_label"]), {"0.5", "target_0.90", "target_0.95"})  # no 0.80 row
        # the filters of the plot: stage finetune, target points, image level, metric fp; the columns difference_roi_minus_full, mean_difference, sd_ddof1, n_negative, n_positive, n_zero
        part = differences[(differences["stage"] == "finetune") & (differences["point"] == "target_0.95") & (differences["level"] == "image") & (differences["metric"] == "fp")]
        self.assertEqual(sorted(part["seed"]), sorted(config.SEEDS))
        stats = summary[(summary["stage"] == "finetune") & (summary["point"] == "target_0.95") & (summary["level"] == "image") & (summary["metric"] == "fp")]
        self.assertEqual(len(stats), 1)
        self.assertTrue({"mean_difference", "sd_ddof1", "n_negative", "n_positive", "n_zero"} <= set(stats.columns))
        self.assertEqual(int(points[points["point_label"] == "target_0.90"]["needed_tp"].iloc[0]), 113)
        self.assertEqual(int(points[points["point_label"] == "target_0.95"]["needed_tp"].iloc[0]), 119)


class AutoRoiAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script("06_auto_roi")

    def prepared(self, tmp: Path):
        copy_primary_inputs(tmp, with_auto=True)
        return mock.patch.object(config, "TABLES_DIR", tmp)

    def test_all_fifteen_tables_are_derived_from_primary_inputs_and_equal_the_stored_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.assertEqual(list(tmp.glob("auto_roi_*.csv")), [])  # no old derived table is available
            with self.prepared(tmp):
                self.script.aggregate()
                written = {p.name: p.read_bytes() for p in tmp.glob("auto_roi_*.csv") if p.name != rs.DIAGNOSTICS_FILE}
                self.assertEqual(sorted(written), sorted(rs.AUTO_FILES.values()))
                with self.assertRaises(FileExistsError):  # a second writing call is blocked, nothing changes
                    self.script.aggregate()
                self.assertEqual({p.name: p.read_bytes() for p in tmp.glob("auto_roi_*.csv") if p.name != rs.DIAGNOSTICS_FILE}, written)
                self.assertEqual(len(written), 15)
            for name in written:
                cells, largest = table_difference((tmp / name).read_text(), (STORED / name).read_text())
                self.assertLessEqual(largest, FLOAT_TOLERANCE, name)
                self.assertEqual(cells, 3 if name == "auto_roi_notable_examples.csv" else 0, name)  # three dice values of the segmentation selection are parsed exactly now (last digit)

    def test_a_later_existing_output_blocks_before_any_input_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / rs.AUTO_FILES["concurrence"]).write_text("x")
            with mock.patch.object(config, "TABLES_DIR", tmp), mock.patch.object(self.script.data, "read_manifest", side_effect=AssertionError("an input was read")):
                with self.assertRaises(FileExistsError):
                    self.script.aggregate()

    def test_group_design_and_plot_contract(self):
        sets = rs.auto_score_sets()
        manifest = data.read_manifest()
        validation = data.subset_frame(manifest, "validation")
        tables = rs.auto_tables(sets, rs.read_diagnostics(rs.diagnostics_path(), validation), pd.read_csv(rs.segmentation_selection_path(), dtype={"ID": str, "Case": str}))
        runs = tables["run_metrics"]
        self.assertEqual(len(runs), 30)
        self.assertEqual(runs.groupby("group", sort=False).size().to_dict(), {g: 5 for g in ("4B_frozen", "4E_frozen", "4FB_frozen", "4B_finetune", "4E_finetune", "4FB_finetune")})
        # the automatic ROI uses the SAME classifier as the expert ROI; the full image uses the 4B variant
        self.assertEqual(runs[runs["group"] == "4FB_finetune"]["model"].iloc[0], runs[runs["group"] == "4E_finetune"]["model"].iloc[0])
        self.assertNotEqual(runs[runs["group"] == "4B_finetune"]["model"].iloc[0], runs[runs["group"] == "4E_finetune"]["model"].iloc[0])
        plot = ROOT / "plots" / "plot_auto_roi_results.py"
        stages = read_literal(plot, "STAGES")
        self.assertEqual({g for _, _, groups in stages for g in groups}, set(runs["group"]))
        paired = tables["paired_differences"]
        for stage, _, groups in stages:
            for pair in ("auto_minus_expert", "auto_minus_full"):
                self.assertEqual(len(paired[(paired["pair"] == f"{pair} ({stage})") & (paired["metric"] == "roc_auc")]), 5)
        points = tables["operating_points"]
        for group in ("4B_frozen", "4E_finetune", "4FB_finetune"):
            for target in ("0.90", "0.95"):
                self.assertEqual(sorted(points[(points["group"] == group) & (points["point_label"] == f"target_{target}")]["seed"]), sorted(config.SEEDS))
        diffs = tables["operating_point_differences"]
        selected = diffs[(diffs["pair"] == "auto_minus_expert") & (diffs["stage"] == "frozen") & (diffs["point"] == "target_0.95") & (diffs["level"] == "image") & (diffs["metric"] == "fp")]
        self.assertEqual(sorted(selected["seed"]), sorted(config.SEEDS))
        self.assertIn("difference_auto_minus_reference", diffs.columns)
        self.assertEqual(set(tables["crop_categories"]["category"]), set(ar.CATEGORIES))
        self.assertEqual(len(tables["box_definitions"]), 4)
        self.assertEqual(tables["notable_examples"]["ID"].tolist()[:3], ["bus_1014-r", "bus_1013-l", "bus_1014-l"])  # the named examples first, then the fixed error-analysis selection
        self.assertEqual(len(tables["notable_examples"]), 12)

    def test_inconsistent_inputs_are_reported_without_any_reconstruction(self):
        manifest = data.read_manifest()
        validation = data.subset_frame(manifest, "validation")
        sets = rs.auto_score_sets()
        rs.check_score_sets(sets, validation)
        key = ("4FB_finetune", 21)
        broken = {
            "missing seed": {k: v for k, v in sets.items() if k != key},
            "duplicated ID": {**sets, key: pd.concat([sets[key].iloc[:-1], sets[key].iloc[[0]]], ignore_index=True)},
            "shuffled order": {**sets, key: sets[key].iloc[::-1].reset_index(drop=True)},
            "wrong label": {**sets, key: sets[key].assign(label=1 - sets[key]["label"])},
            "wrong subset (training rows)": {**sets, key: read_training_rows()},
            "NaN score": {**sets, key: sets[key].assign(malignant_score=[np.nan] + sets[key]["malignant_score"].tolist()[1:])},
            "score outside [0, 1]": {**sets, key: sets[key].assign(malignant_score=sets[key]["malignant_score"] + 1.0)},
        }
        for name, bad in broken.items():
            with self.assertRaises(ValueError, msg=name) as caught:
                rs.check_score_sets(bad, validation)
            self.assertIn("inconsistent", str(caught.exception), name)

    def test_missing_primary_inputs_stop_with_a_clear_error_and_do_not_claim_equality(self):
        missing = {"automatic scores of one seed": lambda: rs.auto_score_path(21, "frozen"), "per-image diagnoses": rs.diagnostics_path, "example selection of the segmentation study": rs.segmentation_selection_path}
        for name, path in missing.items():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                with self.prepared(tmp):
                    path().unlink()  # exactly one primary input is absent
                    with self.assertRaises(FileNotFoundError, msg=name):
                        self.script.derive_tables()
                    shutil.copyfile(STORED / "auto_roi_geometry_validation.csv", tmp / "auto_roi_geometry_validation.csv")
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        self.script.show()
                    self.assertIn("were not derived", out.getvalue(), name)
                    self.assertNotIn("identical to the stored table", out.getvalue(), name)

    def test_without_stored_reference_tables_the_default_view_claims_no_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.prepared(tmp):
                shutil.copyfile(STORED / "auto_roi_geometry_validation.csv", tmp / "auto_roi_geometry_validation.csv")
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.script.show()
                self.assertEqual(out.getvalue().count("no equality claimed"), 15)
                self.assertNotIn("identical to the stored table", out.getvalue())

    def test_unknown_and_incompatible_arguments_are_errors(self):
        for argv in (["--run", "--aggregate"], ["--unknown"], ["aggregate"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                self.script.main(argv)
            self.assertEqual(caught.exception.code, 2, argv)


def read_training_rows() -> pd.DataFrame:
    """Scores of a run that include the training rows (the wrong subset for a validation comparison)."""
    frame = pd.read_csv(config.run_paths(config.VARIANTS["mobilenetv2_higher_lr_160_roi"], 21)["predictions"], dtype={"ID": str, "Case": str})
    return frame[frame["subset"] == "train"][["ID", "Case", "label", "malignant_score"]].reset_index(drop=True)


# ---- the expert geometry --------------------------------------------------------------------------------------------------------------------------------------------
class ExpertGeometryTests(unittest.TestCase):
    """Measured once from synthetic pairs, then read by the training input, the add-on and the launcher check; the reader needs no mask."""

    BOXES = {  # name -> (size (w, h), lesion rectangles)
        "a": ((80, 60), [(10, 12, 30, 32)]),                      # inside, non-square
        "b": ((80, 60), [(0, 0, 7, 5)]),                          # border contact in the corner, enlargement
        "c": ((100, 50), [(5, 5, 20, 20), (60, 10, 90, 40)]),     # two components far apart: one wide box
        "d": ((40, 90), [(2, 3, 38, 87)]),                        # almost the whole image: crop area close to one, padding left and right
        "e": ((64, 64), [(30, 30, 34, 34)]),                      # tiny lesion: strong enlargement
        "f": ((120, 40), [(50, 5, 70, 35)]),                      # wide image, tall crop: padding on the sides
    }

    def synthetic_dataset(self, root: Path):
        (root / "img").mkdir()
        (root / "msk").mkdir()
        rows = []
        for index, (name, ((w, h), rectangles)) in enumerate(self.BOXES.items()):
            image = Image.fromarray(((np.arange(w * h, dtype=np.uint32).reshape(h, w) * (index + 3)) % 251).astype(np.uint8), mode="L")
            mask = np.zeros((h, w), dtype=bool)
            for x0, y0, x1, y1 in rectangles:
                mask[y0:y1, x0:x1] = True
            image.save(root / "img" / f"{name}.png")
            Image.fromarray(mask).save(root / "msk" / f"{name}.png")
            rows.append({"ID": f"bus_{name}", "Case": str(index // 2 + 1), "Pathology": "malignant" if index // 2 % 2 else "benign", "label": str(index // 2 % 2), "Device": "synthetic", "split": ("train", "validation", "test")[index % 3] if False else ("train" if index < 4 else "validation"),
                         "image_relpath": f"img/{name}.png", "mask_relpath": f"msk/{name}.png", "image_width": str(w), "image_height": str(h), "mask_width": str(w), "mask_height": str(h), "csv_size_mismatch": "False",
                         "image_mask_size_match": "True"})
        rows.append({**rows[0], "ID": "bus_test", "Case": "99", "split": "test"})  # a test row that must never be measured (its files do not exist)
        rows[-1].update(image_relpath="img/missing.png", mask_relpath="msk/missing.png")
        return pd.DataFrame(rows)

    def test_geometry_is_measured_once_and_the_geometry_based_crop_equals_the_mask_based_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = self.synthetic_dataset(tmp)
            geometry = data.measure_expert_geometry(tmp, manifest)
            self.assertEqual(geometry["subset"].tolist(), ["train"] * 4 + ["validation"] * 2)  # the test row is never opened
            data.validate_expert_geometry(geometry, manifest)
            for subset in ("train", "validation"):
                frame = data.subset_frame(manifest, subset)
                old_arrays, old_geometry = data.load_roi_arrays(tmp, frame.to_dict("records"), subset)  # measures the masks
                for mask in (tmp / "msk").glob("*.png"):
                    mask.rename(mask.with_suffix(".hidden"))  # the reader must not need a mask
                images, new_arrays, boxes = data.load_roi_inputs(tmp, frame, geometry[geometry["subset"] == subset])
                for hidden in (tmp / "msk").glob("*.hidden"):
                    hidden.rename(hidden.with_suffix(".png"))
                self.assertTrue(np.array_equal(old_arrays, new_arrays), subset)  # bit-identical crops including border contact, padding, enlargement and several components
                self.assertEqual(boxes, [tuple(int(v) for v in r) for r in old_geometry[["exp_x0", "exp_y0", "exp_x1", "exp_y1"]].to_numpy()])
            by = geometry.set_index("ID")
            self.assertEqual(int(by.loc["bus_c", "n_components"]), 2)
            self.assertEqual((int(by.loc["bus_c", "x0"]), int(by.loc["bus_c", "x1"])), (5, 90))
            self.assertTrue(bool(by.loc["bus_b", "touches_left"]) and bool(by.loc["bus_b", "expansion_clipped"]) and bool(by.loc["bus_e", "upscaled"]))
            self.assertEqual((int(by.loc["bus_a", "exp_x0"]), int(by.loc["bus_a", "exp_x1"])), (6, 34))  # 20 % of 20 pixels = 4 per side, rounded up
            self.assertGreater(float(by.loc["bus_d", "crop_area_fraction"]), 0.9)

    def test_prepare_writes_once_and_the_consumers_read_it(self):
        script = load_script("03_classification")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = self.synthetic_dataset(tmp)
            tables = tmp / "tables"
            tables.mkdir()
            with mock.patch.object(config, "TABLES_DIR", tables), mock.patch.object(config, "PROJECT_ROOT", tmp), mock.patch.object(data, "read_manifest", return_value=manifest), \
                    mock.patch.object(data, "check_split"):  # the synthetic split has no real counts
                script.prepare()
                self.assertEqual(sorted(p.name for p in tables.iterdir()), ["expert_roi_geometry_frequencies.csv", "expert_roi_geometry_train_validation.csv"])
                before = {p.name: p.read_bytes() for p in tables.iterdir()}
                with mock.patch.object(data, "measure_expert_geometry", side_effect=AssertionError("masks were measured again")), self.assertRaises(FileExistsError):
                    script.prepare()  # blocked before any measurement
                self.assertEqual({p.name: p.read_bytes() for p in tables.iterdir()}, before)
                for mask in (tmp / "msk").glob("*.png"):
                    mask.unlink()  # from here on no mask exists: every consumer works from the stored geometry
                # (1) the training input adapter for the ROI variants
                inputs = data.load_classification_inputs("roi", 160, project_root=tmp)
                self.assertEqual({k: v[1].shape for k, v in inputs.items()}, {"train": (4, 160, 160, 1), "validation": (2, 160, 160, 1)})
                # (2) the add-on: same crops, only the validity masks are its own
                addon = load_script("08_adversarial_addon")
                geometry = data.read_expert_geometry(manifest)
                with mock.patch.object(config, "PROJECT_ROOT", tmp), mock.patch.object(addon.config, "PROJECT_ROOT", tmp):
                    built = addon.build_inputs(data.subset_frame(manifest, "validation"), geometry)
                self.assertTrue(np.array_equal(built["x"], inputs["validation"][1]))
                self.assertEqual(built["valid"].shape, (2, 160, 160))
                # (3) the launcher check, three planned seeds: the shared geometry is an input, not a per-seed output
                plan = launcher.plan_runs("expert_roi", [7, 21, 42], ["mobilenetv2_frozen_160_roi"])
                launcher.check_inputs(plan)
                self.assertNotIn(data.expert_geometry_path(), config.run_outputs(config.VARIANTS["mobilenetv2_frozen_160_roi"], 7))
                for seed in (7, 21, 42):
                    self.assertNotIn(data.expert_geometry_path(), config.run_outputs(config.VARIANTS["mobilenetv2_frozen_160_roi"], seed))

    def test_a_missing_geometry_stops_everything_with_the_preparation_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            started = []
            with mock.patch.object(config, "TABLES_DIR", tmp), mock.patch.object(data, "read_manifest", return_value=self.synthetic_dataset(tmp)), mock.patch.object(data, "check_split"):
                with self.assertRaises(FileNotFoundError) as caught:
                    launcher.run_plan("expert_roi", Path("x.py"), [7], None, starter=lambda *a, **k: started.append(a), outputs=lambda v, s: [tmp / "absent"])
                self.assertIn("--preset expert_roi --prepare", str(caught.exception))
                self.assertEqual(started, [])
                with self.assertRaises(FileNotFoundError):
                    data.load_classification_inputs("roi", 160, project_root=tmp)
                launcher.check_inputs(launcher.plan_runs("multiseed", [7], ["small_cnn_160"]))  # full-image variants need no geometry
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                load_script("03_classification").main(["--preset", "multiseed", "--prepare"])

    def test_wrong_geometry_is_rejected_without_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = self.synthetic_dataset(tmp)
            good = data.measure_expert_geometry(tmp, manifest)
            data.validate_expert_geometry(good, manifest)
            swapped = good.copy()
            swapped.loc[[0, 1], "ID"] = swapped.loc[[1, 0], "ID"].to_numpy()
            cases = {
                "missing row": good.iloc[:-1],
                "duplicated ID": pd.concat([good.iloc[:-1], good.iloc[[0]]], ignore_index=True),
                "swapped IDs": swapped,
                "wrong case": good.assign(Case=["999"] + good["Case"].tolist()[1:]),
                "wrong label": good.assign(label=1 - good["label"]),
                "wrong subset": good.assign(subset=["validation"] + good["subset"].tolist()[1:]),
                "box outside the image": good.assign(exp_x1=[10_000] + good["exp_x1"].tolist()[1:]),
                "empty box": good.assign(exp_x1=good["exp_x0"]),
                "crop size differs from the box": good.assign(crop_w=good["crop_w"] + 1),
                "wrong resize scale": good.assign(scale_to_160=good["scale_to_160"] * 1.01),
                "decoded size differs from the manifest": good.assign(image_width=good["image_width"] + 1),
                "column missing": good.drop(columns=["exp_y0"]),
            }
            for name, bad in cases.items():
                with self.assertRaises(data.RoiDataError, msg=name):
                    data.validate_expert_geometry(bad, manifest)
            with self.assertRaises(Exception):  # a decoded image size that differs from the stored geometry is found when the image is opened
                data.load_roi_inputs(tmp, data.subset_frame(manifest, "train"), good[good["subset"] == "train"].replace({"image_width": {80: 81}}))
            with self.assertRaises(data.RoiDataError):
                data.load_roi_inputs(tmp, data.subset_frame(manifest, "train"), good[good["subset"] == "validation"])

    def test_the_stored_project_geometry_is_valid_and_reproduces_the_stored_frequencies(self):
        manifest = data.read_manifest()
        geometry = data.read_expert_geometry(manifest)
        self.assertEqual(geometry["subset"].value_counts().to_dict(), {"train": 1114, "validation": 385})
        self.assertTrue(results.same_as_stored(rs.geometry_frequencies(geometry), STORED / "expert_roi_geometry_frequencies.csv"))
        stored = pd.read_csv(STORED / "expert_roi_geometry_train_validation.csv", dtype={"ID": str, "Case": str})
        self.assertEqual(stored["ID"].tolist(), geometry["ID"].tolist())
        self.assertFalse(set(roi_functions_used_by_measure()) - {"read_mask_raw", "decode_mask", "tight_box", "expanded_box", "geometry_problems", "components", "border_contact", "margins", "resize_geometry", "positive_pixels"})


def roi_functions_used_by_measure():
    """The roi functions called by data.measure_pair (one measurement and crop rule for the loader, the preparation and the mask EDA)."""
    tree = ast.parse(Path(data.__file__).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "measure_pair")
    return sorted({c.func.attr for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name) and c.func.value.id == "roi"})


# ---- corrections before the submission run: partial seeds, exact labels, the 20 % expansion ---------------------------------------------------------------------------
class SubmissionCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script("03_classification")
        cls.validation = data.subset_frame(data.read_manifest(), "validation")

    def test_a_partial_seed_view_uses_its_own_seeds_and_says_so(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.script.main(["--preset", "expert_roi", "--seeds", "7", "21"])
        self.assertIn("PARTIAL VIEW of the seeds [7, 21]", out.getvalue())
        self.assertNotIn("identical", out.getvalue())
        tables = rs.expert_operating_tables(rs.expert_score_sets([7, 21]))
        for key in ("threshold_curves", "operating_points", "operating_point_differences", "operating_point_summary"):
            frame = tables[key]
            self.assertEqual(sorted(frame["seed"].unique()) if "seed" in frame else [7, 21], [7, 21], key)
        self.assertEqual(set(tables["operating_point_summary"]["seeds"]), {"7;21"})
        self.assertEqual(set(tables["operating_point_summary"]["n_seeds"]), {2})

    def test_an_incomplete_aggregation_is_refused_before_anything_is_read_or_written(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "TABLES_DIR", Path(tmp)), mock.patch.object(rs, "expert_score_sets", side_effect=AssertionError("an input was read")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                self.script.main(["--preset", "expert_roi", "--aggregate", "--seeds", "7", "21"])
            self.assertEqual(caught.exception.code, 2)
            with self.assertRaises(ValueError):
                self.script.write_tables("expert_roi", [7, 21])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_options_that_do_not_belong_to_an_action_are_errors(self):
        for argv in (["--variants", "small_cnn_160"], ["--aggregate", "--variants", "mobilenetv2_frozen_160_roi"], ["--prepare", "--seeds", "7"], ["--train-one", "mobilenetv2_frozen_160_roi", "--variants", "x"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                self.script.main(["--preset", "expert_roi", *argv])
            self.assertEqual(caught.exception.code, 2, argv)

    def test_class_labels_are_checked_before_any_conversion(self):
        self.assertEqual(data.exact_labels(["0", "1", 1, 0.0], "x").tolist(), [0, 1, 1, 0])
        for bad in ([0.25], [float("nan")], ["a"], [2], [-1], [None]):
            with self.assertRaises(ValueError, msg=str(bad)):
                data.exact_labels(bad, "x")
        sets = rs.expert_score_sets()
        key = ("4B_finetune", 21)
        for bad_label in (0.25, "x", float("nan")):
            frame = sets[key].copy()
            frame["label"] = frame["label"].astype(object)
            frame.loc[frame.index[0], "label"] = bad_label  # the manifest requires 0 here; a truncating cast would have accepted 0.25
            with self.assertRaises(ValueError, msg=str(bad_label)):
                rs.check_score_sets({**sets, key: frame}, self.validation)

    def test_diagnosis_labels_classes_and_columns_are_checked_against_the_split(self):
        source = rs.diagnostics_path()
        rs.read_diagnostics(source, self.validation)  # the stored diagnoses are valid
        frame = pd.read_csv(source, dtype={"ID": str, "Case": str}, float_precision="round_trip")
        index = int(frame.index[frame["ID"] == "bus_0008-l"][0])
        flipped = frame.copy()
        flipped.loc[index, "label"] = 1 - flipped.loc[index, "label"]  # the case used in the review: a reversed label
        cases = {"reversed label": flipped, "fractional label": frame.assign(label=[0.25] + frame["label"].tolist()[1:]), "wrong class name": frame.assign(**{"class": ["malignant"] + frame["class"].tolist()[1:]}),
                 "missing column": frame.drop(columns=["component_dice"]), "missing label column": frame.drop(columns=["label"])}
        with tempfile.TemporaryDirectory() as tmp:
            for name, bad in cases.items():
                path = Path(tmp) / "diagnostics.csv"
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError, msg=name):
                    rs.read_diagnostics(path, self.validation)

    def test_geometry_labels_are_exact_and_the_expansion_is_validated(self):
        manifest = data.read_manifest()
        good = pd.read_csv(data.expert_geometry_path(), dtype={"ID": str, "Case": str})
        data.validate_expert_geometry(good, manifest)
        with self.assertRaises(data.RoiDataError):
            data.validate_expert_geometry(good.assign(label=[0.25] + good["label"].tolist()[1:]), manifest)
        # the counter-example of the review: the expanded box of bus_0001-l replaced by the tight box, the crop sizes and the resize scale adjusted consistently
        index = int(good.index[good["ID"] == "bus_0001-l"][0])
        bad = good.copy()
        x0, y0, x1, y1 = (int(bad.loc[index, c]) for c in ("x0", "y0", "x1", "y1"))
        bad.loc[index, ["exp_x0", "exp_y0", "exp_x1", "exp_y1", "crop_w", "crop_h"]] = [x0, y0, x1, y1, x1 - x0, y1 - y0]
        bad.loc[index, "scale_to_160"] = roi_scale(x1 - x0, y1 - y0)
        with self.assertRaises(data.RoiDataError) as caught:
            data.validate_expert_geometry(bad, manifest)
        self.assertIn("20 % expansion", str(caught.exception))
        shrunk = good.copy()  # one pixel too small is also refused
        shrunk.loc[index, ["exp_x1", "crop_w"]] = [int(good.loc[index, "exp_x1"]) - 1, int(good.loc[index, "crop_w"]) - 1]
        shrunk.loc[index, "scale_to_160"] = roi_scale(int(shrunk.loc[index, "crop_w"]), int(shrunk.loc[index, "crop_h"]))
        with self.assertRaises(data.RoiDataError):
            data.validate_expert_geometry(shrunk, manifest)
        clipped = good[good["expansion_clipped"].astype(str) == "True"]
        self.assertGreater(len(clipped), 0)  # the clipped border cases of the stored geometry stay valid (validated above together with all other rows)


def roi_scale(width: int, height: int) -> float:
    from busbra import roi
    return roi.resize_geometry(width, height)["scale"]


if __name__ == "__main__":
    unittest.main()
