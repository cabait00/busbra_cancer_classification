"""Regression tests at the scientific boundaries of the project (CPU, no TensorFlow, no images).

Run from the project folder:  .venv/bin/python -m unittest discover -s tests
Synthetic arrays check the rules; the stored manifest, the stored bootstrap draws and the stored result tables check that the shared code reproduces what was reported.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from busbra import config, data, evaluation as ev, final_evaluation as fe, roi, test_evaluation as te, thresholds as th  # noqa: E402


class SplitTests(unittest.TestCase):
    def test_stored_manifest_is_valid_and_case_separated(self):
        manifest = data.read_manifest()
        data.check_split(manifest)  # raises on leakage, wrong counts, non-unique IDs, label mismatch
        self.assertEqual(manifest["ID"].iloc[0], "bus_0001-l")  # IDs stay strings with their leading zeros

    def test_case_in_two_subsets_is_rejected(self):
        manifest = data.read_manifest().copy()
        case = manifest.loc[manifest["split"] == "train", "Case"].iloc[0]
        row = manifest[manifest["split"] == "test"].index[0]
        manifest.loc[row, "Case"] = case
        with self.assertRaises(ValueError):
            data.check_split(manifest)

    def test_outer_fold_rule(self):
        self.assertEqual(config.SPLIT_OF_OUTER_FOLD, {1: "test", 2: "validation", 3: "train", 4: "train", 5: "train"})


class ThresholdTests(unittest.TestCase):
    def test_rate_with_zero_denominator_is_undefined(self):
        self.assertIsNone(ev.rate(3, 0))
        self.assertEqual(ev.rate(1, 2), 0.5)
        self.assertIs(th.rate, ev.rate)  # one implementation

    def test_decision_is_inclusive_and_infinity_means_nothing_positive(self):
        counts = th.confusion_counts([0, 1, 1], [0.2, 0.5, 0.9], 0.5)
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"], counts["tn"]), (2, 0, 0, 1))
        self.assertEqual(th.confusion_counts([0, 1], [0.2, 0.9], float("inf"))["tp"], 0)

    def test_target_threshold_is_the_kth_largest_positive_score_with_ties(self):
        labels = np.array([1, 1, 1, 1, 0, 0])
        scores = np.array([0.9, 0.7, 0.7, 0.4, 0.7, 0.1])
        result = th.target_threshold(labels, scores, "0.75")  # ceil(0.75 * 4) = 3 positives needed
        self.assertEqual(result["needed_tp"], 3)
        self.assertEqual(result["threshold"], 0.7)  # the tied benign image at 0.7 is classified together with the positives
        self.assertEqual((result["tp"], result["fp"]), (3, 1))
        self.assertEqual(fe.needed_exact("0.95", 125), 119)
        self.assertEqual(th.needed_true_positives("0.90", 125), 113)

    def test_case_score_is_the_maximum_of_the_image_scores(self):
        cases = th.aggregate_cases(["a", "a", "b"], [1, 1, 0], [0.2, 0.8, 0.4])
        self.assertEqual(cases.set_index("Case").loc["a", "case_score"], 0.8)
        with self.assertRaises(ValueError):
            th.aggregate_cases(["a", "a"], [1, 0], [0.2, 0.8])  # two labels in one case

    def test_frozen_thresholds_round_trip_exactly(self):
        frozen, problems = te.load_frozen_thresholds(config.TABLES_DIR / "final_operating_points.csv", [e["combo_id"] for e in fe.combinations()])
        self.assertEqual(problems, [])
        self.assertEqual(len(frozen), 27 * 2 * 3)


class RoiTests(unittest.TestCase):
    def test_expansion_rounds_up_per_side_and_clips_at_the_border(self):
        self.assertEqual(roi.expanded_box((10, 10, 17, 20), 100, 100), (8, 8, 19, 22))  # w = 7 -> 2 px, h = 10 -> 2 px
        self.assertEqual(roi.expanded_box((10, 10, 15, 20), 100, 100), (9, 8, 16, 22))  # w = 5 -> exactly 1 px
        self.assertEqual(roi.expanded_box((0, 3, 30, 40), 32, 41), (0, 0, 32, 41))  # clipped, never mirrored or padded

    def test_tight_box_covers_all_components_half_open(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[2:4, 3:5] = True
        mask[10:12, 20:25] = True
        self.assertEqual(roi.tight_box(mask), (3, 2, 25, 12))


class BootstrapTests(unittest.TestCase):
    def test_draws_are_deterministic_and_stratified(self):
        labels = np.array([1] * 5 + [0] * 9)
        first, second = te.bootstrap_draws(labels, replicates=20), te.bootstrap_draws(labels, replicates=20)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue((labels[first[:, :5]] == 1).all() and (labels[first[:, 5:]] == 0).all())  # malignant cases first, then benign

    def test_draws_equal_the_stored_reference_when_it_exists(self):
        reference = config.TABLES_DIR / "test_evaluation" / "bootstrap_arrays.npz"  # optional historical reference, not an input of the evaluation
        if not reference.is_file():
            self.skipTest("no stored reference draws")
        manifest = data.read_manifest()
        test = data.subset_frame(manifest, "test")
        structure = te.case_table(pd.DataFrame({"ID": test["ID"], "Case": test["Case"], "label": test["label"].astype(int)}))
        self.assertTrue(np.array_equal(te.bootstrap_draws(structure["case_labels"])[:50], np.load(reference)["draws"][:50].astype(np.int64)))


class StoredResultTests(unittest.TestCase):
    def test_auc_and_ap_of_all_27_combinations_equal_the_stored_metrics(self):
        scores = pd.read_csv(config.TABLES_DIR / "test_evaluation_scores.csv", dtype={"ID": str, "Case": str}, float_precision="round_trip")
        stored = pd.read_csv(config.TABLES_DIR / "test_evaluation_metrics.csv").set_index(["combo_id", "level"])
        self.assertEqual(scores["combo_id"].nunique(), 27)
        for combo, frame in scores.groupby("combo_id", sort=False):
            labels, values = fe.units("image", frame)
            m = ev.binary_metrics(labels, values)
            self.assertAlmostEqual(m["roc_auc"], stored.loc[(combo, "image"), "roc_auc"], places=12)
            self.assertAlmostEqual(m["average_precision"], stored.loc[(combo, "image"), "average_precision"], places=12)
            labels, values = fe.units("case", frame)
            self.assertAlmostEqual(ev.binary_metrics(labels, values)["roc_auc"], stored.loc[(combo, "case"), "roc_auc"], places=12)


if __name__ == "__main__":
    unittest.main()
