"""Synthetic tests for the confidence analysis and validation export (issue #14)."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))

from wildquest.evaluation import evaluate_predictions  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, CODE / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_script("analyze_confidence")
exporter = load_script("export_validation_predictions")
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def row(true_id, pred_id, confidence):
    return {"true": true_id, "pred": pred_id, "confidence": confidence, "margin": None,
            "correct": true_id == pred_id, "sample_id": "", "local_path": ""}


# 10 predictions: 7 correct, 3 wrong. Wrong ones at 0.95, 0.60 and 0.40.
ROWS = [row(0, 0, 0.99), row(1, 1, 0.97), row(2, 3, 0.95), row(4, 4, 0.90),
        row(5, 5, 0.85), row(6, 6, 0.75), row(7, 7, 0.65), row(8, 9, 0.60),
        row(10, 10, 0.55), row(11, 12, 0.40)]


def probs_for(pred_id, confidence, runner_up=None):
    """A 25-class probability vector with `confidence` on pred_id."""
    probs = [0.0] * 25
    probs[pred_id] = confidence
    other = runner_up if runner_up is not None else (pred_id + 1) % 25
    remaining = 1 - confidence
    # The runner-up stays below the prediction; the rest is spread over the other 23 classes.
    probs[other] = min(remaining, 0.9 * confidence)
    rest = [i for i in range(25) if i not in (pred_id, other)]
    for i in rest:
        probs[i] = (remaining - probs[other]) / len(rest)
    return probs


class TestThresholdTable(unittest.TestCase):
    def table(self, thresholds):
        return {r["threshold"]: r for r in analysis.threshold_table(ROWS, thresholds)}

    def test_coverage_accuracy_and_errors_caught(self):
        t = self.table([0.70])[0.70]
        # Accepted: 0.99, 0.97, 0.95(wrong), 0.90, 0.85, 0.75 -> 6 of 10, 5 correct.
        self.assertEqual(t["accepted"], 6)
        self.assertAlmostEqual(t["coverage"], 0.6)
        self.assertAlmostEqual(t["accepted_accuracy"], 5 / 6)
        self.assertEqual(t["accepted_errors"], 1)
        self.assertEqual(t["needs_review"], 4)
        self.assertAlmostEqual(t["review_rate"], 0.4)
        self.assertAlmostEqual(t["errors_caught"], 2 / 3)          # 0.60 and 0.40 flagged
        self.assertAlmostEqual(t["correct_sent_to_review"], 2 / 7)  # 0.65 and 0.55 flagged

    def test_threshold_is_inclusive(self):
        self.assertEqual(self.table([0.95])[0.95]["accepted"], 3)

    def test_extremes(self):
        t = self.table([0.0, 1.0])
        self.assertEqual(t[0.0]["coverage"], 1.0)
        self.assertAlmostEqual(t[0.0]["accepted_accuracy"], 0.7)
        self.assertEqual(t[0.0]["errors_caught"], 0.0)
        self.assertEqual(t[1.0]["accepted"], 0)
        self.assertIsNone(t[1.0]["accepted_accuracy"])
        self.assertEqual(t[1.0]["errors_caught"], 1.0)

    def test_coverage_never_increases_with_threshold(self):
        table = analysis.threshold_table(ROWS, [0.9, 0.3, 0.6, 0.75])
        self.assertEqual([r["threshold"] for r in table], [0.3, 0.6, 0.75, 0.9])
        coverages = [r["coverage"] for r in table]
        self.assertEqual(coverages, sorted(coverages, reverse=True))

    def test_no_errors_gives_undefined_catch_rate(self):
        table = analysis.threshold_table([row(0, 0, 0.9), row(1, 1, 0.5)], [0.7])
        self.assertIsNone(table[0]["errors_caught"])


class TestRecommendation(unittest.TestCase):
    def test_picks_lowest_threshold_meeting_target(self):
        table = analysis.threshold_table(ROWS, [0.5, 0.7, 0.96, 0.98])
        # 0.5: 7/9 correct; 0.7: 5/6; 0.96: 2/2 -> first to reach 0.95 is 0.96.
        self.assertEqual(analysis.recommend_threshold(table, 0.95)["threshold"], 0.96)
        self.assertEqual(analysis.recommend_threshold(table, 0.80)["threshold"], 0.7)

    def test_none_when_target_unreachable(self):
        table = analysis.threshold_table([row(0, 1, 0.99)], [0.5, 0.9])
        self.assertIsNone(analysis.recommend_threshold(table, 0.95))


class TestReliabilityAndErrors(unittest.TestCase):
    def test_bins_and_ece(self):
        rows = [row(0, 0, 0.95), row(1, 2, 0.95), row(3, 3, 0.15), row(4, 4, 1.0)]
        bins, ece = analysis.reliability(rows, n_bins=10)
        self.assertEqual(sum(b["count"] for b in bins), 4)
        last = bins[9]                       # 0.95, 0.95 and 1.0 (1.0 goes in the top bin)
        self.assertEqual(last["count"], 3)
        self.assertAlmostEqual(last["accuracy"], 2 / 3)
        self.assertAlmostEqual(last["mean_confidence"], (0.95 + 0.95 + 1.0) / 3)
        self.assertEqual(bins[1]["count"], 1)
        expected = 3 / 4 * abs(2 / 3 - 2.9 / 3) + 1 / 4 * abs(1 - 0.15)
        self.assertAlmostEqual(ece, expected)

    def test_high_confidence_errors_sorted_and_paired(self):
        errors = analysis.high_confidence_errors(ROWS, 0.50)
        self.assertEqual([r["confidence"] for r in errors], [0.95, 0.60])
        species = [f"s{i}" for i in range(25)]
        pairs = analysis.error_pairs(errors + errors[:1], species)
        self.assertEqual(pairs[0], {"true_species": "s2", "predicted_species": "s3", "count": 2})

    def test_per_class_review_rate(self):
        species = [f"s{i}" for i in range(25)]
        classes = analysis.per_class(ROWS + [row(0, 0, 0.30)], 0.70, species)
        self.assertEqual(classes[0]["support"], 2)
        self.assertAlmostEqual(classes[0]["review_rate"], 0.5)
        self.assertEqual(classes[2]["confident_errors"], 1)
        self.assertIsNone(classes[24]["accuracy"])


class TestExportFormat(unittest.TestCase):
    def test_row_matches_evaluation_contract(self):
        probs = probs_for(17, 0.9412345678)
        r = exporter.prediction_row("data/raw/images/val/17/123.jpeg", 17, probs)
        self.assertEqual((r["sample_id"], r["pred_class_id"], r["confidence"]),
                         ("123", 17, "0.94123457"))
        self.assertEqual(list(r), exporter.FIELDS)

    def test_exported_csv_passes_framework_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val_predictions.csv"
            rows = [exporter.prediction_row(f"data/raw/images/val/{c:02d}/{c}.jpeg", c,
                                            probs_for(p, 1 / 3 + 0.1 * (c % 6), runner_up=c if c != p else None))
                    for c, p in [(0, 0), (1, 2), (2, 2), (3, 3), (4, 5)]]
            exporter.write_predictions_csv(path, rows)
            result = evaluate_predictions(path)  # raises if the CSV breaks the contract
            self.assertEqual(result.metrics["total"], 5)
            self.assertAlmostEqual(result.metrics["top1_accuracy"], 3 / 5)

    def test_export_reads_only_validation(self):
        self.assertEqual(exporter.SPLIT, "val")

    def test_wrong_vector_length_rejected(self):
        with self.assertRaises(ValueError):
            exporter.prediction_row("x.jpeg", 0, [1.0])


class TestEndToEnd(unittest.TestCase):
    def test_analyze_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = tmp / "val_predictions.csv"
            exporter.write_predictions_csv(csv_path, [
                exporter.prediction_row(f"data/raw/images/val/{r['true']:02d}/{i}.jpeg",
                                        r["true"], probs_for(r["pred"], r["confidence"],
                                                             runner_up=r["true"] if not r["correct"] else None))
                for i, r in enumerate(ROWS)])
            out = tmp / "reports"
            summary = analysis.analyze(csv_path, out, target_accuracy=0.80,
                                       thresholds=(0.5, 0.7, 0.9), plots=HAS_MATPLOTLIB)
            self.assertEqual(summary["images"], 10)
            self.assertAlmostEqual(summary["top1_accuracy"], 0.7)
            self.assertEqual(summary["recommended_threshold"]["threshold"], 0.7)
            self.assertEqual(summary["high_confidence_errors"], 1)  # the 0.95 mistake
            for name in ("summary.json", "thresholds.csv", "reliability.csv", "per_class.csv",
                         "high_confidence_errors.csv"):
                self.assertTrue((out / name).exists(), name)
            if HAS_MATPLOTLIB:
                for name in ("reliability_diagram.png", "threshold_tradeoff.png",
                             "confidence_histogram.png"):
                    self.assertTrue((out / name).exists(), name)
            json.loads((out / "summary.json").read_text(encoding="utf-8"))
            with (out / "high_confidence_errors.csv").open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_rejects_output_that_would_overwrite_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.csv"
            exporter.write_predictions_csv(path, [exporter.prediction_row("a.jpeg", 0,
                                                                          probs_for(0, 0.9))])
            with self.assertRaises(ValueError):
                analysis.analyze(path, Path(tmp), plots=False)


if __name__ == "__main__":
    unittest.main()