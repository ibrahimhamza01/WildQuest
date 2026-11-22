"""Offline evaluator tests using synthetic prediction CSVs."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

from PIL import Image

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from wildquest.evaluation import evaluate_predictions, write_reports

REQUIRED = ("true_class_id", "pred_class_id")
PROBABILITIES = tuple(f"prob_{i}" for i in range(25))
KNOWN_ROWS = [(0, 0), (0, 0), (0, 1), (1, 1), (2, 3), (2, 3), (4, 3)]


def probability_vector(*scores):
    return scores + (0,) * (25 - len(scores))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.predictions = self.root / "predictions.csv"
        self.reports = self.root / "reports"

    def write_predictions(self, header, rows):
        with self.predictions.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return self.predictions

    def run_cli(self, *args):
        if not args:
            args = ("--predictions", self.predictions.name, "--output-dir", self.reports.name)
        return subprocess.run(
            [sys.executable, str(CODE / "scripts/evaluate_predictions.py"), *args],
            cwd=self.root, capture_output=True, text=True,
        )

    def test_perfect_predictions_for_all_classes(self):
        result = evaluate_predictions(self.write_predictions(REQUIRED, [(i, i) for i in range(25)]))
        self.assertEqual(result.metrics, {
            "top1_accuracy": 1.0, "macro_precision": 1.0, "macro_recall": 1.0, "macro_f1": 1.0,
            "correct": 25, "total": 25,
        })
        self.assertEqual(result.confusion_matrix, [[int(i == j) for j in range(25)] for i in range(25)])
        self.assertIsNone(result.confidence_analysis)

    def test_known_metrics_and_matrix_orientation(self):
        result = evaluate_predictions(self.write_predictions(REQUIRED, KNOWN_ROWS))
        self.assertAlmostEqual(result.metrics["top1_accuracy"], 3 / 7)
        self.assertAlmostEqual(result.metrics["macro_precision"], (1 + 0.5) / 25)
        self.assertAlmostEqual(result.metrics["macro_recall"], (2 / 3 + 1) / 25)
        self.assertAlmostEqual(result.metrics["macro_f1"], (0.8 + 2 / 3) / 25)
        self.assertEqual(result.confusion_matrix[0][1], 1)
        self.assertEqual(result.confusion_matrix[1][0], 0)
        self.assertEqual(result.confusion_matrix[2][3], 2)

        cls = result.per_class
        self.assertEqual(cls[0], {"class_id": 0, "support": 3, "predicted_count": 2,
                                  "correct": 2, "precision": 1.0, "recall": 2 / 3, "f1": 0.8})
        self.assertAlmostEqual(cls[1]["f1"], 2 / 3)
        for i, support, predicted in [(2, 2, 0), (3, 0, 3), (24, 0, 0)]:
            with self.subTest(class_id=i):
                self.assertEqual(cls[i], {"class_id": i, "support": support, "predicted_count": predicted,
                                          "correct": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0})

    def test_probability_metrics_and_confidence_fallback(self):
        path = self.write_predictions(REQUIRED + PROBABILITIES, [
            (0, 0, *probability_vector(0.7, 0.2, 0.1)),
            (2, 0, *probability_vector(0.6, 0.25, 0.15)),
            (4, 1, *probability_vector(0.2, 0.5, 0.2, 0.09, 0.01)),
        ])
        result = evaluate_predictions(path)
        self.assertAlmostEqual(result.metrics["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(result.metrics["top3_accuracy"], 2 / 3)
        self.assertEqual(result.confidence_analysis, {
            "source": "predicted_class_probability",
            "correct": {"count": 1, "mean_confidence": 0.7},
            "incorrect": {"count": 2, "mean_confidence": 0.55},
        })

    def test_ties_prefer_supplied_prediction_then_lower_class_id(self):
        for true_id, expected_top3 in [(24, 1.0), (0, 1.0), (1, 1.0), (2, 0.0)]:
            with self.subTest(true_id=true_id):
                path = self.write_predictions(REQUIRED + PROBABILITIES,
                                              [(true_id, 24, *([0.04] * 25))])
                result = evaluate_predictions(path)
                self.assertEqual(result.metrics["top3_accuracy"], expected_top3)
                self.assertEqual(result.metrics["top1_accuracy"], float(true_id == 24))

    def test_confidence_without_probabilities_and_empty_group(self):
        path = self.write_predictions(REQUIRED + ("confidence",), [(0, 0, 0), (1, 1, 1)])
        result = evaluate_predictions(path)
        self.assertNotIn("top3_accuracy", result.metrics)
        self.assertEqual(result.confidence_analysis, {
            "source": "confidence",
            "correct": {"count": 2, "mean_confidence": 0.5},
            "incorrect": {"count": 0, "mean_confidence": None},
        })

    def test_explicit_confidence_takes_precedence_and_rounded_sum_is_allowed(self):
        path = self.write_predictions(REQUIRED + ("confidence",) + PROBABILITIES,
                                      [(1, 0, 0.25, *probability_vector(0.60001, 0.4))])
        result = evaluate_predictions(path)
        self.assertEqual(result.metrics["top3_accuracy"], 1.0)
        self.assertEqual(result.confidence_analysis["source"], "confidence")
        self.assertEqual(result.confidence_analysis["incorrect"]["mean_confidence"], 0.25)

    def test_metadata_bom_and_reordered_columns(self):
        self.predictions.write_text(
            'tag,pred_class_id,true_class_id\n"crop, a", 24 ,24\n\n', encoding="utf-8-sig")
        result = evaluate_predictions(self.predictions)
        self.assertEqual(result.metrics["top1_accuracy"], 1.0)
        self.assertEqual(result.confusion_matrix[24][24], 1)
        self.assertNotIn("top3_accuracy", result.metrics)

    def test_invalid_csv_structure(self):
        for content, message in [
            ("", "empty CSV"),
            ("\n\n", "empty CSV"),
            ("true_class_id,pred_class_id\n\n", "no prediction rows"),
            ("true_class_id,label\n0,1\n", "missing required columns: pred_class_id"),
            ("true_class_id,pred_class_id,pred_class_id\n", "duplicate or blank"),
            ("true_class_id,pred_class_id,\n", "duplicate or blank"),
            ("true_class_id,pred_class_id,prob_0\n", "incomplete probability columns.*prob_24"),
            ("true_class_id,pred_class_id,prob_25\n", "unknown probability columns: prob_25"),
            ("true_class_id,pred_class_id\n0\n", "line 2: expected 2 fields, got 1"),
            ("true_class_id,pred_class_id\n0,1,2\n", "line 2: expected 2 fields, got 3"),
            ('true_class_id,pred_class_id\n0,"1\n', "malformed CSV near line 2"),
        ]:
            with self.subTest(content=content):
                self.predictions.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_predictions(self.predictions)

    def test_invalid_class_ids_in_both_columns(self):
        for index, column in enumerate(REQUIRED):
            for value, message in [("", f"missing value for {column}"),
                                   ("x", "must be an integer"), ("1.0", "must be an integer"),
                                   ("1e0", "must be an integer"), (-1, "must be in"),
                                   (25, "must be in")]:
                with self.subTest(column=column, value=value):
                    row = [0, 0]
                    row[index] = value
                    path = self.write_predictions(REQUIRED, [(0, 0), row])
                    with self.assertRaisesRegex(ValueError, f"line 3: .*{message}"):
                        evaluate_predictions(path)

    def test_invalid_confidence_and_probability_values(self):
        for value, message in [
            ("", "missing value for"), ("unknown", "must be a number"), ("NaN", "must be finite"),
            ("+inf", "must be finite"), ("-Infinity", "must be finite"),
            (-0.1, "must be in"), (1.1, "must be in"),
            ("1.0000000000000000001", "must be in"), ("-1e-400", "must be in"),
        ]:
            for header, row in [
                (REQUIRED + ("confidence",), (0, 0, value)),
                (REQUIRED + PROBABILITIES, (0, 0, *probability_vector(value))),
            ]:
                with self.subTest(column=header[2], value=value):
                    path = self.write_predictions(header, [row])
                    with self.assertRaisesRegex(ValueError, f"line 2: .*{message}"):
                        evaluate_predictions(path)

    def test_invalid_probability_sum_and_nonmaximal_prediction(self):
        for pred, probs, message in [
            (0, probability_vector(0.6, 0.39), "must sum to 1"),
            (0, probability_vector(0.60002, 0.4), "must sum to 1"),
            (1, probability_vector(0.5000001, 0.4999999), "pred_class_id 1.*maximum"),
        ]:
            with self.subTest(pred=pred, probs=probs):
                path = self.write_predictions(REQUIRED + PROBABILITIES, [(0, pred, *probs)])
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_predictions(path)

    def test_probability_sum_uses_fixed_precision(self):
        with localcontext() as context:
            context.prec = 3
            for score in ("0.60001", "0.60002"):
                with self.subTest(score=score):
                    path = self.write_predictions(REQUIRED + PROBABILITIES,
                                                  [(0, 0, *probability_vector(score, "0.4"))])
                    if score == "0.60001":
                        self.assertEqual(evaluate_predictions(path).metrics["top3_accuracy"], 1.0)
                    else:
                        with self.assertRaisesRegex(ValueError, "probabilities must sum to 1"):
                            evaluate_predictions(path)
            self.assertEqual(context.prec, 3)

    def test_reports_have_labeled_totals_and_sorted_confusions(self):
        result = evaluate_predictions(self.write_predictions(REQUIRED, KNOWN_ROWS))
        write_reports(result, self.reports)
        self.assertEqual({p.name for p in self.reports.iterdir()}, {
            "summary.json", "per_class.csv", "confusion_matrix.csv", "confusion_matrix.png",
        })
        summary = json.loads((self.reports / "summary.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(summary["metrics"]["top1_accuracy"], 3 / 7)
        self.assertNotIn("top3_accuracy", summary["metrics"])
        self.assertNotIn("confidence_analysis", summary)
        self.assertEqual(summary["top_confusions"], [
            {"true_species": "Branta canadensis", "predicted_species": "Buteo jamaicensis", "count": 2},
            {"true_species": "Anas platyrhynchos", "predicted_species": "Ardea herodias", "count": 1},
            {"true_species": "Cardinalis cardinalis", "predicted_species": "Buteo jamaicensis", "count": 1},
        ])
        with (self.reports / "per_class.csv").open(newline="", encoding="utf-8") as stream:
            classes = list(csv.DictReader(stream))
        self.assertEqual([int(row["class_id"]) for row in classes], list(range(25)))
        self.assertEqual(classes[0]["scientific_name"], "Anas platyrhynchos")
        self.assertEqual(classes[0]["support"], "3")
        self.assertEqual(classes[0]["predicted_count"], "2")
        self.assertAlmostEqual(float(classes[0]["f1"]), 0.8)

        with (self.reports / "confusion_matrix.csv").open(newline="", encoding="utf-8") as stream:
            labels, *matrix = list(csv.reader(stream))
        self.assertEqual(labels[:3], ["true\\predicted", "00:Anas platyrhynchos", "01:Ardea herodias"])
        self.assertEqual([row[0] for row in matrix], labels[1:])
        self.assertEqual([list(map(int, row[1:])) for row in matrix], result.confusion_matrix)
        with Image.open(self.reports / "confusion_matrix.png") as image:
            self.assertEqual(image.format, "PNG")
            image.verify()

    def test_cli_writes_optional_metrics_from_another_working_directory(self):
        self.write_predictions(REQUIRED + PROBABILITIES,
                               [(1, 0, *probability_vector(0.8, 0.2))])
        process = self.run_cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Top-3 accuracy: 1.0000", process.stdout)
        summary = json.loads((self.reports / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["metrics"]["top3_accuracy"], 1.0)
        self.assertEqual(summary["confidence_analysis"]["incorrect"]["mean_confidence"], 0.8)
        self.assertEqual(summary["top_confusions"][0]["true_species"], "Ardea herodias")

    def test_cli_reports_bad_or_missing_input_without_creating_reports(self):
        process = self.run_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("predictions.csv", process.stderr)
        self.write_predictions(REQUIRED, [(0, 25)])
        process = self.run_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("line 2: pred_class_id must be in [0, 24]", process.stderr)
        self.assertNotIn("Traceback", process.stderr)
        self.assertFalse(self.reports.exists())

    def test_cli_rejects_input_at_any_report_path_before_writing(self):
        for filename in ("per_class.csv", "confusion_matrix.csv", "confusion_matrix.png", "summary.json"):
            with self.subTest(filename=filename):
                output = self.root / f"collision-{filename}"
                output.mkdir()
                predictions = output / filename
                content = b"true_class_id,pred_class_id\n0,0\n"
                predictions.write_bytes(content)
                process = self.run_cli("--predictions", str(predictions), "--output-dir", str(output))
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("input CSV collides with report path", process.stderr)
                self.assertEqual(predictions.read_bytes(), content)
                self.assertEqual(set(output.iterdir()), {predictions})

    def test_cli_rejects_report_path_aliasing_input(self):
        self.write_predictions(REQUIRED, [(0, 0)])
        content = self.predictions.read_bytes()
        self.reports.mkdir()
        (self.reports / "per_class.csv").symlink_to(self.predictions)
        (self.reports / "confusion_matrix.csv").hardlink_to(self.predictions)
        for report in ("per_class.csv", "confusion_matrix.csv"):
            with self.subTest(report=report):
                process = self.run_cli()
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("input CSV collides with report path", process.stderr)
                self.assertIn(report, process.stderr)
                self.assertEqual(self.predictions.read_bytes(), content)
                (self.reports / report).unlink()

    def test_demo_cli_generates_deterministic_csv_and_labeled_reports(self):
        process = self.run_cli("--demo")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Synthetic demo metrics", process.stdout)
        output = self.root / "outputs/evaluation/demo"
        self.assertEqual({path.name for path in output.iterdir()}, {
            "predictions.csv", "summary.json", "per_class.csv", "confusion_matrix.csv",
            "confusion_matrix.png",
        })
        predictions = output / "predictions.csv"
        original_csv = predictions.read_bytes()
        with predictions.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, [*REQUIRED, "confidence", *PROBABILITIES])
            rows = list(reader)
        self.assertEqual([int(row["true_class_id"]) for row in rows], list(range(25)))
        self.assertEqual(sum(row["true_class_id"] == row["pred_class_id"] for row in rows), 15)
        for row in rows:
            with self.subTest(true_id=row["true_class_id"]):
                scores = [Decimal(row[column]) for column in PROBABILITIES]
                self.assertTrue(all(0 <= score <= 1 for score in scores))
                self.assertEqual(sum(scores), 1)
                self.assertEqual(scores[int(row["pred_class_id"])], max(scores))
                self.assertEqual(Decimal(row["confidence"]), max(scores))

        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertIs(summary["synthetic"], True)
        self.assertEqual(summary["metrics"]["total"], 25)
        self.assertEqual(summary["metrics"]["top1_accuracy"], 0.6)
        self.assertEqual(summary["metrics"]["top3_accuracy"], 0.8)
        self.assertAlmostEqual(summary["metrics"]["macro_f1"], 8 / 15)
        confidence = summary["confidence_analysis"]
        self.assertEqual(confidence["source"], "confidence")
        self.assertAlmostEqual(confidence["correct"]["mean_confidence"], 0.8)
        self.assertAlmostEqual(confidence["incorrect"]["mean_confidence"], 0.6)
        process = self.run_cli("--demo")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(predictions.read_bytes(), original_csv)

    def test_demo_cli_rejects_incompatible_options(self):
        for option, value in [("--predictions", "predictions.csv"), ("--output-dir", "reports")]:
            with self.subTest(option=option):
                process = self.run_cli("--demo", option, value)
                self.assertEqual(process.returncode, 2)
                self.assertIn(option, process.stderr)
        self.assertFalse((self.root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
