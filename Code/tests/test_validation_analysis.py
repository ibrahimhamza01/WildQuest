"""Focused tests of validation-only metrics and confusion plot output."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "analyze_validation", ROOT / "Code/scripts/analyze_validation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ValidationAnalysisTests(unittest.TestCase):
    def test_metrics_rows_true_columns_predicted(self):
        # true class 0 predicted as 0 twice, 1 once; true class 1 -> 1 once.
        matrix = [[2, 1], [0, 1]]
        totals, classes = module.confusion_metrics(matrix)
        self.assertEqual(totals["correct"], 3)
        self.assertEqual(totals["total"], 4)
        self.assertAlmostEqual(totals["top1_accuracy"], 0.75)
        self.assertAlmostEqual(classes[0]["precision"], 1.0)
        self.assertAlmostEqual(classes[0]["recall"], 2/3)
        self.assertAlmostEqual(classes[1]["precision"], 0.5)
        self.assertAlmostEqual(classes[1]["recall"], 1.0)
        self.assertAlmostEqual(totals["macro_f1"], (0.8 + 2/3)/2)

    def test_empty_matrix_rejected(self):
        with self.assertRaises(ValueError):
            module.confusion_metrics([[0, 0], [0, 0]])

    def test_writes_confusion_image(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "confusion.png"
            matrix = [[4 if i == j else 0 for j in range(25)] for i in range(25)]
            module.save_confusion_png(path, matrix, [f"Species {i}" for i in range(25)])
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
