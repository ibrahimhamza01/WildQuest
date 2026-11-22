#!/usr/bin/env python3
"""Evaluate a WildQuest prediction CSV and write labeled reports."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildquest.evaluation import evaluate_predictions, write_reports  # noqa: E402
from wildquest.evaluation.predictions import REQUIRED_COLUMNS, load_species  # noqa: E402

DEMO_OUTPUT_DIR = Path("outputs/evaluation/demo")


def write_demo_predictions(path):
    """Write one synthetic prediction for every configured true class."""
    class_ids = range(len(load_species()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*REQUIRED_COLUMNS, "confidence", *(f"prob_{i}" for i in class_ids)])
        for true_id in class_ids:
            next_id, second_id, third_id = ((true_id + i) % len(class_ids) for i in (1, 2, 3))
            if true_id % 5 == 0:
                ranked_ids = (next_id, second_id, third_id)
                scores = (0.65, 0.25, 0.10)
            elif true_id % 5 == 1:
                ranked_ids = (next_id, true_id, second_id)
                scores = (0.55, 0.30, 0.15)
            else:
                ranked_ids = (true_id, next_id, second_id)
                scores = (0.80, 0.15, 0.05)
            probabilities = [0] * len(class_ids)
            for class_id, score in zip(ranked_ids, scores):
                probabilities[class_id] = score
            writer.writerow([true_id, ranked_ids[0], scores[0], *probabilities])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path, help="Input prediction CSV")
    mode.add_argument("--demo", action="store_true", help=f"Run a synthetic example in {DEMO_OUTPUT_DIR}/")
    parser.add_argument("--output-dir", type=Path, help="Directory for reports from --predictions")
    args = parser.parse_args()

    if args.demo and args.output_dir is not None:
        parser.error("--output-dir is only used with --predictions")
    if not args.demo and args.output_dir is None:
        parser.error("--predictions requires --output-dir")
    output_dir = DEMO_OUTPUT_DIR if args.demo else args.output_dir
    predictions_path = output_dir / "predictions.csv" if args.demo else args.predictions

    try:
        if args.demo:
            write_demo_predictions(predictions_path)
        result = evaluate_predictions(predictions_path)
        write_reports(result, output_dir, predictions_path=predictions_path, synthetic=args.demo)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"{parser.prog}: error: {exc}\n")

    metrics = result.metrics
    if args.demo:
        print("Synthetic demo metrics (generated predictions only).")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.4f}  Macro F1: {metrics['macro_f1']:.4f}")
    if "top3_accuracy" in metrics:
        print(f"Top-3 accuracy: {metrics['top3_accuracy']:.4f}")
    print(f"Wrote reports to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
