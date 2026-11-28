#!/usr/bin/env python3
"""Analyze classifier confidence on exported validation predictions (issue #14).

Reads a prediction CSV in the evaluation-framework format (as written by
export_validation_predictions.py) and reports:

- reliability: how often predictions at each confidence level are correct,
  plus the expected calibration error (ECE);
- the accuracy / coverage trade-off of auto-accepting predictions at or above
  each candidate threshold and sending the rest to NEEDS_REVIEW;
- high-confidence errors and the species pairs that cause them;
- per-species accuracy, confidence and review rate at the recommended threshold.

Runs offline with the standard library (plots need matplotlib). Never loads a
model or images.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code/src"))

from wildquest.evaluation.predictions import load_species, read_predictions  # noqa: E402

N_CLASSES = 25
DEFAULT_THRESHOLDS = (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                      0.85, 0.90, 0.95, 0.99)
CURRENT_THRESHOLD = 0.70  # provisional review_threshold in Code/configs/gamification.json


def load_rows(path):
    """Validate the CSV with the evaluation framework, then return one dict per row."""
    path = Path(path)
    validated = list(read_predictions(path, N_CLASSES))
    with path.open(newline="", encoding="utf-8-sig") as stream:
        extras = [row for row in csv.DictReader(stream) if any(row.values())]
    if len(extras) != len(validated):
        raise ValueError(f"{path}: could not align rows")
    rows = []
    for (true_id, pred_id, confidence, probs), extra in zip(validated, extras):
        if confidence is None and probs is None:
            raise ValueError(f"{path}: needs a confidence column or prob_0..prob_24 columns")
        conf = float(confidence if confidence is not None else probs[pred_id])
        margin = None
        if probs is not None:
            ordered = sorted((float(p) for p in probs), reverse=True)
            margin = ordered[0] - ordered[1]
        rows.append({
            "sample_id": extra.get("sample_id", ""),
            "local_path": extra.get("local_path", ""),
            "true": true_id,
            "pred": pred_id,
            "confidence": conf,
            "margin": margin,
            "correct": true_id == pred_id,
        })
    return rows


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def threshold_table(rows, thresholds=DEFAULT_THRESHOLDS):
    """Auto-accept predictions with confidence >= t; everything else needs review."""
    total = len(rows)
    total_errors = sum(not r["correct"] for r in rows)
    total_correct = total - total_errors
    table = []
    for t in sorted(thresholds):
        accepted = [r for r in rows if r["confidence"] >= t]
        accepted_correct = sum(r["correct"] for r in accepted)
        accepted_errors = len(accepted) - accepted_correct
        flagged = total - len(accepted)
        flagged_errors = total_errors - accepted_errors
        table.append({
            "threshold": t,
            "accepted": len(accepted),
            "coverage": _ratio(len(accepted), total),
            "accepted_accuracy": _ratio(accepted_correct, len(accepted)),
            "accepted_errors": accepted_errors,
            "needs_review": flagged,
            "review_rate": _ratio(flagged, total),
            "errors_caught": _ratio(flagged_errors, total_errors),
            "correct_sent_to_review": _ratio(flagged - flagged_errors, total_correct),
        })
    return table


def reliability(rows, n_bins=10):
    """Equal-width confidence bins with accuracy, mean confidence and the ECE."""
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        bins[min(int(r["confidence"] * n_bins), n_bins - 1)].append(r)
    table, ece = [], 0.0
    for k, members in enumerate(bins):
        count = len(members)
        accuracy = _ratio(sum(r["correct"] for r in members), count)
        mean_conf = _ratio(sum(r["confidence"] for r in members), count)
        if count:
            ece += count / len(rows) * abs(accuracy - mean_conf)
        table.append({"bin_low": k / n_bins, "bin_high": (k + 1) / n_bins, "count": count,
                      "accuracy": accuracy, "mean_confidence": mean_conf,
                      "gap": None if count == 0 else mean_conf - accuracy})
    return table, ece


def recommend_threshold(table, target_accuracy):
    """Lowest threshold (so the fewest reviews) whose accepted accuracy meets the target."""
    for row in table:  # table is sorted by threshold
        if row["accepted"] and row["accepted_accuracy"] >= target_accuracy:
            return row
    return None


def high_confidence_errors(rows, min_confidence):
    errors = [r for r in rows if not r["correct"] and r["confidence"] >= min_confidence]
    return sorted(errors, key=lambda r: (-r["confidence"], r["true"], r["pred"]))


def error_pairs(errors, species):
    counts = Counter((r["true"], r["pred"]) for r in errors)
    return [{"true_species": species[t], "predicted_species": species[p], "count": n}
            for (t, p), n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def per_class(rows, threshold, species):
    out = []
    for class_id, name in enumerate(species):
        members = [r for r in rows if r["true"] == class_id]
        out.append({
            "class_id": class_id,
            "scientific_name": name,
            "support": len(members),
            "accuracy": _ratio(sum(r["correct"] for r in members), len(members)),
            "mean_confidence": _ratio(sum(r["confidence"] for r in members), len(members)),
            "review_rate": _ratio(sum(r["confidence"] < threshold for r in members), len(members)),
            "confident_errors": sum(not r["correct"] and r["confidence"] >= threshold
                                    for r in members),
        })
    return out


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir, rows, rel_table, thr_table, recommended):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filled = [b for b in rel_table if b["count"]]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.bar([(b["bin_low"] + b["bin_high"]) / 2 for b in filled], [b["accuracy"] for b in filled],
           width=1 / len(rel_table), edgecolor="black", alpha=0.7, label="Accuracy in bin")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Model confidence", ylabel="Accuracy",
           title="Reliability diagram (validation)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=150)
    plt.close(fig)

    xs = [r["threshold"] for r in thr_table]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [r["coverage"] for r in thr_table], "o-", label="Coverage (auto-accepted)")
    ax.plot(xs, [r["accepted_accuracy"] for r in thr_table], "s-",
            label="Accuracy of auto-accepted")
    ax.axvline(CURRENT_THRESHOLD, color="gray", linestyle=":", label="Current 0.70")
    if recommended:
        ax.axvline(recommended["threshold"], color="green", linestyle="--",
                   label=f"Recommended {recommended['threshold']:.2f}")
    ax.set(ylim=(0, 1.02), xlabel="Confidence threshold", ylabel="Share of validation images",
           title="Accuracy vs. coverage by threshold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_tradeoff.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = [i / 20 for i in range(21)]
    ax.hist([r["confidence"] for r in rows if r["correct"]], bins=bins, alpha=0.6, label="Correct")
    ax.hist([r["confidence"] for r in rows if not r["correct"]], bins=bins, alpha=0.6,
            label="Incorrect")
    ax.set(xlabel="Model confidence", ylabel="Validation images",
           title="Confidence of correct vs. incorrect predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_histogram.png", dpi=150)
    plt.close(fig)


def analyze(predictions, output_dir, target_accuracy=0.95, high_confidence=0.90,
            thresholds=DEFAULT_THRESHOLDS, n_bins=10, plots=True):
    predictions = Path(predictions).resolve()
    output_dir = Path(output_dir).resolve()
    rows = load_rows(predictions)
    species = load_species()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = ["summary.json", "thresholds.csv", "reliability.csv", "per_class.csv",
               "high_confidence_errors.csv"]
    if any((output_dir / name).resolve() == predictions for name in outputs):
        raise ValueError("The predictions CSV would be overwritten by a report file")

    thr_table = threshold_table(rows, thresholds)
    rel_table, ece = reliability(rows, n_bins)
    recommended = recommend_threshold(thr_table, target_accuracy)
    policy_threshold = recommended["threshold"] if recommended else CURRENT_THRESHOLD
    confident = high_confidence_errors(rows, high_confidence)
    errors = [r for r in rows if not r["correct"]]
    correct = [r for r in rows if r["correct"]]
    mean = lambda items, key: _ratio(sum(r[key] for r in items), len(items))  # noqa: E731
    margins = [r for r in rows if r["margin"] is not None]

    summary = {
        "predictions": str(predictions),
        "images": len(rows),
        "top1_accuracy": _ratio(len(correct), len(rows)),
        "mean_confidence": mean(rows, "confidence"),
        "mean_confidence_correct": mean(correct, "confidence"),
        "mean_confidence_incorrect": mean(errors, "confidence"),
        "mean_margin_correct": mean([r for r in margins if r["correct"]], "margin"),
        "mean_margin_incorrect": mean([r for r in margins if not r["correct"]], "margin"),
        "expected_calibration_error": ece,
        "target_accuracy": target_accuracy,
        "recommended_threshold": recommended,
        "current_threshold": next(r for r in threshold_table(rows, [CURRENT_THRESHOLD])),
        "high_confidence_cutoff": high_confidence,
        "high_confidence_errors": len(confident),
        "high_confidence_error_pairs": error_pairs(confident, species)[:10],
        "all_error_pairs": error_pairs(errors, species)[:10],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8")
    _write_csv(output_dir / "thresholds.csv", thr_table, list(thr_table[0]))
    _write_csv(output_dir / "reliability.csv", rel_table, list(rel_table[0]))
    classes = per_class(rows, policy_threshold, species)
    _write_csv(output_dir / "per_class.csv", classes, list(classes[0]))
    _write_csv(output_dir / "high_confidence_errors.csv",
               [{**r, "true_species": species[r["true"]], "predicted_species": species[r["pred"]]}
                for r in confident],
               ["sample_id", "local_path", "true_species", "predicted_species", "confidence",
                "margin"])
    if plots:
        save_plots(output_dir, rows, rel_table, thr_table, recommended)

    pct = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
    print(f"{len(rows):,} predictions | Top-1 accuracy {pct(summary['top1_accuracy'])} | "
          f"ECE {ece:.3f}")
    print(f"Mean confidence: correct {pct(summary['mean_confidence_correct'])}, "
          f"incorrect {pct(summary['mean_confidence_incorrect'])}")
    print("\nThreshold  coverage  accepted-accuracy  errors-caught  review-rate")
    for r in thr_table:
        print(f"  {r['threshold']:.2f}     {pct(r['coverage']):>7}   {pct(r['accepted_accuracy']):>10}"
              f"         {pct(r['errors_caught']):>7}      {pct(r['review_rate']):>7}")
    if recommended:
        print(f"\nRecommended threshold for >= {target_accuracy:.0%} accepted accuracy: "
              f"{recommended['threshold']:.2f} (coverage {pct(recommended['coverage'])})")
    else:
        print(f"\nNo candidate threshold reaches {target_accuracy:.0%} accepted accuracy.")
    print(f"High-confidence errors (>= {high_confidence:.0%}): {len(confident)}")
    print(f"Wrote reports to {output_dir}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "outputs/confidence_analysis/val_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/confidence_analysis")
    parser.add_argument("--target-accuracy", type=float, default=0.95,
                        help="accuracy the auto-accepted predictions should reach")
    parser.add_argument("--high-confidence", type=float, default=0.90,
                        help="confidence at or above which an error counts as high-confidence")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    for name in ("target_accuracy", "high_confidence"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.bins < 1:
        parser.error("--bins must be >= 1")
    return analyze(args.predictions, args.output_dir, args.target_accuracy,
                   args.high_confidence, n_bins=args.bins, plots=not args.no_plots)


if __name__ == "__main__":
    main()
    