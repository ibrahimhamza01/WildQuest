# Classifier Confidence Analysis (Issue #14)

This analysis checks how well the EfficientNet-B0 classifier's confidence predicts
whether it is correct. From that, it recommends the confidence threshold below which
an observation is stored as `NEEDS_REVIEW` (no XP until approved). The current value,
0.70 in `Code/configs/gamification.json`, is provisional.

> **Status:** the scripts and tests are complete. Results and the final
> recommendation will be added after running them on the trained checkpoint.

## How to run

From the repository root. The export needs PyTorch (`Code/requirements.txt`), the
trained checkpoint, and the 2,500 **validation** images in `data/raw/images/val/`.
The analysis itself needs only the standard library plus matplotlib for the plots.

```bash
# 1. Export validation predictions (runs on CPU or GPU; the checkpoint is only read)
python Code/scripts/export_validation_predictions.py \
  --checkpoint outputs/baselines/efficientnet_b0_v1/best.pt

# 2. Analyze confidence
python Code/scripts/analyze_confidence.py

# Optional: score the same CSV with the shared evaluation framework (#6)
python Code/scripts/evaluate_predictions.py \
  --predictions outputs/confidence_analysis/val_predictions.csv \
  --output-dir outputs/evaluation/efficientnet_b0_v1_val

# Tests (synthetic data; no model or images needed)
python -m unittest Code/tests/test_confidence_analysis.py -v
```

All outputs go to `outputs/confidence_analysis/`, which Git ignores.

## What the export does

`export_validation_predictions.py` reads only the `val` split. Its split is fixed
in the code, and `WildlifeDataset` refuses the test splits, so no general or DMV test
image is ever loaded. It checks that the checkpoint's manifest hash and class mapping
match the frozen repository versions, then writes `val_predictions.csv` with one row
per image:
- `sample_id` and `local_path`;
- `true_class_id` and `pred_class_id`;
- `confidence`, the softmax probability of the predicted class;
- `prob_0` to `prob_24`, in frozen class-ID order.

This is the evaluation framework's CSV contract. `evaluate_predictions.py` accepts
the file directly, and the tests confirm that.

## What the analysis reports

| Output | Contents |
|---|---|
| `thresholds.csv` | For each candidate threshold *t* (0.30–0.99), predictions with confidence ≥ *t* are auto-accepted and the rest go to review. Reported per threshold: coverage, accuracy of the accepted predictions, accepted errors, review rate, share of all errors caught by review, and share of correct predictions unnecessarily sent to review. |
| `reliability.csv`, `reliability_diagram.png` | Accuracy versus mean confidence in ten confidence bins, plus the expected calibration error (ECE). A well-calibrated model's bars sit on the diagonal. |
| `confidence_histogram.png` | Confidence distributions of correct versus incorrect predictions. |
| `threshold_tradeoff.png` | Coverage and accepted accuracy across thresholds, with the current and recommended values marked. |
| `high_confidence_errors.csv` | Every mistake made with confidence ≥ 0.90: image, true and predicted species, confidence, and Top-1 / Top-2 margin. |
| `per_class.csv` | Per species: accuracy, mean confidence, review rate, and confident errors at the recommended threshold. |
| `summary.json` | Headline numbers, the recommended and current thresholds, and the most common (high-confidence) error pairs. |

**How the threshold is recommended.** The analysis picks the lowest candidate
threshold whose auto-accepted predictions reach the target accuracy (default 95%,
set with `--target-accuracy`). The lowest such threshold sends the fewest correct
predictions to review. The final policy in the Results section weighs this number
against the review rate and the species that produce high-confidence errors.

## Results

*To be completed after running on `efficientnet_b0_v1`.*

## Caveats

- The threshold applies to **full-image** inference, which is how the classifier was
  validated. Once MegaDetector crops feed the classifier (#11), the confidence
  distribution will change, so this analysis must be rerun on crop-based predictions.
- Validation images come from the same sources as training. Real user photos may be
  harder, so the recommended threshold should be revisited with app data.