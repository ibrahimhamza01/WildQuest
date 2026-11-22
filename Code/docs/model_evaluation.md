# Standalone model evaluation

Run the evaluator from the repository root with a prediction CSV and an output directory.

```bash
python Code/scripts/evaluate_predictions.py \
  --predictions path/to/predictions.csv \
  --output-dir outputs/evaluation/example
```

CSV evaluation uses the Python standard library. Report rendering uses Pillow, which is already listed in [requirements.txt](../requirements.txt). The evaluator reads the prediction CSV and [class mapping](../configs/class_mapping.json). It runs offline with no model, checkpoint, dataset images, PyTorch, or detector runtime. Invalid input produces an error on stderr and a nonzero exit code.

## Prediction CSV

Use a UTF-8, comma-delimited file with a header and one row per classification. Column order is flexible. Other named columns such as `sample_id` are ignored. Blank lines are skipped. Each data row must have the same number of fields as the header.

| Column | Definition |
| --- | --- |
| `true_class_id` | Required integer class ID for the ground truth. |
| `pred_class_id` | Required integer class ID for the exported Top-1 prediction. |
| `confidence` | Optional confidence of the exported prediction. Every row must have a finite value in `[0, 1]` if the column is present. |
| `prob_0` through `prob_24` | Optional probabilities in frozen class-ID order. Include all 25 columns together. Each value must be finite and in `[0, 1]`. |

Class IDs are integers from 0 through 24. IDs and scientific names come from the frozen class mapping. A label-only input can be as small as this example.

```csv
true_class_id,pred_class_id
0,0
1,0
2,2
```

For a confidence-only export, append the `confidence` header and a value to each row, such as `1,0,0.62`.

Each probability vector must sum to 1 within an absolute tolerance of 0.00001 to accommodate rounded exports. The predicted class must have a maximum probability. Any class tied exactly for the maximum is a valid predicted class.

## Synthetic demo

From the repository root, run the evaluator with no input CSV.

```bash
python Code/scripts/evaluate_predictions.py --demo
```

The command generates `predictions.csv` and the four reports listed below in `outputs/evaluation/demo/`. The Git-ignored CSV has one synthetic row per true class, with a confidence score and all 25 probability columns. The expected demo results are 15 of 25 Top-1 hits (60%), 20 of 25 Top-3 hits (80%), and a macro F1 of approximately 0.5333.

The CLI and heatmap label the results as synthetic, and `summary.json` includes `"synthetic": true`. Running the demo again regenerates the same predictions and replaces its reports. Use `--demo` on its own.

## Metrics

- **Top-1 accuracy** is the fraction of rows where the exported predicted ID matches the true ID.
- **Per-class precision** is the number of correct predictions for the class divided by its predicted count. **Recall** divides the same correct count by the class's true support. **F1** is the harmonic mean of precision and recall.
- **Macro precision, recall, and F1** are the unweighted means of the corresponding per-class metrics across all 25 configured classes, including classes with zero support. A zero denominator yields 0, equivalent to scikit-learn's `zero_division=0`.
- **Top-3 accuracy** is the fraction of rows whose true ID ranks among the first three class probabilities. Exact ties favor the exported predicted ID, then lower class IDs. This metric is omitted unless all 25 probability columns are present.
- **Confidence analysis** reports the count and mean confidence for correct and incorrect Top-1 predictions. A group with zero rows has a JSON `null` mean. When present, the explicit `confidence` column takes precedence and may contain a separately calibrated score. Otherwise the probability assigned to the predicted class is used. The analysis is omitted when both inputs are absent.

## Reports

| File | Contents |
| --- | --- |
| `summary.json` | Accuracy, macro metrics, counts, and up to ten nonzero misclassification pairs. Pairs are ordered by decreasing count, then by true and predicted class IDs. Includes the confidence analysis when available. |
| `per_class.csv` | Scientific name, support, predicted count, correct count, precision, recall, and F1 for each class in ID order. |
| `confusion_matrix.csv` | A 25×25 count matrix with ID and scientific name on both axes. Rows are true classes and columns are predicted classes. |
| `confusion_matrix.png` | The same counts in a heatmap. Row labels pair class IDs with scientific names. Column IDs use that same key. Shading shows the fraction of each true class assigned to the corresponding predicted class. |

Use a dedicated report directory under `outputs/evaluation/`. Git ignores this tree. A rerun replaces the report files in that directory. The CLI rejects any input path that points to a report file.

## Architecture and future inference exports

The `wildquest.evaluation` package exposes `evaluate_predictions(path)` to validate and score a CSV and `write_reports(result, output_dir)` to render the result. Its prediction reader validates the schema and values. The metrics module accumulates class counts and derives scores. The reporting module handles labels, file formats, and the heatmap. The existing validation analyzer also uses the shared confusion metrics and reporting helpers within its checkpoint-based validation workflow.

Future EfficientNet inference can export one row per labeled image. Use the frozen class indices for the true and predicted labels. To enable Top-3, export all softmax probabilities in class-ID order with enough precision to meet the sum tolerance and preserve the predicted class as a maximum. Integrated computer vision inference can export the same schema for its labeled classification units, such as matched crops. Choose the matching and inclusion rules upstream so each CSV row has one ground-truth species and one predicted species. The evaluator measures species classification on the exported rows.

## Tests

Run the focused evaluator tests or the complete test suite from the repository root.

```bash
python -m unittest discover -s Code/tests -p test_evaluation.py -v
python -m unittest discover -s Code/tests -v
```

The evaluator tests create temporary prediction CSVs and reports from synthetic data.
