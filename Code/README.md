# WildQuest — Code

This directory contains WildQuest's implementation, configurations, and tests.

Run all commands from the **repository root**.

## Environment

Development and GPU training use the course-provided `dl-venv` on AWS EC2.

Python dependencies are listed in `Code/requirements.txt`. On a fresh environment, install the appropriate CUDA-compatible PyTorch build and the listed dependencies.

The dataset downloader requires AWS CLI v2 and read access to the team's S3 bucket.

## Execution Order

**1. Download the dataset**

```bash
python Code/scripts/get_dataset.py
```

Downloads the frozen 25,000-image dataset into `data/raw/images/`.

**2. Verify dataset integrity**

```bash
python Code/scripts/get_dataset.py --verify-only
```

**3. Run tests**

```bash
python -m unittest discover -s Code/tests -v
```

**4. Smoke-test the classifier**

```bash
python Code/scripts/train_classifier.py --smoke-test
```

**5. Train EfficientNet-B0**

```bash
python Code/scripts/train_classifier.py
```

Training settings are defined in `Code/configs/train_efficientnet_b0.json`.

**6. Analyze the saved classifier on validation images**

```bash
python Code/scripts/analyze_validation.py
```

The script reads the frozen `val` split and requires a trained checkpoint with the matching manifest hash and class mapping. Its default checkpoint is `outputs/baselines/efficientnet_b0_v1/best.pt`. Use `--checkpoint outputs/efficientnet_b0/best.pt` for a checkpoint produced under the default training settings.

The default device is CUDA. Pass `--device cpu` to run on CPU. The defaults are `--batch-size 64` and `--num-workers 2`. Reports go to `outputs/analysis/efficientnet_b0_v1/validation/` unless you pass `--output-dir`. The script writes top-1 and top-3 accuracy, cross-entropy, and macro metrics to `summary.json`. It also writes `per_class.csv`, `confusion_matrix.csv`, `confusion_matrix.png`, and `mistakes.csv`.

## Standalone Model Evaluation

Evaluate saved classifier predictions with Python and Pillow. This workflow runs offline with a prediction CSV and the frozen class mapping.

```bash
python Code/scripts/evaluate_predictions.py \
  --predictions path/to/predictions.csv \
  --output-dir outputs/evaluation/example
```

See the [model evaluation guide](docs/model_evaluation.md) for the CSV contract, metric definitions, report formats, and a synthetic example.

## Observations and XP

Record classified animals in SQLite, build each user's species collection, and award XP with encounter deduplication. Standard library only; rules live in `Code/configs/gamification.json`.

```bash
python -m unittest Code/tests/test_observations.py Code/tests/test_gamification.py -v
```

See the [observation system guide](docs/observation_system.md) for the event format, XP and encounter rules, and the service API.

## Standalone Animal Detection

MegaDetector locates animals and saves bounding boxes and crops. Follow the [animal detection guide](docs/animal_detection.md) to set up the separate CPU runtime and define `detector_python` and `detector_bundle`. Choose a new run directory under `outputs/detection/` each time.

```bash
"$detector_python" Code/scripts/test_detector.py \
  --image data/raw/images/train/02/18592269.jpg \
  --model-bundle "$detector_bundle" \
  --output-dir outputs/detection/single-photo
```

## Components

- [EfficientNet classifier](src/wildquest/classifier/README.md) — Architecture, transfer learning, training configuration, and results.
- `scripts/get_dataset.py` — Dataset downloading, integrity verification, and repair.

## Dataset and Reproducibility

The dataset is frozen using `Code/configs/dataset_manifest.csv` and `class_mapping.json`. Do not change the existing split assignments or class IDs.

Training uses only the `train` and `val` splits. The general and DMV test splits are reserved for final evaluation.

The dataset contains licensed iNaturalist photos from the October 27, 2026 Open Dataset snapshot. Preserve source and license metadata; CC-BY redistribution requires appropriate attribution. DMV coverage uses an approximate geographic bounding box.

Local datasets and model outputs are excluded from Git. Do not commit raw images, checkpoints, or credentials.