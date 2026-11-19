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

## Components

- [EfficientNet classifier](src/wildquest/classifier/README.md) — Architecture, transfer learning, training configuration, and results.
- `scripts/get_dataset.py` — Dataset downloading, integrity verification, and repair.

## Dataset and Reproducibility

The dataset is frozen using `Code/configs/dataset_manifest.csv` and `class_mapping.json`. Do not change the existing split assignments or class IDs.

Training uses only the `train` and `val` splits. The general and DMV test splits are reserved for final evaluation.

The dataset contains licensed iNaturalist photos from the October 27, 2026 Open Dataset snapshot. Preserve source and license metadata; CC-BY redistribution requires appropriate attribution. DMV coverage uses an approximate geographic bounding box.

Local datasets and model outputs are excluded from Git. Do not commit raw images, checkpoints, or credentials.