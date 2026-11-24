#!/usr/bin/env python3
"""Export EfficientNet predictions on the VALIDATION split to a CSV (issue #14).

One row per validation image with the true class, the Top-1 prediction, its
confidence, and all 25 softmax probabilities in frozen class-ID order. The CSV
follows the evaluation framework's contract (Code/docs/model_evaluation.md), so
it can be scored with Code/scripts/evaluate_predictions.py and analyzed with
Code/scripts/analyze_confidence.py.

The checkpoint is only read. Test and DMV test images are never loaded.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code/src"))

N_CLASSES = 25
SPLIT = "val"  # fixed: this exporter never reads train, test or dmv_test
PROB_DECIMALS = 8
FIELDS = ["sample_id", "local_path", "true_class_id", "pred_class_id", "confidence",
          *[f"prob_{i}" for i in range(N_CLASSES)]]


def prediction_row(local_path, true_id, probabilities):
    """Build one CSV row from a probability vector (any sequence of 25 floats).

    Probabilities are rounded to PROB_DECIMALS places. Rounding is monotonic, so
    the predicted class still has a maximum rounded probability, and the rounded
    vector sums to 1 well within the evaluator's 1e-5 tolerance.
    """
    if len(probabilities) != N_CLASSES:
        raise ValueError(f"Expected {N_CLASSES} probabilities, got {len(probabilities)}")
    pred_id = max(range(N_CLASSES), key=lambda i: (probabilities[i], -i))
    rounded = [round(float(p), PROB_DECIMALS) for p in probabilities]
    row = {
        "sample_id": Path(local_path).stem,
        "local_path": local_path,
        "true_class_id": int(true_id),
        "pred_class_id": pred_id,
        "confidence": f"{rounded[pred_id]:.{PROB_DECIMALS}f}",
    }
    row.update({f"prob_{i}": f"{p:.{PROB_DECIMALS}f}" for i, p in enumerate(rounded)})
    return row


def write_predictions_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(checkpoint_path):
    """Read the checkpoint and confirm it matches the frozen manifest and class mapping."""
    import torch

    mapping = json.loads((ROOT / "Code/configs/class_mapping.json").read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(
        (ROOT / "Code/configs/dataset_manifest.csv").read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("manifest_sha256") != manifest_sha:
        raise ValueError("Checkpoint manifest hash differs from the frozen local manifest")
    if checkpoint.get("class_mapping") != mapping:
        raise ValueError("Checkpoint class mapping differs from the repository mapping")
    return checkpoint


def export(checkpoint_path, output_path, batch_size=64, num_workers=2, device_name="auto"):
    import torch
    from torch.utils.data import DataLoader

    from wildquest.classifier.dataset import WildlifeDataset
    from wildquest.classifier.model import build_model

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable: pass --device cpu")
    device = torch.device(device_name)

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    dataset = WildlifeDataset(ROOT, SPLIT)
    missing = [p for p, _ in dataset.samples if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(dataset)} validation images are missing, "
                                f"e.g. {missing[0].relative_to(ROOT).as_posix()}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # pretrained=False: the checkpoint replaces every weight, so nothing is downloaded.
    model = build_model(num_classes=N_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    rows, index = [], 0
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)).float(), dim=1).cpu().tolist()
            for true_id, probs in zip(labels.tolist(), probabilities):
                local_path = dataset.samples[index][0].relative_to(ROOT).as_posix()
                rows.append(prediction_row(local_path, true_id, probs))
                index += 1
            print(f"\r{index:,}/{len(dataset):,} validation images", end="", flush=True)
    print()
    if len(rows) != len(dataset):
        raise ValueError("Prediction count does not match the validation split")

    write_predictions_csv(output_path, rows)
    correct = sum(r["true_class_id"] == r["pred_class_id"] for r in rows)
    print(f"Checkpoint epoch {checkpoint.get('epoch')} | split: {SPLIT} | device: {device}")
    print(f"Top-1 accuracy: {correct / len(rows):.4f} ({correct:,}/{len(rows):,})")
    print(f"Wrote {output_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/baselines/efficientnet_b0_v1/best.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/confidence_analysis/val_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be >= 1 and --num-workers >= 0")
    export(args.checkpoint, args.output, args.batch_size, args.num_workers, args.device)


if __name__ == "__main__":
    main()