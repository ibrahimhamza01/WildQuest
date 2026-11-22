#!/usr/bin/env python3
"""Analyze the saved WildQuest EfficientNet checkpoint on VALIDATION images only.

This script never trains, selects test data, or changes the checkpoint. It writes
validation metrics, a confusion-matrix image, and individual mistakes for review.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code/src"))

from wildquest.classifier.dataset import WildlifeDataset  # noqa: E402
from wildquest.classifier.model import build_model  # noqa: E402

from wildquest.evaluation.metrics import confusion_metrics  # noqa: E402
from wildquest.evaluation.reporting import (  # noqa: E402
    label_per_class, save_confusion_png, top_confusions, write_confusion_csv, write_csv,
    write_per_class_csv,
)

N_CLASSES = 25


def analyze(checkpoint_path, output_dir, batch_size=64, num_workers=2, device_name="cuda"):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable: pass --device cpu for a local check")
    device = torch.device(device_name)
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()

    mapping = json.loads((ROOT / "Code/configs/class_mapping.json").read_text(encoding="utf-8"))
    species = [mapping["id_to_label"][str(i)] for i in range(N_CLASSES)]
    actual_sha = hashlib.sha256((ROOT / "Code/configs/dataset_manifest.csv").read_bytes()).hexdigest()
    # Torch 2.13 accepts weights_only=True for this checkpoint's primitive metadata.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("manifest_sha256") != actual_sha:
        raise ValueError("Checkpoint manifest hash differs from frozen local manifest")
    if checkpoint.get("class_mapping") != mapping:
        raise ValueError("Checkpoint class mapping differs from repository mapping")

    dataset = WildlifeDataset(ROOT, "val")  # NEVER train/test/dmv_test
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=num_workers > 0)

    # pretrained=False avoids downloading ImageNet weights; the checkpoint replaces all weights.
    model = build_model(num_classes=N_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    matrix = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int64, device=device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    top3_sum = torch.zeros((), dtype=torch.int64, device=device)
    collected = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss_per_image = F.cross_entropy(logits, labels, reduction="none")
            probabilities = torch.softmax(logits.float(), dim=1)
            confidence, predicted = probabilities.max(dim=1)
            true_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
            matrix += torch.bincount(labels * N_CLASSES + predicted,
                                     minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
            top3_sum += (logits.topk(3, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum()
            loss_sum += loss_per_image.double().sum()
            collected.append(torch.stack((labels, predicted, confidence, true_probability,
                                          loss_per_image.float()), dim=1))

    results = torch.cat(collected).cpu().tolist()  # one GPU->CPU transfer at the end
    confusion = matrix.cpu().tolist()
    metrics, per_class = confusion_metrics(confusion)
    if len(results) != len(dataset):
        raise ValueError("Prediction order/length mismatch")
    metrics["cross_entropy"] = (loss_sum / len(dataset)).item()
    metrics["top3_accuracy"] = (top3_sum / len(dataset)).item()

    output_dir.mkdir(parents=True, exist_ok=True)
    per_class_rows = label_per_class(per_class, species)
    write_per_class_csv(output_dir / "per_class.csv", per_class_rows)
    write_confusion_csv(output_dir / "confusion_matrix.csv", confusion, species)

    mistakes = []
    for index, (true_id, pred_id, confidence, true_prob, per_image_loss) in enumerate(results):
        true_id, pred_id = int(true_id), int(pred_id)
        if true_id != pred_id:
            image_path = dataset.samples[index][0].relative_to(ROOT).as_posix()
            mistakes.append({
                "local_path": image_path,
                "true_id": true_id,
                "true_species": species[true_id],
                "predicted_id": pred_id,
                "predicted_species": species[pred_id],
                "predicted_confidence": round(confidence, 6),
                "true_class_probability": round(true_prob, 6),
                "cross_entropy": round(per_image_loss, 6),
            })
    mistakes.sort(key=lambda row: row["cross_entropy"], reverse=True)
    write_csv(output_dir / "mistakes.csv",
              ["local_path", "true_id", "true_species", "predicted_id", "predicted_species",
               "predicted_confidence", "true_class_probability", "cross_entropy"], mistakes)
    save_confusion_png(output_dir / "confusion_matrix.png", confusion, species,
                       title="WildQuest validation confusion matrix")

    common_confusions = top_confusions(confusion, species)
    summary = {
        "split": "val",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_recorded_val_accuracy": checkpoint.get("val_accuracy"),
        "manifest_sha256": actual_sha,
        "metrics": metrics,
        "top_confusions": common_confusions,
        "lowest_f1_classes": sorted(per_class_rows, key=lambda row: row["f1"])[:5],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')} | split: val ({metrics['total']:,} images)")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.4f}  Top-3: {metrics['top3_accuracy']:.4f}")
    print(f"Cross-entropy: {metrics['cross_entropy']:.4f}  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Correct: {metrics['correct']:,}  Incorrect: {len(mistakes):,}")
    print("\nLowest-F1 classes:")
    for row in summary["lowest_f1_classes"]:
        print(f"  {row['scientific_name']:<32} F1={row['f1']:.3f} recall={row['recall']:.3f}")
    print("\nMost common confusion pairs (true -> predicted):")
    for row in common_confusions[:5]:
        print(f"  {row['true_species']} -> {row['predicted_species']}: {row['count']}")
    print(f"\nWrote reports to: {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/baselines/efficientnet_b0_v1/best.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/analysis/efficientnet_b0_v1/validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be >=1 and --num-workers >=0")
    analyze(args.checkpoint, args.output_dir, args.batch_size, args.num_workers, args.device)


if __name__ == "__main__":
    main()
