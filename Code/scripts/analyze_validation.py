#!/usr/bin/env python3
"""Analyze the saved WildQuest EfficientNet checkpoint on VALIDATION images only.

This script never trains, selects test data, or changes the checkpoint. It writes
validation metrics, a confusion-matrix image, and individual mistakes for review.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code/src"))

from wildquest.classifier.dataset import WildlifeDataset  # noqa: E402
from wildquest.classifier.model import build_model  # noqa: E402

N_CLASSES = 25


def confusion_metrics(confusion):
    """Compute macro metrics and row-wise per-class metrics from a square confusion matrix.

    Rows = true class IDs; columns = predicted class IDs.
    Class with zero predictions receives precision 0 (rather than undefined).
    """
    n = len(confusion)
    if n == 0 or any(len(row) != n for row in confusion):
        raise ValueError("Confusion matrix must be nonempty and square")
    total = sum(map(sum, confusion))
    if total == 0:
        raise ValueError("Cannot calculate metrics on an empty validation set")
    per_class = []
    for i in range(n):
        true_positive = confusion[i][i]
        support = sum(confusion[i])
        predicted_count = sum(confusion[r][i] for r in range(n))
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "class_id": i,
            "support": support,
            "predicted_count": predicted_count,
            "correct": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    diagonal = sum(confusion[i][i] for i in range(n))
    return {
        "top1_accuracy": diagonal / total,
        "macro_precision": sum(v["precision"] for v in per_class) / n,
        "macro_recall": sum(v["recall"] for v in per_class) / n,
        "macro_f1": sum(v["f1"] for v in per_class) / n,
        "correct": diagonal,
        "total": total,
    }, per_class


def save_confusion_png(path, confusion, species):
    """Draw a 25x25 *row-normalized* heatmap using Pillow (no new dependency)."""
    cell, left, top = 32, 315, 115
    n = len(confusion)
    width = left + cell * n + 75
    height = top + cell * n + 105
    picture = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(picture)
    try:
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
        body = ImageFont.truetype("DejaVuSans.ttf", 13)
        title = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        small = body = title = ImageFont.load_default()

    d.text((left, 15), "WildQuest validation confusion matrix", fill=(20, 36, 55), font=title)
    d.text((left, 48), "Rows = true species; columns = predicted class ID", fill=(75, 90, 110), font=body)
    for j in range(n):
        x = left + j * cell
        d.text((x + 8, top - 26), f"{j:02d}", fill=(32, 46, 63), font=small)
    for i, row in enumerate(confusion):
        y = top + i * cell
        d.text((14, y + 7), f"{i:02d}  {species[i]}", fill=(32, 46, 63), font=body)
        support = sum(row)
        for j, count in enumerate(row):
            value = count / support if support else 0.0
            # White-to-blue sequential shade (normalized by true-class support).
            shade = (
                int(247 - 205 * value),
                int(249 - 160 * value),
                int(253 - 87 * value),
            )
            x = left + j * cell
            d.rectangle((x, y, x + cell - 1, y + cell - 1), fill=shade, outline=(232, 237, 243))
            if count:
                color = "white" if value > 0.64 else (24, 45, 65)
                label = str(count)
                bbox = d.textbbox((0, 0), label, font=small)
                d.text((x + (cell - (bbox[2] - bbox[0])) / 2,
                        y + (cell - (bbox[3] - bbox[1])) / 2 - 2),
                       label, fill=color, font=small)
    d.text((left, top + n * cell + 22),
           "Cell text = number of images. Darker cells = larger share of that true class.",
           fill=(75, 90, 110), font=body)
    picture.save(path)


def write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


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
    per_class_rows = [{**p, "scientific_name": species[p["class_id"]]} for p in per_class]
    write_csv(output_dir / "per_class.csv",
              ["class_id", "scientific_name", "support", "predicted_count", "correct",
               "precision", "recall", "f1"], per_class_rows)
    matrix_labels = [f"{i:02d}:{species[i]}" for i in range(N_CLASSES)]
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\predicted"] + matrix_labels)
        for i, row in enumerate(confusion):
            writer.writerow([matrix_labels[i]] + row)

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
    save_confusion_png(output_dir / "confusion_matrix.png", confusion, species)

    top_confusions = sorted(
        ({"true_species": species[i], "predicted_species": species[j], "count": confusion[i][j]}
         for i in range(N_CLASSES) for j in range(N_CLASSES) if i != j and confusion[i][j]),
        key=lambda row: row["count"], reverse=True,
    )[:10]
    summary = {
        "split": "val",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_recorded_val_accuracy": checkpoint.get("val_accuracy"),
        "manifest_sha256": actual_sha,
        "metrics": metrics,
        "top_confusions": top_confusions,
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
    for row in top_confusions[:5]:
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
