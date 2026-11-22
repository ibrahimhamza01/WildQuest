"""Labeled CSV, JSON, and confusion-matrix reports."""

import csv
import json
from pathlib import Path


def save_confusion_png(path, confusion, species, *, title="WildQuest confusion matrix"):
    """Draw counts with row-normalized shading using Pillow."""
    from PIL import Image, ImageDraw, ImageFont
    cell, left, top = 32, 315, 115
    n = len(confusion)
    width = left + cell * n + 75
    height = top + cell * n + 105
    picture = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(picture)
    try:
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
        body = ImageFont.truetype("DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        small = body = title_font = ImageFont.load_default()

    d.text((left, 15), title, fill=(20, 36, 55), font=title_font)
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


def label_per_class(per_class, species):
    return [{**row, "scientific_name": species[row["class_id"]]} for row in per_class]


def write_per_class_csv(path, labeled_rows):
    write_csv(path, ["class_id", "scientific_name", "support", "predicted_count", "correct",
                     "precision", "recall", "f1"], labeled_rows)


def write_confusion_csv(path, confusion, species):
    labels = [f"{i:02d}:{name}" for i, name in enumerate(species)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\predicted", *labels])
        for label, row in zip(labels, confusion):
            writer.writerow([label, *row])


def top_confusions(confusion, species):
    """Return up to ten off-diagonal pairs, with ties ordered by true and predicted ID."""
    pairs = [(i, j, count) for i, row in enumerate(confusion)
             for j, count in enumerate(row) if i != j and count]
    pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    return [{"true_species": species[i], "predicted_species": species[j], "count": count}
            for i, j, count in pairs[:10]]


def write_reports(result, output_dir, *, predictions_path=None, synthetic=False):
    """Write evaluation reports, protecting the input path when supplied."""
    output_dir = Path(output_dir)
    per_class_csv = output_dir / "per_class.csv"
    confusion_csv = output_dir / "confusion_matrix.csv"
    confusion_png = output_dir / "confusion_matrix.png"
    summary_json = output_dir / "summary.json"
    if predictions_path is not None:
        for report_path in (per_class_csv, confusion_csv, confusion_png, summary_json):
            if report_path.exists() and report_path.samefile(predictions_path):
                raise ValueError(f"input CSV collides with report path: {report_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    species = result.species
    confusion = result.confusion_matrix
    write_per_class_csv(per_class_csv, label_per_class(result.per_class, species))
    write_confusion_csv(confusion_csv, confusion, species)
    title = "WildQuest synthetic demo confusion matrix" if synthetic else "WildQuest confusion matrix"
    save_confusion_png(confusion_png, confusion, species, title=title)
    summary = {"metrics": result.metrics, "top_confusions": top_confusions(confusion, species)}
    if synthetic:
        summary["synthetic"] = True
    if result.confidence_analysis is not None:
        summary["confidence_analysis"] = result.confidence_analysis
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
