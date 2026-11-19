#!/usr/bin/env python3
"""Stratified, read-only audit of WildQuest train/val image suitability.

Example from repository root:
    python Code/scripts/audit_image_quality.py sample
    # Manually complete the review fields in outputs/audits/image_quality_v1/review.csv
    python Code/scripts/audit_image_quality.py summarize

No training, test, or dmv_test images are accessed.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("outputs/audits/image_quality_v1")
SPLITS = ("train", "val")
CLASS_COUNT = 25
IMAGE_TYPES = (
    "visible_animal", "indirect_evidence", "remains_or_part",
    "no_animal_or_evidence", "uncertain",
)
VISIBILITY = ("clear", "small", "occluded", "unclear", "not_applicable")
ANIMAL_COUNT = ("single", "multiple", "none", "uncertain")
LABEL_ASSESSMENT = ("plausible", "questionable", "cannot_assess")
REVIEW_FIELDS = ("image_type", "visibility", "animal_count", "label_assessment", "notes")
FIELDS = (
    "sample_id", "sheet", "split", "class_id", "scientific_name",
    "local_path", "observation_uuid", *REVIEW_FIELDS,
)


def manifest_records(root):
    """Read only train/val rows from the committed frozen manifest."""
    cfg = root / "Code/configs"
    manifest = cfg / "dataset_manifest.csv"
    checksum = (cfg / "dataset_manifest.sha256").read_text(encoding="utf-8").split()[0].lower()
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != checksum:
        raise ValueError("Manifest SHA-256 mismatch: refusing to audit modified manifest")
    mapping = json.loads((cfg / "class_mapping.json").read_text(encoding="utf-8"))["id_to_label"]
    if set(mapping) != {str(i) for i in range(CLASS_COUNT)}:
        raise ValueError("Expected the frozen 25-class mapping")
    records = defaultdict(list)
    seen_observations = set()
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"split", "class_id", "scientific_name", "local_path", "observation_uuid"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Required columns missing from frozen manifest")
        for row in reader:
            if row["split"] not in SPLITS:
                continue
            split, class_id = row["split"], int(row["class_id"])
            if class_id not in range(CLASS_COUNT) or row["scientific_name"] != mapping[str(class_id)]:
                raise ValueError("Class mapping mismatch")
            relative = Path(row["local_path"])
            expected_parent = Path("data/raw/images") / split / f"{class_id:02d}"
            if relative.is_absolute() or ".." in relative.parts or relative.parent != expected_parent:
                raise ValueError(f"Unexpected manifest path: {relative}")
            obs = row["observation_uuid"]
            if not obs or obs in seen_observations:
                raise ValueError("Repeated/missing train/val observation UUID")
            seen_observations.add(obs)
            records[(split, class_id)].append({
                "split": split, "class_id": class_id,
                "scientific_name": row["scientific_name"],
                "local_path": relative.as_posix(), "observation_uuid": obs,
            })
    return records


def choose_samples(records, per_class, seed):
    """Exactly per_class images from each of 25 classes in each allowed split."""
    rng = random.Random(seed)
    result = []
    for split in SPLITS:
        chosen = []
        for class_id in range(CLASS_COUNT):
            population = sorted(records[(split, class_id)], key=lambda r: r["local_path"])
            if len(population) < per_class:
                raise ValueError(f"Only {len(population)} images for {split} class {class_id}")
            chosen.extend(rng.sample(population, per_class))
        rng.shuffle(chosen)
        for i, record in enumerate(chosen, start=1):
            entry = dict(record)
            entry["sample_id"] = f"{'TR' if split == 'train' else 'VA'}-{i:03d}"
            entry["sheet"] = f"{split}_{(i - 1) // 25 + 1:02d}.png"
            entry.update({f: "" for f in REVIEW_FIELDS})
            result.append(entry)
    return result


def draw_sheets(samples, root, out, panel_width=270, panel_height=310):
    """Build legible 5x5 sheets; fit whole photos without cropping."""
    font = ImageFont.load_default()
    by_sheet = defaultdict(list)
    for row in samples:
        by_sheet[row["sheet"]].append(row)
    sheets_dir = out / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for sheet_name, rows in sorted(by_sheet.items()):
        canvas = Image.new("RGB", (panel_width * 5, panel_height * 5), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(rows):
            x, y = (i % 5) * panel_width, (i // 5) * panel_height
            image_path = root / row["local_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Required image missing: {image_path}")
            try:
                with Image.open(image_path) as src:
                    img = ImageOps.exif_transpose(src).convert("RGB")
                    img.thumbnail((panel_width - 14, 222), Image.Resampling.LANCZOS)
            except (OSError, UnidentifiedImageError) as e:
                raise ValueError(f"Unable to open {image_path}: {e}") from e
            canvas.paste(img, (x + (panel_width - img.width) // 2, y + 4 + (222 - img.height) // 2))
            label_lines = [row["sample_id"], row["scientific_name"], Path(row["local_path"]).name]
            for k, line in enumerate(label_lines):
                draw.text((x + 7, y + 231 + 17 * k), line, font=font, fill="black")
            draw.rectangle((x, y, x + panel_width - 1, y + panel_height - 1), outline="#B0B0B0")
        canvas.save(sheets_dir / sheet_name)


def sample(root, out, per_class, seed):
    if per_class < 1:
        raise ValueError("--per-class must be >= 1")
    review_path = out / "review.csv"
    if review_path.exists():
        raise FileExistsError(f"{review_path} exists. Refusing to overwrite annotations; use a new --output-dir")
    records = manifest_records(root)
    chosen = choose_samples(records, per_class, seed)
    out.mkdir(parents=True, exist_ok=True)
    draw_sheets(chosen, root, out)
    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(chosen)
    (out / "sampling.json").write_text(json.dumps({
        "seed": seed, "per_class_per_split": per_class,
        "splits": list(SPLITS), "sample_count": len(chosen),
        "note": "Stratified random sample; original manifest and images are unchanged. No test samples.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Sampled {len(chosen)} images ({per_class} per class per split) using seed {seed}.")
    print(f"Review sheets: {out / 'sheets'}")
    print(f"Enter audit labels in: {review_path}")
    print(f"image_type options: {', '.join(IMAGE_TYPES)}")
    print(f"visibility options: {', '.join(VISIBILITY)}")
    print(f"animal_count options: {', '.join(ANIMAL_COUNT)}")
    print(f"label_assessment options: {', '.join(LABEL_ASSESSMENT)}")


def summarize(out):
    path = out / "review.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Audit review.csv is empty")
    if {r["split"] for r in rows} - set(SPLITS):
        raise ValueError("Unexpected split: only train and val are allowed")
    for row in rows:
        for field, choices in (
            ("image_type", IMAGE_TYPES), ("visibility", VISIBILITY),
            ("animal_count", ANIMAL_COUNT), ("label_assessment", LABEL_ASSESSMENT),
        ):
            value = row.get(field, "").strip()
            if value and value not in choices:
                raise ValueError(f"{row['sample_id']}: unexpected {field}={value!r}; options: {choices}")
    summary = {"note": "Reviewed stratified samples; not a full-dataset census.", "splits": {}}
    for split in SPLITS:
        group = [r for r in rows if r["split"] == split]
        annotated = [r for r in group if all(r[f].strip() for f in REVIEW_FIELDS if f != "notes")]
        type_counts = Counter(r["image_type"].strip() for r in annotated)
        visibility_counts = Counter(r["visibility"].strip() for r in annotated)
        label_counts = Counter(r["label_assessment"].strip() for r in annotated)
        n = len(annotated)
        summary["splits"][split] = {
            "sampled": len(group), "fully_reviewed": n,
            "image_type_counts": dict(type_counts),
            "visibility_counts": dict(visibility_counts),
            "label_assessment_counts": dict(label_counts),
            "visible_animal_fraction": type_counts["visible_animal"] / n if n else None,
        }
        print(f"{split}: reviewed {n}/{len(group)}")
        if n:
            print("  Image type:", dict(type_counts))
            print("  Visibility:", dict(visibility_counts))
            print("  Label assessment:", dict(label_counts))
            print(f"  Visible-animal fraction among reviewed: {type_counts['visible_animal']/n:.1%}")
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("sample", "summarize"))
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-class", type=int, default=4,
                   help="4 each from 25 classes in both train and val = 200 total")
    args = p.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if args.action == "sample":
        sample(root, out, args.per_class, args.seed)
    else:
        summarize(out)


if __name__ == "__main__":
    main()
