#!/usr/bin/env python3
"""Run standalone animal detection on explicit images or a train/val sample."""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
import tomllib
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code/src"))

from wildquest.detection import MegaDetector, crop_animal  # noqa: E402
from wildquest.detection.megadetector import MODEL_ID  # noqa: E402

SPLITS = ("train", "val")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_manifest():
    config = ROOT / "Code/configs"
    contents = (config / "dataset_manifest.csv").read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != (config / "dataset_manifest.sha256").read_text().split()[0]:
        raise ValueError("Frozen dataset manifest SHA-256 does not match")
    reader = csv.DictReader(io.StringIO(contents.decode("utf-8"), newline=""))
    return {row["local_path"]: row for row in reader if row["split"] in SPLITS}, digest


def sample_rows(rows, split, count, seed):
    if split not in SPLITS:
        raise ValueError("Detector sampling supports only train and val")
    classes = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            classes[int(row["class_id"])].append(row)
    if not classes:
        raise ValueError(f"No manifest images for split {split}")

    rng = random.Random(seed)
    seen_observations = set()
    sample = []
    for class_id, candidates in sorted(classes.items()):
        if len(candidates) < count:
            raise ValueError(f"Class {class_id} has fewer than {count} images in {split}")
        candidates.sort(key=lambda row: row["photo_id"])
        rng.shuffle(candidates)
        selected, repeated = [], []
        for row in candidates:
            observation = row["observation_uuid"]
            if observation in seen_observations:
                repeated.append(row)
                continue
            selected.append(row)
            seen_observations.add(observation)
            if len(selected) == count:
                break
        sample.extend(selected + repeated[:count - len(selected)])
    return sample


def resolve_image(path, allowed):
    """Accept external images and require an allowlisted path inside the dataset."""
    requested = Path(os.path.abspath(path))
    resolved = requested.resolve()
    data = ROOT / "data"
    resolved_data = data.resolve()
    matches = []
    for name, directory in ((requested, data), (resolved, resolved_data)):
        if name.is_relative_to(directory):
            relative = (Path("data") / name.relative_to(directory)).as_posix()
            row = allowed.get(relative)
            if row is None:
                raise ValueError(f"Repository dataset images must be in the train/val manifest: {path}")
            matches.append(row)
    if matches and (not resolved.is_relative_to(resolved_data) or matches[0] != matches[-1]):
        raise ValueError(f"Dataset path resolves outside its manifest entry: {path}")
    return resolved, matches[0] if matches else None


def output_directory(path):
    base = ROOT / "outputs/detection"
    requested = Path(os.path.abspath(path))
    resolved = requested.resolve()
    if (
        not requested.is_relative_to(base)
        or not resolved.is_relative_to(base)
        or resolved == base
    ):
        raise ValueError("--output-dir must be a new directory under outputs/detection/")
    if requested.exists() or requested.is_symlink():
        raise ValueError(f"Output directory already exists: {requested}")
    return resolved


def model_metadata(bundle):
    manifest = bundle / "manifest.toml"
    with manifest.open("rb") as handle:
        model = tomllib.load(handle).get("model", {})
    if bundle.name != MODEL_ID or model.get("id") != MODEL_ID or model.get("format") != "onnx":
        raise ValueError(f"--model-bundle must point to the unpacked {MODEL_ID} ONNX directory")
    if not model.get("file"):
        raise ValueError("Model manifest is missing its ONNX file path")
    onnx_file = (bundle / model["file"]).resolve()
    if not onnx_file.is_relative_to(bundle) or onnx_file.suffix != ".onnx":
        raise ValueError("Model manifest must point to an ONNX file inside the bundle")
    return {
        "id": MODEL_ID,
        "version": model.get("version"),
        "license": model.get("license"),
        "bundle": str(bundle),
        "manifest_sha256": sha256(manifest),
        "onnx_sha256": sha256(onnx_file),
    }


def box_area(box):
    left, top, right, bottom = box
    return (right - left) * (bottom - top)


def box_iou(left, right):
    width = min(left[2], right[2]) - max(left[0], right[0])
    height = min(left[3], right[3]) - max(left[1], right[1])
    intersection = max(0, width) * max(0, height)
    return intersection / (box_area(left) + box_area(right) - intersection)


def process_image(detector, path, row, output, index):
    source = row["local_path"] if row else str(path)
    record = {
        "index": index,
        "source": source,
        "photo_id": row["photo_id"] if row else None,
        "observation_uuid": row["observation_uuid"] if row else None,
        "split": row["split"] if row else None,
        "class_id": int(row["class_id"]) if row else None,
        "species": row["scientific_name"] if row else None,
        "group": row["group"] if row else None,
        "license": row["license"] if row else None,
        "original_size": None,
        "oriented_size": None,
        "exif_orientation": None,
        "inference_succeeded": False,
        "detection_count": None,
        "detections": [],
        "max_box_iou": None,
        "min_box_area_fraction": None,
        "preview_path": None,
        "error": None,
    }
    stage = "input"
    try:
        if row and sha256(path) != row["file_sha256"]:
            raise ValueError("Image SHA-256 does not match the frozen manifest")
        with Image.open(path) as original:
            record["original_size"] = original.size
            record["exif_orientation"] = original.getexif().get(274)
            image = ImageOps.exif_transpose(original).convert("RGB")
        record["oriented_size"] = image.size

        stage = "inference"
        detections = detector.detect(image)
        record["inference_succeeded"] = True
        record["detection_count"] = len(detections)
        record["detections"] = [
            {"label": item.category, "confidence": item.confidence,
             "box_xyxy": item.box, "crop_path": None}
            for item in detections
        ]
        boxes = [item.box for item in detections]
        if boxes:
            record["min_box_area_fraction"] = min(map(box_area, boxes)) / (image.width * image.height)
        if len(boxes) > 1:
            record["max_box_iou"] = max(box_iou(a, b) for a, b in combinations(boxes, 2))

        stage = "artifacts"
        stem = f"{index:04d}"
        preview = image.copy()
        draw = ImageDraw.Draw(preview)
        for number, detection in enumerate(record["detections"], start=1):
            box = detection["box_xyxy"]
            crop = crop_animal(image, box)
            if crop is None:
                raise ValueError(f"Detection {number} did not yield a valid crop")
            crop_path = f"crops/{stem}-{number:02d}.png"
            crop.save(output / crop_path)
            detection["crop_path"] = crop_path
            left, top, right, bottom = box
            draw.rectangle((left, top, right - 1, bottom - 1), outline="#ffdf38", width=2)
            draw.text((left + 2, top + 1), f"{number}: {detection['confidence']:.2f}",
                      fill="#ffdf38", stroke_width=1, stroke_fill="black")
        preview_path = f"previews/{stem}.jpg"
        preview.save(output / preview_path, quality=90)
        record["preview_path"] = preview_path
    except Exception as exc:
        record["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    return record


def coverage(records):
    successful = [row for row in records if row["inference_succeeded"]]
    attempted = len(records)
    inferred = len(successful)
    with_detections = sum(row["detection_count"] > 0 for row in successful)
    without_detections = inferred - with_detections
    multiple = sum(row["detection_count"] > 1 for row in successful)
    failures = sum(row["error"] is not None for row in records)

    def percent(count, total):
        return round(100 * count / total, 2) if total else None

    return {
        "attempted": attempted,
        "processed_successfully": attempted - failures,
        "processed_successfully_pct": percent(attempted - failures, attempted),
        "inference_succeeded": inferred,
        "with_detections": with_detections,
        "coverage_pct": percent(with_detections, inferred),
        "no_detections": without_detections,
        "no_detections_pct": percent(without_detections, inferred),
        "multiple_detections": multiple,
        "multiple_detections_pct": percent(multiple, inferred),
        "processing_failures": failures,
        "processing_failures_pct": percent(failures, attempted),
    }


def summarize(records):
    groups, species = defaultdict(list), defaultdict(list)
    for row in records:
        if row["group"] is not None:
            groups[row["group"]].append(row)
            species[row["species"]].append(row)
    return {
        "coverage_denominator": "Images with successful inference",
        "processing_denominator": "Images attempted",
        "overall": coverage(records),
        "by_group": {name: coverage(rows) for name, rows in sorted(groups.items())},
        "by_species": {name: coverage(rows) for name, rows in sorted(species.items())},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, action="extend", nargs="+",
                        help="Explicit image paths, relative to the current directory or absolute")
    inputs.add_argument("--split", choices=SPLITS, help="Sample this frozen dataset split")
    parser.add_argument("--sample-per-class", type=int, default=4, help="Manifest sample size per class")
    parser.add_argument("--seed", type=int, default=42, help="Manifest sampling seed")
    parser.add_argument("--threshold", type=float, default=0.20, help="Minimum animal confidence")
    parser.add_argument("--model-bundle", type=Path, required=True,
                        help=f"Unpacked local {MODEL_ID} directory containing manifest.toml")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New run directory under outputs/detection/")
    args = parser.parse_args(argv)
    if args.sample_per_class < 1 or not 0 <= args.threshold <= 1:
        parser.error("--sample-per-class must be positive and --threshold must be between 0 and 1")

    output = output_directory(args.output_dir)
    allowed, manifest_sha256 = load_manifest()
    paths = args.image
    if args.split:
        rows = sample_rows(allowed.values(), args.split, args.sample_per_class, args.seed)
        paths = [ROOT / row["local_path"] for row in rows]
    images = [resolve_image(path, allowed) for path in paths]
    bundle = args.model_bundle.resolve()
    model = model_metadata(bundle)

    import sparrow_engine

    output.mkdir(parents=True, exist_ok=False)
    (output / "crops").mkdir()
    (output / "previews").mkdir()
    sparrow_engine.init(device="cpu", model_dir=str(bundle.parent))
    device = sparrow_engine.active_device()
    if device != "cpu":
        raise RuntimeError(f"Expected CPU inference, but SPARROW-Engine reported {device}")
    detector = MegaDetector(confidence_threshold=args.threshold)
    print(f"Running {MODEL_ID} on {device} at {args.threshold:g} for {len(images)} images", flush=True)
    results = []
    for index, (path, row) in enumerate(images, start=1):
        result = process_image(detector, path, row, output, index)
        results.append(result)
        if result["error"]:
            print(f"{result['source']}: {result['error']['message']}", file=sys.stderr)
        if index % 10 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)} images", flush=True)

    report = {
        "model": {**model, "sparrow_engine_version": sparrow_engine.__version__, "device": device},
        "threshold": args.threshold,
        "box_format": "integer XYXY in oriented source pixels, right and bottom exclusive",
        "manifest_sha256": manifest_sha256,
        "selection": {"split": args.split, "seed": args.seed, "sample_per_class": args.sample_per_class}
        if args.split else {"mode": "explicit"},
        "images": results,
    }
    summary = summarize(results)
    for filename, payload in (("results.json", report), ("summary.json", summary)):
        with (output / filename).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(f"Wrote results and coverage summary to {output}")
    return int(summary["overall"]["processing_failures"] > 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
